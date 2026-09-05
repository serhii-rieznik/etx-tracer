#pragma once

#include "gpu_rt_wavefront_common.hlsl"
#include "gpu_rt_wavefront_vcm_grid_shared.hlsl"
#include <interop/image_filter_shared.hxx>
#include <access/bsdf_resource_gpu.hxx>
#include <access/material_access_gpu.hxx>

uint wavefront_vcm_merge_hash_cell(int3 cell) {
  return ((uint(cell.x) * 73856093u) ^ (uint(cell.y) * 19349663u) ^ (uint(cell.z) * 83492791u)) & constants.vcm_grid_mask;
}

bool wavefront_vcm_merge_load_input(uint dispatch_index, out GPUWavefrontResources resources, out uint path_index, out GPUWavefrontPathState state, out GPUWavefrontHit hit,
  out GPUWavefrontPathVertex camera_vertex, out Material material) {
  resources = wavefront_load_resources();
  path_index = 0u;
  state = (GPUWavefrontPathState)0;
  hit = (GPUWavefrontHit)0;
  camera_vertex = (GPUWavefrontPathVertex)0;
  material = (Material)0;
  if ((resources.vcm_grid_heads_buffer == kInvalidIndex) || (resources.vcm_grid_next_buffer == kInvalidIndex) || (constants.vcm_radius <= 0.0f)) {
    return false;
  }
  if ((constants.dispatch_item_count != 0u) && (dispatch_index >= constants.dispatch_item_count)) {
    return false;
  }
  dispatch_index += constants.dispatch_item_offset;
#if ETX_ENABLE_WORK_QUEUES
  uint material_queue_count = wavefront_material_queue_count(resources, true, constants.work_queue_index);
  if (dispatch_index >= material_queue_count) {
    return false;
  }
  dispatch_index = wavefront_material_queue_load(resources, true, constants.work_queue_index, dispatch_index);
#endif
  uint queue_descriptor = wavefront_queue_current_descriptor(true);
  if (dispatch_index >= wavefront_queue_count(queue_descriptor)) {
    return false;
  }
  path_index = wavefront_queue_load(queue_descriptor, dispatch_index);
  state = wavefront_load_path_state(resources.camera_state_buffer, path_index);
  hit = wavefront_load_hit(resources.camera_hit_buffer, path_index);
  if ((wavefront_path_state_valid(state) == false) || (wavefront_hit_valid(hit) == false) || wavefront_hit_is_miss(hit) || wavefront_hit_is_medium(hit)) {
    return false;
  }
  camera_vertex = wavefront_load_path_vertex(resources.camera_vertex_buffer, wavefront_camera_vertex_slot(path_index, state.path_length));
  if ((wavefront_path_vertex_valid(camera_vertex) == false) || (wavefront_path_vertex_is_surface(camera_vertex) == false) ||
      (wavefront_path_vertex_connectible(camera_vertex) == false)) {
    return false;
  }
  MaterialAccessGPUContext material_context = {constants.scene.materials};
  return material_access_try_load_full(material_context, camera_vertex.material_index, material);
}

struct WavefrontVCMMergeQuery {
  GPUWavefrontResources resources;
  GPUWavefrontPathVertex camera_vertex;
  Material material;
  BSDFResourceContext bsdf_context;
  BSDFData camera_data;
  uint path_length;
  uint sampler_seed;
  float radius_squared;
  float inv_radius_squared;
#if ETX_SPECTRAL_MODE == ETX_SPECTRAL_MODE_SPECTRAL
  float3 spectral_estimate_scale;
#endif
#if ((ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR) || (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC)) && (ETX_SPECTRAL_MODE != ETX_SPECTRAL_MODE_RUNTIME)
  bool use_prepared_material;
  BSDFEnergyCompensatedPreparedMaterial prepared_material;
# if ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR
  LocalFrame conductor_frame;
  float3 conductor_local_w_i;
  bool use_prepared_conductor;
  BSDFEnergyCompensatedConductorIncident conductor_incident;
  SpectralResponse conductor_reflectance;
# endif
#endif
#if ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE
  bool use_prepared_diffuse;
  LocalFrame diffuse_frame;
  float3 diffuse_local_w_i;
  SpectralResponse diffuse_albedo;
  float diffuse_roughness;
#endif
};

bool wavefront_vcm_merge_prepare(uint dispatch_index, out WavefrontVCMMergeQuery query) {
  query = (WavefrontVCMMergeQuery)0;
  uint path_index = 0u;
  GPUWavefrontPathState state = (GPUWavefrontPathState)0;
  GPUWavefrontHit hit = (GPUWavefrontHit)0;
  if ((wavefront_vcm_merge_load_input(dispatch_index, query.resources, path_index, state, hit, query.camera_vertex, query.material) == false) ||
      (wavefront_vcm_merge_stage_matches_material(query.material.cls) == false)) {
    return false;
  }
  query.path_length = state.path_length;
  query.bsdf_context = make_bsdf_resource_gpu_context(constants.scene.images, constants.scene.spectrums, constants.scene.spectral_values,
    constants.scene.energy_compensation_interfaces, constants.scene.scene_globals);
  query.camera_data = bsdf_data_make(hit.vertex, state.spect, hit.medium_index, PathSource::Camera, query.camera_vertex.w_i);
  Sampler sampler = (Sampler)0;
  sampler.seed = state.sampler_seed;
#if ((ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR) || (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC)) && (ETX_SPECTRAL_MODE != ETX_SPECTRAL_MODE_RUNTIME)
  query.use_prepared_material = (query.material.cls == MaterialClass::Conductor) || (query.material.cls == MaterialClass::Dielectric);
  query.prepared_material = (BSDFEnergyCompensatedPreparedMaterial)0;
  if (query.use_prepared_material) {
    query.prepared_material = bsdf_energy_compensated_prepare_material(query.bsdf_context, query.camera_data.spectrum_sample, query.material, query.camera_data.tex, sampler);
  }
# if ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR
  query.conductor_frame = bsdf_data_get_normal_frame(query.camera_data, query.material);
  query.conductor_local_w_i = local_frame_to_local(query.conductor_frame, -query.camera_data.w_i);
  query.use_prepared_conductor = query.use_prepared_material && (query.prepared_material.conductor_delta == false) && (query.conductor_local_w_i.z > kEpsilon) &&
                                 bsdf_energy_compensated_material_interface_valid(query.bsdf_context, query.material, MaterialClass::Conductor);
  query.conductor_incident = (BSDFEnergyCompensatedConductorIncident)0;
  query.conductor_reflectance = spectral_response_zero(state.spect);
  if (query.use_prepared_conductor) {
    query.conductor_incident =
      bsdf_energy_compensated_conductor_prepare_incident(query.bsdf_context, state.spect, query.material, query.conductor_local_w_i.z, query.prepared_material);
    query.conductor_reflectance = bsdf_resource_apply_image(query.bsdf_context, state.spect, query.material.reflectance, query.camera_data.tex);
  }
# endif
#endif
#if ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE
  query.use_prepared_diffuse = query.material.cls == MaterialClass::Diffuse;
  query.diffuse_frame = (LocalFrame)0;
  query.diffuse_local_w_i = float3(0.0f, 0.0f, 0.0f);
  query.diffuse_albedo = spectral_response_zero(state.spect);
  query.diffuse_roughness = 0.0f;
  if (query.use_prepared_diffuse) {
    query.diffuse_frame = bsdf_data_get_normal_frame(query.camera_data, query.material);
    query.diffuse_local_w_i = local_frame_to_local(query.diffuse_frame, -query.camera_data.w_i);
    query.diffuse_albedo = bsdf_resource_apply_image(query.bsdf_context, query.camera_data.spectrum_sample, query.material.scattering, query.camera_data.tex);
    query.diffuse_roughness = bsdf_diffuse_scalar_roughness(query.bsdf_context, query.material, query.camera_data.tex);
  }
#endif
  query.radius_squared = constants.vcm_radius * constants.vcm_radius;
  query.inv_radius_squared = 1.0f / query.radius_squared;
#if ETX_SPECTRAL_MODE == ETX_SPECTRAL_MODE_SPECTRAL
  query.spectral_estimate_scale = wavefront_spectral_estimate(spectral_response_make(state.spect, 1.0f), state.spect);
#endif
  query.sampler_seed = sampler.seed;
  return true;
}

uint wavefront_vcm_merge_cell_hash(float3 position, uint cell_index) {
  float3 grid_position = position / (2.0f * constants.vcm_radius);
  float3 base_float = floor(grid_position);
  float3 fraction = grid_position - base_float;
  int3 base_cell = int3(base_float);
  int3 adjacent = base_cell + int3(fraction.x < 0.5f ? -1 : 1, fraction.y < 0.5f ? -1 : 1, fraction.z < 0.5f ? -1 : 1);
  int3 cell = int3((cell_index & 1u) != 0u ? adjacent.x : base_cell.x, (cell_index & 2u) != 0u ? adjacent.y : base_cell.y, (cell_index & 4u) != 0u ? adjacent.z : base_cell.z);
  return wavefront_vcm_merge_hash_cell(cell);
}

bool wavefront_vcm_merge_candidate_matches(WavefrontVCMMergeQuery query, uint light_index, out float distance_squared) {
  distance_squared = 0.0f;
  ByteAddressBuffer light_vertices = WAVEFRONT_RO_BUFFER(query.resources.light_vertex_buffer);
  const uint light_vertex_offset = light_index * kGPUWavefrontLightPathVertexStride;
  const uint packed_path_and_flags = light_vertices.Load(light_vertex_offset + kGPUWavefrontLightPathVertexPackedPathAndFlagsOffset);
  const uint light_path_length = wavefront_unpack_light_path_vertex_path_length(packed_path_and_flags);
  // VCM counts the first light-surface vertex as depth zero; GPU history includes the emitter root.
  const uint vcm_light_path_length = (light_path_length > 0u) ? (light_path_length - 1u) : 0u;
  if ((vcm_light_path_length + query.path_length + 1u) > query.resources.max_path_length) {
    return false;
  }
  const float3 light_position = wavefront_load_float3(light_vertices, light_vertex_offset + kGPUWavefrontLightPathVertexPositionOffset);
  const float3 delta = light_position - query.camera_vertex.position;
  distance_squared = dot(delta, delta);
  if ((distance_squared <= query.radius_squared) == false) {
    return false;
  }
  const float3 light_normal = wavefront_load_float3(light_vertices, light_vertex_offset + kGPUWavefrontLightPathVertexNormalOffset);
  return dot(query.camera_vertex.normal, light_normal) > kEpsilon;
}
float3 wavefront_vcm_merge_evaluate(WavefrontVCMMergeQuery query, uint light_index, float distance_squared, inout Sampler sampler) {
  GPUWavefrontPathVertex light_vertex = wavefront_load_path_vertex(query.resources.light_vertex_buffer, light_index);
  const bool query_matches = spectral_query_compatible(spectral_response_as_query(light_vertex.throughput), query.camera_data.spectrum_sample);
  if ((wavefront_path_vertex_valid(light_vertex) == false) || (wavefront_path_vertex_is_surface(light_vertex) == false) ||
      (wavefront_path_vertex_connectible(light_vertex) == false) || (query_matches == false)) {
    return float3(0.0f, 0.0f, 0.0f);
  }
  float3 outgoing_direction = -light_vertex.w_i;
  BSDFEval camera_eval = (BSDFEval)0;
#if ((ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR) || (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC)) && (ETX_SPECTRAL_MODE != ETX_SPECTRAL_MODE_RUNTIME)
  if (query.use_prepared_material) {
# if ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR
    if (query.use_prepared_conductor) {
      const float3 conductor_local_w_o = local_frame_to_local(query.conductor_frame, outgoing_direction);
      camera_eval = bsdf_conductor_energy_compensated_evaluate_prepared_local(query.bsdf_context, query.camera_data, query.material, query.conductor_local_w_i, conductor_local_w_o,
        query.prepared_material, query.conductor_reflectance, query.conductor_incident);
    } else {
      camera_eval = bsdf_conductor_energy_compensated_evaluate_prepared(query.bsdf_context, query.camera_data, outgoing_direction, query.material, query.prepared_material);
    }
# else
    camera_eval = bsdf_dielectric_energy_compensated_evaluate_prepared(query.bsdf_context, query.camera_data, outgoing_direction, query.material, query.prepared_material);
# endif
  } else
#elif ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE
  if (query.use_prepared_diffuse) {
    const float3 diffuse_local_w_o = local_frame_to_local(query.diffuse_frame, outgoing_direction);
    if ((query.diffuse_local_w_i.z > kEpsilon) && (diffuse_local_w_o.z > kEpsilon)) {
      camera_eval.func = bsdf_diffuse_eon_brdf(query.camera_data.spectrum_sample, query.diffuse_albedo, query.diffuse_local_w_i, diffuse_local_w_o, query.diffuse_roughness);
      camera_eval.pdf = kInvPi * diffuse_local_w_o.z;
      camera_eval.eta = 1.0f;
    }
  } else
#endif
  {
    camera_eval = wavefront_vcm_merge_stage_bsdf_eval(query.bsdf_context, query.camera_data, outgoing_direction, query.material, sampler);
  }
  if (bsdf_eval_valid(camera_eval)) {
    float reverse_pdf = 0.0f;
#if ((ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR) || (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC)) && (ETX_SPECTRAL_MODE != ETX_SPECTRAL_MODE_RUNTIME)
    if (query.use_prepared_material) {
      BSDFData reverse_data = query.camera_data;
      reverse_data.w_i = -outgoing_direction;
      reverse_data.path_source = PathSource::Light;
# if ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR
      reverse_pdf = bsdf_conductor_energy_compensated_pdf_prepared(query.bsdf_context, reverse_data, -query.camera_data.w_i, query.material, query.prepared_material);
# else
      reverse_pdf = bsdf_dielectric_energy_compensated_pdf_prepared(query.bsdf_context, reverse_data, -query.camera_data.w_i, query.material, query.prepared_material);
# endif
    } else
#endif
    {
      reverse_pdf = wavefront_vcm_merge_stage_reverse_pdf(query.bsdf_context, query.camera_data, outgoing_direction, query.material, sampler);
    }
    float w_light = light_vertex.forward_pdf * constants.vcm_vc_weight + light_vertex.d_vm * camera_eval.pdf;
    float w_camera = query.camera_vertex.forward_pdf * constants.vcm_vc_weight + query.camera_vertex.d_vm * reverse_pdf;
    float mis_weight = scene_multiple_importance_sampling_enabled() ? (1.0f / (1.0f + w_light + w_camera)) : 1.0f;
    float kernel_weight = 1.0f;
    if (constants.vcm_kernel != 0u) {
      kernel_weight = max(2.0f * (1.0f - distance_squared * query.inv_radius_squared), 0.0f);
    }
#if ETX_SPECTRAL_MODE == ETX_SPECTRAL_MODE_RGB
    const float3 estimate = camera_eval.func.integrated * (query.camera_vertex.throughput.integrated * light_vertex.throughput.integrated);
#elif ETX_SPECTRAL_MODE == ETX_SPECTRAL_MODE_SPECTRAL
    const float3 estimate = query.spectral_estimate_scale * (camera_eval.func.value * (query.camera_vertex.throughput.value * light_vertex.throughput.value));
#else
    const SpectralResponse value = spectral_response_mul(camera_eval.func, spectral_response_mul(query.camera_vertex.throughput, light_vertex.throughput));
    const float3 estimate = wavefront_spectral_estimate(value, query.camera_data.spectrum_sample);
#endif
    return estimate * (kernel_weight * mis_weight * constants.vcm_vm_normalization);
  }
  return float3(0.0f, 0.0f, 0.0f);
}

#if ETX_VCM_COOPERATIVE_MERGE
groupshared float3 vcm_merge_contributions[64];

void wavefront_vcm_merge_group(uint dispatch_index, uint lane_index) {
  WavefrontVCMMergeQuery query;
  if (wavefront_vcm_merge_prepare(dispatch_index, query) == false) {
    return;
  }
  const uint cell_index = lane_index / 8u;
  const uint cell_hash = wavefront_vcm_merge_cell_hash(query.camera_vertex.position, cell_index);
  bool duplicate_hash = false;
  [unroll] for (uint previous = 0u; previous < 8u; ++previous) {
    if (previous < cell_index) {
      duplicate_hash = duplicate_hash || (wavefront_vcm_merge_cell_hash(query.camera_vertex.position, previous) == cell_hash);
    }
  }
  const uint count = duplicate_hash ? 0u : wavefront_vcm_grid_range_count(query.resources, cell_hash);
  const uint begin = count > 0u ? wavefront_vcm_grid_range_start(query.resources, cell_hash) : 0u;
  Sampler sampler = (Sampler)0;
  sampler.seed = query.sampler_seed;
  float3 merged = float3(0.0f, 0.0f, 0.0f);
  for (uint local_index = lane_index % 8u; local_index < count; local_index += 8u) {
    const uint light_index = WAVEFRONT_RO_BUFFER(query.resources.vcm_grid_next_buffer).Load((begin + local_index) * sizeof(uint));
    float distance_squared = 0.0f;
    if (wavefront_vcm_merge_candidate_matches(query, light_index, distance_squared)) {
      merged += wavefront_vcm_merge_evaluate(query, light_index, distance_squared, sampler);
    }
  }
  vcm_merge_contributions[lane_index] = merged;
  GroupMemoryBarrierWithGroupSync();
  if (lane_index == 0u) {
    float3 total = float3(0.0f, 0.0f, 0.0f);
    [unroll] for (uint contribution_index = 0u; contribution_index < 64u; ++contribution_index) {
      total += vcm_merge_contributions[contribution_index];
    }
    if (all(isfinite(total)) && any(total != float3(0.0f, 0.0f, 0.0f))) {
      wavefront_film_add(query.camera_vertex.pixel_index, total);
    }
  }
}
#else
void wavefront_vcm_merge(uint dispatch_index) {
  WavefrontVCMMergeQuery query;
  if (wavefront_vcm_merge_prepare(dispatch_index, query) == false) {
    return;
  }
  Sampler sampler = (Sampler)0;
  sampler.seed = query.sampler_seed;
  float3 merged = float3(0.0f, 0.0f, 0.0f);
  uint cell_hashes[8];
  [unroll] for (uint cell_offset = 0u; cell_offset < 8u; ++cell_offset) {
    const uint cell_hash = wavefront_vcm_merge_cell_hash(query.camera_vertex.position, cell_offset);
    cell_hashes[cell_offset] = cell_hash;
    bool duplicate_hash = false;
    [unroll] for (uint previous_offset = 0u; previous_offset < cell_offset; ++previous_offset) {
      duplicate_hash = duplicate_hash || (cell_hashes[previous_offset] == cell_hash);
    }
    if (duplicate_hash) {
      continue;
    }
    const uint count = wavefront_vcm_grid_range_count(query.resources, cell_hash);
    const uint begin = count > 0u ? wavefront_vcm_grid_range_start(query.resources, cell_hash) : 0u;
    for (uint local_index = 0u; local_index < count; ++local_index) {
      const uint light_index = WAVEFRONT_RO_BUFFER(query.resources.vcm_grid_next_buffer).Load((begin + local_index) * sizeof(uint));
      float distance_squared = 0.0f;
      if (wavefront_vcm_merge_candidate_matches(query, light_index, distance_squared)) {
        merged += wavefront_vcm_merge_evaluate(query, light_index, distance_squared, sampler);
      }
    }
  }
  if (all(isfinite(merged)) && any(merged != float3(0.0f, 0.0f, 0.0f))) {
    wavefront_film_add(query.camera_vertex.pixel_index, merged);
  }
}
#endif
