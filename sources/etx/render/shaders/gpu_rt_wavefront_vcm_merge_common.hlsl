#pragma once

#include "gpu_rt_wavefront_common.hlsl"
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

void wavefront_vcm_merge(uint dispatch_index) {
  GPUWavefrontResources resources = (GPUWavefrontResources)0;
  uint path_index = 0u;
  GPUWavefrontPathState state = (GPUWavefrontPathState)0;
  GPUWavefrontHit hit = (GPUWavefrontHit)0;
  GPUWavefrontPathVertex camera_vertex = (GPUWavefrontPathVertex)0;
  Material material = (Material)0;
  if (wavefront_vcm_merge_load_input(dispatch_index, resources, path_index, state, hit, camera_vertex, material) == false ||
      (wavefront_vcm_merge_stage_matches_material(material.cls) == false)) {
    return;
  }

  float cell_size = 2.0f * constants.vcm_radius;
  float3 grid_position = camera_vertex.position / cell_size;
  float3 base_float = floor(grid_position);
  float3 fraction = grid_position - base_float;
  int3 base_cell = int3(base_float);
  int3 adjacent = base_cell + int3(fraction.x < 0.5f ? -1 : 1, fraction.y < 0.5f ? -1 : 1, fraction.z < 0.5f ? -1 : 1);
  int3 cells[8] = {int3(base_cell.x, base_cell.y, base_cell.z), int3(adjacent.x, base_cell.y, base_cell.z), int3(base_cell.x, adjacent.y, base_cell.z),
    int3(adjacent.x, adjacent.y, base_cell.z), int3(base_cell.x, base_cell.y, adjacent.z), int3(adjacent.x, base_cell.y, adjacent.z), int3(base_cell.x, adjacent.y, adjacent.z),
    int3(adjacent.x, adjacent.y, adjacent.z)};
  uint cell_hashes[8];

  BSDFResourceContext bsdf_context = make_bsdf_resource_gpu_context(constants.scene.images, constants.scene.spectrums, constants.scene.spectral_values,
    constants.scene.energy_compensation_interfaces, constants.scene.scene_globals);
  BSDFData camera_data = bsdf_data_make(hit.vertex, state.spect, hit.medium_index, PathSource::Camera, camera_vertex.w_i);
  Sampler sampler = (Sampler)0;
  sampler.seed = state.sampler_seed;
#if ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE
  const bool use_prepared_diffuse = material.cls == MaterialClass::Diffuse;
  LocalFrame diffuse_frame = (LocalFrame)0;
  float3 diffuse_local_w_i = float3(0.0f, 0.0f, 0.0f);
  SpectralResponse diffuse_albedo = spectral_response_zero(state.spect);
  float diffuse_roughness = 0.0f;
  if (use_prepared_diffuse) {
    diffuse_frame = bsdf_data_get_normal_frame(camera_data, material);
    diffuse_local_w_i = local_frame_to_local(diffuse_frame, -camera_data.w_i);
    diffuse_albedo = bsdf_resource_apply_image(bsdf_context, camera_data.spectrum_sample, material.scattering, camera_data.tex);
    diffuse_roughness = bsdf_diffuse_scalar_roughness(bsdf_context, material, camera_data.tex);
  }
#endif
  float radius_squared = constants.vcm_radius * constants.vcm_radius;
  float inv_radius_squared = 1.0f / radius_squared;
  float3 merged = float3(0.0f, 0.0f, 0.0f);
#if ETX_SPECTRAL_MODE == ETX_SPECTRAL_MODE_SPECTRAL
  const float3 spectral_estimate_scale = wavefront_spectral_estimate(spectral_response_make(state.spect, 1.0f), state.spect);
#endif
  ByteAddressBuffer heads = WAVEFRONT_RO_BUFFER(resources.vcm_grid_heads_buffer);
  ByteAddressBuffer next_indices = WAVEFRONT_RO_BUFFER(resources.vcm_grid_next_buffer);
  ByteAddressBuffer light_vertices = WAVEFRONT_RO_BUFFER(resources.light_vertex_buffer);

  [unroll] for (uint cell_offset = 0u; cell_offset < 8u; ++cell_offset) {
    uint cell_hash = wavefront_vcm_merge_hash_cell(cells[cell_offset]);
    cell_hashes[cell_offset] = cell_hash;
    bool duplicate_hash = false;
    [unroll] for (uint previous_offset = 0u; previous_offset < cell_offset; ++previous_offset) {
      duplicate_hash = duplicate_hash || (cell_hashes[previous_offset] == cell_hash);
    }
    if (duplicate_hash) {
      continue;
    }
    uint light_index = heads.Load(cell_hash * 4u);
    uint visited = 0u;
    while ((light_index != kInvalidIndex) && (light_index < constants.vcm_light_vertex_count) && (visited < resources.light_vertex_capacity)) {
      uint next_index = next_indices.Load(light_index * 4u);
      visited += 1u;
      const uint light_vertex_offset = light_index * kGPUWavefrontLightPathVertexStride;
      // CPU VCM numbers the first light-surface vertex as depth zero, while
      // the shared GPU BDPT history numbers it as depth one (after the emitter
      // root). Translate only for VCM's combined merge-path depth test.
      const uint packed_path_and_flags = light_vertices.Load(light_vertex_offset + kGPUWavefrontLightPathVertexPackedPathAndFlagsOffset);
      const uint light_path_length = wavefront_unpack_light_path_vertex_path_length(packed_path_and_flags);
      const uint vcm_light_path_length = (light_path_length > 0u) ? (light_path_length - 1u) : 0u;
      if ((vcm_light_path_length + state.path_length + 1u) <= resources.max_path_length) {
        const float3 light_position = wavefront_load_float3(light_vertices, light_vertex_offset + kGPUWavefrontLightPathVertexPositionOffset);
        const float3 delta = light_position - camera_vertex.position;
        const float distance_squared = dot(delta, delta);
        if (distance_squared <= radius_squared) {
          const float3 light_normal = wavefront_load_float3(light_vertices, light_vertex_offset + kGPUWavefrontLightPathVertexNormalOffset);
          if (dot(camera_vertex.normal, light_normal) > kEpsilon) {
            GPUWavefrontPathVertex light_vertex = wavefront_load_path_vertex(resources.light_vertex_buffer, light_index);
            const bool query_matches = spectral_query_compatible(spectral_response_as_query(light_vertex.throughput), state.spect);
            if (wavefront_path_vertex_valid(light_vertex) && wavefront_path_vertex_is_surface(light_vertex) && wavefront_path_vertex_connectible(light_vertex) && query_matches) {
              float3 outgoing_direction = -light_vertex.w_i;
              BSDFEval camera_eval = (BSDFEval)0;
#if ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE
              if (use_prepared_diffuse) {
                const float3 diffuse_local_w_o = local_frame_to_local(diffuse_frame, outgoing_direction);
                if ((diffuse_local_w_i.z > kEpsilon) && (diffuse_local_w_o.z > kEpsilon)) {
                  camera_eval.func = bsdf_diffuse_eon_brdf(camera_data.spectrum_sample, diffuse_albedo, diffuse_local_w_i, diffuse_local_w_o, diffuse_roughness);
                  camera_eval.pdf = kInvPi * diffuse_local_w_o.z;
                  camera_eval.eta = 1.0f;
                }
              } else
#endif
              {
                camera_eval = wavefront_vcm_merge_stage_bsdf_eval(bsdf_context, camera_data, outgoing_direction, material, sampler);
              }
              if (bsdf_eval_valid(camera_eval)) {
                float reverse_pdf = wavefront_vcm_merge_stage_reverse_pdf(bsdf_context, camera_data, outgoing_direction, material, sampler);
                float w_light = light_vertex.forward_pdf * constants.vcm_vc_weight + light_vertex.d_vm * camera_eval.pdf;
                float w_camera = camera_vertex.forward_pdf * constants.vcm_vc_weight + camera_vertex.d_vm * reverse_pdf;
                float mis_weight = scene_multiple_importance_sampling_enabled() ? (1.0f / (1.0f + w_light + w_camera)) : 1.0f;
                float kernel_weight = 1.0f;
                if (constants.vcm_kernel != 0u) {
                  kernel_weight = max(2.0f * (1.0f - distance_squared * inv_radius_squared), 0.0f);
                }
#if ETX_SPECTRAL_MODE == ETX_SPECTRAL_MODE_RGB
                const float3 estimate = camera_eval.func.integrated * (camera_vertex.throughput.integrated * light_vertex.throughput.integrated);
#elif ETX_SPECTRAL_MODE == ETX_SPECTRAL_MODE_SPECTRAL
                const float3 estimate = spectral_estimate_scale * (camera_eval.func.value * (camera_vertex.throughput.value * light_vertex.throughput.value));
#else
                const SpectralResponse value = spectral_response_mul(camera_eval.func, spectral_response_mul(camera_vertex.throughput, light_vertex.throughput));
                const float3 estimate = wavefront_spectral_estimate(value, state.spect);
#endif
                merged += estimate * (kernel_weight * mis_weight * constants.vcm_vm_normalization);
              }
            }
          }
        }
      }
      light_index = next_index;
    }
  }
  if (all(isfinite(merged)) && any(merged != float3(0.0f, 0.0f, 0.0f))) {
    wavefront_film_add(camera_vertex.pixel_index, merged);
  }
}
