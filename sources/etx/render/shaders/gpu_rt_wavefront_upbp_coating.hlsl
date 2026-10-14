#pragma once

#include "gpu_rt_wavefront_coating_storage.hlsl"
#include "gpu_rt_wavefront_vcm_grid_shared.hlsl"

struct UPBPCoatingQuery {
  GPUWavefrontResources wavefront;
  GPUUPBPResources resources;
  GPUUPBPVertex camera;
  UPBPSurfacePreparedQuery prepared;
};

bool upbp_coating_prepare_query(uint dispatch_index, out UPBPCoatingQuery query) {
  query = (UPBPCoatingQuery)0;
  query.wavefront = wavefront_load_resources();
  query.resources = upbp_load_resources(query.wavefront);
  const GPUUPBPResources resources = query.resources;
  if ((resources.counter_buffer == kInvalidIndex) || (resources.surface_grid_ranges_buffer == kInvalidIndex) || (resources.density_output_surface_point_buffer == kInvalidIndex) ||
      (resources.iteration.surface_radius <= 0.0f)) {
    return false;
  }
  uint vertex_index = constants.dispatch_item_offset + dispatch_index;
  if (constants.work_queue_index == GPUUPBPDensityQueryMode::Compacted) {
    const uint query_family = ETX_UPBP_SURFACE_QUERY_FAMILY;
    const uint query_count = WAVEFRONT_RO_BUFFER(resources.counter_buffer).Load((GPUUPBPCounterIndex::CameraSurfaceVariousQuery + query_family) * sizeof(uint));
    vertex_index = ((vertex_index < query_count) && (resources.point_buffer != kInvalidIndex))
                     ? WAVEFRONT_RO_BUFFER(resources.point_buffer).Load((query_family * resources.point_capacity + vertex_index) * sizeof(uint))
                     : kInvalidIndex;
  }
  const uint vertex_count = min(WAVEFRONT_RO_BUFFER(resources.counter_buffer).Load(GPUUPBPCounterIndex::CameraVertex * sizeof(uint)), resources.camera_vertex_capacity);
  if (vertex_index >= vertex_count) {
    return false;
  }
  query.camera = upbp_load_vertex(resources.vertex_buffer, vertex_index);
  if (((query.camera.flags & GPUUPBPVertexFlags::Valid) == 0u) || (query.camera.path_length == 0u) || (upbp_vertex_is_density_connectible(query.camera) == false)) {
    return false;
  }
  query.prepared = upbp_prepare_surface_vertex(query.camera);
  return query.prepared.valid;
}

bool upbp_coating_candidate(UPBPCoatingQuery query, GPUUPBPVertex light_vertex, out float kernel_value) {
  const GPUUPBPVertex camera_vertex = query.camera;
  kernel_value = 0.0f;
  if (((light_vertex.flags & GPUUPBPVertexFlags::Valid) == 0u) || (light_vertex.path_length == 0u) || (upbp_vertex_is_surface(light_vertex) == false) ||
      (upbp_vertex_is_surface(camera_vertex) == false) || upbp_vertex_is_delta(light_vertex) || upbp_vertex_is_delta(camera_vertex) ||
      (upbp_vertex_is_density_connectible(light_vertex) == false) || (upbp_vertex_is_density_connectible(camera_vertex) == false) ||
      (dot(light_vertex.geo_normal, camera_vertex.geo_normal) <= 0.0f)) {
    return false;
  }
  const uint path_length = light_vertex.path_length + camera_vertex.path_length;
  if ((path_length < load_scene_options_min_path_length()) || (path_length > load_scene_options_max_path_length())) {
    return false;
  }
  const SpectralResponse light_throughput = upbp_unpack_spectral_response(light_vertex.throughput);
  const SpectralResponse camera_throughput = upbp_unpack_spectral_response(camera_vertex.throughput);
  if (spectral_query_compatible(spectral_response_as_query(light_throughput), spectral_response_as_query(camera_throughput)) == false) {
    return false;
  }
  const float3 delta = light_vertex.position - camera_vertex.position;
  const float distance_squared = dot(delta, delta);
  kernel_value = upbp_kernel_value(query.resources.iteration.kernel, 2u, query.resources.iteration.surface_radius, distance_squared);
  if (kernel_value <= 0.0f) {
    return false;
  }
  return scene_math_shared_scattering_direction_valid(camera_vertex.normal, camera_vertex.geo_normal, camera_vertex.w_i, -light_vertex.w_i);
}

void upbp_coating_resolve(UPBPCoatingQuery query, GPUUPBPVertex light_vertex, float kernel_value, BSDFEval evaluation, inout Sampler sampler, inout SpectralResponse accumulated) {
  const GPUUPBPResources resources = query.resources;
  const GPUUPBPVertex camera_vertex = query.camera;
  const SpectralResponse light_throughput = upbp_unpack_spectral_response(light_vertex.throughput);
  const SpectralResponse camera_throughput = upbp_unpack_spectral_response(camera_vertex.throughput);
  const float3 outgoing_direction = -light_vertex.w_i;
  const SpectralResponse scattering_value = evaluation.func;
  const float pdf_forward = evaluation.pdf;
  const float pdf_reverse = upbp_surface_stage_reverse_pdf(query.prepared.context, query.prepared.data, outgoing_direction, query.prepared.material, sampler);
  if ((pdf_forward <= 0.0f) || (pdf_reverse <= 0.0f) || spectral_response_is_zero(scattering_value)) {
    return;
  }
  const float direction_cosine = dot(camera_vertex.w_i, outgoing_direction);
  const float sin_theta = sqrt(max(0.0f, 1.0f - direction_cosine * direction_cosine));
  const float mis_weight = upbp_point_merge_mis_weight(resources.iteration, GPUUPBPTechnique::Surface, light_vertex, camera_vertex, pdf_forward, pdf_reverse, sin_theta);
  if (mis_weight <= 0.0f) {
    return;
  }
  const float estimator_scale = kernel_value / float(resources.iteration.global_light_path_count);
  const SpectralResponse contribution =
    spectral_response_mul(spectral_response_mul(light_throughput, camera_throughput), spectral_response_mul(scattering_value, estimator_scale * mis_weight));
  accumulated = spectral_response_add(accumulated, contribution);
}

uint upbp_coating_cell_hash(float3 position, uint cell_index, float radius, uint mask) {
  const float3 grid_position = position / (2.0f * radius);
  const float3 base_float = floor(grid_position);
  const float3 fraction = grid_position - base_float;
  const int3 base_cell = int3(base_float);
  const int3 adjacent = base_cell + int3(fraction.x < 0.5f ? -1 : 1, fraction.y < 0.5f ? -1 : 1, fraction.z < 0.5f ? -1 : 1);
  const int3 cell =
    int3((cell_index & 1u) != 0u ? adjacent.x : base_cell.x, (cell_index & 2u) != 0u ? adjacent.y : base_cell.y, (cell_index & 4u) != 0u ? adjacent.z : base_cell.z);
  return ((uint(cell.x) * 73856093u) ^ (uint(cell.y) * 19349663u) ^ (uint(cell.z) * 83492791u)) & mask;
}

uint upbp_coating_lane_offset(uint slot) {
  return kGPUVCMCoatingHeaderSize + slot * kGPUVCMCoatingLaneStride;
}

void upbp_coating_scan(UPBPCoatingQuery query, uint slot, uint lane_index, inout uint cursor, inout uint flags, out uint light_index, out float kernel_value) {
  const uint cell_index = lane_index / 8u;
  const float radius = query.resources.iteration.surface_radius;
  const uint mask = query.resources.surface_grid_mask;
  const uint cell = upbp_coating_cell_hash(query.camera.position, cell_index, radius, mask);
  bool duplicate = false;
  [unroll] for (uint previous = 0u; previous < 8u; ++previous) {
    if (previous < cell_index) {
      duplicate = duplicate || (upbp_coating_cell_hash(query.camera.position, previous, radius, mask) == cell);
    }
  }
  GPUWavefrontResources grid = query.wavefront;
  grid.vcm_grid_heads_buffer = query.resources.surface_grid_ranges_buffer;
  grid.vcm_grid_next_buffer = query.resources.surface_grid_indices_buffer;
  const uint count = duplicate ? 0u : wavefront_vcm_grid_range_count(grid, cell);
  const uint begin = count > 0u ? wavefront_vcm_grid_range_start(grid, cell) : 0u;
  light_index = 0u;
  kernel_value = 0.0f;
  for (uint scanned = 0u; (scanned < 64u) && (cursor < count); ++scanned) {
    light_index = WAVEFRONT_RO_BUFFER(grid.vcm_grid_next_buffer).Load((begin + cursor) * sizeof(uint));
    cursor += 8u;
    const GPUUPBPVertex light_vertex = upbp_unpack_density_point(upbp_load_density_point(query.resources.density_output_surface_point_buffer, light_index));
    if (upbp_coating_candidate(query, light_vertex, kernel_value) == false) {
      continue;
    }
    Sampler sampler = make_bsdf_sampler(upbp_pair_seed(query.resources.iteration, query.camera, light_vertex));
    wavefront_coating_begin_evaluation(query.prepared.context, query.wavefront, slot, query.camera.material_index, query.prepared.data, -light_vertex.w_i, query.prepared.material,
      sampler);
    flags = 2u;
    return;
  }
  if (cursor >= count) {
    flags = 0u;
    uint ignored;
    WAVEFRONT_RW_BUFFER(query.wavefront.vcm_coating_buffer).InterlockedAdd(kGPUVCMCoatingPendingOffset, uint(-1), ignored);
  }
}

#if ETX_VCM_COATING_PHASE == 1
[numthreads(64, 1, 1)] void ETX_STAGE_ENTRY(uint3 id : SV_DispatchThreadID) {
  if (id.x == 0u) {
    const GPUWavefrontResources wavefront = wavefront_load_resources();
    WAVEFRONT_RW_BUFFER(wavefront.vcm_coating_buffer).Store(kGPUVCMCoatingPendingOffset, 0u);
  }
}
#elif ETX_VCM_COATING_PHASE == 3
groupshared GPUWavefrontCompactSpectralResponse upbp_coating_subtotals[64u];

[numthreads(64, 1, 1)] void ETX_STAGE_ENTRY(uint3 group_id : SV_GroupID, uint lane_index : SV_GroupIndex) {
  UPBPCoatingQuery query;
  if (upbp_coating_prepare_query(group_id.x, query) == false) {
    return;
  }
  const uint slot = group_id.x * kGPUVCMCoatingLaneCount + lane_index;
  upbp_coating_subtotals[lane_index].payload =
    asfloat(WAVEFRONT_RO_BUFFER(query.wavefront.vcm_coating_buffer).Load4(upbp_coating_lane_offset(slot) + kGPUUPBPCoatingSubtotalOffset));
  GroupMemoryBarrierWithGroupSync();
  if (lane_index == 0u) {
    SpectralResponse total = spectral_response_zero(spectral_response_as_query(upbp_unpack_spectral_response(query.camera.throughput)));
    [unroll] for (uint lane = 0u; lane < 64u; ++lane) {
      total = spectral_response_add(total, upbp_unpack_spectral_response(upbp_coating_subtotals[lane]));
    }
    const uint local_path_index = query.camera.global_path_index - query.resources.iteration.camera_batch_offset;
    if (local_path_index < query.resources.iteration.camera_batch_count) {
      const GPUWavefrontPathState camera_state = wavefront_load_path_state(query.wavefront.camera_state_buffer, local_path_index);
      if (upbp_density_contribution_finite(query.resources, GPUUPBPTechnique::Surface, query.camera.global_path_index, total)) {
        wavefront_film_add(camera_state.pixel_index, wavefront_spectral_estimate(total, spectral_response_as_query(total)));
      }
    }
  }
}
#else
[numthreads(64, 1, 1)] void ETX_STAGE_ENTRY(uint3 group_id : SV_GroupID, uint lane_index : SV_GroupIndex) {
  const GPUWavefrontResources wavefront = wavefront_load_resources();
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(wavefront.vcm_coating_buffer);
  const uint slot = group_id.x * kGPUVCMCoatingLaneCount + lane_index;
  const uint offset = upbp_coating_lane_offset(slot);
# if ETX_VCM_COATING_PHASE == 2
  uint flags = buffer.Load(offset + kGPUVCMCoatingFlagsOffset);
  if (flags == 0u) {
    return;
  }
# else
  buffer.Store(offset + kGPUVCMCoatingFlagsOffset, 0u);
# endif
  UPBPCoatingQuery query;
  if (upbp_coating_prepare_query(group_id.x, query) == false) {
    return;
  }
# if ETX_VCM_COATING_PHASE == 2
  uint cursor = buffer.Load(offset + kGPUVCMCoatingCursorOffset);
  GPUWavefrontCompactSpectralResponse packed;
  packed.payload = asfloat(buffer.Load4(offset + kGPUUPBPCoatingSubtotalOffset));
  SpectralResponse accumulated = upbp_unpack_spectral_response(packed);
  if (flags == 2u) {
    const uint light_index = buffer.Load(offset + kGPUVCMCoatingLightIndexOffset);
    const float kernel_value = asfloat(buffer.Load(offset + kGPUVCMCoatingDistanceOffset));
    const GPUUPBPVertex light_vertex = upbp_unpack_density_point(upbp_load_density_point(query.resources.density_output_surface_point_buffer, light_index));
    Sampler sampler;
    const BSDFEval evaluation = wavefront_coating_load_evaluation(wavefront, slot, sampler);
    upbp_coating_resolve(query, light_vertex, kernel_value, evaluation, sampler, accumulated);
    flags = 1u;
  }
# else
  uint cursor = lane_index % 8u;
  uint flags = 1u;
  SpectralResponse accumulated = spectral_response_zero(spectral_response_as_query(upbp_unpack_spectral_response(query.camera.throughput)));
  uint ignored;
  buffer.InterlockedAdd(kGPUVCMCoatingPendingOffset, 1u, ignored);
# endif
  uint light_index;
  float kernel_value;
  upbp_coating_scan(query, slot, lane_index, cursor, flags, light_index, kernel_value);
  buffer.Store(offset + kGPUVCMCoatingCursorOffset, cursor);
  buffer.Store(offset + kGPUVCMCoatingLightIndexOffset, light_index);
  buffer.Store(offset + kGPUVCMCoatingDistanceOffset, asuint(kernel_value));
  buffer.Store(offset + kGPUVCMCoatingFlagsOffset, flags);
  buffer.Store4(offset + kGPUUPBPCoatingSubtotalOffset, asuint(upbp_pack_spectral_response(accumulated).payload));
}
#endif
