#include "gpu_rt_wavefront_common.hlsl"

uint wavefront_vcm_hash_cell(int3 cell) {
  return ((uint(cell.x) * 73856093u) ^ (uint(cell.y) * 19349663u) ^ (uint(cell.z) * 83492791u)) & constants.vcm_grid_mask;
}

uint wavefront_vcm_position_to_cell(float3 position) {
  float cell_size = 2.0f * constants.vcm_radius;
  int3 cell = int3(floor(position / cell_size));
  return wavefront_vcm_hash_cell(cell);
}

[numthreads(256, 1, 1)] void wavefront_vcm_grid_clear_main(uint3 dtid : SV_DispatchThreadID) {
  GPUWavefrontResources resources = wavefront_load_resources();
  if ((resources.vcm_grid_heads_buffer == kInvalidIndex) || (dtid.x > constants.vcm_grid_mask)) {
    return;
  }
  WAVEFRONT_RW_BUFFER(resources.vcm_grid_heads_buffer).Store(dtid.x * 4u, kInvalidIndex);
}

[numthreads(256, 1, 1)] void wavefront_vcm_grid_build_main(uint3 dtid : SV_DispatchThreadID) {
  GPUWavefrontResources resources = wavefront_load_resources();
  uint vertex_index = dtid.x;
  if ((resources.vcm_grid_heads_buffer == kInvalidIndex) || (resources.vcm_grid_next_buffer == kInvalidIndex) ||
      (vertex_index < resources.path_capacity) || (vertex_index >= constants.vcm_light_vertex_count) ||
      (vertex_index >= resources.light_vertex_capacity) || (constants.vcm_radius <= 0.0f)) {
    return;
  }

  GPUWavefrontPathVertex vertex = wavefront_load_path_vertex(resources.light_vertex_buffer, vertex_index);
  if ((wavefront_path_vertex_valid(vertex) == false) || (wavefront_path_vertex_is_surface(vertex) == false) ||
      (wavefront_path_vertex_connectible(vertex) == false)) {
    return;
  }

  uint cell = wavefront_vcm_position_to_cell(vertex.position);
  uint previous_head = kInvalidIndex;
  WAVEFRONT_RW_BUFFER(resources.vcm_grid_heads_buffer).InterlockedExchange(cell * 4u, vertex_index, previous_head);
  WAVEFRONT_RW_BUFFER(resources.vcm_grid_next_buffer).Store(vertex_index * 4u, previous_head);
}
