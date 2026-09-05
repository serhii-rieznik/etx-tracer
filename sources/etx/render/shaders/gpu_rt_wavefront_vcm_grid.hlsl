#include "gpu_rt_wavefront_common.hlsl"
#include "gpu_rt_wavefront_vcm_grid_shared.hlsl"

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
  const uint cell = dtid.x + dtid.y * (65535u * 256u);
  if ((resources.vcm_grid_heads_buffer == kInvalidIndex) || (cell > constants.vcm_grid_mask)) {
    return;
  }
  WAVEFRONT_RW_BUFFER(resources.vcm_grid_heads_buffer).Store(cell * sizeof(uint), 0u);
  WAVEFRONT_RW_BUFFER(resources.vcm_grid_heads_buffer).Store((2u * (constants.vcm_grid_mask + 1u) + cell) * sizeof(uint), 0u);
}

bool wavefront_vcm_grid_vertex_cell(GPUWavefrontResources resources, uint vertex_index, out uint cell) {
  cell = 0u;
  if ((resources.vcm_grid_heads_buffer == kInvalidIndex) || (resources.vcm_grid_next_buffer == kInvalidIndex) || (vertex_index < resources.path_capacity) ||
      (vertex_index >= constants.vcm_light_vertex_count) || (vertex_index >= resources.light_vertex_capacity) || (constants.vcm_radius <= 0.0f)) {
    return false;
  }

  GPUWavefrontPathVertex vertex = wavefront_load_path_vertex(resources.light_vertex_buffer, vertex_index);
  if ((wavefront_path_vertex_valid(vertex) == false) || (wavefront_path_vertex_is_surface(vertex) == false) || (wavefront_path_vertex_connectible(vertex) == false)) {
    return false;
  }
  cell = wavefront_vcm_position_to_cell(vertex.position);
  return true;
}

[numthreads(256, 1, 1)] void wavefront_vcm_grid_build_main(uint3 dtid : SV_DispatchThreadID) {
  const GPUWavefrontResources resources = wavefront_load_resources();
  uint cell = 0u;
  if (wavefront_vcm_grid_vertex_cell(resources, dtid.x + dtid.y * (65535u * 256u), cell)) {
    WAVEFRONT_RW_BUFFER(resources.vcm_grid_heads_buffer).InterlockedAdd(cell * sizeof(uint), 1u);
  }
}

groupshared uint vcm_grid_scan_values[64u];

[numthreads(64, 1, 1)] void wavefront_vcm_grid_prefix_main(uint3 group_id : SV_GroupID, uint lane : SV_GroupIndex) {
  const GPUWavefrontResources resources = wavefront_load_resources();
  const uint group_index = group_id.x + group_id.y * 65535u;
  const uint index = group_index * 64u + lane;
  const uint count = constants.dispatch_item_count;
  const uint offset = constants.dispatch_item_offset;
  const uint value = index < count ? WAVEFRONT_RO_BUFFER(resources.vcm_grid_heads_buffer).Load((offset + index) * sizeof(uint)) : 0u;
  vcm_grid_scan_values[lane] = value;
  GroupMemoryBarrierWithGroupSync();
  [unroll] for (uint step = 1u; step < 64u; step *= 2u) {
    const uint previous = lane >= step ? vcm_grid_scan_values[lane - step] : 0u;
    GroupMemoryBarrierWithGroupSync();
    vcm_grid_scan_values[lane] += previous;
    GroupMemoryBarrierWithGroupSync();
  }
  if (index < count) {
    WAVEFRONT_RW_BUFFER(resources.vcm_grid_heads_buffer).Store((offset + index) * sizeof(uint), vcm_grid_scan_values[lane] - value);
    if (offset == 0u) {
      WAVEFRONT_RW_BUFFER(resources.vcm_grid_heads_buffer).Store((count + index) * sizeof(uint), value);
    }
  }
  if ((lane == 63u) && (count > 64u) && ((group_index * 64u) < count)) {
    const uint parent_offset = (offset == 0u) ? (3u * count) : (offset + count);
    WAVEFRONT_RW_BUFFER(resources.vcm_grid_heads_buffer).Store((parent_offset + group_index) * sizeof(uint), vcm_grid_scan_values[lane]);
  }
}

  [numthreads(256, 1, 1)] void wavefront_vcm_grid_scatter_main(uint3 dtid : SV_DispatchThreadID) {
  const GPUWavefrontResources resources = wavefront_load_resources();
  const uint vertex_index = dtid.x + dtid.y * (65535u * 256u);
  uint cell = 0u;
  if (wavefront_vcm_grid_vertex_cell(resources, vertex_index, cell) == false) {
    return;
  }
  const uint begin = wavefront_vcm_grid_range_start(resources, cell);
  uint local_index = 0u;
  WAVEFRONT_RW_BUFFER(resources.vcm_grid_heads_buffer).InterlockedAdd((2u * (constants.vcm_grid_mask + 1u) + cell) * sizeof(uint), 1u, local_index);
  WAVEFRONT_RW_BUFFER(resources.vcm_grid_next_buffer).Store((begin + local_index) * sizeof(uint), vertex_index);
}
