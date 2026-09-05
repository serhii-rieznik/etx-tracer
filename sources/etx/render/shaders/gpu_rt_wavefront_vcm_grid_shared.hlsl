#pragma once

uint wavefront_vcm_grid_range_start(GPUWavefrontResources resources, uint cell) {
  const uint cell_count = constants.vcm_grid_mask + 1u;
  uint level_count = cell_count;
  uint level_offset = 0u;
  uint index = cell;
  uint result = 0u;
  for (;;) {
    result += WAVEFRONT_RO_BUFFER(resources.vcm_grid_heads_buffer).Load((level_offset + index) * sizeof(uint));
    if (level_count <= 64u) {
      return result;
    }
    index /= 64u;
    level_offset = (level_offset == 0u) ? (3u * cell_count) : (level_offset + level_count);
    level_count = (level_count + 63u) / 64u;
  }
}

uint wavefront_vcm_grid_range_count(GPUWavefrontResources resources, uint cell) {
  return WAVEFRONT_RO_BUFFER(resources.vcm_grid_heads_buffer).Load((constants.vcm_grid_mask + 1u + cell) * sizeof(uint));
}
