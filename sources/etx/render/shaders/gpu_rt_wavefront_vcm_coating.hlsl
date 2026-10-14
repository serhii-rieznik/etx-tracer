#pragma once

#include "gpu_rt_wavefront_coating_storage.hlsl"

struct WavefrontVCMCoatingLane {
  uint cursor;
  uint light_index;
  float distance_squared;
  uint flags;
  float3 subtotal;
  Sampler sampler;
};

static const uint kVCMCoatingScanning = 1u;
static const uint kVCMCoatingPending = 2u;

uint wavefront_vcm_coating_lane_offset(uint slot) {
  return kGPUVCMCoatingHeaderSize + slot * kGPUVCMCoatingLaneStride;
}

WavefrontVCMCoatingLane wavefront_vcm_coating_load_lane(RWByteAddressBuffer buffer, uint offset) {
  WavefrontVCMCoatingLane lane;
  lane.cursor = buffer.Load(offset + kGPUVCMCoatingCursorOffset);
  lane.light_index = buffer.Load(offset + kGPUVCMCoatingLightIndexOffset);
  lane.distance_squared = asfloat(buffer.Load(offset + kGPUVCMCoatingDistanceOffset));
  lane.flags = buffer.Load(offset + kGPUVCMCoatingFlagsOffset);
  lane.subtotal = asfloat(buffer.Load3(offset + kGPUVCMCoatingSubtotalOffset));
  lane.sampler = wavefront_coating_load_sampler(buffer, offset + kGPUVCMCoatingSamplerOffset);
  return lane;
}

void wavefront_vcm_coating_store_lane(RWByteAddressBuffer buffer, uint offset, WavefrontVCMCoatingLane lane) {
  buffer.Store(offset + kGPUVCMCoatingCursorOffset, lane.cursor);
  buffer.Store(offset + kGPUVCMCoatingLightIndexOffset, lane.light_index);
  buffer.Store(offset + kGPUVCMCoatingDistanceOffset, asuint(lane.distance_squared));
  buffer.Store(offset + kGPUVCMCoatingFlagsOffset, lane.flags);
  buffer.Store3(offset + kGPUVCMCoatingSubtotalOffset, asuint(lane.subtotal));
  wavefront_coating_store_sampler(buffer, offset + kGPUVCMCoatingSamplerOffset, lane.sampler);
}

void wavefront_vcm_coating_scan(WavefrontVCMMergeQuery query, uint slot, uint lane_index, inout WavefrontVCMCoatingLane lane) {
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
  // Bound dispatch work; the persisted cursor resumes every remaining candidate.
  for (uint scanned = 0u; (scanned < 64u) && (lane.cursor < count); ++scanned) {
    const uint light_index = WAVEFRONT_RO_BUFFER(query.resources.vcm_grid_next_buffer).Load((begin + lane.cursor) * sizeof(uint));
    lane.cursor += 8u;
    float distance_squared = 0.0f;
    if (wavefront_vcm_merge_candidate_matches(query, light_index, distance_squared) == false) {
      continue;
    }
    GPUWavefrontPathVertex light_vertex;
    if (wavefront_vcm_merge_load_candidate(query, light_index, light_vertex) == false) {
      continue;
    }
    wavefront_coating_begin_evaluation(query.bsdf_context, query.resources, slot, query.camera_vertex.material_index, query.camera_data, -light_vertex.w_i, query.material,
      lane.sampler);
    lane.light_index = light_index;
    lane.distance_squared = distance_squared;
    lane.flags = kVCMCoatingPending;
    return;
  }
  if (lane.cursor >= count) {
    lane.flags = 0u;
    uint ignored;
    WAVEFRONT_RW_BUFFER(query.resources.vcm_coating_buffer).InterlockedAdd(kGPUVCMCoatingPendingOffset, uint(-1), ignored);
  }
}

#if ETX_VCM_COATING_PHASE == 1
[numthreads(64, 1, 1)] void ETX_STAGE_ENTRY(uint3 id : SV_DispatchThreadID) {
  if (id.x == 0u) {
    const GPUWavefrontResources resources = wavefront_load_resources();
    WAVEFRONT_RW_BUFFER(resources.vcm_coating_buffer).Store(kGPUVCMCoatingPendingOffset, 0u);
  }
}
#elif ETX_VCM_COATING_PHASE == 3
groupshared float3 vcm_coating_contributions[64];

[numthreads(64, 1, 1)] void ETX_STAGE_ENTRY(uint3 group_id : SV_GroupID, uint lane_index : SV_GroupIndex) {
  WavefrontVCMMergeQuery query;
  if (wavefront_vcm_merge_prepare(group_id.x, query) == false) {
    return;
  }
  const uint slot = group_id.x * kGPUVCMCoatingLaneCount + lane_index;
  const uint offset = wavefront_vcm_coating_lane_offset(slot);
  vcm_coating_contributions[lane_index] = asfloat(WAVEFRONT_RO_BUFFER(query.resources.vcm_coating_buffer).Load3(offset + kGPUVCMCoatingSubtotalOffset));
  GroupMemoryBarrierWithGroupSync();
  if (lane_index == 0u) {
    float3 total = float3(0.0f, 0.0f, 0.0f);
    [unroll] for (uint contribution_index = 0u; contribution_index < 64u; ++contribution_index) {
      total += vcm_coating_contributions[contribution_index];
    }
    if (all(isfinite(total)) && any(total != float3(0.0f, 0.0f, 0.0f))) {
      wavefront_film_add(query.camera_vertex.pixel_index, total);
    }
  }
}
#else
[numthreads(64, 1, 1)] void ETX_STAGE_ENTRY(uint3 group_id : SV_GroupID, uint lane_index : SV_GroupIndex) {
  const GPUWavefrontResources resources = wavefront_load_resources();
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(resources.vcm_coating_buffer);
  const uint slot = group_id.x * kGPUVCMCoatingLaneCount + lane_index;
  const uint offset = wavefront_vcm_coating_lane_offset(slot);
# if ETX_VCM_COATING_PHASE == 2
  if (buffer.Load(offset + kGPUVCMCoatingFlagsOffset) == 0u) {
    return;
  }
# else
  buffer.Store(offset + kGPUVCMCoatingFlagsOffset, 0u);
  buffer.Store3(offset + kGPUVCMCoatingSubtotalOffset, uint3(0u, 0u, 0u));
# endif
  WavefrontVCMMergeQuery query;
  if (wavefront_vcm_merge_prepare(group_id.x, query) == false) {
    return;
  }
# if ETX_VCM_COATING_PHASE == 2
  WavefrontVCMCoatingLane lane = wavefront_vcm_coating_load_lane(buffer, offset);
  if (lane.flags == kVCMCoatingPending) {
    Sampler sampler;
    const BSDFEval evaluation = wavefront_coating_load_evaluation(resources, slot, sampler);
    const GPUWavefrontPathVertex light_vertex = wavefront_load_path_vertex(resources.light_vertex_buffer, lane.light_index);
    lane.subtotal += wavefront_vcm_merge_resolve(query, light_vertex, -light_vertex.w_i, lane.distance_squared, evaluation, sampler);
    lane.sampler = sampler;
    lane.flags = kVCMCoatingScanning;
  }
# else
  WavefrontVCMCoatingLane lane = (WavefrontVCMCoatingLane)0;
  lane.cursor = lane_index % 8u;
  lane.sampler.seed = query.sampler_seed;
  lane.flags = kVCMCoatingScanning;
  uint ignored;
  buffer.InterlockedAdd(kGPUVCMCoatingPendingOffset, 1u, ignored);
# endif
  wavefront_vcm_coating_scan(query, slot, lane_index, lane);
  wavefront_vcm_coating_store_lane(buffer, offset, lane);
}
#endif
