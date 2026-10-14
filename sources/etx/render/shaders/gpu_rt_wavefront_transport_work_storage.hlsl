#pragma once

#include "gpu_rt_wavefront_base.hlsl"

bool wavefront_transport_work_request(uint3 id, out uint request_index) {
  request_index = 0u;
  const GPUWavefrontResources resources = wavefront_load_resources();
  ByteAddressBuffer work = WAVEFRONT_RO_BUFFER(resources.transport_work_buffer);
  const uint index = id.x + id.y * (65535u * 64u);
  if (index >= work.Load(kGPUTransportWorkCountOffset)) {
    return false;
  }
  request_index = work.Load(kGPUTransportWorkIndicesOffset + index * sizeof(uint));
  return true;
}
