#include "gpu_rt_wavefront_base.hlsl"
#include <interop/gpu_coating_abi.hxx>
#include <interop/gpu_emission_abi.hxx>

[numthreads(64, 1, 1)] void wavefront_transport_work_clear_main(uint3 id : SV_DispatchThreadID) {
  if (id.x == 0u) {
    const GPUWavefrontResources resources = wavefront_load_resources();
    WAVEFRONT_RW_BUFFER(resources.transport_work_buffer).Store(kGPUTransportWorkCountOffset, 0u);
  }
}

  [numthreads(64, 1, 1)] void wavefront_transport_work_collect_main(uint3 id : SV_DispatchThreadID) {
  if (id.x >= constants.dispatch_item_count) {
    return;
  }
  const GPUWavefrontResources resources = wavefront_load_resources();
  const bool coating = constants.work_queue_index == GPUTransportWorkKind::Coating;
  const uint descriptor = coating ? resources.coating_request_buffer : resources.emission_request_buffer;
  const uint capacity_offset = coating ? kGPUCoatingRequestCapacityOffset : kGPUEmissionRequestCapacityOffset;
  const uint header_size = coating ? kGPUCoatingRequestHeaderSize : kGPUEmissionRequestHeaderSize;
  const uint stride = coating ? kGPUCoatingRequestStride : resources.emission_request_stride;
  const uint active_offset = coating ? kGPUCoatingRequestActiveOffset : kGPUEmissionRequestActiveOffset;
  ByteAddressBuffer requests = WAVEFRONT_RO_BUFFER(descriptor);
  const uint request_index = constants.dispatch_item_offset + id.x;
  if (request_index >= requests.Load(capacity_offset)) {
    return;
  }
  if (requests.Load(header_size + request_index * stride + active_offset) == 0u) {
    return;
  }
  RWByteAddressBuffer work = WAVEFRONT_RW_BUFFER(resources.transport_work_buffer);
  uint index;
  work.InterlockedAdd(kGPUTransportWorkCountOffset, 1u, index);
  work.Store(kGPUTransportWorkIndicesOffset + index * sizeof(uint), request_index);
}

[numthreads(64, 1, 1)] void wavefront_transport_work_build_args_main(uint3 id : SV_DispatchThreadID) {
  if (id.x != 0u) {
    return;
  }
  const GPUWavefrontResources resources = wavefront_load_resources();
  const uint count = WAVEFRONT_RO_BUFFER(resources.transport_work_buffer).Load(kGPUTransportWorkCountOffset);
  const uint groups = (count + 63u) / 64u;
  WAVEFRONT_RW_BUFFER(resources.dispatch_args_buffer).Store4(kGPUWavefrontTransportDispatchArgsOffset, uint4(min(groups, 65535u), max(1u, (groups + 65534u) / 65535u), 1u, 0u));
}
