#include "gpu_rt_wavefront_trace_common.hlsl"

[numthreads(64, 1, 1)] void wavefront_trace_camera_main(uint3 dtid : SV_DispatchThreadID) {
  wavefront_trace_path(true, dtid.x);
}
