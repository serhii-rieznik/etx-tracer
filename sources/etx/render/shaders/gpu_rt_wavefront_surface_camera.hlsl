#include "gpu_rt_wavefront_surface_finalize_common.hlsl"

[numthreads(64, 1, 1)] void wavefront_camera_surface_classify_main(uint3 dtid : SV_DispatchThreadID) {
  wavefront_surface_classify(true, dtid.x);
}

[numthreads(64, 1, 1)] void wavefront_camera_continue_finalize_main(uint3 dtid : SV_DispatchThreadID) {
  wavefront_surface_continue_finalize(true, dtid.x);
}
