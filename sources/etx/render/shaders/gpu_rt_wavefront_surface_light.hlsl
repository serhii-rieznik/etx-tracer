#include "gpu_rt_wavefront_surface_common.hlsl"

[numthreads(64, 1, 1)] void wavefront_light_surface_classify_main(uint3 dtid : SV_DispatchThreadID) {
  wavefront_surface_classify(false, dtid.x);
}

  [numthreads(64, 1, 1)] void wavefront_light_continue_finalize_main(uint3 dtid : SV_DispatchThreadID) {
  wavefront_surface_continue_finalize(false, dtid.x);
}
