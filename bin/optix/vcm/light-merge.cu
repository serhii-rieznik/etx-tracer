#include <etx/rt/shared/optix.hxx>
#include <etx/rt/shared/vcm_shared.hxx>

using namespace etx;

static __constant__ VCMGlobal global;

RAYGEN(main) {
  uint3 idx = optixGetLaunchIndex();
  uint3 dim = optixGetLaunchDimensions();
  uint32_t index = idx.x + idx.y * dim.x;

  auto dst = global.light_final_image;
  auto src = global.light_iteration_image;
  float t = global.iteration->iteration / float(global.iteration->iteration + 1);
  dst[index] = lerp(src[index], dst[index], t);
}
