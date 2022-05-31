#include <etx/rt/shared/optix.hxx>
#include <etx/rt/shared/vcm_shared.hxx>

using namespace etx;

static __constant__ GPUCameraLaunchParams global;

RAYGEN(main) {
  uint3 idx = optixGetLaunchIndex();
  uint3 dim = optixGetLaunchDimensions();
  uint32_t index = idx.x + idx.y * dim.x;
  auto xyz = spectrum::rgb_to_xyz({float(idx.x) / dim.x, float(idx.y) / dim.y, 0.0f});
  global.camera_image[index] = to_float4(xyz);
}
