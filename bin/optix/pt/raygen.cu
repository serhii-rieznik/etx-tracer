#include <etx/render/shared/scene.hxx>
#include <etx/rt/shared/optix.hxx>
#include <etx/rt/shared/path_tracing_shared.hxx>

using namespace etx;

static __constant__ PTGPUData global;

RAYGEN(main) {
  uint3 idx = optixGetLaunchIndex();
  uint3 dim = optixGetLaunchDimensions();

  uint32_t index = idx.x + idx.y * dim.x;
  global.payloads[index] = make_ray_payload(global.scene, {idx.x, dim.y - idx.y - 1u}, {dim.x, dim.y}, 0u);
  global.output[index] = {};
}
