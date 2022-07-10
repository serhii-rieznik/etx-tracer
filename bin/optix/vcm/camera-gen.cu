#include <etx/rt/shared/optix.hxx>
#include <etx/rt/shared/vcm_shared.hxx>

using namespace etx;

static __constant__ VCMGlobal global;

RAYGEN(main) {
  uint3 idx = optixGetLaunchIndex();
  uint3 dim = optixGetLaunchDimensions();
  uint32_t index = idx.x + idx.y * dim.x;

  const auto& light_path = global.light_paths[index];
  global.input_state[index] = vcm_generate_camera_state({idx.x, idx.y}, global.scene, *global.iteration, light_path.spect);
  global.iteration->active_paths = dim.x * dim.y;
}
