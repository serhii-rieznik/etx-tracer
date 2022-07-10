#include <etx/rt/shared/optix.hxx>
#include <etx/rt/shared/vcm_shared.hxx>

using namespace etx;

static __constant__ VCMGlobal global;

RAYGEN(main) {
  uint3 idx = optixGetLaunchIndex();
  uint3 dim = optixGetLaunchDimensions();
  uint32_t index = idx.x + idx.y * dim.x;

  global.input_state[index] = vcm_generate_emitter_state(index, global.scene, *global.iteration);
  global.iteration->active_paths = dim.x * dim.y;
}
