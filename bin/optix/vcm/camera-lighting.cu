#include <etx/rt/shared/optix.hxx>
#include <etx/rt/shared/vcm_shared.hxx>

using namespace etx;

static __constant__ VCMGlobal global;

RAYGEN(main) {
  uint3 idx = optixGetLaunchIndex();
  auto& state = global.input_state[idx.x];
  if (state.ray_action_set()) {
    return;
  }

  const auto& scene = global.scene;
  const auto& options = global.options;
  const auto& light_vertices = global.light_vertices;
  const auto& light_paths = global.light_paths;
  auto& iteration = *global.iteration;

  Raytracing rt;
  vcm_update_camera_vcm(state);
  vcm_handle_direct_hit(scene, options, state);
  vcm_gather_vertices(scene, iteration, light_vertices, global.spatial_grid, options, state);

  vcm_connect_to_light(scene, iteration, options, rt, state);
  vcm_connect_to_light_path(scene, iteration, light_paths, light_vertices, options, rt, state);
}
