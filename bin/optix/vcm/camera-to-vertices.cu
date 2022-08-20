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

  const auto& options = global.options;
  const auto& light_paths = global.light_paths;
  const auto& light_path = light_paths[state.global_index];
  uint32_t i = idx.y;
  if ((i >= light_path.count) || (state.total_path_depth + i + 2 > options.max_depth)) {
    return;
  }

  const auto& scene = global.scene;
  const auto& light_vertices = global.light_vertices;
  auto& iteration = *global.iteration;

  Raytracing rt;

  float3 target_position = {};
  SpectralResponse value = {};
  bool connected = vcm_connect_to_light_vertex(scene, state.spect, state, light_vertices[light_path.index + i],  //
    options, iteration.vm_weight, state.medium_index, target_position, value);

  if (connected == false) {
    return;
  }

  const auto& tri = scene.triangles[state.intersection.triangle_index];
  float3 p0 = shading_pos(scene.vertices, tri, state.intersection.barycentric, normalize(target_position - state.intersection.pos));
  auto tr = value * transmittance(state.spect, state.sampler, p0, target_position, state.medium_index, scene, rt);

  atomicAdd(&state.gathered.components.x, tr.components.x);
  atomicAdd(&state.gathered.components.y, tr.components.y);
  atomicAdd(&state.gathered.components.z, tr.components.z);
}
