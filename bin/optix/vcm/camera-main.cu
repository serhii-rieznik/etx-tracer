#include <etx/rt/shared/optix.hxx>
#include <etx/rt/shared/vcm_shared.hxx>

using namespace etx;

static __constant__ VCMGlobal global;

ETX_GPU_CODE void finish_ray(VCMIteration& iteration, VCMPathState& state) {
  state.merged *= iteration.vm_normalization;
  state.merged += (state.gathered / spectrum::sample_pdf()).to_xyz();

  uint32_t x = state.index % global.scene.camera.image_size.x;
  uint32_t y = state.index / global.scene.camera.image_size.x;
  uint32_t c = x + (global.scene.camera.image_size.y - 1 - y) * global.scene.camera.image_size.x;

  float4 old_value = global.camera_image[c];
  float4 new_value = {state.merged.x, state.merged.y, state.merged.z, 1.0f};

  float t = float(iteration.iteration) / float(iteration.iteration + 1);
  global.camera_image[c] = (iteration.iteration == 0) ? new_value : lerp(new_value, old_value, t);
}

ETX_GPU_CODE void continue_tracing(VCMIteration& iteration, VCMPathState& state) {
  if (state.path_length <= global.options.max_depth) {
    int i = atomicAdd(&iteration.active_paths, 1u);
    global.output_state[i] = state;
  } else {
    finish_ray(iteration, state);
  }
}

RAYGEN(main) {
  Raytracing rt;

  uint3 idx = optixGetLaunchIndex();
  auto& state = global.input_state[idx.x];
  auto& iteration = *global.iteration;

  const auto& light_path = global.light_paths[state.index];
  if (vcm_camera_step(global.scene, iteration, global.options, light_path, global.light_vertices, state, rt, global.spatial_grid)) {
    continue_tracing(iteration, state);
  } else {
    finish_ray(iteration, state);
  }
}
