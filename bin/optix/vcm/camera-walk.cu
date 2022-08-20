#include <etx/rt/shared/optix.hxx>
#include <etx/rt/shared/vcm_shared.hxx>

using namespace etx;

static __constant__ VCMGlobal global;

ETX_GPU_CODE void finish_ray(VCMIteration& iteration, VCMPathState& state) {
  float3 result = state.merged * iteration.vm_normalization + (state.gathered / spectrum::sample_pdf()).to_xyz();

  uint32_t x = state.global_index % global.scene.camera.image_size.x;
  uint32_t y = state.global_index / global.scene.camera.image_size.x;
  uint32_t c = x + (global.scene.camera.image_size.y - 1 - y) * global.scene.camera.image_size.x;
  float4& current = global.camera_iteration_image[c];
  atomicAdd(&current.x, result.x);
  atomicAdd(&current.y, result.y);
  atomicAdd(&current.z, result.z);
}

ETX_GPU_CODE void continue_tracing(VCMIteration& iteration, const VCMPathState& state) {
  int i = atomicAdd(&iteration.active_paths, 1u);
  global.output_state[i] = state;
}

RAYGEN(main) {
  uint3 idx = optixGetLaunchIndex();
  auto& state = global.input_state[idx.x];
  const auto& scene = global.scene;
  const auto& options = global.options;
  auto& iteration = *global.iteration;

  // Last kernel
  if (state.ray_action_set() == false) {
    bool continue_ray = vcm_next_ray(scene, PathSource::Camera, options, state, iteration);
    state.continue_ray(continue_ray);
  }

  if (state.should_continue_ray()) {
    continue_tracing(iteration, state);
  } else {
    finish_ray(iteration, state);
  }
}
