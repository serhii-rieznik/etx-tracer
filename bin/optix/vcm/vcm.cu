#include <etx/rt/shared/vcm_shared.hxx>

#if (ETX_NVCC_COMPILER == 0)
extern uint3 blockIdx;
extern uint3 blockDim;
extern uint3 threadIdx;
#endif

using namespace etx;

ETX_GPU_CALLABLE void gen_light_rays(VCMGlobal* global) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t index = x + y * global->scene.camera.image_size.x;
  if (index >= global->light_final_image.count)
    return;

  global->input_state[index] = vcm_generate_emitter_state(index, global->scene, *global->iteration);
  global->light_iteration_image[index] = {};
  global->iteration->active_paths = global->scene.camera.image_size.x * global->scene.camera.image_size.y;
}

ETX_GPU_CALLABLE void merge_light_image(VCMGlobal* global) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t index = x + y * global->scene.camera.image_size.x;
  if (index >= global->light_final_image.count)
    return;

  auto dst = global->light_final_image;
  auto src = global->light_iteration_image;
  float t = global->iteration->iteration / float(global->iteration->iteration + 1);
  dst[index] = lerp(src[index], dst[index], t);
}

ETX_GPU_CALLABLE void gen_camera_rays(VCMGlobal* global) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t index = x + y * global->scene.camera.image_size.x;
  if (index >= global->light_paths.count)
    return;

  const auto& light_path = global->light_paths[index];
  global->input_state[index] = vcm_generate_camera_state({x, y}, global->scene, *global->iteration, light_path.spect);
  global->camera_iteration_image[index] = {};
  global->iteration->active_paths = global->scene.camera.image_size.x * global->scene.camera.image_size.y;
}

ETX_GPU_CALLABLE void merge_camera_image(VCMGlobal* global) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t index = x + y * global->scene.camera.image_size.x;
  if (index >= global->camera_final_image.count)
    return;

  auto dst = global->camera_final_image;
  auto src = global->camera_iteration_image;
  float t = global->iteration->iteration / float(global->iteration->iteration + 1);
  dst[index] = lerp(src[index], dst[index], t);
}

ETX_GPU_CALLABLE void vcm_continue_camera_path(VCMGlobal* global_ptr) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= global_ptr->launch_dim)
    return;

  auto& global = *global_ptr;
  auto& state = global.input_state[idx];
  auto& iteration = *global.iteration;

  const auto& scene = global.scene;
  const auto& options = global.options;

  // Last kernel
  if (state.ray_action_set() == false) {
    bool continue_ray = vcm_next_ray(scene, PathSource::Camera, options, state, iteration);
    state.continue_ray(continue_ray);
  }

  if (state.should_continue_ray()) {
    int i = atomicAdd(&global.iteration->active_paths, 1u);
    global.output_state[i] = state;
  } else {
    float3 result = state.merged * global.iteration->vm_normalization + (state.gathered / spectrum::sample_pdf()).to_xyz();
    uint32_t x = state.global_index % global.scene.camera.image_size.x;
    uint32_t y = state.global_index / global.scene.camera.image_size.x;
    uint32_t c = x + (global.scene.camera.image_size.y - 1 - y) * global.scene.camera.image_size.x;
    float4& current = global.camera_iteration_image[c];
    atomicAdd(&current.x, result.x);
    atomicAdd(&current.y, result.y);
    atomicAdd(&current.z, result.z);
  }
}
