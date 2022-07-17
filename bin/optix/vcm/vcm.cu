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
  global->iteration->active_paths = global->scene.camera.image_size.x * global->scene.camera.image_size.y;
}
