#include <etx/rt/shared/vcm_shared.hxx>

using namespace etx;

ETX_GPU_CALLABLE void merge_light_image(const VCMGlobal* global) {
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
