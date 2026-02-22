#pragma once

namespace etx {
struct Scene;
struct ImageSceneAccessCPUDesc;
ETX_SHARED_INLINE bool try_load_scene_image_access(const Scene& scene, uint32_t image_index, ImageSceneAccessCPUDesc& image_access);
}  // namespace etx

struct CameraLensSampleSharedCPUContext {
  const etx::Scene& scene;
};

ETX_SHARED_INLINE bool camera_lens_sample_shared_cpu_try_sample_image_uv(
  ETX_IN(CameraLensSampleSharedCPUContext, context), uint32_t lens_image, ETX_IN(float2, rnd), ETX_OUT(float2, image_uv)) {
  image_uv = float2(0.0f, 0.0f);
  etx::ImageSceneAccessCPUDesc image_access = {};
  if (etx::try_load_scene_image_access(context.scene, lens_image, image_access) == false) {
    return false;
  }

  image_uv = context.scene.images[image_access.image_index].sample(rnd);
  return true;
}

#define ETX_CAMERA_LENS_SAMPLE_SHARED_CONTEXT_TYPE CameraLensSampleSharedCPUContext
#define ETX_CAMERA_LENS_SAMPLE_SHARED_TRY_SAMPLE_IMAGE_UV(context, lens_image, rnd, image_uv) \
  camera_lens_sample_shared_cpu_try_sample_image_uv(context, lens_image, rnd, image_uv)
#include <etx/render/interop/camera_lens_sample_shared.hxx>
#undef ETX_CAMERA_LENS_SAMPLE_SHARED_TRY_SAMPLE_IMAGE_UV
#undef ETX_CAMERA_LENS_SAMPLE_SHARED_CONTEXT_TYPE
