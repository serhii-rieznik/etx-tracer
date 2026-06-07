#pragma once

#include <etx/render/interop/image.hxx>

struct Scene;

struct ImageAccessCPUDesc {
  float3 fsize = {};
  uint3 size = {};
  uint32_t options = 0u;
  uint32_t image_index = kInvalidIndex;
};

struct ImageAccessCPUContext {
  const Scene* scene = nullptr;
};

ETX_SHARED_INLINE ImageAccessCPUContext make_image_access_cpu_context(const Scene& scene) {
  ImageAccessCPUContext result = {};
  result.scene = &scene;
  return result;
}

ETX_SHARED_INLINE bool image_access_cpu_has_images(ETX_IN(ImageAccessCPUContext, context)) {
  return (context.scene != nullptr) && (context.scene->images.count > 0u);
}

ETX_SHARED_INLINE bool image_access_cpu_load_desc(ETX_IN(ImageAccessCPUContext, context), uint32_t image_index, ETX_OUT(ImageAccessCPUDesc, image_access)) {
  if ((context.scene == nullptr) || (image_index >= context.scene->images.count)) {
    return false;
  }

  const auto& image = context.scene->images[image_index];
  image_access.fsize = image.fsize;
  image_access.size = image.isize;
  image_access.options = image.options;
  image_access.image_index = image_index;
  return true;
}

ETX_SHARED_INLINE bool image_access_try_load(ETX_IN(ImageAccessCPUContext, context), uint32_t image_index, ETX_OUT(ImageAccessCPUDesc, image_access)) {
  if (image_access_cpu_has_images(context) == false) {
    return false;
  }

  return image_access_cpu_load_desc(context, image_index, image_access);
}

ETX_SHARED_INLINE bool image_access_has_alpha(ETX_IN(ImageAccessCPUContext, context), uint32_t image_index) {
  ImageAccessCPUDesc image_access = {};
  if (image_access_try_load(context, image_index, image_access) == false) {
    return false;
  }

  return (image_access.options & Image::HasAlphaChannel) != 0u;
}
