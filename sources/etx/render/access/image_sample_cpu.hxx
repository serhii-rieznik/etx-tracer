#pragma once

#include <etx/render/access/image_access_cpu.hxx>
#include <etx/render/access/image_sample_shared.hxx>

struct Scene;

struct ImageSampleCPUContext {
  const Scene* scene = nullptr;
};

ETX_SHARED_INLINE ImageSampleCPUContext make_image_sample_cpu_context(const Scene& scene) {
  ImageSampleCPUContext result = {};
  result.scene = &scene;
  return result;
}

ETX_SHARED_INLINE bool image_sample_try_sample(ETX_IN(ImageSampleCPUContext, context), uint32_t image_index, ETX_IN(float2, rnd), ETX_OUT(ImageSampleAccess, sample)) {
  sample = image_sample_access_default(rnd);

  ImageAccessCPUContext access_context = make_image_access_cpu_context(*context.scene);
  ImageAccessCPUDesc image_access = {};
  if (image_access_try_load(access_context, image_index, image_access) == false) {
    return false;
  }

  sample.uv = context.scene->images[image_access.image_index].sample(rnd, sample.pdf, sample.location, sample.value);
  return true;
}
