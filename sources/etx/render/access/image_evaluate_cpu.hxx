#pragma once

#include <etx/render/access/image_access_cpu.hxx>

struct ImageEvaluateCPUContext {
  const Scene* scene = nullptr;
};

ETX_SHARED_INLINE ImageEvaluateCPUContext make_image_evaluate_cpu_context(const Scene& scene) {
  ImageEvaluateCPUContext result = {};
  result.scene = &scene;
  return result;
}

ETX_SHARED_INLINE bool image_evaluate_cpu_try_rgba(ETX_IN(ImageEvaluateCPUContext, context), uint32_t image_index, ETX_IN(float2, uv), ETX_OUT(float, image_pdf),
  ETX_OUT(float4, image_value)) {
  image_pdf = 0.0f;
  image_value = float4(1.0f, 1.0f, 1.0f, 1.0f);

  if (context.scene == nullptr) {
    return false;
  }

  ImageAccessCPUContext access_context = make_image_access_cpu_context(*context.scene);
  ImageAccessCPUDesc image_access = {};
  if (image_access_try_load(access_context, image_index, image_access) == false) {
    return false;
  }

  image_value = context.scene->images[image_access.image_index].evaluate(uv, &image_pdf);
  return true;
}

ETX_SHARED_INLINE bool image_evaluate_try_rgba(ETX_IN(ImageEvaluateCPUContext, context), uint32_t image_index, ETX_IN(float2, uv), ETX_OUT(float, image_pdf),
  ETX_OUT(float4, image_value)) {
  image_pdf = 0.0f;
  image_value = float4(1.0f, 1.0f, 1.0f, 1.0f);
  return image_evaluate_cpu_try_rgba(context, image_index, uv, image_pdf, image_value);
}

ETX_SHARED_INLINE float4 image_evaluate_sample_whole_or_default(ETX_IN(ImageEvaluateCPUContext, context), uint32_t image_index, ETX_IN(float2, uv), ETX_IN(float4, default_value)) {
  if (image_index == kInvalidIndex) {
    return default_value;
  }

  float image_pdf = 0.0f;
  float4 image_value = float4(1.0f, 1.0f, 1.0f, 1.0f);
  if (image_evaluate_try_rgba(context, image_index, uv, image_pdf, image_value) == false) {
    return default_value;
  }

  return default_value * image_value;
}

ETX_SHARED_INLINE float image_evaluate_sample_channel_or_default(ETX_IN(ImageEvaluateCPUContext, context), uint32_t image_index, uint32_t channel, ETX_IN(float2, uv),
  float default_value) {
  if ((image_index == kInvalidIndex) || (channel >= 4u)) {
    return default_value;
  }

  float image_pdf = 0.0f;
  float4 image_value = float4(1.0f, 1.0f, 1.0f, 1.0f);
  if (image_evaluate_try_rgba(context, image_index, uv, image_pdf, image_value) == false) {
    return default_value;
  }

  if (channel == 0u) {
    return image_value.x;
  }
  if (channel == 1u) {
    return image_value.y;
  }
  if (channel == 2u) {
    return image_value.z;
  }

  return image_value.w;
}

ETX_SHARED_INLINE bool image_can_apply(ETX_IN(ImageAccessCPUContext, context), uint32_t image_index) {
  ImageAccessCPUDesc image_access = {};
  return image_access_try_load(context, image_index, image_access);
}
