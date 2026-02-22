#pragma once

#include "interop.hxx"

#ifndef ETX_IMAGE_EVALUATE_SHARED_CONTEXT_TYPE
# error "ETX_IMAGE_EVALUATE_SHARED_CONTEXT_TYPE must be defined before including image_evaluate_shared.hxx"
#endif

#ifndef ETX_IMAGE_EVALUATE_SHARED_TRY_EVALUATE_IMAGE_RGBA
# error "ETX_IMAGE_EVALUATE_SHARED_TRY_EVALUATE_IMAGE_RGBA must be defined before including image_evaluate_shared.hxx"
#endif

ETX_SHARED_INLINE bool image_evaluate_shared_try_evaluate_rgba(
  ETX_IN(ETX_IMAGE_EVALUATE_SHARED_CONTEXT_TYPE, context), uint32_t image_index, ETX_IN(float2, uv), ETX_OUT(float, image_pdf), ETX_OUT(float4, image_value)) {
  image_pdf = 0.0f;
  image_value = float4(1.0f, 1.0f, 1.0f, 1.0f);
  if (ETX_IMAGE_EVALUATE_SHARED_TRY_EVALUATE_IMAGE_RGBA(context, image_index, uv, image_pdf, image_value) == false) {
    return false;
  }
  return true;
}

ETX_SHARED_INLINE float4 image_evaluate_shared_sample_whole_or_default(
  ETX_IN(ETX_IMAGE_EVALUATE_SHARED_CONTEXT_TYPE, context), uint32_t image_index, ETX_IN(float2, uv), ETX_IN(float4, default_value)) {
  if (image_index == kInvalidIndex) {
    return default_value;
  }

  float image_pdf = 0.0f;
  float4 image_value = float4(1.0f, 1.0f, 1.0f, 1.0f);
  if (image_evaluate_shared_try_evaluate_rgba(context, image_index, uv, image_pdf, image_value) == false) {
    return default_value;
  }

  return default_value * image_value;
}

ETX_SHARED_INLINE float image_evaluate_shared_sample_channel_or_default(
  ETX_IN(ETX_IMAGE_EVALUATE_SHARED_CONTEXT_TYPE, context), uint32_t image_index, uint32_t channel, ETX_IN(float2, uv), float default_value) {
  if ((image_index == kInvalidIndex) || (channel >= 4u)) {
    return default_value;
  }

  float image_pdf = 0.0f;
  float4 image_value = float4(1.0f, 1.0f, 1.0f, 1.0f);
  if (image_evaluate_shared_try_evaluate_rgba(context, image_index, uv, image_pdf, image_value) == false) {
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
