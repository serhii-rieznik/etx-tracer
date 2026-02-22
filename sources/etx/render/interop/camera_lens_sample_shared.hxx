#pragma once

#include "math_shared.hxx"

#ifndef ETX_CAMERA_LENS_SAMPLE_SHARED_CONTEXT_TYPE
# error "ETX_CAMERA_LENS_SAMPLE_SHARED_CONTEXT_TYPE must be defined before including camera_lens_sample_shared.hxx"
#endif

#ifndef ETX_CAMERA_LENS_SAMPLE_SHARED_TRY_SAMPLE_IMAGE_UV
# error "ETX_CAMERA_LENS_SAMPLE_SHARED_TRY_SAMPLE_IMAGE_UV must be defined before including camera_lens_sample_shared.hxx"
#endif

ETX_SHARED_INLINE bool camera_lens_sample_shared_enabled(float lens_radius, float focal_distance) {
  return (lens_radius > kEpsilon) && (focal_distance > kEpsilon);
}

ETX_SHARED_INLINE float2 camera_lens_sample_shared(
  ETX_IN(ETX_CAMERA_LENS_SAMPLE_SHARED_CONTEXT_TYPE, context), float lens_radius, float focal_distance, uint32_t lens_image, ETX_IN(float2, rnd)) {
  if (camera_lens_sample_shared_enabled(lens_radius, focal_distance) == false) {
    return float2(0.0f, 0.0f);
  }

  if (lens_image == kInvalidIndex) {
    return sample_disk(rnd);
  }

  float2 image_uv = float2(0.0f, 0.0f);
  bool sampled_image = ETX_CAMERA_LENS_SAMPLE_SHARED_TRY_SAMPLE_IMAGE_UV(context, lens_image, rnd, image_uv);
  if (sampled_image == false) {
    return sample_disk(rnd);
  }

  return image_uv * 2.0f - 1.0f;
}
