#pragma once

#include "image.hxx"

#ifndef ETX_IMAGE_SAMPLE_SHARED_CONTEXT_TYPE
# error "ETX_IMAGE_SAMPLE_SHARED_CONTEXT_TYPE must be defined before including image_sample_shared.hxx"
#endif

#ifndef ETX_IMAGE_SAMPLE_SHARED_Y_COUNT
# error "ETX_IMAGE_SAMPLE_SHARED_Y_COUNT must be defined before including image_sample_shared.hxx"
#endif

#ifndef ETX_IMAGE_SAMPLE_SHARED_X_COUNT
# error "ETX_IMAGE_SAMPLE_SHARED_X_COUNT must be defined before including image_sample_shared.hxx"
#endif

#ifndef ETX_IMAGE_SAMPLE_SHARED_IMAGE_FSIZE
# error "ETX_IMAGE_SAMPLE_SHARED_IMAGE_FSIZE must be defined before including image_sample_shared.hxx"
#endif

#ifndef ETX_IMAGE_SAMPLE_SHARED_SAMPLE_Y
# error "ETX_IMAGE_SAMPLE_SHARED_SAMPLE_Y must be defined before including image_sample_shared.hxx"
#endif

#ifndef ETX_IMAGE_SAMPLE_SHARED_SAMPLE_X
# error "ETX_IMAGE_SAMPLE_SHARED_SAMPLE_X must be defined before including image_sample_shared.hxx"
#endif

#ifndef ETX_IMAGE_SAMPLE_SHARED_CDF_Y
# error "ETX_IMAGE_SAMPLE_SHARED_CDF_Y must be defined before including image_sample_shared.hxx"
#endif

#ifndef ETX_IMAGE_SAMPLE_SHARED_CDF_X
# error "ETX_IMAGE_SAMPLE_SHARED_CDF_X must be defined before including image_sample_shared.hxx"
#endif

ETX_SHARED_INLINE bool image_sample_shared_distribution(
  ETX_INOUT(ETX_IMAGE_SAMPLE_SHARED_CONTEXT_TYPE, context), ETX_IN(float2, rnd), ETX_OUT(float, image_pdf), ETX_OUT(uint2, location), ETX_OUT(float2, uv)) {
  image_pdf = 0.0f;
  location = uint2(0u, 0u);
  uv = rnd;

  uint32_t y_count = ETX_IMAGE_SAMPLE_SHARED_Y_COUNT(context);
  if (y_count == 0u) {
    return false;
  }

  float y_pdf = 0.0f;
  location.y = ETX_IMAGE_SAMPLE_SHARED_SAMPLE_Y(context, rnd.y, y_pdf);
  if ((location.y == kInvalidIndex) || (location.y >= y_count)) {
    return false;
  }

  uint32_t x_count = ETX_IMAGE_SAMPLE_SHARED_X_COUNT(context, location.y);
  if (x_count == 0u) {
    return false;
  }

  float x_pdf = 0.0f;
  location.x = ETX_IMAGE_SAMPLE_SHARED_SAMPLE_X(context, location.y, rnd.x, x_pdf);
  if ((location.x == kInvalidIndex) || (location.x >= x_count)) {
    return false;
  }

  uint32_t x1_index = min(location.x + 1u, x_count - 1u);
  uint32_t y1_index = min(location.y + 1u, y_count - 1u);

  float x0_cdf = ETX_IMAGE_SAMPLE_SHARED_CDF_X(context, location.y, location.x);
  float x1_cdf = ETX_IMAGE_SAMPLE_SHARED_CDF_X(context, location.y, x1_index);
  float y0_cdf = ETX_IMAGE_SAMPLE_SHARED_CDF_Y(context, location.y);
  float y1_cdf = ETX_IMAGE_SAMPLE_SHARED_CDF_Y(context, y1_index);

  uv = image_sample_uv_from_distribution(rnd, location, ETX_IMAGE_SAMPLE_SHARED_IMAGE_FSIZE(context), x0_cdf, x1_cdf, y0_cdf, y1_cdf);
  image_pdf = x_pdf * y_pdf;
  return true;
}
