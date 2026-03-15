#pragma once

struct ETX_ALIGNED ImageSampleAccess {
  float2 uv ETX_INIT({});
  float pdf ETX_INIT(0.0f);
  uint2 location ETX_INIT({});
  float4 value ETX_INIT({});
};

ETX_SHARED_INLINE ImageSampleAccess image_sample_access_default(ETX_IN(float2, uv)) {
  ETX_ZERO_INIT(ImageSampleAccess, result);
  result.uv = uv;
  return result;
}
