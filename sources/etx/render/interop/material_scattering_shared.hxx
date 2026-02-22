#pragma once

#include "spectrum.hxx"

ETX_SHARED_INLINE SpectralResponse material_scattering_shared_apply_spectral(
  ETX_IN(SpectralQuery, spect), ETX_IN(SpectralResponse, value), ETX_IN(float3, image_rgb), bool apply_image) {
  SpectralResponse result = value;
  if (apply_image) {
    result = spectral_response_apply_rgb_scale(spect, result, image_rgb);
  }
  return result;
}

ETX_SHARED_INLINE SpectralResponse material_scattering_shared_apply_spectral_clamped_ao(
  ETX_IN(SpectralQuery, spect), ETX_IN(SpectralResponse, value), ETX_IN(float3, image_rgb), bool apply_image, float ao) {
  SpectralResponse result = material_scattering_shared_apply_spectral(spect, value, image_rgb, apply_image);
  result = spectral_response_mul(result, ao);
  return spectral_response_clamp_non_negative(result);
}

ETX_SHARED_INLINE float3 material_scattering_shared_apply_integrated_clamped_ao(ETX_IN(float3, value), ETX_IN(float3, image_rgb), bool apply_image, float ao) {
  float3 result = spectral_rgb_clamp_non_negative(value);
  if (apply_image) {
    result *= image_rgb;
  }
  result = spectral_rgb_clamp_non_negative(result);
  return result * ao;
}
