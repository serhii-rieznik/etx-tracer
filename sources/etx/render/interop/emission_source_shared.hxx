#pragma once

#include "material_scattering_shared.hxx"

#ifndef ETX_EMISSION_SOURCE_SHARED_CONTEXT_TYPE
# error "ETX_EMISSION_SOURCE_SHARED_CONTEXT_TYPE must be defined before including emission_source_shared.hxx"
#endif

#ifndef ETX_EMISSION_SOURCE_SHARED_CAN_SAMPLE_SPECTRUM
# error "ETX_EMISSION_SOURCE_SHARED_CAN_SAMPLE_SPECTRUM must be defined before including emission_source_shared.hxx"
#endif

#ifndef ETX_EMISSION_SOURCE_SHARED_LOAD_SPECTRUM_INTEGRATED
# error "ETX_EMISSION_SOURCE_SHARED_LOAD_SPECTRUM_INTEGRATED must be defined before including emission_source_shared.hxx"
#endif

#ifndef ETX_EMISSION_SOURCE_SHARED_LOAD_SPECTRUM_SPECTRAL
# error "ETX_EMISSION_SOURCE_SHARED_LOAD_SPECTRUM_SPECTRAL must be defined before including emission_source_shared.hxx"
#endif

#ifndef ETX_EMISSION_SOURCE_SHARED_CAN_APPLY_IMAGE
# error "ETX_EMISSION_SOURCE_SHARED_CAN_APPLY_IMAGE must be defined before including emission_source_shared.hxx"
#endif

#ifndef ETX_EMISSION_SOURCE_SHARED_EVALUATE_IMAGE_RGB
# error "ETX_EMISSION_SOURCE_SHARED_EVALUATE_IMAGE_RGB must be defined before including emission_source_shared.hxx"
#endif

ETX_SHARED_INLINE float3 emission_source_shared_evaluate_integrated(
  ETX_IN(ETX_EMISSION_SOURCE_SHARED_CONTEXT_TYPE, context), uint32_t emission_spectrum_index, uint32_t emission_image_index, ETX_IN(float2, uv)) {
  if (ETX_EMISSION_SOURCE_SHARED_CAN_SAMPLE_SPECTRUM(context, emission_spectrum_index) == false) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  float3 result = ETX_EMISSION_SOURCE_SHARED_LOAD_SPECTRUM_INTEGRATED(context, emission_spectrum_index);
  bool apply_image = ETX_EMISSION_SOURCE_SHARED_CAN_APPLY_IMAGE(context, emission_image_index);
  float3 image_rgb = float3(1.0f, 1.0f, 1.0f);
  if (apply_image) {
    image_rgb = ETX_EMISSION_SOURCE_SHARED_EVALUATE_IMAGE_RGB(context, emission_image_index, uv);
  }

  return material_scattering_shared_apply_integrated_clamped_ao(result, image_rgb, apply_image, 1.0f);
}

ETX_SHARED_INLINE SpectralResponse emission_source_shared_evaluate_spectral(
  ETX_IN(ETX_EMISSION_SOURCE_SHARED_CONTEXT_TYPE, context), uint32_t emission_spectrum_index, uint32_t emission_image_index, ETX_IN(float2, uv), ETX_IN(SpectralQuery, spect)) {
  SpectralResponse zero_value = spectral_response_zero(spect);
  if (ETX_EMISSION_SOURCE_SHARED_CAN_SAMPLE_SPECTRUM(context, emission_spectrum_index) == false) {
    return zero_value;
  }

  SpectralResponse result = ETX_EMISSION_SOURCE_SHARED_LOAD_SPECTRUM_SPECTRAL(context, emission_spectrum_index, spect);
  bool apply_image = ETX_EMISSION_SOURCE_SHARED_CAN_APPLY_IMAGE(context, emission_image_index);
  float3 image_rgb = float3(1.0f, 1.0f, 1.0f);
  if (apply_image) {
    image_rgb = ETX_EMISSION_SOURCE_SHARED_EVALUATE_IMAGE_RGB(context, emission_image_index, uv);
  }

  return material_scattering_shared_apply_spectral_clamped_ao(spect, result, image_rgb, apply_image, 1.0f);
}
