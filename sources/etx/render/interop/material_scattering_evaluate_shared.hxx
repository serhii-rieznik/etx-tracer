#pragma once

#include "material_scattering_shared.hxx"

#ifndef ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_CONTEXT_TYPE
# error "ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_CONTEXT_TYPE must be defined before including material_scattering_evaluate_shared.hxx"
#endif

#ifndef ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_TRY_LOAD_STATE
# error "ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_TRY_LOAD_STATE must be defined before including material_scattering_evaluate_shared.hxx"
#endif

#ifndef ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_LOAD_SPECTRUM_INTEGRATED
# error "ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_LOAD_SPECTRUM_INTEGRATED must be defined before including material_scattering_evaluate_shared.hxx"
#endif

#ifndef ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_LOAD_SPECTRUM_SPECTRAL
# error "ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_LOAD_SPECTRUM_SPECTRAL must be defined before including material_scattering_evaluate_shared.hxx"
#endif

#ifndef ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_CAN_APPLY_IMAGE
# error "ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_CAN_APPLY_IMAGE must be defined before including material_scattering_evaluate_shared.hxx"
#endif

#ifndef ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_EVALUATE_IMAGE_RGB
# error "ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_EVALUATE_IMAGE_RGB must be defined before including material_scattering_evaluate_shared.hxx"
#endif

ETX_SHARED_INLINE SpectralResponse material_scattering_evaluate_shared_spectral(ETX_IN(ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_CONTEXT_TYPE, context), uint32_t material_index,
  ETX_IN(float2, uv), float ao, ETX_IN(SpectralQuery, spect), ETX_IN(SpectralResponse, fallback_value)) {
  uint32_t scattering_spectrum_index = kInvalidIndex;
  uint32_t scattering_image_index = kInvalidIndex;
  if (ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_TRY_LOAD_STATE(context, material_index, scattering_spectrum_index, scattering_image_index) == false) {
    return fallback_value;
  }

  SpectralResponse scattering_value = ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_LOAD_SPECTRUM_SPECTRAL(context, scattering_spectrum_index, spect);
  bool has_image = ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_CAN_APPLY_IMAGE(context, scattering_image_index);
  float3 image_rgb = float3(1.0f, 1.0f, 1.0f);
  if (has_image) {
    image_rgb = ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_EVALUATE_IMAGE_RGB(context, scattering_image_index, uv);
  }

  return material_scattering_shared_apply_spectral_clamped_ao(spect, scattering_value, image_rgb, has_image, ao);
}

ETX_SHARED_INLINE float3 material_scattering_evaluate_shared_integrated_or_fallback(ETX_IN(ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_CONTEXT_TYPE, context), uint32_t material_index,
  ETX_IN(float2, uv), float ao, ETX_IN(float3, fallback_color)) {
  uint32_t scattering_spectrum_index = kInvalidIndex;
  uint32_t scattering_image_index = kInvalidIndex;
  if (ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_TRY_LOAD_STATE(context, material_index, scattering_spectrum_index, scattering_image_index) == false) {
    return fallback_color;
  }

  float3 scattering_integrated = ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_LOAD_SPECTRUM_INTEGRATED(context, scattering_spectrum_index);
  bool has_image = ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_CAN_APPLY_IMAGE(context, scattering_image_index);
  float3 image_rgb = float3(1.0f, 1.0f, 1.0f);
  if (has_image) {
    image_rgb = ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_EVALUATE_IMAGE_RGB(context, scattering_image_index, uv);
  }

  return material_scattering_shared_apply_integrated_clamped_ao(scattering_integrated, image_rgb, has_image, ao);
}
