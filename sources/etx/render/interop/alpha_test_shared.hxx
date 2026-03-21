#pragma once

#include "material.hxx"

ETX_SHARED_INLINE bool alpha_test_shared_pass(ETX_INOUT(AlphaTestContext, context)) {
  uint32_t material_class = alpha_test_material_class(context);
  if (material_class == MaterialClass::Void) {
    return true;
  }

  float material_alpha = alpha_test_material_opacity(context);
  float alpha_diffuse = 1.0f;
  uint32_t scattering_image_index = alpha_test_scattering_image_index(context);
  if ((scattering_image_index != kInvalidIndex) && alpha_test_image_has_alpha(context, scattering_image_index)) {
    alpha_diffuse = alpha_test_evaluate_alpha(context, scattering_image_index);
  }

  float alpha_test_value = alpha_diffuse * material_alpha;
  if (alpha_test_value <= 0.0f) {
    return true;
  }
  if (alpha_test_value >= 1.0f) {
    return false;
  }
  return alpha_test_value <= alpha_test_rnd(context);
}
