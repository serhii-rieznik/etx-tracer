#pragma once

#include "material.hxx"

ETX_SHARED_INLINE bool alpha_test_shared_pass(ETX_INOUT(AlphaTestContext, context)) {
  uint32_t material_class = alpha_test_material_class(context);
  if (material_class == MaterialClass::Void) {
    return true;
  }

  float material_alpha = alpha_test_material_opacity(context);
  float alpha_mask = 1.0f;
  uint32_t mask_image_index = alpha_test_mask_image_index(context);
  if (mask_image_index != kInvalidIndex) {
    alpha_mask = alpha_test_evaluate_mask(context, mask_image_index);
  }

  float alpha_test_value = alpha_mask * material_alpha;
  if (alpha_test_value <= 0.0f) {
    return true;
  }
  if (alpha_test_value >= 1.0f) {
    return false;
  }
  return alpha_test_value <= alpha_test_rnd(context);
}
