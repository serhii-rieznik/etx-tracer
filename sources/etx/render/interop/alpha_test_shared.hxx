#pragma once

#include "material.hxx"

#ifndef ETX_ALPHA_TEST_SHARED_CONTEXT_TYPE
# error "ETX_ALPHA_TEST_SHARED_CONTEXT_TYPE must be defined before including alpha_test_shared.hxx"
#endif

#ifndef ETX_ALPHA_TEST_SHARED_MATERIAL_CLASS
# error "ETX_ALPHA_TEST_SHARED_MATERIAL_CLASS must be defined before including alpha_test_shared.hxx"
#endif

#ifndef ETX_ALPHA_TEST_SHARED_MATERIAL_OPACITY
# error "ETX_ALPHA_TEST_SHARED_MATERIAL_OPACITY must be defined before including alpha_test_shared.hxx"
#endif

#ifndef ETX_ALPHA_TEST_SHARED_SCATTERING_IMAGE_INDEX
# error "ETX_ALPHA_TEST_SHARED_SCATTERING_IMAGE_INDEX must be defined before including alpha_test_shared.hxx"
#endif

#ifndef ETX_ALPHA_TEST_SHARED_IMAGE_HAS_ALPHA
# error "ETX_ALPHA_TEST_SHARED_IMAGE_HAS_ALPHA must be defined before including alpha_test_shared.hxx"
#endif

#ifndef ETX_ALPHA_TEST_SHARED_EVALUATE_ALPHA
# error "ETX_ALPHA_TEST_SHARED_EVALUATE_ALPHA must be defined before including alpha_test_shared.hxx"
#endif

#ifndef ETX_ALPHA_TEST_SHARED_RND
# error "ETX_ALPHA_TEST_SHARED_RND must be defined before including alpha_test_shared.hxx"
#endif

ETX_SHARED_INLINE bool alpha_test_shared_pass(ETX_INOUT(ETX_ALPHA_TEST_SHARED_CONTEXT_TYPE, context)) {
  uint32_t material_class = ETX_ALPHA_TEST_SHARED_MATERIAL_CLASS(context);
  if (material_class == MaterialClass::Void) {
    return true;
  }

  float material_alpha = ETX_ALPHA_TEST_SHARED_MATERIAL_OPACITY(context);
  float alpha_diffuse = 1.0f;
  uint32_t scattering_image_index = ETX_ALPHA_TEST_SHARED_SCATTERING_IMAGE_INDEX(context);
  if ((scattering_image_index != kInvalidIndex) && ETX_ALPHA_TEST_SHARED_IMAGE_HAS_ALPHA(context, scattering_image_index)) {
    alpha_diffuse = ETX_ALPHA_TEST_SHARED_EVALUATE_ALPHA(context, scattering_image_index);
  }

  float alpha_test_value = alpha_diffuse * material_alpha;
  return alpha_test_value <= ETX_ALPHA_TEST_SHARED_RND(context);
}
