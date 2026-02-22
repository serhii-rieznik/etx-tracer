#pragma once

#ifndef ETX_ALPHA_TEST_ACCESS_SHARED_CONTEXT_TYPE
# error "ETX_ALPHA_TEST_ACCESS_SHARED_CONTEXT_TYPE must be defined before including alpha_test_access_shared.hxx"
#endif

#ifndef ETX_ALPHA_TEST_ACCESS_SHARED_ALPHA_CONTEXT_TYPE
# error "ETX_ALPHA_TEST_ACCESS_SHARED_ALPHA_CONTEXT_TYPE must be defined before including alpha_test_access_shared.hxx"
#endif

#ifndef ETX_ALPHA_TEST_ACCESS_SHARED_HAS_REQUIRED_SCENE_BUFFERS
# error "ETX_ALPHA_TEST_ACCESS_SHARED_HAS_REQUIRED_SCENE_BUFFERS must be defined before including alpha_test_access_shared.hxx"
#endif

#ifndef ETX_ALPHA_TEST_ACCESS_SHARED_MAKE_ALPHA_CONTEXT
# error "ETX_ALPHA_TEST_ACCESS_SHARED_MAKE_ALPHA_CONTEXT must be defined before including alpha_test_access_shared.hxx"
#endif

#ifndef ETX_ALPHA_TEST_ACCESS_SHARED_PASS
# error "ETX_ALPHA_TEST_ACCESS_SHARED_PASS must be defined before including alpha_test_access_shared.hxx"
#endif

#ifndef ETX_ALPHA_TEST_ACCESS_SHARED_ALPHA_CONTEXT_SEED
# error "ETX_ALPHA_TEST_ACCESS_SHARED_ALPHA_CONTEXT_SEED must be defined before including alpha_test_access_shared.hxx"
#endif

ETX_SHARED_INLINE bool alpha_test_access_shared_pass(
  ETX_IN(ETX_ALPHA_TEST_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t material_index, ETX_IN(float2, uv), ETX_INOUT(uint32_t, seed)) {
  if (ETX_ALPHA_TEST_ACCESS_SHARED_HAS_REQUIRED_SCENE_BUFFERS(context) == false) {
    return false;
  }

  ETX_ALPHA_TEST_ACCESS_SHARED_ALPHA_CONTEXT_TYPE alpha_context = ETX_ALPHA_TEST_ACCESS_SHARED_MAKE_ALPHA_CONTEXT(context, material_index, uv, seed);
  bool result = ETX_ALPHA_TEST_ACCESS_SHARED_PASS(alpha_context);
  seed = ETX_ALPHA_TEST_ACCESS_SHARED_ALPHA_CONTEXT_SEED(alpha_context);
  return result;
}
