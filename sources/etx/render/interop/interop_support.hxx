#pragma once

#include "interop_constants.hxx"

#if (ETX_CPP)
# include <etx/render/shared/base.hxx>

using complex = std::complex<float>;

ETX_SHARED_INLINE complex make_complex(float r, float i) {
  return {r, i};
}

ETX_SHARED_INLINE complex complex_sqrt(complex c) {
  return std::sqrt(c);
}

ETX_SHARED_INLINE complex complex_cos(complex c) {
  return std::cos(c);
}

ETX_SHARED_INLINE complex complex_exp(complex c) {
  return std::exp(c);
}

ETX_SHARED_INLINE float complex_abs(complex c) {
  return std::abs(c);
}

ETX_SHARED_INLINE float complex_norm(complex c) {
  return std::norm(c);
}

namespace etx {

template <class T>
ETX_SHARED_INLINE void print_value(const char* name, const T& v, const char* filename, uint32_t line);

template <>
ETX_SHARED_INLINE void print_value<bool>(const char* name, const bool& v, const char* filename, uint32_t line) {
  ETX_STD printf("Validation failed: %s (%s) at %s [%u]\n", name, v ? "true" : "false", filename, line);
}

template <>
ETX_SHARED_INLINE void print_value<float>(const char* name, const float& v, const char* filename, uint32_t line) {
  ETX_STD printf("Validation failed: %s (%f) at %s [%u]\n", name, v, filename, line);
}

template <>
ETX_SHARED_INLINE void print_value<float2>(const char* name, const float2& v, const char* filename, uint32_t line) {
  ETX_STD printf("Validation failed: %s (%f %f) at %s [%u]\n", name, v.x, v.y, filename, line);
}

template <>
ETX_SHARED_INLINE void print_value<float3>(const char* name, const float3& v, const char* filename, uint32_t line) {
  ETX_STD printf("Validation failed: %s (%f %f %f) at %s [%u]\n", name, v.x, v.y, v.z, filename, line);
}

template <>
ETX_SHARED_INLINE void print_value<float4>(const char* name, const float4& v, const char* filename, uint32_t line) {
  ETX_STD printf("Validation failed: %s (%f %f %f %f) at %s [%u]\n", name, v.x, v.y, v.z, v.w, filename, line);
}

ETX_SHARED_INLINE bool isfinite(complex t) {
  return ETX_STD isfinite(t.real()) && ETX_STD isfinite(t.imag());
}

ETX_SHARED_INLINE bool value_is_correct(const complex& v) {
  return value_is_correct(v.real()) && value_is_correct(v.imag());
}

ETX_SHARED_INLINE bool valid_value(complex t) {
  return isfinite(t.real()) && isfinite(t.imag());
}

}  // namespace etx

#else

struct complex {
  float re;
  float im;
};

ETX_SHARED_INLINE complex make_complex(float r, float i) {
  complex result;
  result.re = r;
  result.im = i;
  return result;
}

#endif

ETX_SHARED_INLINE float3 make_float3(float x, float y, float z) {
  return float3(x, y, z);
}
