#pragma once

#if defined(__cplusplus)

# define ETX_SHARED_INLINE     inline
# define ETX_IN(type, name)    const type& name
# define ETX_OUT(type, name)   type& name
# define ETX_INOUT(type, name) type& name
# define ETX_ALIGNED           alignas(16)
# define ETX_INIT(...)         = __VA_ARGS__
# define ETX_ZERO(type) \
   {                    \
   }
# define ETX_ZERO_INIT(type, name) type name = ETX_ZERO(type)
# define ETX_STATIC_CONST          constexpr

#else

# define ETX_SHARED_INLINE         inline
# define ETX_IN(type, name)        in type name
# define ETX_OUT(type, name)       out type name
# define ETX_INOUT(type, name)     inout type name
# define ETX_ALIGNED               /* */
# define ETX_INIT(...)             /* */
# define ETX_ZERO(type)            (type)0
# define ETX_ZERO_INIT(type, name) type name = ETX_ZERO(type)
# define ETX_STATIC_CONST          static const

#endif

ETX_STATIC_CONST float kQuarterPi = 0.78539816339744830961566084581988f;
ETX_STATIC_CONST float kHalfPi = 1.5707963267948966192313216916398f;
ETX_STATIC_CONST float kPi = 3.1415926535897932384626433832795f;
ETX_STATIC_CONST float kDoublePi = 6.283185307179586476925286766559f;
ETX_STATIC_CONST float kSqrt2 = 1.4142135623730950488016887242097f;
ETX_STATIC_CONST float kInvPi = 0.31830988618379067153776752674503f;
ETX_STATIC_CONST float kSqrtPI = 1.7724538509055160272981674833411f;
ETX_STATIC_CONST float kEpsilon = 1.192092896e-07f;
ETX_STATIC_CONST float kMaxFloat = 3.402823466e+38f;
ETX_STATIC_CONST float kMaxHalf = 65504.0f;
ETX_STATIC_CONST float kInvMaxHalf = 1.0f / kMaxHalf;
ETX_STATIC_CONST float kRayEpsilon = 15.0f / (kMaxHalf - 1.0f);
ETX_STATIC_CONST float kDeltaAlphaTreshold = 1.0e-4f;
ETX_STATIC_CONST float kGoldenRatio = 1.6180339887498948482f;
ETX_STATIC_CONST uint32_t kInvalidIndex = ~0u;

#if defined(__cplusplus)
# include <stdint.h>
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
  printf("Validation failed: %s (%s) at %s [%u]\n", name, v ? "true" : "false", filename, line);
}

template <>
ETX_SHARED_INLINE void print_value<float>(const char* name, const float& v, const char* filename, uint32_t line) {
  printf("Validation failed: %s (%f) at %s [%u]\n", name, v, filename, line);
}

template <>
ETX_SHARED_INLINE void print_value<float2>(const char* name, const float2& v, const char* filename, uint32_t line) {
  printf("Validation failed: %s (%f %f) at %s [%u]\n", name, v.x, v.y, filename, line);
}

template <>
ETX_SHARED_INLINE void print_value<float3>(const char* name, const float3& v, const char* filename, uint32_t line) {
  printf("Validation failed: %s (%f %f %f) at %s [%u]\n", name, v.x, v.y, v.z, filename, line);
}

template <>
ETX_SHARED_INLINE void print_value<float4>(const char* name, const float4& v, const char* filename, uint32_t line) {
  printf("Validation failed: %s (%f %f %f %f) at %s [%u]\n", name, v.x, v.y, v.z, v.w, filename, line);
}

ETX_SHARED_INLINE bool isfinite(complex t) {
  return isfinite(t.real()) && isfinite(t.imag());
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

ETX_STATIC_CONST ETX_SHARED_INLINE float3 make_float3(float x, float y, float z) {
  return float3(x, y, z);
}
