#pragma once

#if (ETX_RENDER_BASE_INCLUDED)
#else
# error This file should not be included separately. Use etx/render/shared/base.hxx instead
#endif

#include <math.h>
#include <complex.h>
#include <string.h>

template <class t>
struct vector2 {
  t x, y;
};

template <class t>
struct vector3 {
  t x, y, z;
};

template <class t>
struct vector4 {
  t x, y, z, w;
};

struct SphericalCoordinates {
  float phi;
  float theta;
  float r;
};

#if (ETX_NVCC_COMPILER)
# if defined(__NVCC__)
#  include <thrust/complex.h>
#  define STD_NS thrust
# else
#  define STD_NS cuda::std
# endif
#else
# define STD_NS std
using float2 = vector2<float>;
using float3 = vector3<float>;
using float4 = vector4<float>;
using int2 = vector2<int32_t>;
using int3 = vector3<int32_t>;
using int4 = vector4<int32_t>;
using uint2 = vector2<uint32_t>;
using uint3 = vector3<uint32_t>;
using uint4 = vector4<uint32_t>;
using short2 = vector2<int16_t>;
using short3 = vector3<int16_t>;
using short4 = vector4<int16_t>;
using ushort2 = vector2<uint16_t>;
using ushort3 = vector3<uint16_t>;
using ushort4 = vector4<uint16_t>;
using byte2 = vector2<int8_t>;
using byte3 = vector3<int8_t>;
using byte4 = vector4<int8_t>;
using ubyte2 = vector2<uint8_t>;
using ubyte3 = vector3<uint8_t>;
using ubyte4 = vector4<uint8_t>;
using char2 = byte2;
using char3 = byte3;
using char4 = byte4;
using uchar2 = ubyte2;
using uchar3 = ubyte3;
using uchar4 = ubyte4;
#endif

using complex = STD_NS::complex<float>;

ETX_GPU_CODE complex complex_sqrt(complex c) {
  return STD_NS::sqrt(c);
}
ETX_GPU_CODE complex complex_cos(complex c) {
  return STD_NS::cos(c);
}
ETX_GPU_CODE complex complex_exp(complex c) {
  return STD_NS::exp(c);
}
ETX_GPU_CODE float complex_abs(complex c) {
  return STD_NS::abs(c);
}
ETX_GPU_CODE float complex_norm(complex c) {
  return STD_NS::norm(c);
}

struct float3x3 {
  float3 col[3] ETX_EMPTY_INIT;
};

union float4x4 {
  float4 col[4] ETX_EMPTY_INIT;
  float val[16];
};

namespace etx {

constexpr float kQuarterPi = 0.78539816339744830961566084581988f;
constexpr float kHalfPi = 1.5707963267948966192313216916398f;
constexpr float kPi = 3.1415926535897932384626433832795f;
constexpr float kDoublePi = 6.283185307179586476925286766559f;
constexpr float kSqrt2 = 1.4142135623730950488016887242097f;
constexpr float kInvPi = 0.31830988618379067153776752674503f;
constexpr float kSqrtPI = 1.7724538509055160272981674833411f;
constexpr float kEpsilon = 1.192092896e-07f;
constexpr float kMaxFloat = 3.402823466e+38f;
constexpr float kMaxHalf = 65504.0f;
constexpr float kRayEpsilon = 1.0f / 65535.0f;
constexpr float kDeltaAlphaTreshold = 1.0e-4f;

constexpr uint32_t kInvalidIndex = ~0u;

template <class t>
ETX_GPU_CODE constexpr t min(t a, t b) {
  return a < b ? a : b;
}

template <class t>
ETX_GPU_CODE constexpr t max(t a, t b) {
  return a > b ? a : b;
}

template <class t>
ETX_GPU_CODE constexpr t clamp(t val, t min_val, t max_val) {
  return (val < min_val) ? min_val : (val > max_val ? max_val : val);
}

ETX_GPU_CODE constexpr float saturate(float val) {
  return clamp(val, 0.0f, 1.0f);
}
ETX_GPU_CODE constexpr float sign(float val) {
  return val >= 0.0f ? 1.0f : -1.0f;
}

/*
 * Float2
 */
#define ETX_V2(V, C)                                               \
  constexpr ETX_GPU_CODE V operator+(const V& a, C b) {            \
    return {a.x + b, a.y + b};                                     \
  }                                                                \
  constexpr ETX_GPU_CODE V operator+(C b, const V& a) {            \
    return {a.x + b, a.y + b};                                     \
  }                                                                \
  constexpr ETX_GPU_CODE V operator+(const V& a, const V& b) {     \
    return {a.x + b.x, a.y + b.y};                                 \
  }                                                                \
  constexpr ETX_GPU_CODE V& operator+=(V& a, const V& b) {         \
    a.x += b.x;                                                    \
    a.y += b.y;                                                    \
    return a;                                                      \
  }                                                                \
  constexpr ETX_GPU_CODE V operator-(const V& a) {                 \
    return {-a.x, -a.y};                                           \
  }                                                                \
  constexpr ETX_GPU_CODE V operator-(const V& a, C b) {            \
    return {a.x - b, a.y - b};                                     \
  }                                                                \
  constexpr ETX_GPU_CODE V operator-(C b, const V& a) {            \
    return {b - a.x, b - a.y};                                     \
  }                                                                \
  constexpr ETX_GPU_CODE V operator-(const V& a, const V& b) {     \
    return {a.x - b.x, a.y - b.y};                                 \
  }                                                                \
  constexpr ETX_GPU_CODE V& operator-=(V& a, const V& b) {         \
    a.x -= b.x;                                                    \
    a.y -= b.y;                                                    \
    return a;                                                      \
  }                                                                \
  constexpr ETX_GPU_CODE V operator*(const V& a, C b) {            \
    return {a.x * b, a.y * b};                                     \
  }                                                                \
  constexpr ETX_GPU_CODE V operator*(C b, const V& a) {            \
    return {a.x * b, a.y * b};                                     \
  }                                                                \
  constexpr ETX_GPU_CODE V operator*(const V& a, const V& b) {     \
    return {a.x * b.x, a.y * b.y};                                 \
  }                                                                \
  constexpr ETX_GPU_CODE V& operator*=(V& a, const V& b) {         \
    a.x *= b.x;                                                    \
    a.y *= b.y;                                                    \
    return a;                                                      \
  }                                                                \
  constexpr ETX_GPU_CODE V& operator*=(V& a, C b) {                \
    a.x *= b;                                                      \
    a.y *= b;                                                      \
    return a;                                                      \
  }                                                                \
  constexpr ETX_GPU_CODE V operator/(const V& a, C b) {            \
    return {a.x / b, a.y / b};                                     \
  }                                                                \
  constexpr ETX_GPU_CODE V operator/(C b, const V& a) {            \
    return {a.x / b, a.y / b};                                     \
  }                                                                \
  constexpr ETX_GPU_CODE V operator/(const V& a, const V& b) {     \
    return {a.x / b.x, a.y / b.y};                                 \
  }                                                                \
  constexpr ETX_GPU_CODE V& operator/=(V& a, const V& b) {         \
    a.x /= b.x;                                                    \
    a.y /= b.y;                                                    \
    return a;                                                      \
  }                                                                \
  constexpr ETX_GPU_CODE V& operator/=(V& a, const C b) {          \
    a.x /= b;                                                      \
    a.y /= b;                                                      \
    return a;                                                      \
  }                                                                \
  constexpr ETX_GPU_CODE bool operator==(const V& a, const V& b) { \
    return (a.x == b.x) && (a.y == b.y);                           \
  }

/*
 * Float3
 */
#define ETX_V3(V, C)                                               \
  constexpr ETX_GPU_CODE V operator+(const V& a, C b) {            \
    return {a.x + b, a.y + b, a.z + b};                            \
  }                                                                \
  constexpr ETX_GPU_CODE V operator+(C b, const V& a) {            \
    return {a.x + b, a.y + b, a.z + b};                            \
  }                                                                \
  constexpr ETX_GPU_CODE V operator+(const V& a, const V& b) {     \
    return {a.x + b.x, a.y + b.y, a.z + b.z};                      \
  }                                                                \
  constexpr ETX_GPU_CODE V& operator+=(V& a, const V& b) {         \
    a.x += b.x;                                                    \
    a.y += b.y;                                                    \
    a.z += b.z;                                                    \
    return a;                                                      \
  }                                                                \
  constexpr ETX_GPU_CODE V& operator+=(V& a, const C b) {          \
    a.x += b;                                                      \
    a.y += b;                                                      \
    a.z += b;                                                      \
    return a;                                                      \
  }                                                                \
  constexpr ETX_GPU_CODE V operator-(const V& a) {                 \
    return {-a.x, -a.y, -a.z};                                     \
  }                                                                \
  constexpr ETX_GPU_CODE V operator-(const V& a, C b) {            \
    return {a.x - b, a.y - b, a.z - b};                            \
  }                                                                \
  constexpr ETX_GPU_CODE V operator-(C b, const V& a) {            \
    return {b - a.x, b - a.y, b - a.z};                            \
  }                                                                \
  constexpr ETX_GPU_CODE V operator-(const V& a, const V& b) {     \
    return {a.x - b.x, a.y - b.y, a.z - b.z};                      \
  }                                                                \
  constexpr ETX_GPU_CODE V& operator-=(V& a, const V& b) {         \
    a.x -= b.x;                                                    \
    a.y -= b.y;                                                    \
    a.z -= b.z;                                                    \
    return a;                                                      \
  }                                                                \
  constexpr ETX_GPU_CODE V& operator-=(V& a, C b) {                \
    a.x -= b;                                                      \
    a.y -= b;                                                      \
    a.z -= b;                                                      \
    return a;                                                      \
  }                                                                \
  constexpr ETX_GPU_CODE V operator*(const V& a, C b) {            \
    return {a.x * b, a.y * b, a.z * b};                            \
  }                                                                \
  constexpr ETX_GPU_CODE V operator*(C b, const V& a) {            \
    return {a.x * b, a.y * b, a.z * b};                            \
  }                                                                \
  constexpr ETX_GPU_CODE V operator*(const V& a, const V& b) {     \
    return {a.x * b.x, a.y * b.y, a.z * b.z};                      \
  }                                                                \
  constexpr ETX_GPU_CODE V& operator*=(V& a, const V& b) {         \
    return (a = {a.x * b.x, a.y * b.y, a.z * b.z});                \
  }                                                                \
  constexpr ETX_GPU_CODE V& operator*=(V& a, C b) {                \
    a.x *= b;                                                      \
    a.y *= b;                                                      \
    a.z *= b;                                                      \
    return a;                                                      \
  }                                                                \
  constexpr ETX_GPU_CODE V operator/(const V& a, C b) {            \
    return {a.x / b, a.y / b, a.z / b};                            \
  }                                                                \
  constexpr ETX_GPU_CODE V operator/(C b, const V& a) {            \
    return {b / a.x, b / a.y, b / a.z};                            \
  }                                                                \
  constexpr ETX_GPU_CODE V operator/(const V& a, const V& b) {     \
    return {a.x / b.x, a.y / b.y, a.z / b.z};                      \
  }                                                                \
  constexpr ETX_GPU_CODE V& operator/=(V& a, const V& b) {         \
    return (a = {a.x / b.x, a.y / b.y, a.z / b.z});                \
  }                                                                \
  constexpr ETX_GPU_CODE V& operator/=(V& a, const C b) {          \
    return (a = {a.x / b, a.y / b, a.z / b});                      \
  }                                                                \
  constexpr ETX_GPU_CODE bool operator==(const V& a, const V& b) { \
    return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);           \
  }

/*
 * Float4
 */
#define ETX_V4(V, C)                                           \
  constexpr ETX_GPU_CODE V operator+(const V& a, C b) {        \
    return {a.x + b, a.y + b, a.z + b, a.w + b};               \
  }                                                            \
  constexpr ETX_GPU_CODE V operator+(C b, const V& a) {        \
    return {a.x + b, a.y + b, a.z + b, a.w + b};               \
  }                                                            \
  constexpr ETX_GPU_CODE V operator+(const V& a, const V& b) { \
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};       \
  }                                                            \
  constexpr V& operator+=(V& a, const V& b) {                  \
    a.x += b.x;                                                \
    a.y += b.y;                                                \
    a.z += b.z;                                                \
    a.w += b.w;                                                \
    return a;                                                  \
  }                                                            \
  constexpr ETX_GPU_CODE V& operator+=(V& a, C b) {            \
    a.x += b;                                                  \
    a.y += b;                                                  \
    a.z += b;                                                  \
    a.w += b;                                                  \
    return a;                                                  \
  }                                                            \
  constexpr ETX_GPU_CODE V operator-(const V& a) {             \
    return {-a.x, -a.y, -a.z, -a.w};                           \
  }                                                            \
  constexpr ETX_GPU_CODE V operator-(const V& a, C b) {        \
    return {a.x - b, a.y - b, a.z - b, a.w - b};               \
  }                                                            \
  constexpr ETX_GPU_CODE V operator-(C b, const V& a) {        \
    return {b - a.x, b - a.y, b - a.z, b - a.w};               \
  }                                                            \
  constexpr ETX_GPU_CODE V operator-(const V& a, const V& b) { \
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};       \
  }                                                            \
  constexpr ETX_GPU_CODE V& operator-=(V& a, const V& b) {     \
    a.x -= b.x;                                                \
    a.y -= b.y;                                                \
    a.z -= b.z;                                                \
    a.w -= b.w;                                                \
    return a;                                                  \
  }                                                            \
  constexpr ETX_GPU_CODE V& operator-=(V& a, C b) {            \
    a.x -= b;                                                  \
    a.y -= b;                                                  \
    a.z -= b;                                                  \
    a.w -= b;                                                  \
    return a;                                                  \
  }                                                            \
  constexpr ETX_GPU_CODE V operator*(const V& a, C b) {        \
    return {a.x * b, a.y * b, a.z * b, a.w * b};               \
  }                                                            \
  constexpr ETX_GPU_CODE V operator*(C b, const V& a) {        \
    return {a.x * b, a.y * b, a.z * b, a.w * b};               \
  }                                                            \
  constexpr ETX_GPU_CODE V operator*(const V& a, const V& b) { \
    return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};       \
  }                                                            \
  constexpr ETX_GPU_CODE V& operator*=(V& a, const V& b) {     \
    a.x *= b.x;                                                \
    a.y *= b.y;                                                \
    a.z *= b.z;                                                \
    a.w *= b.w;                                                \
    return a;                                                  \
  }                                                            \
  constexpr ETX_GPU_CODE V& operator*=(V& a, C b) {            \
    a.x *= b;                                                  \
    a.y *= b;                                                  \
    a.z *= b;                                                  \
    a.w *= b;                                                  \
    return a;                                                  \
  }                                                            \
  constexpr ETX_GPU_CODE V operator/(const V& a, C b) {        \
    return {a.x / b, a.y / b, a.z / b, a.w / b};               \
  }                                                            \
  constexpr ETX_GPU_CODE V operator/(C b, const V& a) {        \
    return {b / a.x, b / a.y, b / a.z, b / a.w};               \
  }                                                            \
  constexpr ETX_GPU_CODE V operator/(const V& a, const V& b) { \
    return {a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};       \
  }                                                            \
  constexpr ETX_GPU_CODE V& operator/=(V& a, const V& b) {     \
    a.x /= b.x;                                                \
    a.y /= b.y;                                                \
    a.z /= b.z;                                                \
    a.w /= b.w;                                                \
    return a;                                                  \
  }                                                            \
  constexpr ETX_GPU_CODE V& operator/=(V& a, C b) {            \
    a.x /= b;                                                  \
    a.y /= b;                                                  \
    a.z /= b;                                                  \
    a.w /= b;                                                  \
    return a;                                                  \
  }

ETX_V2(float2, float)
ETX_V3(float3, float)
ETX_V4(float4, float)

ETX_V2(int2, int32_t)
ETX_V3(int3, int32_t)
ETX_V4(int4, int32_t)

#pragma warning(push)
#pragma warning(disable : 4146)
ETX_V2(uint2, uint32_t)
ETX_V3(uint3, uint32_t)
ETX_V4(uint4, uint32_t)
#pragma warning(pop)

#define ETX_FUNC_2(func, cfunc)               \
  ETX_GPU_CODE float2 func(const float2& a) { \
    return {cfunc(a.x), cfunc(a.y)};          \
  }
#define ETX_FUNC_3(func, cfunc)                  \
  ETX_GPU_CODE float3 func(const float3& a) {    \
    return {cfunc(a.x), cfunc(a.y), cfunc(a.z)}; \
  }
#define ETX_FUNC_4(func, cfunc)                              \
  ETX_GPU_CODE float4 func(const float4& a) {                \
    return {cfunc(a.x), cfunc(a.y), cfunc(a.z), cfunc(a.w)}; \
  }

#define ETX_UNARY_FUNC(func, cfunc) \
  ETX_FUNC_2(func, cfunc)           \
  ETX_FUNC_3(func, cfunc)           \
  ETX_FUNC_4(func, cfunc)

#define ETX_BINARY_FUNC_2(name, impl)                          \
  ETX_GPU_CODE float2 name(const float2& a, const float2& b) { \
    return {                                                   \
      impl(a.x, b.x),                                          \
      impl(a.y, b.y),                                          \
    };                                                         \
  }                                                            \
  ETX_GPU_CODE float2 name(const float2& a, const float b) {   \
    return {                                                   \
      impl(a.x, b),                                            \
      impl(a.y, b),                                            \
    };                                                         \
  }

#define ETX_BINARY_FUNC_3(name, impl)                          \
  ETX_GPU_CODE float3 name(const float3& a, const float3& b) { \
    return {                                                   \
      impl(a.x, b.x),                                          \
      impl(a.y, b.y),                                          \
      impl(a.z, b.z),                                          \
    };                                                         \
  }                                                            \
  ETX_GPU_CODE float3 name(const float3& a, const float b) {   \
    return {                                                   \
      impl(a.x, b),                                            \
      impl(a.y, b),                                            \
      impl(a.z, b),                                            \
    };                                                         \
  }

#define ETX_BINARY_FUNC_4(name, impl)                          \
  ETX_GPU_CODE float4 name(const float4& a, const float4& b) { \
    return {                                                   \
      impl(a.x, b.x),                                          \
      impl(a.y, b.y),                                          \
      impl(a.z, b.z),                                          \
      impl(a.w, b.w),                                          \
    };                                                         \
  }                                                            \
  ETX_GPU_CODE float4 name(const float4& a, const float b) {   \
    return {                                                   \
      impl(a.x, b),                                            \
      impl(a.y, b),                                            \
      impl(a.z, b),                                            \
      impl(a.w, b),                                            \
    };                                                         \
  }

#define ETX_BINARY_FUNC(name, impl) \
  ETX_BINARY_FUNC_2(name, impl)     \
  ETX_BINARY_FUNC_3(name, impl)     \
  ETX_BINARY_FUNC_4(name, impl)

ETX_UNARY_FUNC(abs, fabsf);
ETX_UNARY_FUNC(exp, expf);
ETX_UNARY_FUNC(sqrt, sqrtf);
ETX_UNARY_FUNC(sin, sinf);
ETX_UNARY_FUNC(cos, cosf);
ETX_UNARY_FUNC(floor, floorf);
ETX_UNARY_FUNC(saturate, saturate);
ETX_UNARY_FUNC(sign, sign);
ETX_UNARY_FUNC(atan, atanf);

ETX_BINARY_FUNC(max, fmaxf)
ETX_BINARY_FUNC(min, fminf)
ETX_BINARY_FUNC(pow, powf)
ETX_BINARY_FUNC(mod, fmodf)

ETX_GPU_CODE float dot(const float2& a, const float b) {
  return a.x * b + a.y * b;
}
ETX_GPU_CODE float dot(const float3& a, const float b) {
  return a.x * b + a.y * b + a.z * b;
}
ETX_GPU_CODE float dot(const float4& a, const float b) {
  return a.x * b + a.y * b + a.z * b + a.w * b;
}
ETX_GPU_CODE float dot(const float2& a, const float2& b) {
  return a.x * b.x + a.y * b.y;
}
ETX_GPU_CODE float dot(const float3& a, const float3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
ETX_GPU_CODE float dot(const float4& a, const float4& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

ETX_GPU_CODE float length(const float2& v) {
  return sqrtf(dot(v, v));
}
ETX_GPU_CODE float length(const float3& v) {
  return sqrtf(dot(v, v));
}

ETX_GPU_CODE float3 normalize(const float3& v) {
  return v / length(v);
}

ETX_GPU_CODE float3 reflect(const float3& v, const float3& n) {
  return v - (2.0f * dot(v, n)) * n;
}

ETX_GPU_CODE float3 cross(const float3& a, const float3& b) {
  return {
    a.y * b.z - b.y * a.z,
    a.z * b.x - b.z * a.x,
    a.x * b.y - b.x * a.y,
  };
}

ETX_GPU_CODE float2 lerp(const float2& a, const float2& b, float t) {
  float inv_t = 1.0f - t;
  return {
    a.x * inv_t + b.x * t,
    a.y * inv_t + b.y * t,
  };
}

ETX_GPU_CODE float3 lerp(const float3& a, const float3& b, float t) {
  float inv_t = 1.0f - t;
  return {
    a.x * inv_t + b.x * t,
    a.y * inv_t + b.y * t,
    a.z * inv_t + b.z * t,
  };
}

ETX_GPU_CODE float4 lerp(const float4& a, const float4& b, float t) {
  float inv_t = 1.0f - t;
  return {
    a.x * inv_t + b.x * t,
    a.y * inv_t + b.y * t,
    a.z * inv_t + b.z * t,
    a.w * inv_t + b.w * t,
  };
}

struct ETX_ALIGNED BoundingBox {
  float3 p_min ETX_EMPTY_INIT;
  float3 p_max ETX_EMPTY_INIT;

  ETX_GPU_CODE float3 to_local(const float3& p) const {
    return (p - p_min) / (p_max - p_min);
  }

  ETX_GPU_CODE float3 from_local(const float3& p) const {
    return p * (p_max - p_min) + p_min;
  }

  ETX_GPU_CODE bool contains(const float3& p) const {
    return (p.x >= p_min.x) && (p.y >= p_min.y) && (p.z >= p_min.z) &&  //
           (p.x <= p_max.x) && (p.y <= p_max.y) && (p.z <= p_max.z);
  }
};

#define ETX_MATH_INCLUDES 1
#include <etx/render/shared/vector_math.hxx>
#undef ETX_MATH_INCLUDES

struct Vertex {
  float3 pos;
  float3 nrm;
  float3 tan;
  float3 btn;
  float2 tex;
};

struct ETX_ALIGNED Triangle {
  uint32_t i[3] = {kInvalidIndex, kInvalidIndex, kInvalidIndex};
  float3 geo_n = {};
};

struct ETX_ALIGNED LocalFrame {
  enum : uint32_t {
    EnteringMaterial = 1u << 0u,
  };

  float3 tan = {};
  float3 btn = {};
  float3 nrm = {};
  uint32_t flags = 0u;

  ETX_GPU_CODE float3 to_local(const float3& v) const {
    return float3x3{float3{tan.x, btn.x, nrm.x}, float3{tan.y, btn.y, nrm.y}, float3{tan.z, btn.z, nrm.z}} * v;
  }

  ETX_GPU_CODE float3 from_local(const float3& v) const {
    return float3x3{float3{tan.x, tan.y, tan.z}, float3{btn.x, btn.y, btn.z}, float3{nrm.x, nrm.y, nrm.z}} * v;
  }

  ETX_GPU_CODE static float cos_theta(const float3& v) {
    return v.z;
  }

  ETX_GPU_CODE static float sin_theta(const float3& v) {
    return sqrtf(fmaxf(0.0f, 1.0f - cos_theta(v)));
  }

  bool entering_material() const {
    return (flags & EnteringMaterial) == EnteringMaterial;
  }
};

struct ETX_ALIGNED Ray {
  Ray() = default;

  ETX_GPU_CODE Ray(const float3& origin, const float3& direction)
    : o(origin)
    , d(direction) {
  }

  ETX_GPU_CODE Ray(const float3& origin, const float3& direction, float t_min, float t_max)
    : o(origin)
    , min_t(t_min)
    , d(direction)
    , max_t(t_max) {
  }

  float3 o = {};
  float min_t = kRayEpsilon;
  float3 d = {};
  float max_t = kMaxFloat;
};

struct ETX_ALIGNED IntersectionBase {
  float2 barycentric = {};
  uint32_t triangle_index = kInvalidIndex;
  float t = kMaxFloat;
};

struct ETX_ALIGNED Intersection : public Vertex {
  float3 barycentric = {};
  uint32_t triangle_index = ~0u;
  float3 w_i = {};
  float t = 0.0f;
  uint32_t material_index = ~0u;
  uint32_t emitter_index = ~0u;

  Intersection() = default;

  ETX_GPU_CODE Intersection(const Vertex& v)
    : Vertex(v) {
  }

  ETX_GPU_CODE float distance() const {
    return t;
  }
};

template <typename T>
ETX_GPU_CODE constexpr T lerp(T a, T b, float t) {
  return a * (1.0f - t) + b * t;
}

template <typename T>
ETX_GPU_CODE constexpr T sqr(T t) {
  return t * t;
}

ETX_GPU_CODE float3 to_float3(const float4& v) {
  return {v.x, v.y, v.z};
}

ETX_GPU_CODE float4 to_float4(const float3& v) {
  return {v.x, v.y, v.z, 1.0f};
}

ETX_GPU_CODE float4 to_float4(const ubyte4& v) {
  return {v.x / 255.0f, v.y / 255.0f, v.z / 255.0f, v.w / 255.0f};
}

ETX_GPU_CODE constexpr ubyte4 to_ubyte4(const float4& v) {
  return {
    static_cast<uint8_t>(saturate(v.x) * 255.0f),
    static_cast<uint8_t>(saturate(v.y) * 255.0f),
    static_cast<uint8_t>(saturate(v.z) * 255.0f),
    static_cast<uint8_t>(saturate(v.w) * 255.0f),
  };
}

ETX_GPU_CODE float3 hsv_to_rgb(const float3& hsv) {
  float h = hsv.x;
  float s = hsv.y;
  float v = hsv.z;
  return v * (1.0f - s * saturate(2.0f - abs(mod(h * 6.0f + float3{0.0f, 4.0f, 2.0f}, 6.0f) - 3.0f)));
}

ETX_GPU_CODE float luminance(const float3& value) {
  return value.x * 0.212671f + value.y * 0.715160f + value.z * 0.072169f;
}

ETX_GPU_CODE float3 orthonormalize(const float3& nrm, const float3& tan) {
  return normalize(tan - dot(tan, nrm) * nrm);
}

ETX_GPU_CODE auto orthonormal_basis(const float3& n) {
  struct basis {
    float3 u, v;
  };
  float3 a = normalize((n.x != n.y) || (n.x != n.z)                  //
                         ? float3{n.z - n.y, n.x - n.z, +n.y - n.x}  //
                         : float3{n.z - n.y, n.x + n.z, -n.y - n.x});
  float3 b = normalize(cross(n, a));
  return basis{a, b};
}

ETX_GPU_CODE float3 sample_cosine_distribution(const float2 rnd, float exponent) {
  float cos_theta = powf(rnd.x, 1.0f / (exponent + 1.0f));
  float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
  return {cosf(rnd.y * kDoublePi) * sin_theta, sinf(rnd.y * kDoublePi) * sin_theta, cos_theta};
}

ETX_GPU_CODE float3 sample_cosine_distribution(const float2 rnd, const float3& n, const float3& u, const float3& v, float exponent) {
  float3 l = sample_cosine_distribution(rnd, exponent);
  return u * l.x + v * l.y + n * l.z;
}

ETX_GPU_CODE float3 sample_cosine_distribution(const float2& rnd, const float3& n, float exponent) {
  auto basis = orthonormal_basis(n);
  return sample_cosine_distribution(rnd, n, basis.u, basis.v, exponent);
}

ETX_GPU_CODE float3 barycentrics(float2 bc) {
  return {1.0f - bc.x - bc.y, bc.x, bc.y};
}

ETX_GPU_CODE float3 random_barycentric(const float2 rnd) {
  float r1 = sqrtf(rnd.x);
  return {1.0f - r1, r1 * (1.0f - rnd.y), r1 * rnd.y};
}

ETX_GPU_CODE float2 sample_disk(const float2& rnd) {
  float2 offset = {2.0f * rnd.x - 1.0f, 2.0f * rnd.y - 1.0f};
  if ((offset.x == 0.0f) && (offset.y == 0.0f))
    return {};

  float r = 0.0f;
  float theta = 0.0f;
  if (fabsf(offset.x) > fabsf(offset.y)) {
    r = offset.x;
    theta = kQuarterPi * (offset.y / offset.x);
  } else {
    r = offset.y;
    theta = kHalfPi - kQuarterPi * (offset.x / offset.y);
  }

  return {r * cosf(theta), r * sinf(theta)};
}

ETX_GPU_CODE float2 sample_disk_uv(float xi0, float xi1) {
  float2 offset = {2.0f * xi0 - 1.0f, 2.0f * xi1 - 1.0f};
  if ((offset.x == 0.0f) && (offset.y == 0.0f))
    return {};

  float r = 0.0f;
  float theta = 0.0f;
  if (fabsf(offset.x) > fabsf(offset.y)) {
    r = offset.x;
    theta = kQuarterPi * (offset.y / offset.x);
  } else {
    r = offset.y;
    theta = kHalfPi - kQuarterPi * (offset.x / offset.y);
  }

  return {r * cosf(theta) * 0.5f + 0.5f, r * sinf(theta) * 0.5f + 0.5f};
}

ETX_GPU_CODE float2 projecected_coords(const float3& normal, const float3& in_dir, float sz, float csz) {
  if (sz == 0.0f) {
    return {};
  }

  auto basis = orthonormal_basis(normal);
  float result_u = dot(basis.u, in_dir) / (0.5f * sz * csz);
  float result_v = dot(basis.v, in_dir) / (0.5f * sz * csz);
  return {result_u, result_v};
}

ETX_GPU_CODE float2 disk_uv(const float3& normal, const float3& in_dir, float sz, float csz) {
  auto pc = projecected_coords(normal, in_dir, sz, csz);
  return saturate(pc * 0.5f + 0.5f);
}

ETX_GPU_CODE float3 orthogonalize(const float3& t, const float3& n) {
  return normalize(t - n * dot(n, t));
};

ETX_GPU_CODE float3 orthogonalize(const float3& t, const float3& b, const float3& n) {
  return normalize(t - n * dot(n, t)) * (dot(cross(n, t), b) < 0.0f ? -1.0f : 1.0f);
}

ETX_GPU_CODE bool isfinite(float t) {
  return ::isfinite(t);
}

ETX_GPU_CODE bool valid_value(float t) {
  return (t >= 0.0f) && isfinite(t);
}

ETX_GPU_CODE bool valid_value(const float2& v) {
  return valid_value(v.x) && valid_value(v.y);
}

ETX_GPU_CODE bool valid_value(const float3& v) {
  return valid_value(v.x) && valid_value(v.y) && valid_value(v.z);
}

ETX_GPU_CODE bool valid_value(const float4& v) {
  return valid_value(v.x) && valid_value(v.y) && valid_value(v.z) && valid_value(v.w);
}

ETX_GPU_CODE bool value_is_correct(float t) {
  return !std::isnan(t) && !std::isinf(t);
}

ETX_GPU_CODE bool value_is_correct(const float3& v) {
  return value_is_correct(v.x) && value_is_correct(v.y) && value_is_correct(v.z);
}

ETX_GPU_CODE bool is_valid_vector(const float3& v) {
  return value_is_correct(v) && (dot(v, v) > 0.0f);
}

ETX_GPU_CODE bool isfinite(const float2& v) {
  return isfinite(v.x) && isfinite(v.y);
}

ETX_GPU_CODE bool isfinite(const float3& v) {
  return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

ETX_GPU_CODE bool isfinite(const float4& v) {
  return isfinite(v.x) && isfinite(v.y) && isfinite(v.z) && valid_value(v.w);
}

ETX_GPU_CODE bool valid_value(complex t) {
  return isfinite(t.real()) && isfinite(t.imag());
}

ETX_GPU_CODE bool isfinite(complex t) {
  return isfinite(t.real()) && isfinite(t.imag());
}

ETX_GPU_CODE float to_float(uint32_t value) {
  float result;
  memcpy(&result, &value, sizeof(float));
  return result;
}

ETX_GPU_CODE float to_float(int32_t value) {
  float result;
  memcpy(&result, &value, sizeof(float));
  return result;
}

ETX_GPU_CODE uint32_t to_uint(float value) {
  uint32_t result;
  memcpy(&result, &value, sizeof(float));
  return result;
}

ETX_GPU_CODE uint32_t to_int(float value) {
  int32_t result;
  memcpy(&result, &value, sizeof(float));
  return result;
}

ETX_GPU_CODE float3 offset_ray(const float3& p, const float3& n) {
  constexpr float int_scale = 256.0f;
  constexpr float float_scale = 1.0f / 65536.0f;
  constexpr float origin = 1.0f / 32.0f;

  int32_t of_i_x = static_cast<int32_t>(int_scale * n.x);
  int32_t of_i_y = static_cast<int32_t>(int_scale * n.y);
  int32_t of_i_z = static_cast<int32_t>(int_scale * n.z);

  float p_i_x = to_float(to_int(p.x) + ((p.x > 0.0f) ? of_i_x : -of_i_x));
  float p_i_y = to_float(to_int(p.y) + ((p.y > 0.0f) ? of_i_y : -of_i_y));
  float p_i_z = to_float(to_int(p.z) + ((p.z > 0.0f) ? of_i_z : -of_i_z));

  return {
    fabsf(p.x) < origin ? p.x + float_scale * n.x : p_i_x,
    fabsf(p.y) < origin ? p.y + float_scale * n.y : p_i_y,
    fabsf(p.z) < origin ? p.z + float_scale * n.z : p_i_z,
  };
}

ETX_GPU_CODE float power_heuristic(float f, float g) {
  float f2 = f * f;
  float g2 = g * g;
  return saturate(f2 / (f2 + g2));
}

ETX_GPU_CODE float area_to_solid_angle_probability(const float3& dp, const float3& n, float collimation) {
  float distance_squared = dot(dp, dp);
  if (distance_squared <= kEpsilon)
    return 0.0f;

  float cos_t = powf(fabsf(dot(dp, n) / sqrtf(distance_squared)), collimation);
  return (cos_t > kEpsilon) ? (distance_squared / cos_t) : 0.0f;
}

ETX_GPU_CODE SphericalCoordinates to_spherical(const float3& dir) {
  float r = length(dir);
  return SphericalCoordinates{
    .phi = atan2f(dir.z, dir.x),
    .theta = asinf(dir.y / r),
    .r = r,
  };
}

ETX_GPU_CODE float3 from_spherical(const SphericalCoordinates& s) {
  float cos_p = cosf(s.phi);
  float sin_p = sinf(s.phi);
  float cos_t = cosf(s.theta);
  float sin_t = sinf(s.theta);
  return {
    s.r * cos_p * cos_t,
    s.r * sin_t,
    s.r * sin_p * cos_t,
  };
}

ETX_GPU_CODE float3 from_spherical(float phi, float theta) {
  return from_spherical({phi, theta, 1.0f});
}

ETX_GPU_CODE float3 uv_to_direction(const float2& uv, const float2& offset) {
  float phi = (uv.x * 2.0f - 1.0f) * kPi;
  float theta = (0.5f - uv.y) * kPi;
  return from_spherical(phi, theta);
}

ETX_GPU_CODE float2 direction_to_uv(const float3& dir, const float2& offset) {
  auto s = to_spherical(dir);
  float u = (s.phi / kPi + 1.0f) / 2.0f;
  float v = 0.5f - s.theta / kPi;
  return {u, v};
}

ETX_GPU_CODE uint64_t next_power_of_two(uint64_t v) {
  v--;
  v |= v >> 1llu;
  v |= v >> 2llu;
  v |= v >> 4llu;
  v |= v >> 8llu;
  v |= v >> 16llu;
  v++;
  return v;
}

ETX_GPU_CODE uint32_t next_power_of_two(uint32_t v) {
  v--;
  v |= v >> 1u;
  v |= v >> 2u;
  v |= v >> 4u;
  v |= v >> 8u;
  v |= v >> 16u;
  v++;
  return v;
}

ETX_GPU_CODE float distance_to_sphere(const float3& r_origin, const float3& r_direction, const float3& center, float radius) {
  float3 e = r_origin - center;
  float b = dot(r_direction, e);
  float d = (b * b) - dot(e, e) + (radius * radius);
  if (d < 0.0f) {
    return 0.0f;
  }
  d = sqrtf(d);
  float a0 = -b - d;
  float a1 = -b + d;
  return (a0 < 0.0f) ? ((a1 < 0.0f) ? 0.0f : a1) : a0;
}

ETX_GPU_CODE float gamma_to_linear(float value) {
  return value <= 0.04045f ? value / 12.92f : powf((value + 0.055f) / 1.055f, 2.4f);
}

ETX_GPU_CODE float linear_to_gamma(float value) {
  return value <= 0.0031308f ? 12.92f * value : 1.055f * powf(value, 1.0f / 2.4f) - 0.055f;
}

ETX_GPU_CODE float3 gamma_to_linear(const float3& value) {
  return {
    gamma_to_linear(value.x),
    gamma_to_linear(value.y),
    gamma_to_linear(value.z),
  };
}

ETX_GPU_CODE float3 linear_to_gamma(const float3& value) {
  return {
    linear_to_gamma(value.x),
    linear_to_gamma(value.y),
    linear_to_gamma(value.z),
  };
}

ETX_GPU_CODE float3 slerp(const float3& start, const float3& end, float t) {
  constexpr float kThreshold = 100.0f * kEpsilon;

  float dot_product = clamp(dot(start, end), -1.0f, 1.0f);
  float theta = acosf(dot_product);
  if (fabsf(theta) <= kThreshold) {
    return lerp(start, end, theta);
  }

  float sinTheta = sinf(theta);
  float sinThetaInv = 1.0f / sinTheta;
  float a = sinf((1.0f - t) * theta) * sinThetaInv;
  float b = sinf(t * theta) * sinThetaInv;
  return float3(start.x * a + end.x * b, start.y * a + end.y * b, start.z * a + end.z * b);
}

ETX_GPU_CODE float3 project_point_plane(const float3& v, const float3& plane_n, const float3& plane_o) {
  float3 v_to_o = v - plane_o;
  float distance = dot(v_to_o, plane_n);
  return v - distance * plane_n;
}

ETX_GPU_CODE float intersect_ray_plane(const float3& ray_o, const float3& ray_d, const float3& plane_n, const float3& plane_o) {
  float denom = dot(plane_n, ray_d);
  return (denom > -kEpsilon) ? -1.0f : (dot(plane_n, plane_o - ray_o)) / denom;
}

}  // namespace etx
