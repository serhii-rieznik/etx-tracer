#pragma once

#if (ETX_RENDER_BASE_INCLUDED)
#else
# error This file should not be included separately. Use etx/render/shared/base.hxx instead
#endif

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

template <class t>
ETX_SHARED_INLINE constexpr t min(t a, t b) {
  return a < b ? a : b;
}

template <class t>
ETX_SHARED_INLINE constexpr t max(t a, t b) {
  return a > b ? a : b;
}

template <class t>
ETX_SHARED_INLINE constexpr t clamp(t val, t min_val, t max_val) {
  return (val < min_val) ? min_val : (val > max_val ? max_val : val);
}

ETX_SHARED_INLINE constexpr float saturate(const float& val) {
  return clamp(val, 0.0f, 1.0f);
}

ETX_SHARED_INLINE constexpr float sign(float val) {
  return val >= 0.0f ? 1.0f : -1.0f;
}

struct float3x3 {
  float3 col[3] ETX_EMPTY_INIT;
};

union float4x4 {
  float4 col[4] ETX_EMPTY_INIT;
  float val[16];
};

/*
 * Float2
 */
#define ETX_V2(V, C)                                                    \
  constexpr ETX_SHARED_INLINE V operator+(const V& a, const C b) {      \
    return {a.x + b, a.y + b};                                          \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V operator+(const C b, const V& a) {      \
    return {a.x + b, a.y + b};                                          \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V operator+(const V& a, const V& b) {     \
    return {a.x + b.x, a.y + b.y};                                      \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V& operator+=(V& a, const V& b) {         \
    a.x += b.x;                                                         \
    a.y += b.y;                                                         \
    return a;                                                           \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V operator-(const V& a) {                 \
    return {-a.x, -a.y};                                                \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V operator-(const V& a, const C b) {      \
    return {a.x - b, a.y - b};                                          \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V operator-(const C b, const V& a) {      \
    return {b - a.x, b - a.y};                                          \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V operator-(const V& a, const V& b) {     \
    return {a.x - b.x, a.y - b.y};                                      \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V& operator-=(V& a, const V& b) {         \
    a.x -= b.x;                                                         \
    a.y -= b.y;                                                         \
    return a;                                                           \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V operator*(const V& a, const C b) {      \
    return {a.x * b, a.y * b};                                          \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V operator*(const C b, const V& a) {      \
    return {a.x * b, a.y * b};                                          \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V operator*(const V& a, const V& b) {     \
    return {a.x * b.x, a.y * b.y};                                      \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V& operator*=(V& a, const V& b) {         \
    a.x *= b.x;                                                         \
    a.y *= b.y;                                                         \
    return a;                                                           \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V& operator*=(V& a, const C b) {          \
    a.x *= b;                                                           \
    a.y *= b;                                                           \
    return a;                                                           \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V operator/(const V& a, const C b) {      \
    return {a.x / b, a.y / b};                                          \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V operator/(const C b, const V& a) {      \
    return {a.x / b, a.y / b};                                          \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V operator/(const V& a, const V& b) {     \
    return {a.x / b.x, a.y / b.y};                                      \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V& operator/=(V& a, const V& b) {         \
    a.x /= b.x;                                                         \
    a.y /= b.y;                                                         \
    return a;                                                           \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V& operator/=(V& a, const C b) {          \
    a.x /= b;                                                           \
    a.y /= b;                                                           \
    return a;                                                           \
  }                                                                     \
  constexpr ETX_SHARED_INLINE bool operator==(const V& a, const V& b) { \
    return (a.x == b.x) && (a.y == b.y);                                \
  }

/*
 * Float3
 */
#define ETX_V3(V, C)                                                    \
  constexpr ETX_SHARED_INLINE V operator+(const V& a, const C b) {      \
    return {a.x + b, a.y + b, a.z + b};                                 \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V operator+(const C b, const V& a) {      \
    return {a.x + b, a.y + b, a.z + b};                                 \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V operator+(const V& a, const V& b) {     \
    return {a.x + b.x, a.y + b.y, a.z + b.z};                           \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V& operator+=(V& a, const V& b) {         \
    a.x += b.x;                                                         \
    a.y += b.y;                                                         \
    a.z += b.z;                                                         \
    return a;                                                           \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V& operator+=(V& a, const C b) {          \
    a.x += b;                                                           \
    a.y += b;                                                           \
    a.z += b;                                                           \
    return a;                                                           \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V operator-(const V& a) {                 \
    return {-a.x, -a.y, -a.z};                                          \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V operator-(const V& a, const C b) {      \
    return {a.x - b, a.y - b, a.z - b};                                 \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V operator-(C b, const V& a) {            \
    return {b - a.x, b - a.y, b - a.z};                                 \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V operator-(const V& a, const V& b) {     \
    return {a.x - b.x, a.y - b.y, a.z - b.z};                           \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V& operator-=(V& a, const V& b) {         \
    a.x -= b.x;                                                         \
    a.y -= b.y;                                                         \
    a.z -= b.z;                                                         \
    return a;                                                           \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V& operator-=(V& a, const C b) {          \
    a.x -= b;                                                           \
    a.y -= b;                                                           \
    a.z -= b;                                                           \
    return a;                                                           \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V operator*(const V& a, const C b) {      \
    return {a.x * b, a.y * b, a.z * b};                                 \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V operator*(const C b, const V& a) {      \
    return {a.x * b, a.y * b, a.z * b};                                 \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V operator*(const V& a, const V& b) {     \
    return {a.x * b.x, a.y * b.y, a.z * b.z};                           \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V& operator*=(V& a, const V& b) {         \
    return (a = {a.x * b.x, a.y * b.y, a.z * b.z});                     \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V& operator*=(V& a, C b) {                \
    a.x *= b;                                                           \
    a.y *= b;                                                           \
    a.z *= b;                                                           \
    return a;                                                           \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V operator/(const V& a, const C b) {      \
    return {a.x / b, a.y / b, a.z / b};                                 \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V operator/(const C b, const V& a) {      \
    return {b / a.x, b / a.y, b / a.z};                                 \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V operator/(const V& a, const V& b) {     \
    return {a.x / b.x, a.y / b.y, a.z / b.z};                           \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V& operator/=(V& a, const V& b) {         \
    return (a = {a.x / b.x, a.y / b.y, a.z / b.z});                     \
  }                                                                     \
  constexpr ETX_SHARED_INLINE V& operator/=(V& a, const C b) {          \
    return (a = {a.x / b, a.y / b, a.z / b});                           \
  }                                                                     \
  constexpr ETX_SHARED_INLINE bool operator==(const V& a, const V& b) { \
    return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);                \
  }

/*
 * Float4
 */
#define ETX_V4(V, C)                                                \
  constexpr ETX_SHARED_INLINE V operator+(const V& a, const C b) {  \
    return {a.x + b, a.y + b, a.z + b, a.w + b};                    \
  }                                                                 \
  constexpr ETX_SHARED_INLINE V operator+(const C b, const V& a) {  \
    return {a.x + b, a.y + b, a.z + b, a.w + b};                    \
  }                                                                 \
  constexpr ETX_SHARED_INLINE V operator+(const V& a, const V& b) { \
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};            \
  }                                                                 \
  constexpr V& operator+=(V& a, const V& b) {                       \
    a.x += b.x;                                                     \
    a.y += b.y;                                                     \
    a.z += b.z;                                                     \
    a.w += b.w;                                                     \
    return a;                                                       \
  }                                                                 \
  constexpr ETX_SHARED_INLINE V& operator+=(V& a, const C b) {      \
    a.x += b;                                                       \
    a.y += b;                                                       \
    a.z += b;                                                       \
    a.w += b;                                                       \
    return a;                                                       \
  }                                                                 \
  constexpr ETX_SHARED_INLINE V operator-(const V& a) {             \
    return {-a.x, -a.y, -a.z, -a.w};                                \
  }                                                                 \
  constexpr ETX_SHARED_INLINE V operator-(const V& a, const C b) {  \
    return {a.x - b, a.y - b, a.z - b, a.w - b};                    \
  }                                                                 \
  constexpr ETX_SHARED_INLINE V operator-(const C b, const V& a) {  \
    return {b - a.x, b - a.y, b - a.z, b - a.w};                    \
  }                                                                 \
  constexpr ETX_SHARED_INLINE V operator-(const V& a, const V& b) { \
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};            \
  }                                                                 \
  constexpr ETX_SHARED_INLINE V& operator-=(V& a, const V& b) {     \
    a.x -= b.x;                                                     \
    a.y -= b.y;                                                     \
    a.z -= b.z;                                                     \
    a.w -= b.w;                                                     \
    return a;                                                       \
  }                                                                 \
  constexpr ETX_SHARED_INLINE V& operator-=(V& a, const C b) {      \
    a.x -= b;                                                       \
    a.y -= b;                                                       \
    a.z -= b;                                                       \
    a.w -= b;                                                       \
    return a;                                                       \
  }                                                                 \
  constexpr ETX_SHARED_INLINE V operator*(const V& a, const C b) {  \
    return {a.x * b, a.y * b, a.z * b, a.w * b};                    \
  }                                                                 \
  constexpr ETX_SHARED_INLINE V operator*(const C b, const V& a) {  \
    return {a.x * b, a.y * b, a.z * b, a.w * b};                    \
  }                                                                 \
  constexpr ETX_SHARED_INLINE V operator*(const V& a, const V& b) { \
    return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};            \
  }                                                                 \
  constexpr ETX_SHARED_INLINE V& operator*=(V& a, const V& b) {     \
    a.x *= b.x;                                                     \
    a.y *= b.y;                                                     \
    a.z *= b.z;                                                     \
    a.w *= b.w;                                                     \
    return a;                                                       \
  }                                                                 \
  constexpr ETX_SHARED_INLINE V& operator*=(V& a, const C b) {      \
    a.x *= b;                                                       \
    a.y *= b;                                                       \
    a.z *= b;                                                       \
    a.w *= b;                                                       \
    return a;                                                       \
  }                                                                 \
  constexpr ETX_SHARED_INLINE V operator/(const V& a, const C b) {  \
    return {a.x / b, a.y / b, a.z / b, a.w / b};                    \
  }                                                                 \
  constexpr ETX_SHARED_INLINE V operator/(const C b, const V& a) {  \
    return {b / a.x, b / a.y, b / a.z, b / a.w};                    \
  }                                                                 \
  constexpr ETX_SHARED_INLINE V operator/(const V& a, const V& b) { \
    return {a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};            \
  }                                                                 \
  constexpr ETX_SHARED_INLINE V& operator/=(V& a, const V& b) {     \
    a.x /= b.x;                                                     \
    a.y /= b.y;                                                     \
    a.z /= b.z;                                                     \
    a.w /= b.w;                                                     \
    return a;                                                       \
  }                                                                 \
  constexpr ETX_SHARED_INLINE V& operator/=(V& a, const C b) {      \
    a.x /= b;                                                       \
    a.y /= b;                                                       \
    a.z /= b;                                                       \
    a.w /= b;                                                       \
    return a;                                                       \
  }

ETX_V2(float2, float)
ETX_V3(float3, float)
ETX_V4(float4, float)

ETX_V2(int2, int32_t)
ETX_V3(int3, int32_t)
ETX_V4(int4, int32_t)

// #pragma warning(push)
// #pragma warning(disable : 4146)
ETX_V2(uint2, uint32_t)
ETX_V3(uint3, uint32_t)
ETX_V4(uint4, uint32_t)
// #pragma warning(pop)

#define ETX_FUNC_2(func, cfunc)                    \
  ETX_SHARED_INLINE float2 func(const float2& a) { \
    return {cfunc(a.x), cfunc(a.y)};               \
  }
#define ETX_FUNC_3(func, cfunc)                    \
  ETX_SHARED_INLINE float3 func(const float3& a) { \
    return {cfunc(a.x), cfunc(a.y), cfunc(a.z)};   \
  }
#define ETX_FUNC_4(func, cfunc)                              \
  ETX_SHARED_INLINE float4 func(const float4& a) {           \
    return {cfunc(a.x), cfunc(a.y), cfunc(a.z), cfunc(a.w)}; \
  }

#define ETX_UNARY_FUNC(func, cfunc) \
  ETX_FUNC_2(func, cfunc)           \
  ETX_FUNC_3(func, cfunc)           \
  ETX_FUNC_4(func, cfunc)

#define ETX_BINARY_FUNC_2(name, impl)                               \
  ETX_SHARED_INLINE float2 name(const float2& a, const float2& b) { \
    return {                                                        \
      impl(a.x, b.x),                                               \
      impl(a.y, b.y),                                               \
    };                                                              \
  }                                                                 \
  ETX_SHARED_INLINE float2 name(const float2& a, const float b) {   \
    return {                                                        \
      impl(a.x, b),                                                 \
      impl(a.y, b),                                                 \
    };                                                              \
  }

#define ETX_BINARY_FUNC_3(name, impl)                               \
  ETX_SHARED_INLINE float3 name(const float3& a, const float3& b) { \
    return {                                                        \
      impl(a.x, b.x),                                               \
      impl(a.y, b.y),                                               \
      impl(a.z, b.z),                                               \
    };                                                              \
  }                                                                 \
  ETX_SHARED_INLINE float3 name(const float3& a, const float b) {   \
    return {                                                        \
      impl(a.x, b),                                                 \
      impl(a.y, b),                                                 \
      impl(a.z, b),                                                 \
    };                                                              \
  }

#define ETX_BINARY_FUNC_4(name, impl)                               \
  ETX_SHARED_INLINE float4 name(const float4& a, const float4& b) { \
    return {                                                        \
      impl(a.x, b.x),                                               \
      impl(a.y, b.y),                                               \
      impl(a.z, b.z),                                               \
      impl(a.w, b.w),                                               \
    };                                                              \
  }                                                                 \
  ETX_SHARED_INLINE float4 name(const float4& a, const float b) {   \
    return {                                                        \
      impl(a.x, b),                                                 \
      impl(a.y, b),                                                 \
      impl(a.z, b),                                                 \
      impl(a.w, b),                                                 \
    };                                                              \
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

ETX_SHARED_INLINE float dot(const float2& a, const float b) {
  return a.x * b + a.y * b;
}
ETX_SHARED_INLINE float dot(const float3& a, const float b) {
  return a.x * b + a.y * b + a.z * b;
}
ETX_SHARED_INLINE float dot(const float4& a, const float b) {
  return a.x * b + a.y * b + a.z * b + a.w * b;
}
ETX_SHARED_INLINE float dot(const float2& a, const float2& b) {
  return a.x * b.x + a.y * b.y;
}
ETX_SHARED_INLINE float dot(const float3& a, const float3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
ETX_SHARED_INLINE float dot(const float4& a, const float4& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
ETX_SHARED_INLINE float length(const float2& v) {
  return sqrtf(dot(v, v));
}
ETX_SHARED_INLINE float length(const float3& v) {
  return sqrtf(dot(v, v));
}
ETX_SHARED_INLINE float2 normalize(const float2& v) {
  return v / length(v);
}
ETX_SHARED_INLINE float3 normalize(const float3& v) {
  return v / length(v);
}
ETX_SHARED_INLINE float3 reflect(const float3& v, const float3& n) {
  return v - (2.0f * dot(v, n)) * n;
}
ETX_SHARED_INLINE float3 cross(const float3& a, const float3& b) {
  return {
    a.y * b.z - b.y * a.z,
    a.z * b.x - b.z * a.x,
    a.x * b.y - b.x * a.y,
  };
}
ETX_SHARED_INLINE float2 lerp(const float2& a, const float2& b, float t) {
  float inv_t = 1.0f - t;
  return {
    a.x * inv_t + b.x * t,
    a.y * inv_t + b.y * t,
  };
}
ETX_SHARED_INLINE float3 lerp(const float3& a, const float3& b, float t) {
  float inv_t = 1.0f - t;
  return {
    a.x * inv_t + b.x * t,
    a.y * inv_t + b.y * t,
    a.z * inv_t + b.z * t,
  };
}
ETX_SHARED_INLINE float4 lerp(const float4& a, const float4& b, float t) {
  float inv_t = 1.0f - t;
  return {
    a.x * inv_t + b.x * t,
    a.y * inv_t + b.y * t,
    a.z * inv_t + b.z * t,
    a.w * inv_t + b.w * t,
  };
}

#include <etx/render/interop/math_shared.hxx>

namespace etx {

#define ETX_MATH_INCLUDES 1
#include <etx/render/shared/vector_math.hxx>
#undef ETX_MATH_INCLUDES

template <typename T>
ETX_SHARED_INLINE constexpr T lerp(T a, T b, float t) {
  return a * (1.0f - t) + b * t;
}

template <typename T>
ETX_SHARED_INLINE constexpr T sqr(T t) {
  return t * t;
}

ETX_SHARED_INLINE float3 to_float3(const float4& v) {
  return {v.x, v.y, v.z};
}

ETX_SHARED_INLINE float4 to_float4(const float3& v) {
  return {v.x, v.y, v.z, 1.0f};
}

ETX_SHARED_INLINE float4 to_float4(const ubyte4& v) {
  return {v.x / 255.0f, v.y / 255.0f, v.z / 255.0f, v.w / 255.0f};
}

ETX_SHARED_INLINE constexpr ubyte4 to_ubyte4(const float4& v) {
  return {
    static_cast<uint8_t>(saturate(v.x) * 255.0f),
    static_cast<uint8_t>(saturate(v.y) * 255.0f),
    static_cast<uint8_t>(saturate(v.z) * 255.0f),
    static_cast<uint8_t>(saturate(v.w) * 255.0f),
  };
}

ETX_SHARED_INLINE float3 hsv_to_rgb(const float3& hsv) {
  float h = hsv.x;
  float s = hsv.y;
  float v = hsv.z;
  return v * (1.0f - s * saturate(2.0f - abs(mod(h * 6.0f + float3{0.0f, 4.0f, 2.0f}, 6.0f) - 3.0f)));
}

ETX_SHARED_INLINE bool isfinite(float t) {
  return ::isfinite(t);
}

ETX_SHARED_INLINE bool valid_value(bool t) {
  return t;
}

ETX_SHARED_INLINE bool valid_value(float t) {
  return (t >= 0.0f) && isfinite(t);
}

ETX_SHARED_INLINE bool valid_value(const float2& v) {
  return valid_value(v.x) && valid_value(v.y);
}

ETX_SHARED_INLINE bool valid_value(const float3& v) {
  return valid_value(v.x) && valid_value(v.y) && valid_value(v.z);
}

ETX_SHARED_INLINE bool valid_value(const float4& v) {
  return valid_value(v.x) && valid_value(v.y) && valid_value(v.z) && valid_value(v.w);
}

ETX_SHARED_INLINE bool value_is_correct(float t) {
  return !std::isnan(t) && !std::isinf(t);
}

ETX_SHARED_INLINE bool value_is_correct(const float2& v) {
  return value_is_correct(v.x) && value_is_correct(v.y);
}

ETX_SHARED_INLINE bool value_is_correct(const float3& v) {
  return value_is_correct(v.x) && value_is_correct(v.y) && value_is_correct(v.z);
}

ETX_SHARED_INLINE bool value_is_correct(const float4& v) {
  return value_is_correct(v.x) && value_is_correct(v.y) && value_is_correct(v.z) && value_is_correct(v.w);
}

ETX_SHARED_INLINE bool is_valid_vector(const float3& v) {
  return value_is_correct(v) && (dot(v, v) > 0.0f);
}

ETX_SHARED_INLINE bool isfinite(const float2& v) {
  return isfinite(v.x) && isfinite(v.y);
}

ETX_SHARED_INLINE bool isfinite(const float3& v) {
  return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

ETX_SHARED_INLINE bool isfinite(const float4& v) {
  return isfinite(v.x) && isfinite(v.y) && isfinite(v.z) && valid_value(v.w);
}

ETX_SHARED_INLINE float3 slerp(const float3& start, const float3& end, float t) {
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

}  // namespace etx
