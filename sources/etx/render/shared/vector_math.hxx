#pragma once

#if !defined(__CUDACC__)
# pragma warning(push)
# pragma warning(disable : 4146)
#endif

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

ETX_V2(float2, float)
ETX_V2(int2, int32_t)
ETX_V2(uint2, uint32_t)

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

ETX_V3(float3, float)
ETX_V3(int3, int32_t)
ETX_V3(uint3, uint32_t)

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

ETX_V4(float4, float)
ETX_V4(int4, int32_t)
ETX_V4(uint4, uint32_t)

#if !defined(__CUDACC__)
# pragma warning(pop)
#endif

ETX_GPU_CODE float2 abs(const float2& a) {
  return {fabsf(a.x), fabsf(a.y)};
}
ETX_GPU_CODE float3 abs(const float3& a) {
  return {fabsf(a.x), fabsf(a.y), fabsf(a.z)};
}
ETX_GPU_CODE float4 abs(const float4& a) {
  return {fabsf(a.x), fabsf(a.y), fabsf(a.z), fabsf(a.w)};
}

ETX_GPU_CODE float2 exp(const float2& a) {
  return {expf(a.x), expf(a.y)};
}
ETX_GPU_CODE float3 exp(const float3& a) {
  return {expf(a.x), expf(a.y), expf(a.z)};
}
ETX_GPU_CODE float4 exp(const float4& a) {
  return {expf(a.x), expf(a.y), expf(a.z), expf(a.w)};
}
ETX_GPU_CODE float2 sqrt(const float2& a) {
  return {sqrtf(a.x), sqrtf(a.y)};
}
ETX_GPU_CODE float3 sqrt(const float3& a) {
  return {sqrtf(a.x), sqrtf(a.y), sqrtf(a.z)};
}
ETX_GPU_CODE float4 sqrt(const float4& a) {
  return {sqrtf(a.x), sqrtf(a.y), sqrtf(a.z), sqrtf(a.w)};
}

ETX_GPU_CODE float2 sin(const float2& a) {
  return {sinf(a.x), sinf(a.y)};
}
ETX_GPU_CODE float3 sin(const float3& a) {
  return {sinf(a.x), sinf(a.y), sinf(a.z)};
}
ETX_GPU_CODE float4 sin(const float4& a) {
  return {sinf(a.x), sinf(a.y), sinf(a.z), sinf(a.w)};
}

ETX_GPU_CODE float2 cos(const float2& a) {
  return {cosf(a.x), cosf(a.y)};
}
ETX_GPU_CODE float3 cos(const float3& a) {
  return {cosf(a.x), cosf(a.y), cosf(a.z)};
}
ETX_GPU_CODE float4 cos(const float4& a) {
  return {cosf(a.x), cosf(a.y), cosf(a.z), cosf(a.w)};
}
ETX_GPU_CODE float dot(const float4& a, const float4& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w + b.w;
}

ETX_GPU_CODE float dot(const float2& a, const float2& b) {
  return a.x * b.x + a.y * b.y;
}

ETX_GPU_CODE float length(const float2& v) {
  return sqrtf(dot(v, v));
}

ETX_GPU_CODE float dot(const float3& a, const float3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
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

ETX_GPU_CODE float2 floor(const float2& v) {
  return {floorf(v.x), floorf(v.y)};
}
ETX_GPU_CODE float3 floor(const float3& v) {
  return {floorf(v.x), floorf(v.y), floorf(v.z)};
}

ETX_GPU_CODE float4 floor(const float4& v) {
  return {floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w)};
}

ETX_GPU_CODE float4x4 operator*(const float4x4& m1, const float4x4& m2) {
  float4 srca0 = m1.col[0];
  float4 srca1 = m1.col[1];
  float4 srca2 = m1.col[2];
  float4 srca3 = m1.col[3];
  float4 srcb0 = m2.col[0];
  float4 srcb1 = m2.col[1];
  float4 srcb2 = m2.col[2];
  float4 srcb3 = m2.col[3];

  float4x4 result;
  result.col[0] = srca0 * srcb0.x + srca1 * srcb0.y + srca2 * srcb0.z + srca3 * srcb0.w;
  result.col[1] = srca0 * srcb1.x + srca1 * srcb1.y + srca2 * srcb1.z + srca3 * srcb1.w;
  result.col[2] = srca0 * srcb2.x + srca1 * srcb2.y + srca2 * srcb2.z + srca3 * srcb2.w;
  result.col[3] = srca0 * srcb3.x + srca1 * srcb3.y + srca2 * srcb3.z + srca3 * srcb3.w;
  return result;
}

ETX_GPU_CODE float3 operator*(const float3x3& m, const float3& v) {
  return float3{
    m.col[0].x * v.x + m.col[1].x * v.y + m.col[2].x * v.z,
    m.col[0].y * v.x + m.col[1].y * v.y + m.col[2].y * v.z,
    m.col[0].z * v.x + m.col[1].z * v.y + m.col[2].z * v.z,
  };
}

ETX_GPU_CODE float4 operator*(const float4x4& m, const float4& v) {
  return float4{
    m.col[0].x * v.x + m.col[1].x * v.y + m.col[2].x * v.z + m.col[3].x * v.w,
    m.col[0].y * v.x + m.col[1].y * v.y + m.col[2].y * v.z + m.col[3].y * v.w,
    m.col[0].z * v.x + m.col[1].z * v.y + m.col[2].z * v.z + m.col[3].z * v.w,
    m.col[0].w * v.x + m.col[1].w * v.y + m.col[2].w * v.z + m.col[3].w * v.w,
  };
}

ETX_GPU_CODE float4x4 inverse(const float4x4& m) {
  float coef00 = m.col[2].z * m.col[3].w - m.col[3].z * m.col[2].w;
  float coef02 = m.col[1].z * m.col[3].w - m.col[3].z * m.col[1].w;
  float coef03 = m.col[1].z * m.col[2].w - m.col[2].z * m.col[1].w;
  float coef04 = m.col[2].y * m.col[3].w - m.col[3].y * m.col[2].w;
  float coef06 = m.col[1].y * m.col[3].w - m.col[3].y * m.col[1].w;
  float coef07 = m.col[1].y * m.col[2].w - m.col[2].y * m.col[1].w;
  float coef08 = m.col[2].y * m.col[3].z - m.col[3].y * m.col[2].z;
  float coef10 = m.col[1].y * m.col[3].z - m.col[3].y * m.col[1].z;
  float coef11 = m.col[1].y * m.col[2].z - m.col[2].y * m.col[1].z;
  float coef12 = m.col[2].x * m.col[3].w - m.col[3].x * m.col[2].w;
  float coef14 = m.col[1].x * m.col[3].w - m.col[3].x * m.col[1].w;
  float coef15 = m.col[1].x * m.col[2].w - m.col[2].x * m.col[1].w;
  float coef16 = m.col[2].x * m.col[3].z - m.col[3].x * m.col[2].z;
  float coef18 = m.col[1].x * m.col[3].z - m.col[3].x * m.col[1].z;
  float coef19 = m.col[1].x * m.col[2].z - m.col[2].x * m.col[1].z;
  float coef20 = m.col[2].x * m.col[3].y - m.col[3].x * m.col[2].y;
  float coef22 = m.col[1].x * m.col[3].y - m.col[3].x * m.col[1].y;
  float coef23 = m.col[1].x * m.col[2].y - m.col[2].x * m.col[1].y;

  float4 fac0 = {coef00, coef00, coef02, coef03};
  float4 fac1 = {coef04, coef04, coef06, coef07};
  float4 fac2 = {coef08, coef08, coef10, coef11};
  float4 fac3 = {coef12, coef12, coef14, coef15};
  float4 fac4 = {coef16, coef16, coef18, coef19};
  float4 fac5 = {coef20, coef20, coef22, coef23};

  float4 vec0 = {m.col[1].x, m.col[0].x, m.col[0].x, m.col[0].x};
  float4 vec1 = {m.col[1].y, m.col[0].y, m.col[0].y, m.col[0].y};
  float4 vec2 = {m.col[1].z, m.col[0].z, m.col[0].z, m.col[0].z};
  float4 vec3 = {m.col[1].w, m.col[0].w, m.col[0].w, m.col[0].w};

  float4 inv0 = {vec1 * fac0 - vec2 * fac1 + vec3 * fac2};
  float4 inv1 = {vec0 * fac0 - vec2 * fac3 + vec3 * fac4};
  float4 inv2 = {vec0 * fac1 - vec1 * fac3 + vec3 * fac5};
  float4 inv3 = {vec0 * fac2 - vec1 * fac4 + vec2 * fac5};

  float4 signa = {+1.0f, -1.0f, +1.0f, -1.0f};
  float4 signb = {-1.0f, +1.0f, -1.0f, +1.0f};

  float4x4 inverse = {inv0 * signa, inv1 * signb, inv2 * signa, inv3 * signb};
  float4 row0 = {inverse.col[0].x, inverse.col[1].x, inverse.col[2].x, inverse.col[3].x};
  float4 dot0 = {m.col[0] * row0};
  float dot1 = (dot0.x + dot0.y) + (dot0.z + dot0.w);
  inverse.col[0] /= dot1;
  inverse.col[1] /= dot1;
  inverse.col[2] /= dot1;
  inverse.col[3] /= dot1;
  return inverse;
}

ETX_GPU_CODE float4x4 look_at(const float3& origin, const float3& target, const float3& up) {
  auto f = normalize(target - origin);
  auto s = normalize(cross(f, up));
  auto u = cross(s, f);

  float4x4 result = {};
  result.col[0].x = s.x;
  result.col[1].x = s.y;
  result.col[2].x = s.z;
  result.col[0].y = u.x;
  result.col[1].y = u.y;
  result.col[2].y = u.z;
  result.col[0].z = -f.x;
  result.col[1].z = -f.y;
  result.col[2].z = -f.z;
  result.col[3].x = -dot(s, origin);
  result.col[3].y = -dot(u, origin);
  result.col[3].z = dot(f, origin);
  result.col[3].w = 1.0f;
  return result;
}

ETX_GPU_CODE float4x4 perspective(float fov, uint32_t width, uint32_t height, float z_near, float z_far) {
  ETX_ASSERT(width > 0);
  ETX_ASSERT(height > 0);
  ETX_ASSERT(fov > 0);

  float h = cosf(0.5f * fov) / sinf(0.5f * fov);
  float aspect = float(width) / float(height);

  float4x4 result = {};
  result.col[0].x = h / aspect;
  result.col[1].y = h;
  result.col[2].z = z_far / (z_near - z_far);
  result.col[2].w = -1.0f;
  result.col[3].z = -(z_far * z_near) / (z_far - z_near);
  return result;
}
