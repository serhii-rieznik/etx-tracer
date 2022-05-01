#pragma once

#if (ETX_RENDER_BASE_INCLUDED)
#else
#error This file should not be included separately. Use etx/render/shared/base.hxx instead
#endif

#include <math.h>
#include <string.h>
#include <complex>

#if (ETX_NVCC_COMPILER)

#include <thrust/complex.h>
using complex = thrust::complex<float>;

ETX_GPU_CODE complex complex_sqrt(complex c) {
  return thrust::sqrt(c);
}
ETX_GPU_CODE complex complex_cos(complex c) {
  return thrust::cos(c);
}
ETX_GPU_CODE float complex_abs(complex c) {
  return thrust::abs(c);
}
ETX_GPU_CODE float complex_norm(complex c) {
  return thrust::norm(c);
}

#else

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
using ubyte2 = vector2<uint8_t>;
using ubyte3 = vector3<uint8_t>;
using ubyte4 = vector4<uint8_t>;
using complex = std::complex<float>;

ETX_GPU_CODE complex complex_sqrt(complex c) {
  return std::sqrt(c);
}
ETX_GPU_CODE complex complex_cos(complex c) {
  return std::cos(c);
}
ETX_GPU_CODE float complex_abs(complex c) {
  return std::abs(c);
}
ETX_GPU_CODE float complex_norm(complex c) {
  return std::norm(c);
}

#endif

struct float3x3 {
  float3 col[3] ETX_EMPTY_INIT;
};

struct float4x4 {
  float4 col[4] ETX_EMPTY_INIT;
};

#include <etx/render/shared/vector_math.hxx>

namespace etx {

constexpr float kQuarterPi = 0.78539816339744830961566084581988f;
constexpr float kHalfPi = 1.5707963267948966192313216916398f;
constexpr float kPi = 3.1415926535897932384626433832795f;
constexpr float kDoublePi = 6.283185307179586476925286766559f;
constexpr float kSqrt2 = 1.4142135623730950488016887242097f;
constexpr float kInvPi = 0.31830988618379067153776752674503f;
constexpr float kEpsilon = 1.192092896e-07f;
constexpr float kMaxFloat = 3.402823466e+38f;
constexpr float kRayEpsilon = 1.0e-4f;
constexpr float kDeltaAlphaTreshold = 1.0e-4f;

constexpr float kPlanetRadius = 6371e+3f;
constexpr float kAtmosphereRadius = 120e+3f;

constexpr uint32_t kInvalidIndex = ~0u;

struct ETX_ALIGNED BoundingBox {
  float3 p_min ETX_EMPTY_INIT;
  float3 p_max ETX_EMPTY_INIT;

  ETX_GPU_CODE float3 to_bounding_box(const float3& p) const {
    return (p - p_min) / (p_max - p_min);
  }

  ETX_GPU_CODE float3 from_bounding_box(const float3& p) const {
    return p * (p_max - p_min) + p_min;
  }

  ETX_GPU_CODE bool contains(const float3& p) const {
    return (p.x >= p_min.x) && (p.y >= p_min.y) && (p.z >= p_min.z) &&  //
           (p.x <= p_max.x) && (p.y <= p_max.y) && (p.z <= p_max.z);
  }
};

struct ETX_ALIGNED Vertex {
  float3 pos = {};
  float3 nrm = {};
  float3 tan = {};
  float3 btn = {};
  float2 tex = {};
};

struct ETX_ALIGNED Triangle {
  uint32_t i[3] = {kInvalidIndex, kInvalidIndex, kInvalidIndex};
  uint32_t material_index = kInvalidIndex;
  float3 geo_n = {};
  float area = {};
  uint32_t emitter_index = kInvalidIndex;
  uint32_t pad[3] = {};
};

struct ETX_ALIGNED LocalFrame {
  float3 tan = {};
  float3 btn = {};
  float3 nrm = {};

  ETX_GPU_CODE float3 to_local(const float3& v) const {
    return float3x3{float3{tan.x, btn.x, nrm.x}, float3{tan.y, btn.y, nrm.y}, float3{tan.z, btn.z, nrm.z}} * v;
  }
  ETX_GPU_CODE float3 from_local(const float3& v) const {
    return float3x3{float3{tan.x, tan.y, tan.z}, float3{btn.x, btn.y, btn.z}, float3{nrm.x, nrm.y, nrm.z}} * v;
  }
  ETX_GPU_CODE static float cos_theta(const float3& v) {
    return v.z;
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

struct ETX_ALIGNED Intersection : public Vertex {
  float3 barycentric = {};
  uint32_t triangle_index = kInvalidIndex;
  float3 w_i = {};
  float t = -kMaxFloat;

  Intersection() = default;

  ETX_GPU_CODE Intersection(const Vertex& v)
    : Vertex(v) {
  }

  ETX_GPU_CODE float distance() const {
    return t;
  }
};

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

ETX_GPU_CODE float2 max(const float2& a, const float2& b) {
  return {
    max(a.x, b.x),
    max(a.y, b.y),
  };
}

ETX_GPU_CODE float3 max(const float3& a, const float3& b) {
  return {
    max(a.x, b.x),
    max(a.y, b.y),
    max(a.z, b.z),
  };
}

ETX_GPU_CODE float4 max(const float4& a, const float4& b) {
  return {
    max(a.x, b.x),
    max(a.y, b.y),
    max(a.z, b.z),
    max(a.w, b.w),
  };
}

ETX_GPU_CODE float2 min(const float2& a, const float2& b) {
  return {
    min(a.x, b.x),
    min(a.y, b.y),
  };
}

ETX_GPU_CODE float3 min(const float3& a, const float3& b) {
  return {
    min(a.x, b.x),
    min(a.y, b.y),
    min(a.z, b.z),
  };
}

ETX_GPU_CODE float4 min(const float4& a, const float4& b) {
  return {
    min(a.x, b.x),
    min(a.y, b.y),
    min(a.z, b.z),
    max(a.w, b.w),
  };
}

ETX_GPU_CODE constexpr float saturate(float val) {
  return clamp(val, 0.0f, 1.0f);
}

ETX_GPU_CODE constexpr float2 saturate(float2 val) {
  return {clamp(val.x, 0.0f, 1.0f), clamp(val.y, 0.0f, 1.0f)};
}

ETX_GPU_CODE constexpr float3 saturate(float3 val) {
  return {clamp(val.x, 0.0f, 1.0f), clamp(val.y, 0.0f, 1.0f), clamp(val.z, 0.0f, 1.0f)};
}

ETX_GPU_CODE constexpr float4 saturate(float4 val) {
  return {clamp(val.x, 0.0f, 1.0f), clamp(val.y, 0.0f, 1.0f), clamp(val.z, 0.0f, 1.0f), clamp(val.w, 0.0f, 1.0f)};
}

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

ETX_GPU_CODE float luminance(const float3& value) {
  return value.x * 0.212671f + value.y * 0.715160f + value.z * 0.072169f;
}

ETX_GPU_CODE auto orthonormal_basis(const float3& n) {
  struct basis {
    float3 u, v;
  };
  float s = (n.z < 0.0 ? -1.0f : 1.0f);
  float a = -1.0f / (s + n.z);
  float b = n.x * n.y * a;
  return basis{
    {1.0f + s * n.x * n.x * a, s * b, -s * n.x},
    {b, s + n.y * n.y * a, -n.y},
  };
}

ETX_GPU_CODE float3 sample_cosine_distribution(float xi0, float xi1, const float3& n, const float3& u, const float3& v, float exponent) {
  float cos_theta = powf(xi0, 1.0f / (exponent + 1.0f));
  float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
  return (u * cosf(xi1 * kDoublePi) + v * sinf(xi1 * kDoublePi)) * sin_theta + n * cos_theta;
}

ETX_GPU_CODE float3 sample_cosine_distribution(float xi0, float xi1, const float3& n, float exponent) {
  auto basis = orthonormal_basis(n);
  return sample_cosine_distribution(xi0, xi1, n, basis.u, basis.v, exponent);
}

ETX_GPU_CODE float3 barycentrics(float2 bc) {
  return {1.0f - bc.x - bc.y, bc.x, bc.y};
}

ETX_GPU_CODE float3 random_barycentric(float r1, float r2) {
  r1 = sqrtf(r1);
  return {1.0f - r1, r1 * (1.0f - r2), r1 * r2};
}

ETX_GPU_CODE float2 sample_disk(float xi0, float xi1) {
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

ETX_GPU_CODE float isfinite(float t) {
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

ETX_GPU_CODE float area_to_solid_angle_probability(float pdf_pos, const float3& source_point, const float3& target_normal, const float3& target_point, float p) {
  auto w_o = source_point - target_point;
  float a_distance_squared = dot(w_o, w_o);
  if (a_distance_squared == 0.0f) {
    return 0.0f;
  }

  w_o = w_o / sqrtf(a_distance_squared);
  float cos_t = fabsf(clamp(dot(w_o, target_normal), -1.0f, 1.0f));
  if (p != 1.0f) {
    cos_t = powf(cos_t, p);
  }
  return (cos_t > 1.0e-5f) ? (pdf_pos * a_distance_squared / cos_t) : 0.0f;
}

ETX_GPU_CODE float2 uv_to_phi_theta(float u, float v) {
  float phi = (u * 2.0f - 1.0f) * kPi;
  float theta = (0.5f - v) * kPi;
  return {phi, theta};
}

ETX_GPU_CODE float2 phi_theta_to_uv(float phi, float theta) {
  float u = (phi / kPi + 1.0f) / 2.0f;
  float v = 0.5f - theta / kPi;
  return {u, v};
}

ETX_GPU_CODE float2 direction_to_phi_theta(const float3& dir) {
  return {atan2f(dir.z, dir.x), asinf(dir.y)};
}

ETX_GPU_CODE float2 direction_to_uv(const float3& dir) {
  float2 p_t = direction_to_phi_theta(dir);
  return phi_theta_to_uv(p_t.x, p_t.y);
}

ETX_GPU_CODE float3 phi_theta_to_direction(float phi, float theta) {
  float cos_p = cosf(phi);
  float sin_p = sinf(phi);
  float cos_t = cosf(theta);
  float sin_t = sinf(theta);
  return {
    cos_p * cos_t,
    sin_t,
    sin_p * cos_t,
  };
}

ETX_GPU_CODE float3 uv_to_direction(const float2& uv) {
  float2 p_t = uv_to_phi_theta(uv.x, uv.y);
  return phi_theta_to_direction(p_t.x, p_t.y);
}

ETX_GPU_CODE uint64_t next_power_of_two(uint64_t v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

ETX_GPU_CODE float distance_to_sphere(const float3& r_origin, const float3& r_direction, float s) {
  float b = dot(r_direction, r_origin);
  float d = (b * b) - dot(r_origin, r_origin) + (s * s);
  if (d < 0.0f) {
    return 0.0f;
  }
  d = sqrtf(d);
  float a0 = -b - d;
  float a1 = -b + d;
  return (a0 < 0.0f) ? ((a1 < 0.0f) ? 0.0f : a1) : a0;
}

}  // namespace etx
