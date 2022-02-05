#pragma once

#if (ETX_RENDER_BASE_INCLUDED)

#if (ETX_NVCC_COMPILER)

#else

#define GLM_FORCE_XYZW_ONLY 1
#define GLM_FORCE_DEPTH_ZERO_TO_ONE 1
#include <glm/glm.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtc/quaternion.hpp>

using float2 = glm::vec2;
using float3 = glm::vec3;
using float4 = glm::vec4;
using int2 = glm::ivec2;
using int3 = glm::ivec3;
using uint2 = glm::uvec2;
using uint3 = glm::uvec3;
using uint4 = glm::uvec4;
using float3x3 = glm::mat3x3;
using float4x4 = glm::mat4x4;
using ubyte2 = glm::u8vec2;
using ubyte3 = glm::u8vec3;
using ubyte4 = glm::u8vec4;

template <typename T>
inline constexpr T clamp(T val, T min_val, T max_val) {
  return glm::clamp(val, min_val, max_val);
}

template <typename T>
inline constexpr T saturate(T val) {
  return glm::clamp(val, 0.0f, 1.0f);
}

template <typename T>
inline constexpr T min(T a, T b) {
  return glm::min(a, b);
}

template <typename T>
inline constexpr T max(T a, T b) {
  return glm::max(a, b);
}

template <typename T>
inline constexpr T lerp(T a, T b, float t) {
  return glm::mix(a, b, t);
}

template <typename T>
inline constexpr T sqr(T t) {
  return t * t;
}

#endif

#else

#error This file should not be included separately. Use etx/render/shared/base.hxx instead

#endif

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

constexpr uint32_t kInvalidIndex = ~0u;

struct alignas(16) BoundingBox {
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

struct alignas(16) Vertex {
  float3 pos = {};
  float3 nrm = {};
  float3 tan = {};
  float3 btn = {};
  float2 tex = {};
};

struct alignas(16) Triangle {
  uint32_t i[3] = {kInvalidIndex, kInvalidIndex, kInvalidIndex};
  uint32_t material_index = kInvalidIndex;
  float3 geo_n = {};
  float area = {};
  uint32_t emitter_index = kInvalidIndex;
  uint32_t pad[3] = {};
};

struct alignas(16) Frame {
  float3 tan = {};
  float3 btn = {};
  float3 nrm = {};
};

struct Ray {
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

struct alignas(16) Intersection : public Vertex {
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

  ETX_GPU_CODE operator bool() const {
    return t >= 0.0f;
  }
};

ETX_GPU_CODE float3 to_float3(const float4& v) {
  return {v.x, v.y, v.z};
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

  return {r * std::cos(theta), r * std::sin(theta)};
}

ETX_GPU_CODE float3 orthogonalize(const float3& t, const float3& n) {
  return normalize(t - n * dot(n, t));
};

ETX_GPU_CODE float3 orthogonalize(const float3& t, const float3& b, const float3& n) {
  return normalize(t - n * dot(n, t)) * (dot(cross(n, t), b) < 0.0f ? -1.0f : 1.0f);
}

ETX_GPU_CODE bool valid_value(float t) {
  return (isnan(t) == false) && (isinf(t) == false) && (t >= 0.0f);
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

}  // namespace etx
