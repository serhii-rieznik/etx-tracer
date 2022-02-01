#pragma once

#if (ETX_RENDER_BASE_INCLUDED)

#if (ETX_NVCC_COMPILER)

#else

#define GLM_FORCE_XYZW_ONLY 1
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

constexpr uint32_t kInvalidIndex = static_cast<uint32_t>(-1);

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

}  // namespace etx
