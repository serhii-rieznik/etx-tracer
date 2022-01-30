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

ETX_GPU_CODE float3 to_float3(const float4& v) {
  return {v.x, v.y, v.z};
}

ETX_GPU_CODE float luminance(const float3& value) {
  return value.x * 0.212671f + value.y * 0.715160f + value.z * 0.072169f;
}

}  // namespace etx
