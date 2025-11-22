#pragma once

#include <unordered_map>
#include <functional>

namespace etx {

constexpr float kGeometryEpsilon = 1e-6f;

inline int64_t quantize_float(float val, float eps) {
  return static_cast<int64_t>(std::floor(val / eps));
}

template <typename Vec>
bool float_vec_equal_eps(const Vec& a, const Vec& b, float eps);

template <>
inline bool float_vec_equal_eps<float3>(const float3& a, const float3& b, float eps) {
  return quantize_float(a.x, eps) == quantize_float(b.x, eps) && quantize_float(a.y, eps) == quantize_float(b.y, eps) && quantize_float(a.z, eps) == quantize_float(b.z, eps);
}

template <>
inline bool float_vec_equal_eps<float2>(const float2& a, const float2& b, float eps) {
  return quantize_float(a.x, eps) == quantize_float(b.x, eps) && quantize_float(a.y, eps) == quantize_float(b.y, eps);
}

// Generic hash function for float vectors
template <typename Vec>
uint64_t float_vec_hash(const Vec& v, float eps);

template <>
inline uint64_t float_vec_hash<float3>(const float3& v, float eps) {
  uint64_t h1 = std::hash<int64_t>()(quantize_float(v.x, eps));
  uint64_t h2 = std::hash<int64_t>()(quantize_float(v.y, eps));
  uint64_t h3 = std::hash<int64_t>()(quantize_float(v.z, eps));
  return h1 ^ (h2 << 1) ^ (h3 << 2);
}

template <>
inline uint64_t float_vec_hash<float2>(const float2& v, float eps) {
  uint64_t h1 = std::hash<int64_t>()(quantize_float(v.x, eps));
  uint64_t h2 = std::hash<int64_t>()(quantize_float(v.y, eps));
  return h1 ^ (h2 << 1);
}

struct VertexKey {
  float3 position;
  float3 normal;
  float2 uv;
  bool has_normal;
  bool has_uv;

  bool operator==(const VertexKey& other) const {
    if (!float_vec_equal_eps(position, other.position, kGeometryEpsilon))
      return false;
    if (has_normal != other.has_normal)
      return false;
    if (has_normal && !float_vec_equal_eps(normal, other.normal, kGeometryEpsilon))
      return false;
    if (has_uv != other.has_uv)
      return false;
    if (has_uv && !float_vec_equal_eps(uv, other.uv, kGeometryEpsilon))
      return false;
    return true;
  }
};

struct VertexKeyHash {
  uint64_t operator()(const VertexKey& key) const {
    uint64_t h = 0;
    // Hash position
    h = hash_combine(h, std::hash<float>()(key.position.x));
    h = hash_combine(h, std::hash<float>()(key.position.y));
    h = hash_combine(h, std::hash<float>()(key.position.z));
    // Hash normal if present
    if (key.has_normal) {
      h = hash_combine(h, std::hash<float>()(key.normal.x));
      h = hash_combine(h, std::hash<float>()(key.normal.y));
      h = hash_combine(h, std::hash<float>()(key.normal.z));
    }
    // Hash UV if present
    if (key.has_uv) {
      h = hash_combine(h, std::hash<float>()(key.uv.x));
      h = hash_combine(h, std::hash<float>()(key.uv.y));
    }
    // Hash flags
    h = hash_combine(h, std::hash<bool>()(key.has_normal));
    h = hash_combine(h, std::hash<bool>()(key.has_uv));
    return h;
  }

 private:
  static uint64_t hash_combine(uint64_t seed, uint64_t value) {
    constexpr uint64_t magic = 0x9e3779b9;  // Golden ratio constant
    return seed ^ (value + magic + (seed << 6) + (seed >> 2));
  }
};

// Unordered map type with float keys using epsilon-based comparison
template <typename T>
using FloatUnorderedMap = std::unordered_map<T, uint32_t, std::function<uint64_t(const T&)>, std::function<bool(const T&, const T&)>>;

// Factory functions for creating maps with epsilon-based comparison
inline FloatUnorderedMap<float3> create_float3_map(float eps = kGeometryEpsilon) {
  return FloatUnorderedMap<float3>(
    0,
    [eps](const float3& v) {
      return float_vec_hash<float3>(v, eps);
    },
    [eps](const float3& a, const float3& b) {
      return float_vec_equal_eps<float3>(a, b, eps);
    });
}

inline FloatUnorderedMap<float2> create_float2_map(float eps = kGeometryEpsilon) {
  return FloatUnorderedMap<float2>(
    0,
    [eps](const float2& v) {
      return float_vec_hash<float2>(v, eps);
    },
    [eps](const float2& a, const float2& b) {
      return float_vec_equal_eps<float2>(a, b, eps);
    });
}

}  // namespace etx
