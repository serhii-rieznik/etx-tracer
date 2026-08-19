#pragma once

#include "interop.hxx"
#include "bounding_box.hxx"
#include "geometry.hxx"
#include "ray.hxx"

ETX_ENUM_U32(ProjectionType){
  Equirectangular = 0u,
  EqualArea = 1u,
};

ETX_ENUM_U32(InteractionType){
  Surface = 0u,
  Medium = 1u,

  Count,
};

ETX_STATIC_CONST uint32_t kProjectionEqualArea = 1u;

struct SphericalCoordinates {
  float phi ETX_INIT(0.0f);
  float theta ETX_INIT(0.0f);
  float r ETX_INIT(0.0f);
};

struct ETX_ALIGNED Mesh {
  float3 bbox_min ETX_INIT({});
  uint32_t triangle_offset ETX_INIT(0u);
  float3 bbox_max ETX_INIT({});
  uint32_t triangle_count ETX_INIT(0u);
};

struct ETX_ALIGNED AffineTransform {
  float4 rows[3] ETX_INIT({
    {1.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 1.0f, 0.0f},
  });
};

struct ETX_ALIGNED SceneInstance {
  enum : uint32_t {
    Mirrored = 1u << 0u,
    Enabled = 1u << 1u,
  };

  AffineTransform object_to_world ETX_INIT({});
  AffineTransform world_to_object ETX_INIT({});
  uint32_t mesh_index ETX_INIT(kInvalidIndex);
  uint32_t flags ETX_INIT(0u);
  uint32_t emitter_offset ETX_INIT(0u);
  uint32_t emitter_count ETX_INIT(0u);
};

struct ETX_ALIGNED LocalFrame {
  enum : uint32_t {
    EnteringMaterial = 1u << 0u,
  };

  float3 tan ETX_INIT({});
  float3 btn ETX_INIT({});
  float3 nrm ETX_INIT({});
  uint32_t flags ETX_INIT(0u);

  ETX_SHARED_INLINE static float cos_theta(float3 v) {
    return v.z;
  }

  ETX_SHARED_INLINE static float sin_theta(float3 v) {
    return sqrt(max(0.0f, 1.0f - cos_theta(v)));
  }
};

ETX_SHARED_INLINE float3 local_frame_to_local(ETX_IN(LocalFrame, local_frame), float3 v) {
  return float3(local_frame.tan.x * v.x + local_frame.tan.y * v.y + local_frame.tan.z * v.z, local_frame.btn.x * v.x + local_frame.btn.y * v.y + local_frame.btn.z * v.z,
    local_frame.nrm.x * v.x + local_frame.nrm.y * v.y + local_frame.nrm.z * v.z);
}

ETX_SHARED_INLINE float3 local_frame_from_local(ETX_IN(LocalFrame, local_frame), float3 v) {
  return float3(local_frame.tan.x * v.x + local_frame.btn.x * v.y + local_frame.nrm.x * v.z, local_frame.tan.y * v.x + local_frame.btn.y * v.y + local_frame.nrm.y * v.z,
    local_frame.tan.z * v.x + local_frame.btn.z * v.y + local_frame.nrm.z * v.z);
}

ETX_SHARED_INLINE bool local_frame_entering_material(ETX_IN(LocalFrame, local_frame)) {
  return (local_frame.flags & LocalFrame::EnteringMaterial) == LocalFrame::EnteringMaterial;
}

struct ETX_ALIGNED IntersectionBase {
  float2 barycentric ETX_INIT({});
  uint32_t triangle_index ETX_INIT(kInvalidIndex);
  float t ETX_INIT(kMaxFloat);
  uint32_t instance_index ETX_INIT(kInvalidIndex);
};

struct ETX_ALIGNED Intersection {
  float3 pos ETX_INIT({});
  float3 nrm ETX_INIT({});
  float3 tan ETX_INIT({});
  float3 btn ETX_INIT({});
  float2 tex ETX_INIT({});
  float3 barycentric ETX_INIT({});
  uint32_t triangle_index ETX_INIT(kInvalidIndex);
  float3 w_i ETX_INIT({});
  float t ETX_INIT(0.0f);
  uint32_t material_index ETX_INIT(kInvalidIndex);
  uint32_t emitter_index ETX_INIT(kInvalidIndex);
  uint32_t instance_index ETX_INIT(kInvalidIndex);

#if (ETX_CPP)
  ETX_SHARED_INLINE operator Vertex() const {
    Vertex result = {};
    result.pos = pos;
    result.nrm = nrm;
    result.tan = tan;
    result.btn = btn;
    result.tex = tex;
    return result;
  }
#endif
};

struct OrthonormalBasis {
  float3 u ETX_INIT({});
  float3 v ETX_INIT({});
};

ETX_SHARED_INLINE float3 orthonormalize(ETX_IN(float3, nrm), ETX_IN(float3, tan)) {
  return normalize(tan - dot(tan, nrm) * nrm);
}

ETX_SHARED_INLINE OrthonormalBasis orthonormal_basis(ETX_IN(float3, n)) {
  float3 a = normalize(((n.x != n.y) || (n.x != n.z))                //
                         ? float3(n.z - n.y, n.x - n.z, +n.y - n.x)  //
                         : float3(n.z - n.y, n.x + n.z, -n.y - n.x));
  float3 b = normalize(cross(n, a));
  OrthonormalBasis result;
  result.u = a;
  result.v = b;
  return result;
}

ETX_SHARED_INLINE float3 sample_cosine_distribution(ETX_IN(float2, rnd), float exponent) {
  float cos_theta = pow(max(rnd.x, kEpsilon), 1.0f / (exponent + 1.0f));
  float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
  return float3(cos(rnd.y * kDoublePi) * sin_theta, sin(rnd.y * kDoublePi) * sin_theta, cos_theta);
}

ETX_SHARED_INLINE float3 sample_cosine_distribution(ETX_IN(float2, rnd), ETX_IN(float3, n), ETX_IN(float3, u), ETX_IN(float3, v), float exponent) {
  float3 local = sample_cosine_distribution(rnd, exponent);
  return u * local.x + v * local.y + n * local.z;
}

ETX_SHARED_INLINE float3 sample_cosine_distribution(ETX_IN(float2, rnd), ETX_IN(float3, n), float exponent) {
  OrthonormalBasis basis = orthonormal_basis(n);
  return sample_cosine_distribution(rnd, n, basis.u, basis.v, exponent);
}

ETX_SHARED_INLINE float3 barycentrics(ETX_IN(float2, bc)) {
  return float3(1.0f - bc.x - bc.y, bc.x, bc.y);
}

ETX_SHARED_INLINE float3 random_barycentric(ETX_IN(float2, rnd)) {
  float r1 = sqrt(rnd.x);
  return float3(1.0f - r1, r1 * (1.0f - rnd.y), r1 * rnd.y);
}

ETX_SHARED_INLINE float2 sample_disk(ETX_IN(float2, rnd)) {
  float2 offset = rnd * 2.0f - 1.0f;
  if ((offset.x == 0.0f) && (offset.y == 0.0f)) {
    return float2(0.0f, 0.0f);
  }

  float r = 0.0f;
  float theta = 0.0f;
  if (abs(offset.x) > abs(offset.y)) {
    r = offset.x;
    theta = kQuarterPi * (offset.y / offset.x);
  } else {
    r = offset.y;
    theta = kHalfPi - kQuarterPi * (offset.x / offset.y);
  }

  return float2(r * cos(theta), r * sin(theta));
}

ETX_SHARED_INLINE float2 sample_disk_uv(float xi0, float xi1) {
  float2 offset = float2(2.0f * xi0 - 1.0f, 2.0f * xi1 - 1.0f);
  if ((offset.x == 0.0f) && (offset.y == 0.0f)) {
    return float2(0.0f, 0.0f);
  }

  float r = 0.0f;
  float theta = 0.0f;
  if (abs(offset.x) > abs(offset.y)) {
    r = offset.x;
    theta = kQuarterPi * (offset.y / offset.x);
  } else {
    r = offset.y;
    theta = kHalfPi - kQuarterPi * (offset.x / offset.y);
  }

  return float2(r * cos(theta) * 0.5f + 0.5f, r * sin(theta) * 0.5f + 0.5f);
}

ETX_SHARED_INLINE float2 projecected_coords(ETX_IN(float3, normal), ETX_IN(float3, in_dir), float sz, float csz) {
  if (sz == 0.0f) {
    return float2(0.0f, 0.0f);
  }

  OrthonormalBasis basis = orthonormal_basis(normal);
  float result_u = dot(basis.u, in_dir) / (0.5f * sz * csz);
  float result_v = dot(basis.v, in_dir) / (0.5f * sz * csz);
  return float2(result_u, result_v);
}

ETX_SHARED_INLINE float2 disk_uv(ETX_IN(float3, normal), ETX_IN(float3, in_dir), float sz, float csz) {
  float2 projected = projecected_coords(normal, in_dir, sz, csz);
  return saturate(projected * 0.5f + 0.5f);
}

ETX_SHARED_INLINE float3 orthogonalize(ETX_IN(float3, t), ETX_IN(float3, n)) {
  return normalize(t - n * dot(n, t));
}

ETX_SHARED_INLINE float3 orthogonalize(ETX_IN(float3, t), ETX_IN(float3, b), ETX_IN(float3, n)) {
  return normalize(t - n * dot(n, t)) * (dot(cross(n, t), b) < 0.0f ? -1.0f : 1.0f);
}

ETX_SHARED_INLINE float to_float(uint32_t value) {
#if (ETX_CPP)
  float result = 0.0f;
  ETX_STD memcpy(&result, &value, sizeof(float));
  return result;
#else
  return asfloat(value);
#endif
}

ETX_SHARED_INLINE float to_float(int32_t value) {
#if (ETX_CPP)
  float result = 0.0f;
  ETX_STD memcpy(&result, &value, sizeof(float));
  return result;
#else
  return asfloat(value);
#endif
}

ETX_SHARED_INLINE uint32_t to_uint(float value) {
#if (ETX_CPP)
  uint32_t result = 0u;
  ETX_STD memcpy(&result, &value, sizeof(float));
  return result;
#else
  return asuint(value);
#endif
}

ETX_SHARED_INLINE int32_t to_int(float value) {
#if (ETX_CPP)
  int32_t result = 0;
  ETX_STD memcpy(&result, &value, sizeof(float));
  return result;
#else
  return asint(value);
#endif
}

ETX_SHARED_INLINE float3 offset_ray(ETX_IN(float3, p), ETX_IN(float3, n)) {
  float int_scale = 256.0f;
  float float_scale = 1.0f / 65536.0f;
  float origin = 1.0f / 32.0f;

  int32_t of_i_x = int32_t(int_scale * n.x);
  int32_t of_i_y = int32_t(int_scale * n.y);
  int32_t of_i_z = int32_t(int_scale * n.z);

  float p_i_x = to_float(to_int(p.x) + ((p.x > 0.0f) ? of_i_x : -of_i_x));
  float p_i_y = to_float(to_int(p.y) + ((p.y > 0.0f) ? of_i_y : -of_i_y));
  float p_i_z = to_float(to_int(p.z) + ((p.z > 0.0f) ? of_i_z : -of_i_z));

  return float3(abs(p.x) < origin ? p.x + float_scale * n.x : p_i_x, abs(p.y) < origin ? p.y + float_scale * n.y : p_i_y, abs(p.z) < origin ? p.z + float_scale * n.z : p_i_z);
}

ETX_SHARED_INLINE float balance_heuristic(float value_0, float value_1) {
  float denom = value_0 + value_1;
  return (denom == 0.0f) ? 0.0f : (value_0 / denom);
}

ETX_SHARED_INLINE float balance_heuristic(float value_0, float value_1, float value_2) {
  float denom = value_0 + value_1 + value_2;
  return (denom == 0.0f) ? 0.0f : (value_0 / denom);
}

ETX_SHARED_INLINE float power_heuristic(float value_0, float value_1) {
  float value_0_sq = value_0 * value_0;
  float value_1_sq = value_1 * value_1;
  float denom = value_0_sq + value_1_sq;
  return (denom == 0.0f) ? 0.0f : (value_0_sq / denom);
}

ETX_SHARED_INLINE float power_heuristic(float value_0, float value_1, float value_2) {
  float value_0_sq = value_0 * value_0;
  float value_1_sq = value_1 * value_1;
  float value_2_sq = value_2 * value_2;
  float denom = value_0_sq + value_1_sq + value_2_sq;
  return (denom == 0.0f) ? 0.0f : (value_0_sq / denom);
}

ETX_SHARED_INLINE SphericalCoordinates to_spherical(ETX_IN(float3, dir)) {
  float r = length(dir);
  SphericalCoordinates result;
  result.phi = atan2(dir.z, dir.x);
  result.theta = asin(dir.y / r);
  result.r = r;
  return result;
}

ETX_SHARED_INLINE float3 from_spherical(ETX_IN(SphericalCoordinates, spherical)) {
  float cos_phi = cos(spherical.phi);
  float sin_phi = sin(spherical.phi);
  float cos_theta = cos(spherical.theta);
  float sin_theta = sin(spherical.theta);
  return float3(spherical.r * cos_phi * cos_theta, spherical.r * sin_theta, spherical.r * sin_phi * cos_theta);
}

ETX_SHARED_INLINE float3 from_spherical(float phi, float theta) {
  SphericalCoordinates spherical;
  spherical.phi = phi;
  spherical.theta = theta;
  spherical.r = 1.0f;
  return from_spherical(spherical);
}

ETX_SHARED_INLINE bool projection_is_equal_area(uint32_t projection) {
  return projection == kProjectionEqualArea;
}

ETX_SHARED_INLINE float3 uv_to_direction(ETX_IN(float2, uv), ETX_IN(float2, offset), float u_scale, uint32_t projection) {
  float u = uv.x;
  if (u_scale < 0.0f) {
    u = 1.0f - u;
  }

  u = u - offset.x;
  u = u - floor(u);

  float phi = (u * 2.0f - 1.0f) * kPi;
  float theta = (0.5f - uv.y) * kPi;
  if (projection_is_equal_area(projection)) {
    float v_mapped = 1.0f - uv.y * 2.0f;
    theta = asin(max(-1.0f, min(1.0f, v_mapped)));
  }

  return from_spherical(phi, theta);
}

ETX_SHARED_INLINE float2 direction_to_uv(ETX_IN(float3, dir), ETX_IN(float2, offset), float u_scale, uint32_t projection) {
  SphericalCoordinates spherical = to_spherical(dir);

  float u = (spherical.phi / kPi + 1.0f) * 0.5f;
  if (u_scale < 0.0f) {
    u = 1.0f - u;
  }

  u = u + offset.x;
  u = u - floor(u);

  float v = 0.5f - spherical.theta / kPi;
  if (projection_is_equal_area(projection)) {
    float sin_theta = sin(spherical.theta);
    v = 0.5f - sin_theta * 0.5f;
  }

  return float2(u, v);
}

ETX_SHARED_INLINE float quaternion_to_yaw_rotation_offset(ETX_IN(float4, quat)) {
  if ((quat.x == 0.0f) && (quat.y == 0.0f) && (quat.z == 0.0f) && (quat.w == 1.0f)) {
    return 0.0f;
  }

  float yaw = atan2(2.0f * (quat.w * quat.y + quat.x * quat.z), 1.0f - 2.0f * (quat.y * quat.y + quat.z * quat.z));
  return -yaw / kDoublePi;
}

#if (ETX_CPP)
ETX_SHARED_INLINE uint64_t next_power_of_two(uint64_t value) {
  value--;
  value |= value >> 1llu;
  value |= value >> 2llu;
  value |= value >> 4llu;
  value |= value >> 8llu;
  value |= value >> 16llu;
  value |= value >> 32llu;
  value++;
  return value;
}
#endif

ETX_SHARED_INLINE uint32_t next_power_of_two(uint32_t value) {
  value--;
  value |= value >> 1u;
  value |= value >> 2u;
  value |= value >> 4u;
  value |= value >> 8u;
  value |= value >> 16u;
  value++;
  return value;
}

ETX_SHARED_INLINE float distance_to_sphere(ETX_IN(float3, ray_origin), ETX_IN(float3, ray_direction), ETX_IN(float3, center), float radius) {
  float3 e = ray_origin - center;
  float b = dot(ray_direction, e);
  float d = (b * b) - dot(e, e) + (radius * radius);
  if (d < 0.0f) {
    return 0.0f;
  }

  d = sqrt(d);
  float a0 = -b - d;
  float a1 = -b + d;
  return (a0 < 0.0f) ? ((a1 < 0.0f) ? 0.0f : a1) : a0;
}

ETX_SHARED_INLINE float gamma_to_linear(float value) {
  return value <= 0.04045f ? value / 12.92f : pow((value + 0.055f) / 1.055f, 2.4f);
}

ETX_SHARED_INLINE float linear_to_gamma(float value) {
  return value <= 0.0031308f ? 12.92f * value : 1.055f * pow(value, 1.0f / 2.4f) - 0.055f;
}

ETX_SHARED_INLINE float3 gamma_to_linear(ETX_IN(float3, value)) {
  return float3(gamma_to_linear(value.x), gamma_to_linear(value.y), gamma_to_linear(value.z));
}

ETX_SHARED_INLINE float3 linear_to_gamma(ETX_IN(float3, value)) {
  return float3(linear_to_gamma(value.x), linear_to_gamma(value.y), linear_to_gamma(value.z));
}

ETX_SHARED_INLINE float3 project_point_plane(ETX_IN(float3, v), ETX_IN(float3, plane_n), ETX_IN(float3, plane_o)) {
  float3 v_to_o = v - plane_o;
  float distance = dot(v_to_o, plane_n);
  return v - distance * plane_n;
}

ETX_SHARED_INLINE bool intersect_ray_plane(ETX_IN(Ray, ray), ETX_IN(float3, plane_n), ETX_IN(float3, plane_o), ETX_OUT(float, t)) {
  t = -kMaxFloat;

  float denom = dot(plane_n, ray.d);
  if (abs(denom) < kEpsilon) {
    return false;
  }

  float t0 = dot(plane_n, plane_o - ray.o) / denom;
  if ((t0 < ray.min_t) || (t0 > ray.max_t)) {
    return false;
  }

  t = t0;
  return true;
}

ETX_SHARED_INLINE bool direction_matches(ETX_IN(float3, ideal), ETX_IN(float3, actual), float cosine_threshold) {
  float kDefaultThreshold = 1.0f - kInvMaxHalf;
  float3 i = normalize(ideal);
  float3 a = normalize(actual);
  return dot(i, a) >= min(kDefaultThreshold, cosine_threshold);
}

#if (ETX_CPP)
namespace etx {
using ::balance_heuristic;
using ::barycentrics;
using ::direction_matches;
using ::direction_to_uv;
using ::disk_uv;
using ::distance_to_sphere;
using ::from_spherical;
using ::gamma_to_linear;
using ::InteractionType;
using ::intersect_ray_plane;
using ::Intersection;
using ::IntersectionBase;
using ::linear_to_gamma;
using ::LocalFrame;
using ::Mesh;
using ::next_power_of_two;
using ::offset_ray;
using ::orthogonalize;
using ::orthonormal_basis;
using ::OrthonormalBasis;
using ::orthonormalize;
using ::power_heuristic;
using ::projecected_coords;
using ::project_point_plane;
using ::projection_is_equal_area;
using ::ProjectionType;
using ::quaternion_to_yaw_rotation_offset;
using ::random_barycentric;
using ::sample_cosine_distribution;
using ::sample_disk;
using ::sample_disk_uv;
using ::SphericalCoordinates;
using ::to_float;
using ::to_int;
using ::to_spherical;
using ::to_uint;
using ::uv_to_direction;
}  // namespace etx
#endif
