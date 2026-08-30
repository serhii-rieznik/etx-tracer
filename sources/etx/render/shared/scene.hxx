#pragma once

#include <etx/core/profiler.hxx>

#include <etx/render/interop/gpu_abi_constants.hxx>
#include <etx/render/interop/material_scattering_shared.hxx>
#include <etx/render/interop/surface_point_shared.hxx>
#include <etx/render/interop/scene_math_shared.hxx>
#include <etx/render/shared/spectrum.hxx>
#include <etx/render/shared/camera.hxx>
#include <etx/render/shared/image.hxx>
#include <etx/render/shared/medium.hxx>
#include <etx/render/shared/material.hxx>
#include <etx/render/shared/emitter.hxx>
#include <etx/render/shared/sampler.hxx>
#include <etx/render/shared/bsdf.hxx>

namespace etx {

constexpr uint32_t kMaximumPathLength = 1024u;

struct ETX_ALIGNED Scene {
  enum class LightSampling : uint32_t {
    Uniform,
    FromDistribution,
    RIS_Uniform,
    RIS_FromDistribution,

    Count,
  };

  struct Properties {
    enum : uint32_t {
      Committed = SceneProperty::Committed,
      Spectral = SceneProperty::Spectral,
      MultipleImportanceSampling = SceneProperty::MultipleImportanceSampling,
      BlueNoise = SceneProperty::BlueNoise,

      Count = SceneProperty::Count,
    };
  };

  struct Strategy {
    constexpr static uint32_t DirectHit = 1u << 0u;
    constexpr static uint32_t ConnectToLight = 1u << 1u;
    constexpr static uint32_t ConnectToCamera = 1u << 2u;
    constexpr static uint32_t ConnectVertices = 1u << 3u;
    constexpr static uint32_t MergeVertices = 1u << 4u;
    constexpr static uint32_t Default = DirectHit | ConnectToLight | ConnectToCamera | ConnectVertices | MergeVertices;
  };

  struct Options {
    uint32_t min_path_length = 0u;
    uint32_t max_path_length = kMaximumPathLength;
    uint32_t samples = 256u;
    uint32_t random_path_termination = 6u;
    uint32_t random_seed = 0u;
    float noise_threshold = 0.0f;
    float radiance_clamp = 0.0f;
    uint32_t strategy_flags = 1u << 0u | 1u << 1u | 1u << 2u | 1u << 3u | 1u << 4u;  // DirectHit | ConnectToLight | ConnectToCamera | ConnectVertices | MergeVertices
    bool properties[Properties::Count] = {};
    LightSampling light_sampling = LightSampling::RIS_FromDistribution;
  } options = {};

  struct {
    ArrayView<float3> pos;
    ArrayView<float3> nrm;
    ArrayView<float3> tan;
    ArrayView<float3> btn;
    ArrayView<float2> tex;
  } vertices ETX_EMPTY_INIT;

  ArrayView<Triangle> triangles ETX_EMPTY_INIT;
  ArrayView<Material> materials ETX_EMPTY_INIT;
  ArrayView<Mesh> meshes ETX_EMPTY_INIT;
  ArrayView<SceneInstance> instances ETX_EMPTY_INIT;
  ArrayView<EmitterProfile> emitter_profiles ETX_EMPTY_INIT;
  ArrayView<Emitter> emitter_instances ETX_EMPTY_INIT;
  ArrayView<Image> images ETX_EMPTY_INIT;
  ArrayView<Medium> mediums ETX_EMPTY_INIT;
  ArrayView<SpectralDistribution> spectrums ETX_EMPTY_INIT;

  struct EnergyCompensationInterface {
    uint32_t cls = MaterialClass::Undefined;
    uint32_t cache_mode = 0u;
    uint32_t spectral_wavelength_count = 0u;
    uint32_t thinfilm_slice_count = 1u;
    uint32_t directional_lut = kInvalidIndex;
    uint32_t average_lut = kInvalidIndex;
    uint32_t geometric_lut = kInvalidIndex;
    uint32_t geometric_average_lut = kInvalidIndex;
    uint32_t conductor_fms_lut = kInvalidIndex;
    uint32_t probability_lut = kInvalidIndex;
    float spectral_shortest_wavelength = 0.0f;
    float spectral_longest_wavelength = 0.0f;
  };

  ArrayView<EnergyCompensationInterface> energy_compensation_interfaces ETX_EMPTY_INIT;

  struct EnvironmentEmitters {
    uint32_t emitters[SceneLimits::MaxEnvironmentEmitters] ETX_EMPTY_INIT;
    uint32_t count ETX_EMPTY_INIT;
  } environment_emitters ETX_EMPTY_INIT;

  Distribution emitters_distribution ETX_EMPTY_INIT;

  struct Defaults {
    uint32_t black_spectrum = kInvalidIndex;
    uint32_t white_spectrum = kInvalidIndex;
    uint32_t rayleigh_spectrum = kInvalidIndex;
    uint32_t mie_spectrum = kInvalidIndex;
    uint32_t ozone_spectrum = kInvalidIndex;
    uint32_t subsurface_scatter_material = kInvalidIndex;
    uint32_t subsurface_exit_material = kInvalidIndex;
    uint32_t missing_material = kInvalidIndex;
    uint32_t dielectric_eta = kInvalidIndex;
    uint32_t conductor_eta = kInvalidIndex;
    uint32_t conductor_k = kInvalidIndex;
  } defaults = {};

  float3 bounding_sphere_center ETX_EMPTY_INIT;
  float bounding_sphere_radius ETX_EMPTY_INIT;
  float3 bounding_box_min ETX_EMPTY_INIT;
  float3 bounding_box_max ETX_EMPTY_INIT;

  PixelFilter pixel_sampler ETX_EMPTY_INIT;
  uint32_t strategy_flags ETX_INIT_WITH(Strategy::Default);

  bool committed() const {
    return options.properties[Properties::Committed];
  }
  bool spectral() const {
    return options.properties[Properties::Spectral];
  }
  bool multiple_importance_sampling() const {
    return options.properties[Properties::MultipleImportanceSampling];
  }
  bool blue_noise() const {
    return options.properties[Properties::BlueNoise];
  }
  LightSampling light_sampling_method() const {
    return options.light_sampling;
  }

  bool reservoir_sampling() const {
    return (options.light_sampling == LightSampling::RIS_Uniform) || (options.light_sampling == LightSampling::RIS_FromDistribution);
  }

  bool sample_lights_from_distribution() const {
    return (options.light_sampling == LightSampling::FromDistribution) || (options.light_sampling == LightSampling::RIS_FromDistribution);
  }
  ETX_SHARED_INLINE bool strategy_enabled(uint32_t flag) const {
    return (options.strategy_flags & flag) != 0u;
  }

  ETX_SHARED_INLINE uint32_t sampler_seed(uint32_t value_0, uint32_t value_1) const {
    return sampler_scene_seed(value_0, value_1, options.random_seed);
  }

  ETX_SHARED_INLINE uint32_t sampler_seed(uint32_t value_0, uint32_t value_1, uint32_t domain) const {
    return sampler_scene_domain_seed(value_0, value_1, options.random_seed, domain);
  }
};

#include <etx/render/access/image_access_cpu.hxx>
#include <etx/render/access/image_evaluate_cpu.hxx>
#include <etx/render/access/material_access_cpu.hxx>
#include <etx/render/access/medium_access_cpu.hxx>

ETX_SHARED_INLINE float3 lerp_pos(const Scene& scene, const Triangle& t, const float3& bc) {
  return scene.vertices.pos[t.i[0]] * bc.x +  //
         scene.vertices.pos[t.i[1]] * bc.y +  //
         scene.vertices.pos[t.i[2]] * bc.z;   //
}

ETX_SHARED_INLINE float3 lerp_normal(const Scene& scene, const Triangle& t, const float3& bc) {
  return normalize(scene.vertices.nrm[t.i[0]] * bc.x +  //
                   scene.vertices.nrm[t.i[1]] * bc.y +  //
                   scene.vertices.nrm[t.i[2]] * bc.z);  //
}

ETX_SHARED_INLINE float2 lerp_uv(const Scene& scene, const Triangle& t, const float3& b) {
  return scene.vertices.tex[t.i[0]] * b.x +  //
         scene.vertices.tex[t.i[1]] * b.y +  //
         scene.vertices.tex[t.i[2]] * b.z;   //
}

ETX_SHARED_INLINE void lerp_vertex(const Scene& scene, const Triangle& t, const float3& bc, Vertex& vertex) {
  const uint32_t i0 = t.i[0];
  const uint32_t i1 = t.i[1];
  const uint32_t i2 = t.i[2];

  vertex.pos = scene.vertices.pos[i0] * bc.x + scene.vertices.pos[i1] * bc.y + scene.vertices.pos[i2] * bc.z;
  vertex.nrm = normalize(scene.vertices.nrm[i0] * bc.x + scene.vertices.nrm[i1] * bc.y + scene.vertices.nrm[i2] * bc.z);
  vertex.tex = scene.vertices.tex[i0] * bc.x + scene.vertices.tex[i1] * bc.y + scene.vertices.tex[i2] * bc.z;

  const auto t0 = scene.vertices.tan[i0] * bc.x + scene.vertices.tan[i1] * bc.y + scene.vertices.tan[i2] * bc.z;
  const auto b0 = scene.vertices.btn[i0] * bc.x + scene.vertices.btn[i1] * bc.y + scene.vertices.btn[i2] * bc.z;
  scene_math_shared_build_sampling_frame(vertex.nrm, t0, b0, vertex.nrm, vertex.tan, vertex.btn);
}

ETX_SHARED_INLINE void lerp_vertex(const Scene& scene, const Triangle& t, const float3& bc, Intersection& vertex) {
  Vertex result = {};
  lerp_vertex(scene, t, bc, result);
  vertex.pos = result.pos;
  vertex.nrm = result.nrm;
  vertex.tan = result.tan;
  vertex.btn = result.btn;
  vertex.tex = result.tex;
}

ETX_SHARED_INLINE Vertex lerp_vertex(const Scene& scene, const Triangle& t, const float3& bc) {
  Vertex vertex = {};
  lerp_vertex(scene, t, bc, vertex);
  return vertex;
}

ETX_SHARED_INLINE float3 barycentrics(const Scene& scene, const Triangle& t, const float3& p) {
  const float3& a = scene.vertices.pos[t.i[0]];
  const float3& b = scene.vertices.pos[t.i[1]];
  const float3& c = scene.vertices.pos[t.i[2]];

  const float3 v0 = b - a;
  const float3 v1 = c - a;
  const float3 v2 = p - a;

  float d00 = dot(v0, v0);
  float d01 = dot(v0, v1);
  float d11 = dot(v1, v1);
  float d20 = dot(v2, v0);
  float d21 = dot(v2, v1);

  float denom = d00 * d11 - d01 * d01;
  float u = (d11 * d20 - d01 * d21) / denom;
  float v = (d00 * d21 - d01 * d20) / denom;
  return {1.0f - u - v, u, v};
}

ETX_SHARED_INLINE bool valid_barycentrics(const float3& p) {
  return (p.x >= 0.0f) && (p.x <= 1.0f) &&  //
         (p.y >= 0.0f) && (p.y <= 1.0f) &&  //
         (p.z >= 0.0f) && (p.z <= 1.0f);
}

ETX_SHARED_INLINE float3 shading_pos_project(const float3& position, const float3& origin, const float3& normal) {
  return position - dot(position - origin, normal) * normal;
}

ETX_SHARED_INLINE float3 shading_pos(const Scene& scene, const Triangle& t, const float3& bc, const float3& w_o) {
  const float3& g0 = scene.vertices.pos[t.i[0]];
  const float3& g1 = scene.vertices.pos[t.i[1]];
  const float3& g2 = scene.vertices.pos[t.i[2]];
  const float3& n0 = scene.vertices.nrm[t.i[0]];
  const float3& n1 = scene.vertices.nrm[t.i[1]];
  const float3& n2 = scene.vertices.nrm[t.i[2]];
  const float3 geo_pos = g0 * bc.x + g1 * bc.y + g2 * bc.z;
  const float3 sh_normal = normalize(n0 * bc.x + n1 * bc.y + n2 * bc.z);
  const float direction = (dot(sh_normal, w_o) >= 0.0f) ? +1.0f : -1.0f;
  const float3 p0 = shading_pos_project(geo_pos, g0, direction * n0);
  const float3 p1 = shading_pos_project(geo_pos, g1, direction * n1);
  const float3 p2 = shading_pos_project(geo_pos, g2, direction * n2);
  const float3 sh_pos = p0 * bc.x + p1 * bc.y + p2 * bc.z;
  bool convex = dot(sh_pos - geo_pos, sh_normal) * direction > 0.0f;
  return offset_ray(convex ? sh_pos : geo_pos, t.geo_n * direction);
}

ETX_SHARED_INLINE float3 scene_instance_transform_point(ETX_IN(AffineTransform, transform), float3 point) {
  return float3(dot(float3(transform.rows[0].x, transform.rows[0].y, transform.rows[0].z), point) + transform.rows[0].w,
    dot(float3(transform.rows[1].x, transform.rows[1].y, transform.rows[1].z), point) + transform.rows[1].w,
    dot(float3(transform.rows[2].x, transform.rows[2].y, transform.rows[2].z), point) + transform.rows[2].w);
}

ETX_SHARED_INLINE float3 scene_instance_transform_vector(ETX_IN(AffineTransform, transform), float3 vector) {
  return float3(dot(float3(transform.rows[0].x, transform.rows[0].y, transform.rows[0].z), vector),
    dot(float3(transform.rows[1].x, transform.rows[1].y, transform.rows[1].z), vector), dot(float3(transform.rows[2].x, transform.rows[2].y, transform.rows[2].z), vector));
}

ETX_SHARED_INLINE float3 scene_instance_transform_normal(ETX_IN(AffineTransform, world_to_object), float3 normal) {
  return normalize(float3(world_to_object.rows[0].x * normal.x + world_to_object.rows[1].x * normal.y + world_to_object.rows[2].x * normal.z,
    world_to_object.rows[0].y * normal.x + world_to_object.rows[1].y * normal.y + world_to_object.rows[2].y * normal.z,
    world_to_object.rows[0].z * normal.x + world_to_object.rows[1].z * normal.y + world_to_object.rows[2].z * normal.z));
}

ETX_SHARED_INLINE Vertex scene_instance_transform_vertex(const SceneInstance& instance, Vertex vertex) {
  const float orientation = ((instance.flags & SceneInstance::Mirrored) != 0u) ? -1.0f : 1.0f;
  vertex.pos = scene_instance_transform_point(instance.object_to_world, vertex.pos);
  vertex.nrm = scene_instance_transform_normal(instance.world_to_object, vertex.nrm) * orientation;
  const float3 tangent_hint = scene_instance_transform_vector(instance.object_to_world, vertex.tan);
  const float3 bitangent_hint = scene_instance_transform_vector(instance.object_to_world, vertex.btn);
  scene_math_shared_build_sampling_frame(vertex.nrm, tangent_hint, bitangent_hint, vertex.nrm, vertex.tan, vertex.btn);
  return vertex;
}

ETX_SHARED_INLINE uint32_t scene_instance_emitter_index(const Scene& scene, uint32_t instance_index, uint32_t triangle_index) {
  if ((instance_index == kInvalidIndex) || (instance_index >= scene.instances.count)) {
    return kInvalidIndex;
  }

  const SceneInstance& instance = scene.instances[instance_index];
  uint32_t begin = instance.emitter_offset;
  uint32_t end = begin + instance.emitter_count;
  if (end > scene.emitter_instances.count) {
    return kInvalidIndex;
  }

  while (begin < end) {
    const uint32_t middle = begin + (end - begin) / 2u;
    const uint32_t candidate_triangle = scene.emitter_instances[middle].triangle_index;
    if (candidate_triangle < triangle_index) {
      begin = middle + 1u;
    } else {
      end = middle;
    }
  }
  if ((begin < scene.emitter_instances.count) && (scene.emitter_instances[begin].triangle_index == triangle_index)) {
    return begin;
  }
  return kInvalidIndex;
}

ETX_SHARED_INLINE float3 scene_triangle_world_position(const Scene& scene, const Triangle& triangle, uint32_t vertex_index, uint32_t instance_index) {
  const float3 position = scene.vertices.pos[triangle.i[vertex_index]];
  if ((instance_index == kInvalidIndex) || (instance_index >= scene.instances.count)) {
    return position;
  }
  return scene_instance_transform_point(scene.instances[instance_index].object_to_world, position);
}

ETX_SHARED_INLINE float3 scene_triangle_world_normal(const Scene& scene, const Triangle& triangle, uint32_t vertex_index, uint32_t instance_index) {
  const float3 normal = scene.vertices.nrm[triangle.i[vertex_index]];
  if ((instance_index == kInvalidIndex) || (instance_index >= scene.instances.count)) {
    return normal;
  }
  const SceneInstance& instance = scene.instances[instance_index];
  const float orientation = ((instance.flags & SceneInstance::Mirrored) != 0u) ? -1.0f : 1.0f;
  return scene_instance_transform_normal(instance.world_to_object, normal) * orientation;
}

ETX_SHARED_INLINE float3 scene_triangle_world_geometric_normal(const Scene& scene, const Triangle& triangle, uint32_t instance_index) {
  if ((instance_index == kInvalidIndex) || (instance_index >= scene.instances.count)) {
    return triangle.geo_n;
  }
  const SceneInstance& instance = scene.instances[instance_index];
  const float orientation = ((instance.flags & SceneInstance::Mirrored) != 0u) ? -1.0f : 1.0f;
  return scene_instance_transform_normal(instance.world_to_object, triangle.geo_n) * orientation;
}

ETX_SHARED_INLINE float3 shading_pos(const Scene& scene, const Triangle& triangle, const float3& bc, const float3& w_o, uint32_t instance_index) {
  const float3 g0 = scene_triangle_world_position(scene, triangle, 0u, instance_index);
  const float3 g1 = scene_triangle_world_position(scene, triangle, 1u, instance_index);
  const float3 g2 = scene_triangle_world_position(scene, triangle, 2u, instance_index);
  const float3 n0 = scene_triangle_world_normal(scene, triangle, 0u, instance_index);
  const float3 n1 = scene_triangle_world_normal(scene, triangle, 1u, instance_index);
  const float3 n2 = scene_triangle_world_normal(scene, triangle, 2u, instance_index);
  const float3 geo_pos = g0 * bc.x + g1 * bc.y + g2 * bc.z;
  const float3 sh_normal = normalize(n0 * bc.x + n1 * bc.y + n2 * bc.z);
  const float direction = (dot(sh_normal, w_o) >= 0.0f) ? +1.0f : -1.0f;
  const float3 p0 = shading_pos_project(geo_pos, g0, direction * n0);
  const float3 p1 = shading_pos_project(geo_pos, g1, direction * n1);
  const float3 p2 = shading_pos_project(geo_pos, g2, direction * n2);
  const float3 sh_pos = p0 * bc.x + p1 * bc.y + p2 * bc.z;
  const bool convex = dot(sh_pos - geo_pos, sh_normal) * direction > 0.0f;
  return offset_ray(convex ? sh_pos : geo_pos, scene_triangle_world_geometric_normal(scene, triangle, instance_index) * direction);
}

ETX_SHARED_INLINE Intersection make_intersection(const Scene& scene, const float3& w_i, const IntersectionBase& base) {
  float3 bc = barycentrics(base.barycentric);
  const auto& tri = scene.triangles[base.triangle_index];
  Intersection result_intersection = {};
  lerp_vertex(scene, tri, bc, result_intersection);
  result_intersection.barycentric = bc;
  result_intersection.triangle_index = base.triangle_index;
  result_intersection.w_i = w_i;
  result_intersection.t = base.t;
  result_intersection.material_index = tri.material_index;
  result_intersection.emitter_index = scene_instance_emitter_index(scene, base.instance_index, base.triangle_index);
  result_intersection.instance_index = base.instance_index;

  float3 world_geo_n = tri.geo_n;
  if ((base.instance_index != kInvalidIndex) && (base.instance_index < scene.instances.count)) {
    const SceneInstance& instance = scene.instances[base.instance_index];
    result_intersection.pos = scene_instance_transform_point(instance.object_to_world, result_intersection.pos);
    const float orientation = (instance.flags & SceneInstance::Mirrored) != 0u ? -1.0f : 1.0f;
    result_intersection.nrm = scene_instance_transform_normal(instance.world_to_object, result_intersection.nrm) * orientation;
    world_geo_n = scene_triangle_world_geometric_normal(scene, tri, base.instance_index);
    const float3 tangent_hint = scene_instance_transform_vector(instance.object_to_world, result_intersection.tan);
    const float3 bitangent_hint = scene_instance_transform_vector(instance.object_to_world, result_intersection.btn);
    scene_math_shared_build_sampling_frame(result_intersection.nrm, tangent_hint, bitangent_hint, result_intersection.nrm, result_intersection.tan, result_intersection.btn);
  }

  scene_math_shared_finalize_shading_frame(result_intersection.nrm, result_intersection.nrm, result_intersection.tan, result_intersection.btn, world_geo_n, w_i,
    result_intersection.nrm, result_intersection.tan, result_intersection.btn);
  ETX_ASSERT(is_valid_vector(result_intersection.nrm));
  ETX_ASSERT(is_valid_vector(result_intersection.tan));
  ETX_ASSERT(is_valid_vector(result_intersection.btn));

  const auto& mat = scene.materials[result_intersection.material_index];
  if ((mat.normal_image_index != kInvalidIndex) && (mat.normal_image_index < scene.images.count) && (mat.normal_scale > kEpsilon)) {
    auto sampled_normal = scene.images[mat.normal_image_index].evaluate_normal(result_intersection.tex, mat.normal_scale);
    const float3 mapped_normal_value = result_intersection.tan * sampled_normal.x + result_intersection.btn * sampled_normal.y + result_intersection.nrm * sampled_normal.z;
    const float mapped_normal_length_sq = dot(mapped_normal_value, mapped_normal_value);
    if (mapped_normal_length_sq > kEpsilon) {
      const float3 mapped_normal = mapped_normal_value / sqrt(mapped_normal_length_sq);
      scene_math_shared_finalize_shading_frame(mapped_normal, result_intersection.nrm, result_intersection.tan, result_intersection.btn, world_geo_n, w_i,
        result_intersection.nrm, result_intersection.tan, result_intersection.btn);
    }
    ETX_ASSERT(is_valid_vector(result_intersection.nrm));
    ETX_ASSERT(is_valid_vector(result_intersection.tan));
    ETX_ASSERT(is_valid_vector(result_intersection.btn));
  }

  return result_intersection;
}

ETX_SHARED_INLINE bool random_continue(uint32_t path_length, uint32_t start_path_length, float eta_scale, Sampler& smp, SpectralResponse& throughput) {
  const float max_t = throughput.maximum();
  if (max_t == 0.0f) {
    return false;
  }

  if (path_length < start_path_length) {
    return true;
  }

  const float eta_scaled_max_t = max_t * sqr(eta_scale);
  if (valid_value(eta_scaled_max_t) == false) {
    return false;
  }

  const float p = clamp(eta_scaled_max_t, 0.01f, kSamplerMaximumContinuationProbability);
  if (smp.next() > p) {
    return false;
  }

  throughput *= 1.0f / p;
  return true;
}

ETX_SHARED_INLINE SpectralResponse apply_rgb(const SpectralQuery spect, SpectralResponse response, const float4& value) {
  if (spect.spectral()) {
    SpectralResponse scale = rgb_response(spect, {value.x, value.y, value.z});
    ETX_VALIDATE(scale);
    response *= scale;
    ETX_VALIDATE(response);
  } else {
    response.integrated *= float3{value.x, value.y, value.z};
  }

  return response;
}

float4 sample_whole_image(const SampledImage& img, const float2& uv);
float evaluate_image_channel(uint32_t image_index, uint32_t channel, const float2& uv, float default_value);
bool image_has_alpha_channel(uint32_t image_index);
float2 sample_image_uv(uint32_t image_index, const float2& rnd);
float2 sample_image_uv(uint32_t image_index, const float2& rnd, float& pdf, uint2& location, float4& value);
float evaluate_image(const SampledImage& img, const float2& uv, float default_value);

ETX_SHARED_INLINE float evaluate_metalness(const Material& material, const float2& uv) {
  return material.metalness.value.x * evaluate_image(material.metalness, uv, 1.0f);
}

ETX_SHARED_INLINE float2 evaluate_roughness(const Material& material, const float2& uv) {
  return float2{material.roughness.value.x, material.roughness.value.y} * evaluate_image(material.roughness, uv, 1.0f);
}

ETX_SHARED_INLINE float evaluate_transmission(const Material& material, const float2& uv) {
  return material.transmission.value.x * evaluate_image(material.transmission, uv, 1.0f);
}

RefractiveIndexSample evaluate_refractive_index(const RefractiveIndex& ri, SpectralQuery q);
uint32_t default_dielectric_eta_index();
uint32_t default_conductor_eta_index();
uint32_t default_conductor_k_index();
SpectralResponse medium_load_spectrum_or_zero(uint32_t spectrum_index, SpectralQuery spect);
SpectralResponse apply_image(SpectralQuery spect, const SpectralImage& img, const float2& uv, float& image_pdf);
SpectralResponse apply_image(SpectralQuery spect, const SpectralImage& img, const float2& uv);

}  // namespace etx

#include <etx/render/shared/scene_bsdf.hxx>
#include <etx/render/shared/scene_bssrdf_subsurface.hxx>
#include <etx/render/shared/scene_camera.hxx>
#include <etx/render/shared/scene_emitters.hxx>
#include <etx/render/shared/scene_medium.hxx>
