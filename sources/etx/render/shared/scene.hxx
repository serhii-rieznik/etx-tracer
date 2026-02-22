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
    uint32_t max_path_length = 65535u;
    uint32_t samples = 256u;
    uint32_t random_path_termination = 6u;
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
  ArrayView<EmitterProfile> emitter_profiles ETX_EMPTY_INIT;
  ArrayView<Emitter> emitter_instances ETX_EMPTY_INIT;
  ArrayView<Image> images ETX_EMPTY_INIT;
  ArrayView<Medium> mediums ETX_EMPTY_INIT;
  ArrayView<SpectralDistribution> spectrums ETX_EMPTY_INIT;

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
};

struct ImageSceneAccessCPUDesc {
  float2 fsize = {};
  uint2 size = {};
  uint32_t options = 0u;
  uint32_t pixel_data_offset = kInvalidIndex;
  uint32_t pixel_data_stride = 0u;
  uint32_t pixel_data_chunk_index = kInvalidIndex;
  uint32_t x_distribution_entries_offset = kInvalidIndex;
  uint32_t y_distribution_entries_offset = kInvalidIndex;
  uint32_t x_entries_stride = 0u;
  uint32_t x_distribution_count = 0u;
  uint32_t y_entries_count = 0u;
  uint32_t x_distribution_chunk_index = kInvalidIndex;
  uint32_t y_distribution_chunk_index = kInvalidIndex;
  uint32_t image_index = kInvalidIndex;
};

ETX_SHARED_INLINE bool try_load_scene_image_access(const Scene& scene, uint32_t image_index, ImageSceneAccessCPUDesc& image_access);

ETX_SHARED_INLINE float collimation_to_exponent(float normalized) {
  return scene_math_shared_collimation_to_exponent(normalized);
}

ETX_SHARED_INLINE float3 lerp_pos(const Scene& scene, const Triangle& t, const float3& bc) {
  return surface_point_shared_lerp_float3(scene.vertices.pos[t.i[0]], scene.vertices.pos[t.i[1]], scene.vertices.pos[t.i[2]], bc);
}

ETX_SHARED_INLINE float3 lerp_normal(const Scene& scene, const Triangle& t, const float3& bc) {
  return normalize(surface_point_shared_lerp_float3(scene.vertices.nrm[t.i[0]], scene.vertices.nrm[t.i[1]], scene.vertices.nrm[t.i[2]], bc));
}

ETX_SHARED_INLINE float3 lerp_tangent(const Scene& scene, const Triangle& t, const float3& bc) {
  return normalize(surface_point_shared_lerp_float3(scene.vertices.tan[t.i[0]], scene.vertices.tan[t.i[1]], scene.vertices.tan[t.i[2]], bc));
}

ETX_SHARED_INLINE float3 lerp_bitangent(const Scene& scene, const Triangle& t, const float3& bc) {
  return normalize(surface_point_shared_lerp_float3(scene.vertices.btn[t.i[0]], scene.vertices.btn[t.i[1]], scene.vertices.btn[t.i[2]], bc));
}

ETX_SHARED_INLINE float2 lerp_uv(const Scene& scene, const Triangle& t, const float3& b) {
  return surface_point_shared_lerp_float2(scene.vertices.tex[t.i[0]], scene.vertices.tex[t.i[1]], scene.vertices.tex[t.i[2]], b);
}

ETX_SHARED_INLINE void lerp_vertex(const Scene& scene, const Triangle& t, const float3& bc, Vertex& vertex) {
  const uint32_t i0 = t.i[0];
  const uint32_t i1 = t.i[1];
  const uint32_t i2 = t.i[2];

  surface_point_shared_interpolate_vertex(scene.vertices.pos[i0], scene.vertices.pos[i1], scene.vertices.pos[i2], scene.vertices.nrm[i0], scene.vertices.nrm[i1],
    scene.vertices.nrm[i2], scene.vertices.tan[i0], scene.vertices.tan[i1], scene.vertices.tan[i2], scene.vertices.btn[i0], scene.vertices.btn[i1],
    scene.vertices.btn[i2], scene.vertices.tex[i0], scene.vertices.tex[i1], scene.vertices.tex[i2], bc, true, true, vertex.pos, vertex.nrm, vertex.tan, vertex.btn, vertex.tex);
}

ETX_SHARED_INLINE Vertex lerp_vertex(const Scene& scene, const Triangle& t, const float3& bc) {
  Vertex vertex = {};
  lerp_vertex(scene, t, bc, vertex);
  return vertex;
}

ETX_SHARED_INLINE void orthogonalize(Vertex& v) {
  v.nrm = normalize(v.nrm);
  surface_point_shared_orthogonalize_frame(v.nrm, v.tan, v.btn, v.tan, v.btn);
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
  return scene_math_shared_shading_pos_project(position, origin, normal);
}

ETX_SHARED_INLINE float3 shading_pos(const Scene& scene, const Triangle& t, const float3& bc, const float3& w_o) {
  const float3& g0 = scene.vertices.pos[t.i[0]];
  const float3& g1 = scene.vertices.pos[t.i[1]];
  const float3& g2 = scene.vertices.pos[t.i[2]];
  const float3& n0 = scene.vertices.nrm[t.i[0]];
  const float3& n1 = scene.vertices.nrm[t.i[1]];
  const float3& n2 = scene.vertices.nrm[t.i[2]];
  return scene_math_shared_shading_pos(g0, g1, g2, n0, n1, n2, t.geo_n, bc, w_o);
}

ETX_SHARED_INLINE float3 orient_normals_to_hemisphere(float3 n_s, const float3& n_g, const float3& v) {
  const float3 result = scene_math_shared_orient_normals_to_hemisphere(n_s, n_g, v);
  ETX_ASSERT(is_valid_vector(result));
  return result;
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
  result_intersection.emitter_index = tri.emitter_index;

  const auto& mat = scene.materials[result_intersection.material_index];
  if ((mat.normal_image_index != kInvalidIndex) && (mat.normal_scale > kEpsilon)) {
    ImageSceneAccessCPUDesc image_access = {};
    if (try_load_scene_image_access(scene, mat.normal_image_index, image_access)) {
      auto sampled_normal = scene.images[image_access.image_index].evaluate_normal(result_intersection.tex, mat.normal_scale);
      result_intersection.nrm = normalize(result_intersection.tan * sampled_normal.x + result_intersection.btn * sampled_normal.y + result_intersection.nrm * sampled_normal.z);
      result_intersection.nrm = orient_normals_to_hemisphere(result_intersection.nrm, tri.geo_n, w_i);
      ETX_ASSERT(is_valid_vector(result_intersection.nrm));
      result_intersection.tan = orthonormalize(result_intersection.nrm, result_intersection.tan);
      ETX_ASSERT(is_valid_vector(result_intersection.tan));
      result_intersection.btn = normalize(cross(result_intersection.nrm, result_intersection.tan));
      ETX_ASSERT(is_valid_vector(result_intersection.btn));
    }
  }

  return result_intersection;
}

ETX_SHARED_INLINE bool random_continue(uint32_t path_length, uint32_t start_path_length, float eta_scale, Sampler& smp, SpectralResponse& throughput) {
  float max_t = throughput.maximum();
  if (max_t == 0.0f)
    return false;

  if (path_length < start_path_length)
    return true;

  max_t *= sqr(eta_scale);
  if (valid_value(max_t) == false) {
    return false;
  }

  float p = clamp(max_t, 0.01f, 0.95f);
  if (smp.next() > p)
    return false;

  throughput *= 1.0f / p;
  return true;
}

ETX_SHARED_INLINE SpectralResponse apply_rgb(const SpectralQuery spect, SpectralResponse response, const float4& value, const Scene& scene) {
  const ::SpectralResponse shared_response =
    ::material_scattering_shared_apply_spectral(static_cast<const ::SpectralQuery&>(spect), static_cast<const ::SpectralResponse&>(response), float3{value.x, value.y, value.z}, true);
  SpectralQuery response_query{shared_response.wavelength, shared_response.flags};
  response =
    ::spectral_response_is_spectral(shared_response) ? SpectralResponse{response_query, shared_response.value} : SpectralResponse{response_query, shared_response.integrated};
  ETX_VALIDATE(response);
  return response;
}

struct ImageSceneAccessCPUContext {
  const Scene& scene;
};

ETX_SHARED_INLINE bool image_scene_access_shared_cpu_has_images(ETX_IN(ImageSceneAccessCPUContext, context)) {
  return context.scene.images.count > 0u;
}

ETX_SHARED_INLINE bool image_scene_access_shared_cpu_load_desc(
  ETX_IN(ImageSceneAccessCPUContext, context), uint32_t image_index, ETX_OUT(ImageSceneAccessCPUDesc, image_access)) {
  if (image_index >= context.scene.images.count) {
    return false;
  }

  const auto& image = context.scene.images[image_index];
  image_access.fsize = image.fsize;
  image_access.size = image.isize;
  image_access.options = image.options;
  image_access.pixel_data_offset = image.pixel_data_offset;
  image_access.pixel_data_stride = image.pixel_data_stride;
  image_access.pixel_data_chunk_index = image.pixel_data_chunk_index;
  image_access.x_distribution_entries_offset = image.x_distribution_entries_offset;
  image_access.y_distribution_entries_offset = image.y_distribution_entries_offset;
  image_access.x_entries_stride = image.x_entries_stride;
  image_access.x_distribution_count = image.x_distribution_count;
  image_access.y_entries_count = image.y_entries_count;
  image_access.x_distribution_chunk_index = image.x_distribution_chunk_index;
  image_access.y_distribution_chunk_index = image.y_distribution_chunk_index;
  image_access.image_index = image_index;
  return true;
}

ETX_SHARED_INLINE uint32_t image_scene_access_shared_cpu_chunk_descriptor(ETX_IN(ImageSceneAccessCPUContext, context), uint32_t chunk_index) {
  (void)context;
  return chunk_index;
}

#define ETX_IMAGE_SCENE_ACCESS_SHARED_CONTEXT_TYPE ImageSceneAccessCPUContext
#define ETX_IMAGE_SCENE_ACCESS_SHARED_DESC_TYPE ImageSceneAccessCPUDesc
#define ETX_IMAGE_SCENE_ACCESS_SHARED_HAS_IMAGES(context) image_scene_access_shared_cpu_has_images(context)
#define ETX_IMAGE_SCENE_ACCESS_SHARED_LOAD_DESC(context, image_index, image_access) image_scene_access_shared_cpu_load_desc(context, image_index, image_access)
#define ETX_IMAGE_SCENE_ACCESS_SHARED_CHUNK_DESCRIPTOR(context, chunk_index) image_scene_access_shared_cpu_chunk_descriptor(context, chunk_index)
#include <etx/render/interop/image_scene_access_shared.hxx>
#undef ETX_IMAGE_SCENE_ACCESS_SHARED_CHUNK_DESCRIPTOR
#undef ETX_IMAGE_SCENE_ACCESS_SHARED_LOAD_DESC
#undef ETX_IMAGE_SCENE_ACCESS_SHARED_HAS_IMAGES
#undef ETX_IMAGE_SCENE_ACCESS_SHARED_DESC_TYPE
#undef ETX_IMAGE_SCENE_ACCESS_SHARED_CONTEXT_TYPE

ETX_SHARED_INLINE bool try_load_scene_image_access(const Scene& scene, uint32_t image_index, ImageSceneAccessCPUDesc& image_access) {
  ImageSceneAccessCPUContext access_context = {scene};
  return image_scene_access_shared_try_load_desc(access_context, image_index, image_access);
}

struct ImageEvaluateCPUSharedContext {
  const Scene& scene;
};

ETX_SHARED_INLINE bool image_evaluate_shared_cpu_try_evaluate_image_rgba(
  ETX_IN(ImageEvaluateCPUSharedContext, context), uint32_t image_index, ETX_IN(float2, uv), ETX_OUT(float, image_pdf), ETX_OUT(float4, image_value)) {
  image_pdf = 0.0f;
  image_value = float4(1.0f, 1.0f, 1.0f, 1.0f);

  ImageSceneAccessCPUDesc image_access = {};
  if (try_load_scene_image_access(context.scene, image_index, image_access) == false) {
    return false;
  }

  image_value = context.scene.images[image_access.image_index].evaluate(uv, &image_pdf);
  return true;
}

#define ETX_IMAGE_EVALUATE_SHARED_CONTEXT_TYPE ImageEvaluateCPUSharedContext
#define ETX_IMAGE_EVALUATE_SHARED_TRY_EVALUATE_IMAGE_RGBA(context, image_index, uv, image_pdf, image_value) \
  image_evaluate_shared_cpu_try_evaluate_image_rgba(context, image_index, uv, image_pdf, image_value)
#include <etx/render/interop/image_evaluate_shared.hxx>
#undef ETX_IMAGE_EVALUATE_SHARED_TRY_EVALUATE_IMAGE_RGBA
#undef ETX_IMAGE_EVALUATE_SHARED_CONTEXT_TYPE

ETX_SHARED_INLINE float4 sample_whole_image(const SampledImage& img, const float2& uv, const Scene& scene) {
  ImageEvaluateCPUSharedContext context = {scene};
  return image_evaluate_shared_sample_whole_or_default(context, img.image_index, uv, img.value);
}

ETX_SHARED_INLINE float evaluate_image(const SampledImage& img, const float2& uv, const Scene& scene, const float default_value) {
  ImageEvaluateCPUSharedContext context = {scene};
  return image_evaluate_shared_sample_channel_or_default(context, img.image_index, img.channel, uv, default_value);
}

ETX_SHARED_INLINE float evaluate_metalness(const Material& material, const float2& uv, const Scene& scene) {
  return material.metalness.value.x * evaluate_image(material.metalness, uv, scene, 1.0f);
}

ETX_SHARED_INLINE float2 evaluate_roughness(const Material& material, const float2& uv, const Scene& scene) {
  return float2{material.roughness.value.x, material.roughness.value.y} * evaluate_image(material.roughness, uv, scene, 1.0f);
}

ETX_SHARED_INLINE float evaluate_transmission(const Material& material, const float2& uv, const Scene& scene) {
  return material.transmission.value.x * evaluate_image(material.transmission, uv, scene, 1.0f);
}

ETX_SHARED_INLINE SpectralResponse apply_image(SpectralQuery spect, const SpectralImage& img, const float2& uv, const Scene& scene, float* image_pdf) {
  if (image_pdf != nullptr) {
    *image_pdf = 0.0f;
  }

  auto result = scene.spectrums[img.spectrum_index](spect);
  ETX_VALIDATE(result);
  if (img.image_index == kInvalidIndex) {
    return result;
  }

  ImageEvaluateCPUSharedContext context = {scene};
  float local_image_pdf = 0.0f;
  float4 eval = float4(1.0f, 1.0f, 1.0f, 1.0f);
  if (image_evaluate_shared_try_evaluate_rgba(context, img.image_index, uv, local_image_pdf, eval) == false) {
    return result;
  }

  if (image_pdf != nullptr) {
    *image_pdf = local_image_pdf;
  }

  ETX_VALIDATE(eval);
  const ::SpectralResponse shared_response =
    ::material_scattering_shared_apply_spectral(static_cast<const ::SpectralQuery&>(spect), static_cast<const ::SpectralResponse&>(result), float3{eval.x, eval.y, eval.z}, true);
  SpectralQuery response_query{shared_response.wavelength, shared_response.flags};
  return ::spectral_response_is_spectral(shared_response) ? SpectralResponse{response_query, shared_response.value} : SpectralResponse{response_query, shared_response.integrated};
}

ETX_SHARED_INLINE RefractiveIndexSample evaluate_refractive_index(const Scene& scene, const RefractiveIndex& ri, const SpectralQuery q) {
  RefractiveIndexSample result = {};
  result.cls = ri.cls;
  result.eta = (ri.eta_index == kInvalidIndex) ? SpectralResponse(q, 1.0f) : scene.spectrums[ri.eta_index](q);
  result.k = (ri.k_index == kInvalidIndex) ? SpectralResponse(q, 0.0f) : scene.spectrums[ri.k_index](q);
  return result;
}

}  // namespace etx

#include <etx/render/shared/scene_bsdf.hxx>
#include <etx/render/shared/scene_bssrdf_subsurface.hxx>
#include <etx/render/shared/scene_camera.hxx>
#include <etx/render/shared/scene_emitters.hxx>
#include <etx/render/shared/scene_medium.hxx>
