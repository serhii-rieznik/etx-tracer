#pragma once

#include <etx/core/profiler.hxx>

#include <etx/render/shared/spectrum.hxx>
#include <etx/render/shared/camera.hxx>
#include <etx/render/shared/image.hxx>
#include <etx/render/shared/medium.hxx>
#include <etx/render/shared/material.hxx>
#include <etx/render/shared/emitter.hxx>
#include <etx/render/shared/sampler.hxx>
#include <etx/render/shared/bsdf.hxx>

namespace etx {

struct ETX_ALIGNED EnvironmentEmitters {
  constexpr static const uint32_t kMaxCount = 63;
  uint32_t emitters[kMaxCount] ETX_EMPTY_INIT;
  uint32_t count ETX_EMPTY_INIT;
};

struct ETX_ALIGNED Scene {
  ArrayView<Vertex> vertices ETX_EMPTY_INIT;
  ArrayView<Triangle> triangles ETX_EMPTY_INIT;
  ArrayView<uint32_t> triangle_to_material ETX_EMPTY_INIT;
  ArrayView<uint32_t> triangle_to_emitter ETX_EMPTY_INIT;
  ArrayView<Material> materials ETX_EMPTY_INIT;
  ArrayView<Emitter> emitters ETX_EMPTY_INIT;
  ArrayView<Image> images ETX_EMPTY_INIT;
  ArrayView<Medium> mediums ETX_EMPTY_INIT;
  ArrayView<SpectralDistribution> spectrums ETX_EMPTY_INIT;
  Distribution emitters_distribution ETX_EMPTY_INIT;
  EnvironmentEmitters environment_emitters ETX_EMPTY_INIT;
  float3 bounding_sphere_center ETX_EMPTY_INIT;
  float bounding_sphere_radius ETX_EMPTY_INIT;
  PixelFilter pixel_sampler ETX_EMPTY_INIT;
  uint32_t max_path_length ETX_INIT_WITH(65535u);
  uint32_t samples ETX_INIT_WITH(256u);
  uint32_t random_path_termination ETX_INIT_WITH(6u);
  float noise_threshold ETX_INIT_WITH(0.1f);
  float radiance_clamp ETX_INIT_WITH(0.0f);
  uint8_t spectral ETX_INIT_WITH(0);
  uint8_t pad[3] ETX_INIT_WITH({});
};

struct ContinousTraceOptions {
  IntersectionBase* intersection_buffer = nullptr;
  uint32_t max_intersections = 0;
  uint32_t material_id = kInvalidIndex;
};

ETX_GPU_CODE float3 lerp_pos(const ArrayView<Vertex>& vertices, const Triangle& t, const float3& bc) {
  return vertices[t.i[0]].pos * bc.x +  //
         vertices[t.i[1]].pos * bc.y +  //
         vertices[t.i[2]].pos * bc.z;   //
}

ETX_GPU_CODE float3 lerp_normal(const ArrayView<Vertex>& vertices, const Triangle& t, const float3& bc) {
  return normalize(vertices[t.i[0]].nrm * bc.x +  //
                   vertices[t.i[1]].nrm * bc.y +  //
                   vertices[t.i[2]].nrm * bc.z);  //
}

ETX_GPU_CODE float2 lerp_uv(const ArrayView<Vertex>& vertices, const Triangle& t, const float3& b) {
  return vertices[t.i[0]].tex * b.x +  //
         vertices[t.i[1]].tex * b.y +  //
         vertices[t.i[2]].tex * b.z;   //
}

ETX_GPU_CODE Vertex lerp_vertex(const ArrayView<Vertex>& vertices, const Triangle& t, const float3& bc) {
  const auto& v0 = vertices[t.i[0]];
  const auto& v1 = vertices[t.i[1]];
  const auto& v2 = vertices[t.i[2]];
  return {
    v0.pos * bc.x + v1.pos * bc.y + v2.pos * bc.z,
    v0.nrm * bc.x + v1.nrm * bc.y + v2.nrm * bc.z,
    v0.tan * bc.x + v1.tan * bc.y + v2.tan * bc.z,
    v0.btn * bc.x + v1.btn * bc.y + v2.btn * bc.z,
    v0.tex * bc.x + v1.tex * bc.y + v2.tex * bc.z,
  };
}

ETX_GPU_CODE float3 barycentrics(const ArrayView<Vertex>& vertices, const Triangle& t, const float3& p) {
  const float3& a = vertices[t.i[0]].pos;
  const float3& b = vertices[t.i[1]].pos;
  const float3& c = vertices[t.i[2]].pos;

  const float3 v0 = b - a;
  const float3 v1 = c - a;
  const float3 v2 = p - a;

  // Compute dot products
  float d00 = dot(v0, v0);
  float d01 = dot(v0, v1);
  float d11 = dot(v1, v1);
  float d20 = dot(v2, v0);
  float d21 = dot(v2, v1);

  // Compute denominator
  float denom = d00 * d11 - d01 * d01;

  // Compute barycentric coordinates
  float u = (d11 * d20 - d01 * d21) / denom;
  float v = (d00 * d21 - d01 * d20) / denom;
  return {1.0f - u - v, u, v};
}

ETX_GPU_CODE bool valid_barycentrics(const float3& p) {
  return (p.x >= 0.0f) && (p.x <= 1.0f) &&  //
         (p.y >= 0.0f) && (p.y <= 1.0f) &&  //
         (p.z >= 0.0f) && (p.z <= 1.0f);
}

ETX_GPU_CODE float3 shading_pos_project(const float3& position, const float3& origin, const float3& normal) {
  return position - dot(position - origin, normal) * normal;
}

ETX_GPU_CODE float3 shading_pos(const ArrayView<Vertex>& vertices, const Triangle& t, const float3& bc, const float3& w_o) {
  float3 geo_pos = lerp_pos(vertices, t, bc);
  float3 sh_normal = lerp_normal(vertices, t, bc);
  float direction = (dot(sh_normal, w_o) >= 0.0f) ? +1.0f : -1.0f;
  float3 p0 = shading_pos_project(geo_pos, vertices[t.i[0]].pos, direction * vertices[t.i[0]].nrm);
  float3 p1 = shading_pos_project(geo_pos, vertices[t.i[1]].pos, direction * vertices[t.i[1]].nrm);
  float3 p2 = shading_pos_project(geo_pos, vertices[t.i[2]].pos, direction * vertices[t.i[2]].nrm);
  float3 sh_pos = p0 * bc.x + p1 * bc.y + p2 * bc.z;
  bool convex = dot(sh_pos - geo_pos, sh_normal) * direction > 0.0f;
  return offset_ray(convex ? sh_pos : geo_pos, t.geo_n * direction);
}

ETX_GPU_CODE float3 orient_normals_to_hemisphere(float3 n_s, const float3& n_g, const float3& v) {
  constexpr uint32_t kMaxAttempts = 16u;
  const float i_dot_g = dot(v, n_g);

  float i_dot_s = dot(v, n_s);
  for (uint32_t i = 0; ((i_dot_s * i_dot_g) <= kEpsilon) && (i < kMaxAttempts); ++i) {
    n_s = normalize(8.0 * n_s + n_g);
    ETX_ASSERT(is_valid_vector(n_s));
    i_dot_s = dot(v, n_s);
  }

  return n_s;
}

ETX_GPU_CODE Intersection make_intersection(const Scene& scene, const float3& w_i, const IntersectionBase& base) {
  float3 bc = barycentrics(base.barycentric);
  const auto& tri = scene.triangles[base.triangle_index];
  Intersection result_intersection = lerp_vertex(scene.vertices, tri, bc);
  result_intersection.barycentric = bc;
  result_intersection.triangle_index = static_cast<uint32_t>(base.triangle_index);
  result_intersection.w_i = w_i;
  result_intersection.t = base.t;
  result_intersection.material_index = scene.triangle_to_material[result_intersection.triangle_index];
  result_intersection.emitter_index = scene.triangle_to_emitter[result_intersection.triangle_index];

  const auto& mat = scene.materials[result_intersection.material_index];
  if ((mat.normal_image_index != kInvalidIndex) && (mat.normal_scale > kEpsilon)) {
    auto sampled_normal = scene.images[mat.normal_image_index].evaluate_normal(result_intersection.tex, mat.normal_scale);
    result_intersection.nrm = normalize(result_intersection.tan * sampled_normal.x + result_intersection.btn * sampled_normal.y + result_intersection.nrm * sampled_normal.z);
    result_intersection.nrm = orient_normals_to_hemisphere(result_intersection.nrm, tri.geo_n, w_i);
    ETX_ASSERT(is_valid_vector(result_intersection.nrm));
    result_intersection.tan = orthonormalize(result_intersection.nrm, result_intersection.tan);
    ETX_ASSERT(is_valid_vector(result_intersection.tan));
    result_intersection.btn = normalize(cross(result_intersection.nrm, result_intersection.tan));
    ETX_ASSERT(is_valid_vector(result_intersection.btn));
  }

  return result_intersection;
}

ETX_GPU_CODE bool random_continue(uint32_t path_length, uint32_t start_path_length, float eta_scale, Sampler& smp, SpectralResponse& throughput) {
  ETX_FUNCTION_SCOPE();

  if (path_length < start_path_length)
    return true;

  float max_t = throughput.maximum() * (eta_scale * eta_scale);
  if (valid_value(max_t) == false) {
    return false;
  }

  float q = min(0.95f, max_t);
  if ((q > 0.0f) && (smp.next() < q)) {
    throughput *= (1.0f / q);
    return true;
  }

  return false;
}

ETX_GPU_CODE SpectralResponse apply_rgb(const SpectralQuery spect, SpectralResponse response, const float4& value, const Scene& scene) {
  if (spect.spectral()) {
    auto scale = rgb_response(spect, {value.x, value.y, value.z});
    ETX_VALIDATE(scale);
    response *= scale;
    ETX_VALIDATE(response);
  } else {
    response.integrated *= float3{value.x, value.y, value.z};
  }

  return response;
}

ETX_GPU_CODE float4 sample_whole_image(const SampledImage& img, const float2& uv, const Scene& scene) {
  if (img.image_index == kInvalidIndex) {
    return img.value;
  }

  float4 eval = scene.images[img.image_index].evaluate(uv, nullptr);
  return img.value * eval;
}

ETX_GPU_CODE float evaluate_image(const SampledImage& img, const float2& uv, const Scene& scene, const float default_value) {
  float result = default_value;
  if ((img.image_index == kInvalidIndex) || (img.channel >= 4u)) {
    return result;
  }

  float4 eval = scene.images[img.image_index].evaluate(uv, nullptr);
  const float* data = reinterpret_cast<const float*>(&eval);
  return data[img.channel];
}

ETX_GPU_CODE float evaluate_metalness(const Material& material, const float2& uv, const Scene& scene) {
  return material.metalness.value.x * evaluate_image(material.metalness, uv, scene, 1.0f);
}

ETX_GPU_CODE float2 evaluate_roughness(const Material& material, const float2& uv, const Scene& scene) {
  return float2{material.roughness.value.x, material.roughness.value.y} * evaluate_image(material.roughness, uv, scene, 1.0f);
}

ETX_GPU_CODE SpectralResponse apply_image(SpectralQuery spect, const SpectralImage& img, const float2& uv, const Scene& scene, float* image_pdf) {
  if (image_pdf) {
    *image_pdf = 0.0f;
  }

  auto result = scene.spectrums[img.spectrum_index](spect);
  ETX_VALIDATE(result);
  if (img.image_index == kInvalidIndex) {
    return result;
  }

  float4 eval = scene.images[img.image_index].evaluate(uv, image_pdf);
  ETX_VALIDATE(eval);
  return apply_rgb(spect, result, eval, scene);
}

}  // namespace etx

#include <etx/render/shared/scene_bsdf.hxx>
#include <etx/render/shared/scene_bssrdf_subsurface.hxx>
#include <etx/render/shared/scene_camera.hxx>
#include <etx/render/shared/scene_emitters.hxx>
