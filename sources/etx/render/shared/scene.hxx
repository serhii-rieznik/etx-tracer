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
#include <etx/render/shared/bssrdf_subsurface.hxx>

namespace etx {

struct ETX_ALIGNED EnvironmentEmitters {
  constexpr static const uint32_t kMaxCount = 7;
  uint32_t emitters[kMaxCount] ETX_EMPTY_INIT;
  uint32_t count ETX_EMPTY_INIT;
};

struct ETX_ALIGNED Scene {
  Camera camera ETX_EMPTY_INIT;
  ArrayView<Vertex> vertices ETX_EMPTY_INIT;
  ArrayView<Triangle> triangles ETX_EMPTY_INIT;
  ArrayView<uint32_t> triangle_to_material ETX_EMPTY_INIT;
  ArrayView<uint32_t> triangle_to_emitter ETX_EMPTY_INIT;
  ArrayView<Material> materials ETX_EMPTY_INIT;
  ArrayView<Emitter> emitters ETX_EMPTY_INIT;
  ArrayView<Image> images ETX_EMPTY_INIT;
  ArrayView<Medium> mediums ETX_EMPTY_INIT;
  Distribution emitters_distribution ETX_EMPTY_INIT;
  EnvironmentEmitters environment_emitters ETX_EMPTY_INIT;
  Pointer<Spectrums> spectrums ETX_EMPTY_INIT;
  float3 bounding_sphere_center ETX_EMPTY_INIT;
  float bounding_sphere_radius ETX_EMPTY_INIT;
  uint64_t acceleration_structure ETX_EMPTY_INIT;
  uint32_t camera_medium_index ETX_INIT_WITH(kInvalidIndex);
  uint32_t camera_lens_shape_image_index ETX_INIT_WITH(kInvalidIndex);
  uint32_t max_path_length ETX_INIT_WITH(65535u);
  uint32_t samples ETX_INIT_WITH(256u);
  uint32_t random_path_termination ETX_INIT_WITH(6u);
  bool spectral ETX_INIT_WITH(false);
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
    /*     */ v0.pos * bc.x + v1.pos * bc.y + v2.pos * bc.z,
    normalize(v0.nrm * bc.x + v1.nrm * bc.y + v2.nrm * bc.z),
    normalize(v0.tan * bc.x + v1.tan * bc.y + v2.tan * bc.z),
    normalize(v0.btn * bc.x + v1.btn * bc.y + v2.btn * bc.z),
    /*     */ v0.tex * bc.x + v1.tex * bc.y + v2.tex * bc.z,
  };
}

ETX_GPU_CODE void lerp_vertex(Vertex& v, const ArrayView<Vertex>& vertices, const Triangle& t, const float3& bc) {
  const auto& v0 = vertices[t.i[0]];
  const auto& v1 = vertices[t.i[1]];
  const auto& v2 = vertices[t.i[2]];
  v.pos = /*     */ v0.pos * bc.x + v1.pos * bc.y + v2.pos * bc.z;
  v.nrm = normalize(v0.nrm * bc.x + v1.nrm * bc.y + v2.nrm * bc.z);
  v.tan = normalize(v0.tan * bc.x + v1.tan * bc.y + v2.tan * bc.z);
  v.btn = normalize(v0.btn * bc.x + v1.btn * bc.y + v2.btn * bc.z);
  v.tex = /*     */ v0.tex * bc.x + v1.tex * bc.y + v2.tex * bc.z;
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
  if ((mat.normal_image_index != kInvalidIndex) && (mat.normal_scale > 0.0f)) {
    auto sampled_normal = scene.images[mat.normal_image_index].evaluate_normal(result_intersection.tex, mat.normal_scale);
    float3x3 from_local = {
      float3{result_intersection.tan.x, result_intersection.tan.y, result_intersection.tan.z},
      float3{result_intersection.btn.x, result_intersection.btn.y, result_intersection.btn.z},
      float3{result_intersection.nrm.x, result_intersection.nrm.y, result_intersection.nrm.z},
    };
    result_intersection.nrm = normalize(from_local * sampled_normal);
    result_intersection.tan = normalize(result_intersection.tan - result_intersection.nrm * dot(result_intersection.tan, result_intersection.nrm));
    result_intersection.btn = normalize(cross(result_intersection.nrm, result_intersection.tan));
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

ETX_GPU_CODE SpectralResponse apply_image(SpectralQuery spect, const SpectralImage& img, const float2& uv, const Scene& scene, rgb::SpectrumClass cls) {
  auto result = img.spectrum(spect);
  ETX_VALIDATE(result);
  if (img.image_index == kInvalidIndex) {
    return result;
  }

  float4 eval = scene.images[img.image_index].evaluate(uv);
  ETX_VALIDATE(eval);
  if (spect.spectral()) {
    auto scale = rgb::query_spd(spect, {eval.x, eval.y, eval.z}, cls == rgb::SpectrumClass::Illuminant ? scene.spectrums->rgb_illuminant : scene.spectrums->rgb_reflection);
    ETX_VALIDATE(scale);
    result *= scale;
    ETX_VALIDATE(result);
  } else {
    float3 result_rgb = spectrum::xyz_to_rgb(result.components.xyz);
    result.components.xyz = spectrum::rgb_to_xyz(result_rgb * float3{eval.x, eval.y, eval.z});
  }

  return result;
}

}  // namespace etx

#include <etx/render/shared/scene_bsdf.hxx>
#include <etx/render/shared/scene_camera.hxx>
#include <etx/render/shared/scene_emitters.hxx>
