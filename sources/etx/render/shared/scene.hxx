#pragma once

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

  const auto& mat = scene.materials[tri.material_index];
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

namespace subsurface {

template <class RT>
ETX_GPU_CODE Gather gather(SpectralQuery spect, const Scene& scene, const Intersection& in_intersection, const uint32_t material_index, const RT& rt, Sampler& smp) {
  const auto& mtl = scene.materials[material_index].subsurface;

  Sample ss_samples[3] = {
    sample(in_intersection, mtl, 0u, smp),
    sample(in_intersection, mtl, 1u, smp),
    sample(in_intersection, mtl, 2u, smp),
  };

  IntersectionBase intersections[kTotalIntersection] = {};

  ContinousTraceOptions ct = {intersections, kIntersectionsPerDirection, material_index};
  uint32_t intersections_0 = rt.continuous_trace(scene, ss_samples[0].ray, ct, smp);
  ct.intersection_buffer += intersections_0;
  uint32_t intersections_1 = rt.continuous_trace(scene, ss_samples[1].ray, ct, smp);
  ct.intersection_buffer += intersections_1;
  uint32_t intersections_2 = rt.continuous_trace(scene, ss_samples[2].ray, ct, smp);

  uint32_t intersection_count = intersections_0 + intersections_1 + intersections_2;
  ETX_CRITICAL(intersection_count <= kTotalIntersection);
  if (intersection_count == 0) {
    return {};
  }

  Gather result = {};

  float total_weight = 0.0f;
  for (uint32_t i = 0; i < intersection_count; ++i) {
    Sample& ss_sample = (i < intersections_0) ? ss_samples[0] : (i < intersections_0 + intersections_1 ? ss_samples[1] : ss_samples[2]);

    auto out_intersection = make_intersection(scene, ss_sample.ray.d, intersections[i]);

    float gw = geometric_weigth(out_intersection.nrm, ss_sample);
    float pdf = evaluate(spect, mtl, ss_sample.sampled_radius).average();
    ETX_VALIDATE(pdf);
    if (pdf <= 0.0f)
      continue;

    auto eval = evaluate(spect, mtl, length(out_intersection.pos - in_intersection.pos));
    ETX_VALIDATE(eval);

    auto weight = eval / pdf * gw;
    ETX_VALIDATE(weight);

    if (weight.is_zero())
      continue;

    total_weight += weight.average();
    result.intersections[result.intersection_count] = out_intersection;
    result.weights[result.intersection_count] = weight;
    result.intersection_count += 1u;
  }

  if (total_weight > 0.0f) {
    float rnd = smp.next() * total_weight;
    float partial_sum = 0.0f;
    float sample_weight = 0.0f;
    for (uint32_t i = 0; i < result.intersection_count; ++i) {
      sample_weight = result.weights[i].average();
      float next_sum = partial_sum + sample_weight;
      if (rnd < next_sum) {
        result.selected_intersection = i;
        result.selected_sample_weight = total_weight / sample_weight;
        break;
      }
      partial_sum = next_sum;
    }
    ETX_ASSERT(result.selected_intersection != kInvalidIndex);
  }

  return result;
}

}  // namespace subsurface
}  // namespace etx

#include <etx/render/shared/scene_bsdf.hxx>
#include <etx/render/shared/scene_camera.hxx>
#include <etx/render/shared/scene_emitters.hxx>
