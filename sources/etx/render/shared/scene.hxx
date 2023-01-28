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

template <class RT>
ETX_GPU_CODE SpectralResponse transmittance(SpectralQuery spect, Sampler& smp, const float3& p0, const float3& p1, uint32_t medium_index, const Scene& scene, const RT& rt) {
  SpectralResponse result = {spect.wavelength, 1.0f};

  float3 w_o = p1 - p0;
  ETX_CHECK_FINITE(w_o);

  float max_t = dot(w_o, w_o);
  if (max_t < kRayEpsilon) {
    return result;
  }

  max_t = sqrtf(max_t);
  w_o /= max_t;
  max_t -= kRayEpsilon;

  float3 origin = p0;

  for (;;) {
    Intersection intersection;
    if (rt.trace(scene, {origin, w_o, kRayEpsilon, max_t}, intersection, smp) == false) {
      if (medium_index != kInvalidIndex) {
        result *= scene.mediums[medium_index].transmittance(spect, smp, origin, w_o, max_t);
        ETX_VALIDATE(result);
      }
      break;
    }

    const auto& tri = scene.triangles[intersection.triangle_index];
    const auto& mat = scene.materials[tri.material_index];
    if (mat.cls != Material::Class::Boundary) {
      result = {spect.wavelength, 0.0f};
      break;
    }

    if (medium_index != kInvalidIndex) {
      result *= scene.mediums[medium_index].transmittance(spect, smp, origin, w_o, intersection.t);
      ETX_VALIDATE(result);
    }

    medium_index = (dot(intersection.nrm, w_o) < 0.0f) ? mat.int_medium : mat.ext_medium;
    origin = intersection.pos;
    max_t -= intersection.t;
  }

  return result;
}

namespace subsurface {

template <class RT>
ETX_GPU_CODE Gather gather(SpectralQuery spect, const Scene& scene, const Intersection& in_intersection, const uint32_t material_index, const RT& rt, Sampler& smp) {
  const auto& mtl = scene.materials[material_index];
  Ray ss_ray = {};
  if (subsurface::sample(in_intersection, mtl, smp, ss_ray) == false)
    return {};

  IntersectionBase intersections[subsurface::kMaxIntersections] = {};
  uint32_t intersection_count = rt.continuous_trace(scene, ss_ray, {intersections, subsurface::kMaxIntersections, material_index, in_intersection.triangle_index}, smp);
  ETX_CRITICAL(intersection_count <= subsurface::kMaxIntersections);
  if (intersection_count == 0) {
    return {};
  }

  Gather result = {};
  ArrayView<Intersection> iv = {result.intersections, subsurface::kMaxIntersections};
  ArrayView<SpectralResponse> wv = {result.weights, subsurface::kMaxIntersections};

  float total_weight = 0.0f;
  for (uint32_t i = 0; i < intersection_count; ++i) {
    auto out_intersection = rt.make_intersection(scene, ss_ray.d, intersections[i]);
    out_intersection.w_i = -out_intersection.nrm;
    auto pdf = subsurface::pdf_s_p(in_intersection, out_intersection, mtl.subsurface);
    ETX_VALIDATE(pdf);
    if (pdf > 0.0f) {
      float actual_distance = length(out_intersection.pos - in_intersection.pos);
      auto eval = subsurface::eval_s_r(spect, mtl.subsurface, actual_distance);
      ETX_VALIDATE(eval);
      auto weight = eval / pdf;
      ETX_VALIDATE(weight);
      if (weight.is_zero() == false) {
        total_weight += weight.average();
        iv[result.intersection_count] = out_intersection;
        wv[result.intersection_count] = weight;
        result.intersection_count += 1u;
      }
    }
  }

  if (total_weight > 0.0f) {
    float rnd = smp.next() * total_weight;
    float partial_sum = 0.0f;
    float sample_weight = 0.0f;
    for (uint32_t i = 0; i < result.intersection_count; ++i) {
      sample_weight = wv[i].average();
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
