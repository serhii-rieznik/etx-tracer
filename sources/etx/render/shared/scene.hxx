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

ETX_GPU_CODE SpectralResponse apply_image(SpectralQuery spect, const SpectralImage& img, const float2& uv, const Scene& scene) {
  SpectralResponse result = img.spectrum(spect);

  if (img.image_index != kInvalidIndex) {
    float4 eval = scene.images[img.image_index].evaluate(uv);
    result *= rgb::query_spd(spect, {eval.x, eval.y, eval.z}, scene.spectrums->rgb_reflection);
    ETX_VALIDATE(result);
  }
  return result;
}

namespace subsurface {

template <class RT>
ETX_GPU_CODE bool gather_rw(SpectralQuery spect, const Scene& scene, const Intersection& in_intersection, const RT& rt, Sampler& smp, Gather& result) {
  constexpr uint32_t kMaxIterations = 1024u;
  constexpr float kUniformPDF = 1.0f / float(SpectralResponse::component_count());

  const auto& mat = scene.materials[in_intersection.material_index];
  if (mat.int_medium == kInvalidIndex)
    return false;

  const Medium& medium = scene.mediums[mat.int_medium];

  float anisotropy = medium.phase_function_g;
  SpectralResponse scattering = medium.s_scattering(spect);
  SpectralResponse absorption = medium.s_absorption(spect);

  SpectralResponse extinction = scattering + absorption;
  SpectralResponse albedo = {
    spect.wavelength,
    {
      scattering.components.x > 0.0f ? (extinction.components.x / scattering.components.x) : 0.0f,
      scattering.components.y > 0.0f ? (extinction.components.y / scattering.components.y) : 0.0f,
      scattering.components.z > 0.0f ? (extinction.components.z / scattering.components.z) : 0.0f,
    },
  };

  float3 n = in_intersection.nrm * ((dot(in_intersection.w_i, in_intersection.nrm) < 0.0f) ? -1.0f : +1.0f);

  Ray ray = {};
  ray.d = sample_cosine_distribution(smp.next_2d(), n, 1.0f);
  ray.min_t = kRayEpsilon;
  ray.o = shading_pos(scene.vertices, scene.triangles[in_intersection.triangle_index], in_intersection.barycentric, ray.d);
  ray.max_t = kMaxFloat;

  SpectralResponse throughput = {spect.wavelength, 1.0f};
  for (uint32_t i = 0; i < kMaxIterations; ++i) {
    SpectralResponse pdf = {};
    uint32_t channel = Medium::sample_spectrum_component(spect, albedo, throughput, smp, pdf);
    float scattering_distance = extinction.component(channel);

    ray.max_t = scattering_distance > 0.0f ? (-logf(1.0f - smp.next()) / scattering_distance) : kMaxFloat;
    ETX_VALIDATE(ray.max_t);

    Intersection local_i;
    bool intersection_found = rt.trace_material(scene, ray, in_intersection.material_index, local_i, smp);
    if (intersection_found) {
      ray.max_t = local_i.t;
    }

    SpectralResponse tr = exp(-ray.max_t * extinction);
    ETX_VALIDATE(tr);

    pdf *= intersection_found ? tr : tr * extinction;
    ETX_VALIDATE(pdf);

    if (pdf.is_zero())
      return false;

    SpectralResponse weight = intersection_found ? tr : tr * scattering;
    ETX_VALIDATE(weight);

    throughput *= weight / pdf.sum();
    ETX_VALIDATE(throughput);

    if (throughput.maximum() <= kEpsilon)
      return false;

    if (intersection_found) {
      result.intersections[0] = local_i;
      result.intersections[0].w_i = in_intersection.w_i;
      result.weights[0] = throughput * apply_image(spect, mat.transmittance, local_i.tex, scene);
      result.intersection_count = 1u;
      result.selected_intersection = 0;
      result.selected_sample_weight = 1.0f;
      result.total_weight = 1.0f;
      return true;
    }

    ray.o = ray.o + ray.d * ray.max_t;
    ray.d = Medium::sample_phase_function(spect, smp, ray.d, anisotropy);
  }

  return false;
}

template <class RT>
ETX_GPU_CODE bool gather_cb(SpectralQuery spect, const Scene& scene, const Intersection& in_intersection, const RT& rt, Sampler& smp, Gather& result) {
  const auto& mtl = scene.materials[in_intersection.material_index].subsurface;

  Sample ss_samples[kIntersectionDirections] = {
    sample(spect, in_intersection, mtl, 0u, smp),
    sample(spect, in_intersection, mtl, 1u, smp),
    sample(spect, in_intersection, mtl, 2u, smp),
  };

  IntersectionBase intersections[kTotalIntersections] = {};
  ContinousTraceOptions ct = {intersections, kIntersectionsPerDirection, in_intersection.material_index};
  uint32_t intersections_0 = rt.continuous_trace(scene, ss_samples[0].ray, ct, smp);
  ct.intersection_buffer += intersections_0;
  uint32_t intersections_1 = rt.continuous_trace(scene, ss_samples[1].ray, ct, smp);
  ct.intersection_buffer += intersections_1;
  uint32_t intersections_2 = rt.continuous_trace(scene, ss_samples[2].ray, ct, smp);

  uint32_t intersection_count = intersections_0 + intersections_1 + intersections_2;
  ETX_CRITICAL(intersection_count <= kTotalIntersections);
  if (intersection_count == 0) {
    return false;
  }

  result = {};
  for (uint32_t i = 0; i < intersection_count; ++i) {
    const Sample& ss_sample = (i < intersections_0) ? ss_samples[0] : (i < intersections_0 + intersections_1 ? ss_samples[1] : ss_samples[2]);

    auto out_intersection = make_intersection(scene, in_intersection.w_i, intersections[i]);

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

    result.total_weight += weight.average();
    result.intersections[result.intersection_count] = out_intersection;
    result.weights[result.intersection_count] = weight;
    result.intersection_count += 1u;
  }

  if (result.total_weight > 0.0f) {
    float rnd = smp.next() * result.total_weight;
    float partial_sum = 0.0f;
    float sample_weight = 0.0f;
    for (uint32_t i = 0; i < result.intersection_count; ++i) {
      sample_weight = result.weights[i].average();
      float next_sum = partial_sum + sample_weight;
      if (rnd < next_sum) {
        result.selected_intersection = i;
        result.selected_sample_weight = result.total_weight / sample_weight;
        break;
      }
      partial_sum = next_sum;
    }
    ETX_ASSERT(result.selected_intersection != kInvalidIndex);
  }

  return result.intersection_count > 0;
}

template <class RT>
ETX_GPU_CODE bool gather(SpectralQuery spect, const Scene& scene, const Intersection& in_intersection, const RT& rt, Sampler& smp, Gather& result) {
  const auto& mtl = scene.materials[in_intersection.material_index].subsurface;
  switch (mtl.cls) {
    case SubsurfaceMaterial::Class::ChristensenBurley:
      return gather_cb(spect, scene, in_intersection, rt, smp, result);
    default:
      return gather_rw(spect, scene, in_intersection, rt, smp, result);
  }
}

}  // namespace subsurface
}  // namespace etx

#include <etx/render/shared/scene_bsdf.hxx>
#include <etx/render/shared/scene_camera.hxx>
#include <etx/render/shared/scene_emitters.hxx>
