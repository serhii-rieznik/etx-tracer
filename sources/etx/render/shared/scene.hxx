#pragma once

#include <etx/render/shared/spectrum.hxx>
#include <etx/render/shared/camera.hxx>
#include <etx/render/shared/image.hxx>
#include <etx/render/shared/medium.hxx>
#include <etx/render/shared/material.hxx>
#include <etx/render/shared/emitter.hxx>
#include <etx/render/shared/sampler.hxx>
#include <etx/render/shared/bsdf.hxx>

namespace etx {

struct alignas(16) EnvironmentEmitters {
  constexpr static const uint32_t kMaxCount = 7;
  uint32_t emitters[kMaxCount];
  uint32_t count;
};

struct alignas(16) Scene {
  Camera camera;
  ArrayView<Vertex> vertices ETX_EMPTY_INIT;
  ArrayView<Triangle> triangles ETX_EMPTY_INIT;
  ArrayView<Material> materials ETX_EMPTY_INIT;
  ArrayView<Emitter> emitters ETX_EMPTY_INIT;
  ArrayView<Image> images ETX_EMPTY_INIT;
  ArrayView<Medium> mediums ETX_EMPTY_INIT;
  Distribution emitters_distribution ETX_EMPTY_INIT;
  EnvironmentEmitters environment_emitters ETX_EMPTY_INIT;
  Spectrums* spectrums ETX_EMPTY_INIT;
  float3 bounding_sphere_center ETX_EMPTY_INIT;
  float bounding_sphere_radius ETX_EMPTY_INIT;
  uint64_t acceleration_structure ETX_EMPTY_INIT;
  uint32_t camera_medium_index ETX_INIT_WITH(kInvalidIndex);
  uint32_t camera_lens_shape_image_index ETX_INIT_WITH(kInvalidIndex);

  ETX_GPU_CODE bool valid() const {
    return (vertices.count > 0) && (triangles.count > 0) && (materials.count > 0) && (emitters.count > 0) && (bounding_sphere_radius > 0.0f);
  }
};

ETX_GPU_CODE float2 get_jittered_uv(Sampler& smp, const uint2& pixel, const uint2& dim) {
  return {
    (float(pixel.x) + smp.next()) / float(dim.x) * 2.0f - 1.0f,
    (float(pixel.y) + smp.next()) / float(dim.y) * 2.0f - 1.0f,
  };
}

ETX_GPU_CODE Ray generate_ray(Sampler& smp, const Scene& scene, const float2& uv) {
  float3 s = (uv.x * scene.camera.aspect) * scene.camera.side;
  float3 u = (uv.y) * scene.camera.up;
  float3 w_o = normalize(scene.camera.tan_half_fov * (s + u) + scene.camera.direction);

  float3 origin = scene.camera.position;
  if (scene.camera.lens_radius > 0.0f) {
    float2 sensor_sample = {};
    if (scene.camera_lens_shape_image_index == kInvalidIndex) {
      sensor_sample = sample_disk(smp.next(), smp.next());
    } else {
      float pdf = {};
      uint2 location = {};
      sensor_sample = scene.images[scene.camera_lens_shape_image_index].sample(smp.next(), smp.next(), pdf, location);
      sensor_sample = sensor_sample * 2.0f - 1.0f;
    }
    sensor_sample *= scene.camera.lens_radius;
    origin = origin + scene.camera.side * sensor_sample.x + scene.camera.up * sensor_sample.y;
    float focal_plane_distance = scene.camera.focal_distance / dot(w_o, scene.camera.direction);
    float3 p = scene.camera.position + focal_plane_distance * w_o;
    w_o = normalize(p - origin);
  }

  return {origin, w_o, kRayEpsilon, kMaxFloat};
}

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

ETX_GPU_CODE bool apply_rr(float eta_scale, float rnd, SpectralResponse& throughput) {
  float max_t = throughput.maximum() * (eta_scale * eta_scale);
  if (valid_value(max_t) == false) {
    return false;
  }

  float q = min(0.95f, max_t);
  if ((q > 0.0f) && (rnd < q)) {
    throughput *= (1.0f / q);
    return true;
  }

  return false;
}

ETX_GPU_CODE float emitter_pdf_area_local(const Emitter& em, const Scene& scene) {
  ETX_ASSERT(em.is_local());
  const auto& tri = scene.triangles[em.triangle_index];
  return 1.0f / tri.area;
}

ETX_GPU_CODE SpectralResponse emitter_evaluate_in_local(const Emitter& em, const SpectralQuery spect, const float2& uv, const float3& pos, const float3& to_point, float& pdf_area,
  float& pdf_dir, float& pdf_dir_out, const Scene& scene, const bool no_collimation) {
  ETX_ASSERT(em.is_local());

  const auto& tri = scene.triangles[em.triangle_index];
  if ((em.emission_direction == Emitter::Direction::Single) && (dot(tri.geo_n, to_point - pos) >= 0.0f)) {
    return {spect.wavelength, 0.0f};
  }

  auto dp = pos - to_point;

  pdf_area = emitter_pdf_area_local(em, scene);
  if (em.emission_direction == Emitter::Direction::Omni) {
    pdf_dir = pdf_area * dot(dp, dp);
    pdf_dir_out = pdf_area;
  } else {
    pdf_dir = area_to_solid_angle_probability(pdf_area, to_point, tri.geo_n, pos, no_collimation ? 1.0f : em.collimation);
    pdf_dir_out = pdf_area * fabsf(dot(tri.geo_n, normalize(dp))) * kInvPi;
  }

  SpectralResponse result = em.emission(spect);
  ETX_VALIDATE(result);

  if (em.image_index != kInvalidIndex) {
    auto sampled_value = scene.images[em.image_index].evaluate(uv);
    result *= rgb::query_spd(spect, {sampled_value.x, sampled_value.y, sampled_value.z}, scene.spectrums->rgb_illuminant);
    ETX_VALIDATE(result);
  }

  return result;
}

ETX_GPU_CODE SpectralResponse emitter_evaluate_out_local(const Emitter& em, const SpectralQuery spect, const float2& uv, const float3& emitter_normal, const float3& direction,
  float& pdf_area, float& pdf_dir, float& pdf_dir_out, const Scene& scene) {
  ETX_ASSERT(em.is_local());

  switch (em.emission_direction) {
    case Emitter::Direction::Single: {
      pdf_dir = max(0.0f, dot(emitter_normal, direction)) * kInvPi;
      break;
    }
    case Emitter::Direction::TwoSided: {
      pdf_dir = 0.5f * fabsf(dot(emitter_normal, direction)) * kInvPi;
      break;
    }
    case Emitter::Direction::Omni: {
      pdf_dir = kInvPi;
      break;
    }
  }
  ETX_ASSERT(pdf_dir > 0.0f);

  pdf_area = emitter_pdf_area_local(em, scene);
  ETX_ASSERT(pdf_area > 0.0f);

  pdf_dir_out = pdf_dir * pdf_area;
  ETX_ASSERT(pdf_dir_out > 0.0f);

  SpectralResponse result = em.emission(spect);
  if (em.image_index != kInvalidIndex) {
    auto sampled_value = scene.images[em.image_index].evaluate(uv);
    result *= rgb::query_spd(spect, {sampled_value.x, sampled_value.y, sampled_value.z}, scene.spectrums->rgb_illuminant);
  }

  ETX_VALIDATE(result);
  return result;
}

ETX_GPU_CODE SpectralResponse emitter_evaluate_in_dist(const Emitter& em, const SpectralQuery spect, const float3& in_direction, float& pdf_area, float& pdf_dir,
  float& pdf_dir_out, const Scene& scene) {
  ETX_ASSERT(em.is_distant());

  if (em.cls == Emitter::Class::Directional) {
    if ((em.angular_size > 0.0f) && (dot(in_direction, em.direction) >= em.angular_size_cosine)) {
      pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
      pdf_dir = 1.0f;
      pdf_dir_out = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
      return em.emission(spect);
    } else {
      pdf_area = 0.0f;
      pdf_dir = 0.0f;
      pdf_dir_out = 0.0f;
      return {spect.wavelength, 0.0f};
    }
  }

  float2 uv = direction_to_uv(in_direction);
  float sin_t = sinf(uv.y * kPi);
  if (sin_t == 0.0f) {
    pdf_area = 0.0f;
    pdf_dir = 0.0f;
    pdf_dir_out = 0.0f;
    return {spect.wavelength, 0.0f};
  }

  const auto& img = scene.images[em.image_index];
  pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
  
  pdf_dir = img.pdf(uv) / (2.0f * kPi * kPi * sin_t);
  ETX_VALIDATE(pdf_dir);

  pdf_dir_out = pdf_area * pdf_dir;
  return em.emission(spect) * rgb::query_spd(spect, to_float3(img.evaluate(uv)), scene.spectrums->rgb_illuminant);
}

ETX_GPU_CODE SpectralResponse emitter_evaluate_env_dist(const Emitter& em, const SpectralQuery spect, const float2& uv, float& pdf_area, float& pdf_dir, float& pdf_dir_out,
  const Scene& scene) {
  ETX_ASSERT(em.cls == Emitter::Class::Environment);

  float sin_t = sinf(uv.y * kPi);
  if (sin_t == 0.0f) {
    pdf_area = 0.0f;
    pdf_dir = 0.0f;
    pdf_dir_out = 0.0f;
    return {spect.wavelength, 0.0f};
  }

  const auto& img = scene.images[em.image_index];
  pdf_dir = img.pdf(uv) / (2.0f * kPi * kPi * sin_t);
  pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
  pdf_dir_out = pdf_area * pdf_dir;

  auto eval = to_float3(img.evaluate(uv));
  return SpectralResponse(spect.wavelength, eval);
}

ETX_GPU_CODE SpectralResponse emitter_evaluate_out_dist(const Emitter& em, const SpectralQuery spect, const float3& in_direction, float& pdf_area, float& pdf_dir,
  float& pdf_dir_out, const Scene& scene) {
  ETX_ASSERT(em.is_distant());

  float2 uv = direction_to_uv(in_direction);
  float sin_t = sinf(uv.y * kPi);
  if (sin_t == 0.0f) {
    pdf_dir = 0.0f;
    pdf_area = 0.0f;
    pdf_dir_out = pdf_dir;
    return {spect.wavelength, 0.0f};
  }

  switch (em.cls) {
    case Emitter::Class::Environment: {
      const auto& img = scene.images[em.image_index];
      pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
      pdf_dir = img.pdf(uv) / (2.0f * kPi * kPi * sin_t);
      pdf_dir_out = pdf_dir * pdf_area;
      return em.emission(spect) * rgb::query_spd(spect, to_float3(img.evaluate(uv)), scene.spectrums->rgb_illuminant);
    }
    case Emitter::Class::Directional: {
      pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
      pdf_dir = 1.0f;
      pdf_dir_out = pdf_dir * pdf_area;
      return em.emission(spect);
    }

    default:
      return {spect.wavelength, 0.0f};
  }
}

ETX_GPU_CODE float emitter_pdf_in_dist(const Emitter& em, const float3& in_direction, const Scene& scene) {
  ETX_ASSERT(em.is_distant());

  if (em.cls == Emitter::Class::Directional) {
    if ((em.angular_size > 0.0f) && (dot(in_direction, em.direction) >= em.angular_size_cosine)) {
      return 1.0f;
    } else {
      return 0.0f;
    }
  }

  float2 uv = direction_to_uv(in_direction);
  float sin_t = sinf(uv.y * kPi);
  if (sin_t <= kEpsilon) {
    return 0.0f;
  }

  const auto& img = scene.images[em.image_index];
  return img.pdf(uv) / (2.0f * kPi * kPi * sin_t);
}

ETX_GPU_CODE EmitterSample emitter_sample_in(const Emitter& em, const SpectralQuery spect, Sampler& smp, const float3& from_point, const Scene& scene) {
  constexpr float kDisantRadiusScale = 2.0f;

  EmitterSample result;
  switch (em.cls) {
    case Emitter::Class::Area: {
      const auto& tri = scene.triangles[em.triangle_index];
      result.barycentric = random_barycentric(smp.next(), smp.next());
      result.origin = lerp_pos(scene.vertices, tri, result.barycentric);
      result.normal = lerp_normal(scene.vertices, tri, result.barycentric);
      result.direction = normalize(result.origin - from_point);
      result.value = emitter_evaluate_in_local(em, spect, lerp_uv(scene.vertices, tri, result.barycentric), from_point, result.origin, result.pdf_area, result.pdf_dir,
        result.pdf_dir_out, scene, false);
      break;
    }

    case Emitter::Class::Environment: {
      const auto& img = scene.images[em.image_index];
      float pdf_image = 0.0f;
      uint2 image_location = {};
      result.image_uv = img.sample(smp.next(), smp.next(), pdf_image, image_location);
      result.direction = uv_to_direction(result.image_uv);
      result.normal = -result.direction;
      result.origin = from_point + kDisantRadiusScale * scene.bounding_sphere_radius * result.direction;
      result.value = emitter_evaluate_env_dist(em, spect, result.image_uv, result.pdf_area, result.pdf_dir, result.pdf_dir_out, scene);
      break;
    }

    case Emitter::Class::Directional: {
      if (em.angular_size > 0.0f) {
        auto basic = orthonormal_basis(em.direction);
        float2 disk = em.equivalent_disk_size * sample_disk(smp.next(), smp.next());
        result.direction = normalize(em.direction + basic.u * disk.x + basic.v * disk.y);
      } else {
        result.direction = em.direction;
      }
      result.pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
      result.pdf_dir = 1.0f;
      result.pdf_dir_out = result.pdf_dir * result.pdf_area;
      result.origin = from_point + kDisantRadiusScale * scene.bounding_sphere_radius * result.direction;
      result.normal = em.direction * (-1.0f);
      result.value = em.emission(spect);
      break;
    }
    default: {
      ETX_FAIL("Invalid emitter");
    }
  }
  return result;
}

ETX_GPU_CODE EmitterSample sample_emitter(SpectralQuery spect, Sampler& smp, const float3& from_point, const Scene& scene) {
  float pdf_sample = 0.0f;
  uint32_t emitter_index = static_cast<uint32_t>(scene.emitters_distribution.sample(smp.next(), pdf_sample));
  ETX_ASSERT(emitter_index < scene.emitters_distribution.size);

  const auto& emitter = scene.emitters[emitter_index];
  EmitterSample sample = emitter_sample_in(emitter, spect, smp, from_point, scene);
  sample.pdf_sample = pdf_sample;
  sample.emitter_index = emitter_index;
  sample.triangle_index = emitter.triangle_index;
  sample.is_delta = emitter.is_delta();
  return sample;
}

ETX_GPU_CODE float emitter_discrete_pdf(const Emitter& emitter, const Distribution& dist) {
  return emitter.weight / dist.total_weight;
}

#define ETX_DECLARE_BSDF(Class)                                                                                             \
  namespace Class##BSDF {                                                                                                   \
    ETX_GPU_CODE BSDFSample sample(struct Sampler&, const struct BSDFData&, const Scene& scene);                            \
    ETX_GPU_CODE BSDFEval evaluate(const struct BSDFData& data, const Scene& scene);                                        \
    ETX_GPU_CODE float pdf(const struct BSDFData& data, const Scene& scene);                                                \
    ETX_GPU_CODE bool continue_tracing(const struct Material&, const float2& tex, const Scene& scene, struct Sampler& smp); \
  }

ETX_DECLARE_BSDF(Diffuse);
ETX_DECLARE_BSDF(Plastic);
ETX_DECLARE_BSDF(Conductor);
ETX_DECLARE_BSDF(MultiscatteringConductor);
ETX_DECLARE_BSDF(Dielectric);
ETX_DECLARE_BSDF(Thinfilm);
ETX_DECLARE_BSDF(Translucent);
ETX_DECLARE_BSDF(Mirror);
ETX_DECLARE_BSDF(Boundary);
ETX_DECLARE_BSDF(Generic);
ETX_DECLARE_BSDF(Coating);

namespace bsdf {

#define CASE_IMPL(CLS, FUNC, ...) \
  case Material::Class::CLS:      \
    return CLS##BSDF::FUNC(__VA_ARGS__)

#define CASE_IMPL_SAMPLE(A) CASE_IMPL(A, sample, smp, data, scene)
#define CASE_IMPL_EVALUATE(A) CASE_IMPL(A, evaluate, data, scene)
#define CASE_IMPL_PDF(A) CASE_IMPL(A, pdf, data, scene)
#define CASE_IMPL_CONTINUE(A) CASE_IMPL(A, continue_tracing, material, tex, scene, smp)

#define ALL_CASES(MACRO, CLS)               \
  switch (CLS) {                            \
    MACRO(Diffuse);                         \
    MACRO(Plastic);                         \
    MACRO(Conductor);                       \
    MACRO(MultiscatteringConductor);        \
    MACRO(Dielectric);                      \
    MACRO(Thinfilm);                        \
    MACRO(Translucent);                     \
    MACRO(Mirror);                          \
    MACRO(Boundary);                        \
    MACRO(Generic);                         \
    MACRO(Coating);                         \
    default:                                \
      ETX_FAIL("Unhandled material class"); \
      return {};                            \
  }

[[nodiscard]] ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const struct Scene& scene, struct Sampler& smp) {
  ALL_CASES(CASE_IMPL_SAMPLE, data.material.cls);
};
[[nodiscard]] ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const struct Scene& scene) {
  ALL_CASES(CASE_IMPL_EVALUATE, data.material.cls);
}

[[nodiscard]] ETX_GPU_CODE float pdf(const BSDFData& data, const struct Scene& scene) {
  ALL_CASES(CASE_IMPL_PDF, data.material.cls);
}

[[nodiscard]] ETX_GPU_CODE bool continue_tracing(const Material& material, const float2& tex, const struct Scene& scene, struct Sampler& smp) {
  ALL_CASES(CASE_IMPL_CONTINUE, material.cls);
}

#undef CASE_IMPL

ETX_GPU_CODE SpectralResponse apply_image(SpectralQuery spect, const SpectralResponse& value, uint32_t image_index, const float2& uv, const Scene& scene) {
  SpectralResponse result = value;

  if (image_index != kInvalidIndex) {
    float4 eval = scene.images[image_index].evaluate(uv);
    result *= rgb::query_spd(spect, {eval.x, eval.y, eval.z}, scene.spectrums->rgb_reflection);
    ETX_VALIDATE(result);
  }
  return result;
}

}  // namespace bsdf
}  // namespace etx

#include <etx/render/shared/bsdf_external.hxx>
#include <etx/render/shared/bsdf_conductor.hxx>
#include <etx/render/shared/bsdf_dielectric.hxx>
#include <etx/render/shared/bsdf_generic.hxx>
#include <etx/render/shared/bsdf_plastic.hxx>
#include <etx/render/shared/bsdf_various.hxx>
