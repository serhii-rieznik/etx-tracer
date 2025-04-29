#pragma once

namespace etx {

struct PathVertex {
  enum class Class : uint16_t {
    Invalid,
    Camera,
    Emitter,
    Surface,
    Medium,
  };

  Intersection intersection = {};
  SpectralResponse throughput = {};
  uint32_t medium_index = kInvalidIndex;
  struct {
    float forward = 0.0f;
    float backward = 0.0f;
  } pdf;
  Class cls = Class::Invalid;
  bool delta_connection = false;
  bool delta_emitter = false;
  bool is_subsurface = false;

  PathVertex() = default;

  PathVertex(Class c, const Intersection& i)
    : intersection(i)
    , cls(c) {
  }

  PathVertex(const Medium::Sample& i, const float3& a_w_i)
    : cls(Class::Medium) {
    intersection.pos = i.pos;
    intersection.w_i = a_w_i;
  }

  PathVertex(Class c)
    : cls(c) {
  }

  bool is_specific_emitter() const {
    return (intersection.emitter_index != kInvalidIndex);
  }

  bool is_environment_emitter() const {
    return (cls == Class::Emitter) && (intersection.triangle_index == kInvalidIndex);
  }

  bool is_emitter() const {
    return is_specific_emitter() || is_environment_emitter();
  }

  bool is_surface_interaction() const {
    return (intersection.triangle_index != kInvalidIndex);
  }

  bool is_medium_interaction() const {
    return (cls == Class::Medium) && (medium_index != kInvalidIndex);
  }

  static bool safe_normalize(const float3& a, const float3& b, float3& n) {
    n = a - b;
    float len = dot(n, n);
    if (len == 0.0f)
      return false;

    n *= 1.0f / sqrtf(len);
    return true;
  }

  float pdf_area(SpectralQuery spect, PathSource mode, const PathVertex* prev, const PathVertex* next, const Scene& scene, Sampler& smp) const {
    if (cls == Class::Emitter) {
      ETX_ASSERT(prev == nullptr);
      return pdf_to_light_out(spect, next, scene);
    }

    ETX_ASSERT(prev != nullptr);
    ETX_ASSERT(next != nullptr);
    ETX_ASSERT(is_surface_interaction() || is_medium_interaction());

    float3 w_i = {};
    float3 w_o = {};
    if (safe_normalize(intersection.pos, prev->intersection.pos, w_i) == false)
      return 0.0f;
    if (safe_normalize(next->intersection.pos, intersection.pos, w_o) == false)
      return 0.0f;

    float eval_pdf = 0.0f;
    if (is_surface_interaction()) {
      ETX_ASSERT((is_subsurface == false) || (is_subsurface && (intersection.material_index == 0)));
      const auto& mat = scene.materials[intersection.material_index];
      eval_pdf = bsdf::pdf({spect, medium_index, mode, intersection, w_i}, w_o, mat, scene, smp);
    } else if (is_medium_interaction()) {
      eval_pdf = scene.mediums[medium_index].phase_function(spect, intersection.pos, w_i, w_o);
    } else {
      ETX_FAIL("Invalid vertex class");
    }
    ETX_VALIDATE(eval_pdf);

    return pdf_solid_angle_to_area(eval_pdf, *next);
  }

  float pdf_to_light_out(SpectralQuery spect, const PathVertex* next, const Scene& scene) const {
    ETX_ASSERT(next != nullptr);
    ETX_ASSERT(is_emitter());

    float pdf_area = 0.0f;
    float pdf_dir = 0.0f;
    float pdf_dir_out = 0.0f;

    if (is_specific_emitter()) {
      const auto& emitter = scene.emitters[intersection.emitter_index];
      if (emitter.is_local()) {
        auto w_o = normalize(next->intersection.pos - intersection.pos);
        emitter_evaluate_out_local(emitter, spect, intersection.tex, intersection.nrm, w_o, pdf_area, pdf_dir, pdf_dir_out, scene);
        pdf_area = pdf_solid_angle_to_area(pdf_dir, *next);
      } else if (emitter.is_distant()) {
        auto w_o = normalize(intersection.pos - next->intersection.pos);
        emitter_evaluate_out_dist(emitter, spect, w_o, pdf_area, pdf_dir, pdf_dir_out, scene);
        if (next->is_surface_interaction()) {
          pdf_area *= fabsf(dot(scene.triangles[next->intersection.triangle_index].geo_n, w_o));
        }
      }
    } else if (scene.environment_emitters.count > 0) {
      auto w_o = normalize(intersection.pos - next->intersection.pos);
      for (uint32_t ie = 0; ie < scene.environment_emitters.count; ++ie) {
        const auto& emitter = scene.emitters[scene.environment_emitters.emitters[ie]];
        float local_pdf_area = 0.0f;
        emitter_evaluate_out_dist(emitter, spect, w_o, local_pdf_area, pdf_dir, pdf_dir_out, scene);
        pdf_area += local_pdf_area;
      }
      float w_o_dot_n = next->is_surface_interaction() ? fabsf(dot(scene.triangles[next->intersection.triangle_index].geo_n, w_o)) : 1.0f;
      pdf_area = w_o_dot_n * pdf_area / float(scene.environment_emitters.count);
    }

    return pdf_area;
  }

  float pdf_to_light_in(SpectralQuery spect, const PathVertex* next, const Scene& scene) const {
    ETX_ASSERT(is_emitter());

    float result = 0.0f;
    if (is_specific_emitter()) {
      const auto& emitter = scene.emitters[intersection.emitter_index];
      float pdf_discrete = emitter_discrete_pdf(emitter, scene.emitters_distribution);
      result =
        pdf_discrete * (emitter.is_local() ? emitter_pdf_area_local(emitter, scene) : emitter_pdf_in_dist(emitter, normalize(intersection.pos - next->intersection.pos), scene));
    } else if (scene.environment_emitters.count > 0) {
      for (uint32_t ie = 0; ie < scene.environment_emitters.count; ++ie) {
        const auto& emitter = scene.emitters[scene.environment_emitters.emitters[ie]];
        float pdf_discrete = emitter_discrete_pdf(emitter, scene.emitters_distribution);
        result += pdf_discrete * emitter_pdf_in_dist(emitter, normalize(intersection.pos - next->intersection.pos), scene);
      }
      result = result / float(scene.environment_emitters.count);
    }
    return result;
  }

  float pdf_solid_angle_to_area(float pdf_dir, const PathVertex& to_vertex) const {
    if ((pdf_dir == 0.0f) || to_vertex.is_environment_emitter()) {
      return pdf_dir;
    }

    auto w_o = to_vertex.intersection.pos - intersection.pos;
    float d_squared = dot(w_o, w_o);
    if (d_squared == 0.0f) {
      return 0.0f;
    }

    float inv_d_squared = 1.0f / d_squared;
    w_o *= sqrtf(inv_d_squared);

    float cos_t = (to_vertex.is_surface_interaction() ? fabsf(dot(w_o, to_vertex.intersection.nrm)) : 1.0f);

    float result = cos_t * pdf_dir * inv_d_squared;
    ETX_VALIDATE(result);
    return result;
  }

  SpectralResponse bsdf_in_direction(SpectralQuery spect, PathSource mode, const float3& w_o, const Scene& scene, Sampler& smp) const {
    ETX_ASSERT(is_surface_interaction() || is_medium_interaction());

    if (is_surface_interaction()) {
      const auto& tri = scene.triangles[intersection.triangle_index];
      ETX_ASSERT((is_subsurface == false) || (is_subsurface && (intersection.material_index == 0)));
      const auto& mat = scene.materials[intersection.material_index];
      BSDFEval eval = bsdf::evaluate({spect, medium_index, mode, intersection, intersection.w_i}, w_o, mat, scene, smp);
      ETX_VALIDATE(eval.bsdf);
      if (mode == PathSource::Light) {
        eval.bsdf *= fix_shading_normal(tri.geo_n, intersection.nrm, intersection.w_i, w_o);
        ETX_VALIDATE(eval.bsdf);
      }
      return eval.bsdf;
    }

    if (is_medium_interaction()) {
      return {spect, scene.mediums[medium_index].phase_function(spect, intersection.pos, intersection.w_i, w_o)};
    }

    ETX_FAIL("Invalid path vertex");
    return {spect, 0.0f};
  }
};

struct PathData {
  std::vector<PathVertex> emitter_path;

  struct {
    float pdf_ratio = 0.0f;
    uint32_t delta = false;
  } history[3] = {};
  float camera_mis_value = 0.0f;
  uint32_t camera_path_size = 0u;

  PathData() = default;
  PathData(const PathData&) = delete;
  PathData& operator=(const PathData&) = delete;
};

}  // namespace etx
