#pragma once

namespace etx {

struct PathVertex : public Intersection {
  enum class Class : uint32_t {
    Invalid,
    Camera,
    Emitter,
    Surface,
    Medium,
  };

  SpectralResponse throughput = {};
  struct {
    float forward = 0.0f;
    float backward = 0.0f;
  } pdf;

  Class cls = Class::Invalid;
  uint32_t emitter_index = kInvalidIndex;
  uint32_t medium_index = kInvalidIndex;
  bool delta_connection = false;
  bool delta_emitter = false;

  PathVertex() = default;

  PathVertex(Class c, const Intersection& i)
    : Intersection(i)
    , cls(c) {
  }

  PathVertex(const Medium::Sample& i, const float3& a_w_i)
    : cls(Class::Medium) {
    pos = i.pos;
    w_i = a_w_i;
  }

  PathVertex(Class c)
    : cls(c) {
  }

  bool is_specific_emitter() const {
    return (emitter_index != kInvalidIndex);
  }

  bool is_environment_emitter() const {
    return (cls == Class::Emitter) && (triangle_index == kInvalidIndex);
  }

  bool is_emitter() const {
    return is_specific_emitter() || is_environment_emitter();
  }

  bool is_surface_interaction() const {
    return (triangle_index != kInvalidIndex);
  }

  bool is_medium_interaction() const {
    return (cls == Class::Medium) && (medium_index != kInvalidIndex);
  }

  float pdf_area(SpectralQuery spect, PathSource mode, const PathVertex* prev, const PathVertex* next, const Scene& scene, Sampler& smp) const {
    if (cls == Class::Emitter) {
      return pdf_to_light_out(spect, next, scene);
    }

    ETX_ASSERT(prev != nullptr);
    ETX_ASSERT(next != nullptr);
    ETX_ASSERT(is_surface_interaction() || is_medium_interaction());

    auto w_i = pos - prev->pos;
    {
      float w_i_len = length(w_i);
      if (w_i_len == 0.0f) {
        return 0.0f;
      }
      w_i *= 1.0f / w_i_len;
    }

    auto w_o = next->pos - pos;
    {
      float w_o_len = length(w_o);
      if (w_o_len == 0.0f) {
        return 0.0f;
      }
      w_o *= 1.0f / w_o_len;
    }

    float eval_pdf = 0.0f;
    if (is_surface_interaction()) {
      const auto& mat = scene.materials[material_index];
      eval_pdf = bsdf::pdf({spect, medium_index, mode, *this, w_i}, w_o, mat, scene, smp);
    } else if (is_medium_interaction()) {
      eval_pdf = scene.mediums[medium_index].phase_function(spect, pos, w_i, w_o);
    } else {
      ETX_FAIL("Invalid vertex class");
    }
    ETX_VALIDATE(eval_pdf);

    if (next->is_environment_emitter()) {
      return eval_pdf;
    }

    return pdf_solid_angle_to_area(eval_pdf, *next);
  }

  float pdf_to_light_out(SpectralQuery spect, const PathVertex* next, const Scene& scene) const {
    ETX_ASSERT(next != nullptr);
    ETX_ASSERT(is_emitter());

    float pdf_area = 0.0f;
    float pdf_dir = 0.0f;
    float pdf_dir_out = 0.0f;

    if (is_specific_emitter()) {
      const auto& emitter = scene.emitters[emitter_index];
      if (emitter.is_local()) {
        auto w_o = normalize(next->pos - pos);
        emitter_evaluate_out_local(emitter, spect, tex, nrm, w_o, pdf_area, pdf_dir, pdf_dir_out, scene);
        pdf_area = pdf_solid_angle_to_area(pdf_dir, *next);
      } else if (emitter.is_distant()) {
        auto w_o = normalize(pos - next->pos);
        emitter_evaluate_out_dist(emitter, spect, w_o, pdf_area, pdf_dir, pdf_dir_out, scene);
        if (next->is_surface_interaction()) {
          pdf_area *= fabsf(dot(scene.triangles[next->triangle_index].geo_n, w_o));
        }
      }
    } else if (scene.environment_emitters.count > 0) {
      auto w_o = normalize(pos - next->pos);
      float w_o_dot_n = next->is_surface_interaction() ? fabsf(dot(scene.triangles[next->triangle_index].geo_n, w_o)) : 1.0f;
      for (uint32_t ie = 0; ie < scene.environment_emitters.count; ++ie) {
        const auto& emitter = scene.emitters[scene.environment_emitters.emitters[ie]];
        float local_pdf_area = 0.0f;
        emitter_evaluate_out_dist(emitter, spect, w_o, local_pdf_area, pdf_dir, pdf_dir_out, scene);
        pdf_area += local_pdf_area * w_o_dot_n;
      }
      pdf_area = pdf_area / float(scene.environment_emitters.count);
    }

    return pdf_area;
  }

  float pdf_to_light_in(SpectralQuery spect, const PathVertex* next, const Scene& scene) const {
    ETX_ASSERT(is_emitter());

    float result = 0.0f;
    if (is_specific_emitter()) {
      const auto& emitter = scene.emitters[emitter_index];
      float pdf_discrete = emitter_discrete_pdf(emitter, scene.emitters_distribution);
      result = pdf_discrete * (emitter.is_local() ? emitter_pdf_area_local(emitter, scene) : emitter_pdf_in_dist(emitter, normalize(pos - next->pos), scene));
    } else if (scene.environment_emitters.count > 0) {
      for (uint32_t ie = 0; ie < scene.environment_emitters.count; ++ie) {
        const auto& emitter = scene.emitters[scene.environment_emitters.emitters[ie]];
        float pdf_discrete = emitter_discrete_pdf(emitter, scene.emitters_distribution);
        result += pdf_discrete * emitter_pdf_in_dist(emitter, normalize(pos - next->pos), scene);
      }
      result = result / float(scene.environment_emitters.count);
    }
    return result;
  }

  float pdf_solid_angle_to_area(float pdf_dir, const PathVertex& to_vertex) const {
    if ((pdf_dir == 0.0f) || to_vertex.is_environment_emitter()) {
      return pdf_dir;
    }

    auto w_o = to_vertex.pos - pos;

    float d_squared = dot(w_o, w_o);
    if (d_squared == 0.0f) {
      return 0.0f;
    }

    float inv_d_squared = 1.0f / d_squared;
    w_o *= std::sqrt(inv_d_squared);

    float cos_t = (to_vertex.is_surface_interaction() ? fabsf(dot(w_o, to_vertex.nrm)) : 1.0f);

    float result = cos_t * pdf_dir * inv_d_squared;
    ETX_VALIDATE(result);
    return result;
  }

  SpectralResponse bsdf_in_direction(SpectralQuery spect, PathSource mode, const float3& w_o, const Scene& scene, Sampler& smp) const {
    ETX_ASSERT(is_surface_interaction() || is_medium_interaction());

    if (is_surface_interaction()) {
      const auto& tri = scene.triangles[triangle_index];
      const auto& mat = scene.materials[material_index];

      BSDFEval eval = bsdf::evaluate({spect, medium_index, mode, *this, w_i}, w_o, mat, scene, smp);
      ETX_VALIDATE(eval.bsdf);

      if (mode == PathSource::Light) {
        eval.bsdf *= fix_shading_normal(tri.geo_n, nrm, w_i, w_o);
        ETX_VALIDATE(eval.bsdf);
      }

      ETX_VALIDATE(eval.bsdf);
      return eval.bsdf;
    }

    if (is_medium_interaction()) {
      return {spect, scene.mediums[medium_index].phase_function(spect, pos, w_i, w_o)};
    }

    ETX_FAIL("Invalid vertex class");
    return {spect, 0.0f};
  }
};

struct PathData {
  std::vector<PathVertex> camera_path;
  std::vector<PathVertex> emitter_path;
  float camera_mis = 0.0f;

  PathData() = default;
  PathData(const PathData&) = delete;
  PathData& operator=(const PathData&) = delete;
};

}  // namespace etx
