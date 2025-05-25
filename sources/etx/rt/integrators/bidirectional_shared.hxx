#pragma once

namespace etx {

struct PathVertex {
  enum class Class : uint16_t {
    Invalid,
    Camera,
    Emitter,
    Surface,
    SceneMedium,
    RuntimeMedium,
  };

  enum class SubsurfaceClass : uint16_t {
    None,
    Enter,
    Exit,
  };

  Intersection intersection = {};
  SpectralResponse throughput = {};

  struct {
    float next = 0.0f;
    float forward = 0.0f;
    float backward = 0.0f;
    float accumulated = 0.0f;
  } pdf;

  Class cls = Class::Invalid;
  uint32_t scene_medium_index = kInvalidIndex;
  float medium_anisotropy = 0.0f;
  bool delta_connection = false;
  bool delta_emitter = false;
  bool connectible = true;

  PathVertex() = default;

  PathVertex(Class c, const Intersection& i)
    : intersection(i)
    , cls(c) {
  }

  PathVertex(const float3& medium_sample_pos, const float3& a_w_i, const uint32_t medium_index, const float anisotropy)
    : cls(medium_index == kInvalidIndex ? Class::RuntimeMedium : Class::SceneMedium)
    , scene_medium_index(medium_index)
    , medium_anisotropy(anisotropy) {
    intersection.pos = medium_sample_pos;
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
    return ((cls == Class::SceneMedium) && (scene_medium_index != kInvalidIndex)) || (cls == Class::RuntimeMedium);
  }

  static bool safe_normalize(const float3& a, const float3& b, float3& n) {
    n = a - b;
    float len = dot(n, n);
    if (len == 0.0f)
      return false;

    n *= 1.0f / sqrtf(len);
    return true;
  }

  static float pdf_area(SpectralQuery spect, PathSource path_source, const PathVertex& prev, const PathVertex& curr, const PathVertex& next, const Scene& scene, Sampler& smp) {
    ETX_CRITICAL(curr.is_surface_interaction() || curr.is_medium_interaction());

    float3 w_i = {};
    float3 w_o = {};
    if (safe_normalize(curr.intersection.pos, prev.intersection.pos, w_i) == false)
      return 0.0f;

    if (safe_normalize(next.intersection.pos, curr.intersection.pos, w_o) == false)
      return 0.0f;

    float eval_pdf = 0.0f;
    if (curr.is_surface_interaction()) {
      const auto& mat = scene.materials[curr.intersection.material_index];
      eval_pdf = bsdf::pdf({spect, kInvalidIndex, path_source, curr.intersection, w_i}, w_o, mat, scene, smp);
      ETX_VALIDATE(eval_pdf);
    } else if (curr.is_medium_interaction()) {
      eval_pdf = medium::phase_function(w_i, w_o, curr.medium_anisotropy);
      ETX_VALIDATE(eval_pdf);
    }

    return convert_solid_angle_pdf_to_area(eval_pdf, curr, next);
  }

  static float pdf_from_emitter(SpectralQuery spect, const PathVertex& emitter_vertex, const PathVertex& target_vertex, const Scene& scene) {
    ETX_ASSERT(emitter_vertex.is_emitter());

    float pdf_area = 0.0f;

    if (emitter_vertex.is_specific_emitter()) {
      const auto& emitter = scene.emitters[emitter_vertex.intersection.emitter_index];
      if (emitter.is_local()) {
        float pdf_dir_out = 0.0f;
        float pdf_dir = 0.0f;
        auto w_o = normalize(target_vertex.intersection.pos - emitter_vertex.intersection.pos);
        emitter_evaluate_out_local(emitter, spect, emitter_vertex.intersection.tex, emitter_vertex.intersection.nrm, w_o, pdf_area, pdf_dir, pdf_dir_out, scene);
        pdf_area = convert_solid_angle_pdf_to_area(pdf_dir, emitter_vertex, target_vertex);
      } else if (emitter.is_distant()) {
        float pdf_dir = 0.0f;
        auto w_o = normalize(emitter_vertex.intersection.pos - target_vertex.intersection.pos);
        emitter_evaluate_out_dist(emitter, spect, w_o, pdf_area, pdf_dir, scene);
        if (target_vertex.is_surface_interaction()) {
          pdf_area *= fabsf(dot(scene.triangles[target_vertex.intersection.triangle_index].geo_n, w_o));
        }
      }

      return pdf_area;
    }

    if (scene.environment_emitters.count == 0)
      return 0.0f;

    auto w_o = normalize(emitter_vertex.intersection.pos - target_vertex.intersection.pos);
    for (uint32_t ie = 0; ie < scene.environment_emitters.count; ++ie) {
      const auto& emitter = scene.emitters[scene.environment_emitters.emitters[ie]];
      float local_pdf_dir = 0.0f;
      float local_pdf_area = 0.0f;
      emitter_evaluate_out_dist(emitter, spect, w_o, local_pdf_area, local_pdf_dir, scene);
      pdf_area += local_pdf_area;
    }
    float w_o_dot_n = target_vertex.is_surface_interaction() ? fabsf(dot(scene.triangles[target_vertex.intersection.triangle_index].geo_n, w_o)) : 1.0f;
    pdf_area = w_o_dot_n * pdf_area / float(scene.environment_emitters.count);
    return pdf_area;
  }

  static float pdf_to_emitter(SpectralQuery spect, const PathVertex& interaction, const PathVertex& emitter_vertex, const Scene& scene) {
    ETX_ASSERT(emitter_vertex.is_emitter());

    if (emitter_vertex.is_specific_emitter()) {
      const auto& emitter = scene.emitters[emitter_vertex.intersection.emitter_index];
      float pdf_discrete = emitter_discrete_pdf(emitter, scene.emitters_distribution);
      return pdf_discrete * (emitter.is_local()                                                                                                     //
                                ? emitter_pdf_area_local(emitter, scene)                                                                            //
                                : emitter_pdf_in_dist(emitter, normalize(emitter_vertex.intersection.pos - interaction.intersection.pos), scene));  //
    }

    if (scene.environment_emitters.count == 0)
      return 0.0f;

    float result = 0.0f;
    for (uint32_t ie = 0; ie < scene.environment_emitters.count; ++ie) {
      const auto& emitter = scene.emitters[scene.environment_emitters.emitters[ie]];
      float pdf_discrete = emitter_discrete_pdf(emitter, scene.emitters_distribution);
      result += pdf_discrete * emitter_pdf_in_dist(emitter, normalize(emitter_vertex.intersection.pos - interaction.intersection.pos), scene);
    }

    return result / float(scene.environment_emitters.count);
  }

  static float convert_solid_angle_pdf_to_area(float pdf_dir, const PathVertex& from_vertex, const PathVertex& to_vertex) {
    if ((pdf_dir == 0.0f) || to_vertex.is_environment_emitter()) {
      return pdf_dir;
    }

    auto w_o = to_vertex.intersection.pos - from_vertex.intersection.pos;
    float d_squared = dot(w_o, w_o);
    if (d_squared < kRayEpsilon) {
      return 0.0f;
    }

    float inv_d_squared = 1.0f / d_squared;
    w_o *= sqrtf(inv_d_squared);

    float cos_t = (to_vertex.is_surface_interaction() ? fabsf(dot(w_o, to_vertex.intersection.nrm)) : 1.0f);

    float result = cos_t * pdf_dir * inv_d_squared;
    ETX_VALIDATE(result);
    return result;
  }

  auto bsdf_in_direction(SpectralQuery spect, PathSource mode, const float3& w_o, const Scene& scene, Sampler& smp) const {
    ETX_ASSERT(is_surface_interaction() || is_medium_interaction());

    struct Result {
      SpectralResponse bsdf = {};
      float pdf = {};
    };

    if (is_surface_interaction()) {
      const auto& mat = scene.materials[intersection.material_index];
      BSDFEval eval = bsdf::evaluate({spect, kInvalidIndex, mode, intersection, intersection.w_i}, w_o, mat, scene, smp);
      ETX_VALIDATE(eval.bsdf);
      if (mode == PathSource::Light) {
        const auto& tri = scene.triangles[intersection.triangle_index];
        eval.bsdf *= fix_shading_normal(tri.geo_n, intersection.nrm, intersection.w_i, w_o);
        ETX_VALIDATE(eval.bsdf);
      }

      return Result{eval.bsdf, eval.pdf};
    }

    if (is_medium_interaction()) {
      float eval_pdf = medium::phase_function(intersection.w_i, w_o, medium_anisotropy);
      return Result{{spect, eval_pdf}, eval_pdf};
    }

    ETX_FAIL("Invalid path vertex");
    return Result{{spect, 0.0f}, 0.0f};
  }
};

struct PathData {
  std::vector<PathVertex> emitter_path;
  std::vector<PathVertex> camera_path;

  struct History {
    float pdf_forward = 0.0f;
    float pdf_ratio = 0.0f;
    float mis_accumulated = 0.0f;
    uint32_t delta = false;
  };

  History camera_history[3] = {};
  History emitter_history[3] = {};
  uint32_t camera_path_size = 0u;
  uint32_t emitter_path_size = 0u;

  PathData() = default;
  PathData(const PathData&) = delete;
  PathData& operator=(const PathData&) = delete;

  uint32_t camera_path_length() const {
    return camera_path_size >= 2 ? camera_path_size - 2u : 0;
  }

  uint32_t emitter_path_length() const {
    return emitter_path_size >= 2 ? emitter_path_size - 2u : 0;
  }
};

}  // namespace etx
