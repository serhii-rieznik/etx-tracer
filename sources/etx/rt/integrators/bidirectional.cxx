#include <etx/core/core.hxx>
#include <etx/render/host/film.hxx>
#include <etx/rt/integrators/bidirectional.hxx>
#include <atomic>

#include <etx/rt/shared/path_tracing_shared.hxx>

namespace etx {

namespace {

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

  struct {
    float next = 0.0f;
    float from_prev = 0.0f;
    float from_next = 0.0f;
    float accumulated = 0.0f;
  } pdf;

  Class cls = Class::Invalid;
  Medium::Instance medium = {};
  bool delta_emitter = false;
  bool connectible = true;

  PathVertex() = default;

  PathVertex(Class c, const Intersection& i)
    : intersection(i)
    , cls(c) {
  }

  PathVertex(const float3& medium_sample_pos, const float3& a_w_i, const Medium::Instance m)
    : cls(Class::Medium)
    , medium(m) {
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
    return (cls == Class::Medium) && medium.valid();
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
      eval_pdf = medium::phase_function(w_i, w_o, curr.medium.anisotropy);
      ETX_VALIDATE(eval_pdf);
    }

    return convert_solid_angle_pdf_to_area(eval_pdf, curr, next);
  }

  static float pdf_from_emitter(SpectralQuery spect, const PathVertex& emitter_vertex, const PathVertex& target_vertex, const Scene& scene) {
    ETX_ASSERT(emitter_vertex.is_emitter());
    ETX_CRITICAL(emitter_vertex.is_specific_emitter());

    float pdf_area = 0.0f;
    const auto& emitter = scene.emitters[emitter_vertex.intersection.emitter_index];
    if (emitter.is_local()) {
      float pdf_dir = 0.0f;
      float pdf_dir_out = 0.0f;
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

  static float pdf_to_emitter(SpectralQuery spect, const PathVertex& interaction, const PathVertex& emitter_vertex, const Scene& scene) {
    ETX_ASSERT(emitter_vertex.is_emitter());
    ETX_CRITICAL(emitter_vertex.is_specific_emitter());

    const auto& emitter = scene.emitters[emitter_vertex.intersection.emitter_index];
    float pdf_discrete = emitter_discrete_pdf(emitter, scene.emitters_distribution);
    return pdf_discrete * (emitter.is_local()                                                                                                     //
                              ? emitter_pdf_area_local(emitter, scene)                                                                            //
                              : emitter_pdf_in_dist(emitter, normalize(emitter_vertex.intersection.pos - interaction.intersection.pos), scene));  //
  }

  static float2 pdf_for_environment_emitter(SpectralQuery spect, const PathVertex& emitter_vertex, const PathVertex& target_vertex, const Scene& scene) {
    ETX_ASSERT(emitter_vertex.is_emitter());
    ETX_CRITICAL(emitter_vertex.is_environment_emitter());

    if (scene.environment_emitters.count == 0)
      return {};

    float pdf_dir = 0.0f;
    const auto w_o = -emitter_vertex.intersection.w_i;
    for (uint32_t ie = 0; ie < scene.environment_emitters.count; ++ie) {
      const auto& emitter = scene.emitters[scene.environment_emitters.emitters[ie]];
      const float pdf_discrete = emitter_discrete_pdf(emitter, scene.emitters_distribution);
      const float pdf_emitter_dir = emitter_pdf_in_dist(emitter, w_o, scene);
      pdf_dir += pdf_discrete * pdf_emitter_dir;
    }

    float w_o_dot_n = target_vertex.is_surface_interaction() ? fabsf(dot(scene.triangles[target_vertex.intersection.triangle_index].geo_n, w_o)) : 1.0f;
    float pdf_area = w_o_dot_n / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
    pdf_dir = pdf_dir / float(scene.environment_emitters.count);

    return {pdf_area, pdf_dir};
  }

  static float convert_solid_angle_pdf_to_area(float pdf_dir, const PathVertex& from_vertex, const PathVertex& to_vertex) {
    if ((pdf_dir == 0.0f) || to_vertex.is_environment_emitter()) {
      return pdf_dir;
    }

    auto w_o = to_vertex.intersection.pos - from_vertex.intersection.pos;
    float d_squared = dot(w_o, w_o);
    if (d_squared <= kEpsilon) {
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
      float eval_pdf = medium::phase_function(intersection.w_i, w_o, medium.anisotropy);
      return Result{{spect, eval_pdf}, eval_pdf};
    }

    ETX_FAIL("Invalid path vertex");
    return Result{{spect, 0.0f}, 0.0f};
  }
};

struct PathData {
  std::vector<PathVertex> emitter_path;

  struct History {
    float pdf_forward = 0.0f;
    float pdf_ratio = 0.0f;
    float mis_accumulated = 0.0f;
    float pad = 0.0f;
  };

  History camera_history[3] = {};
  History emitter_history[3] = {};
  uint32_t camera_path_size = 0u;
  uint32_t emitter_path_size = 0u;

  PathData() = default;
  PathData(const PathData&) = delete;
  PathData& operator=(const PathData&) = delete;

  uint32_t camera_path_length() const {
    return camera_path_size - 1u;
  }

  uint32_t emitter_path_length() const {
    return emitter_path_size - 1u;
  }
};

inline float map0(float t) {
  return t == 0.0f ? 1.0f : t;
}

inline float safe_div(float a, float b) {
  float result = map0(a) / map0(b);
  ETX_VALIDATE(result);
  return result;
}

inline float strict_div(float a, float b) {
  if ((a <= 0.0f) || (b <= 0.0f))
    return 0.0f;

  float result = a / b;
  ETX_VALIDATE(result);
  return result;
}

inline float balance_heuristic(float a, float b, float c) {
  float denom = a + b + c;
  return denom == 0.0f ? 0.0f : a / denom;
}

}  // namespace

struct CPUBidirectionalImpl : public Task {
  Raytracing& rt;
  std::vector<PathData> per_thread_path_data;
  std::atomic<Integrator::State>* state = {};
  TimeMeasure iteration_time = {};
  Handle current_task = {};
  Integrator::Status status = {};

  enum class Mode : uint32_t {
    PathTracing,
    LightTracing,
    BDPTFast,
    BDPTFull,

    Count,
  };

  Mode mode = Mode::BDPTFull;

  bool enable_direct_hit = true;
  bool enable_connect_to_light = true;
  bool enable_connect_to_camera = true;
  bool enable_connect_vertices = true;
  bool enable_mis = true;
  bool enable_blue_noise = true;

  struct GBuffer {
    SpectralResponse albedo = {};
    float3 normal = {0.0f, 0.0f, 1.0f};
    bool recorded = false;
  };

  enum class InteractionResult : uint32_t {
    Continue,
    Break,
    NextIteration,
    SampleSubsurface,
  };

  struct Payload {
    SpectralQuery spect = {};
    SpectralResponse result = {};
    SpectralResponse throughput = {};
    float eta = 1.0f;
    float pdf_dir = 0.0f;
    uint32_t medium_index = kInvalidIndex;
    PathSource mode = PathSource::Undefined;
    uint2 pixel = {};
    uint32_t iteration = 0;
    bool use_blue_noise = false;
  };

  CPUBidirectionalImpl(Raytracing& r, std::atomic<Integrator::State>* st)
    : rt(r)
    , per_thread_path_data(rt.scheduler().max_thread_count())
    , state(st) {
  }

  void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) {
    auto& path_data = per_thread_path_data[thread_id];
    auto& film = rt.film();
    auto& scene = rt.scene();

    for (uint32_t i = begin; (state->load() != Integrator::State::Stopped) && (i < end); ++i) {
      uint2 pixel = {};
      if (film.active_pixel(i, pixel) == false)
        continue;

      // Create separate samplers for camera and light paths (like VCM)
      // Both use same index like VCM, but are used sequentially
      auto camera_smp = Sampler(i, status.current_iteration);
      auto light_smp = Sampler(i, status.current_iteration);

      // Generate spectrum from light path first (like VCM)
      SpectralQuery spect = SpectralQuery::sample();
      if (mode != Mode::PathTracing) {
        // Light path generates the spectrum
        spect = scene.spectral() ? SpectralQuery::spectral_sample(light_smp.next()) : SpectralQuery::sample();
        build_emitter_path(light_smp, spect, path_data);
        // Camera path uses the light path spectrum but still consumes a sample (like VCM)
        if (scene.spectral()) {
          camera_smp.next();  // Consume sample to match VCM pattern
        }
      } else {
        // PathTracing mode: camera generates its own spectrum
        spect = scene.spectral() ? SpectralQuery::spectral_sample(camera_smp.next()) : SpectralQuery::sample();
      }

      GBuffer gbuffer = {};
      SpectralResponse result = {spect, 0.0f};

      if (mode != Mode::LightTracing) {
        float2 uv = film.sample(rt.scene(), status.current_iteration == 0u ? PixelFilter::empty() : rt.scene().pixel_sampler, pixel, camera_smp.next_2d());
        result = build_camera_path(camera_smp, spect, uv, path_data, gbuffer, pixel, status.current_iteration);
      }

      auto xyz = (result / spect.sampling_pdf()).to_rgb();
      auto albedo = (gbuffer.albedo / spect.sampling_pdf()).to_rgb();
      film.accumulate(pixel, {{xyz, Film::CameraImage}, {gbuffer.normal, Film::Normals}, {albedo, Film::Albedo}});
    }
  }

  void completed() {
    status.last_iteration_time = iteration_time.measure();
    status.total_time += status.last_iteration_time;
    status.completed_iterations += 1u;
    status.current_iteration += 1u;
    iteration_time = {};
  }

  bool running() const {
    return state->load() != Integrator::State::Stopped;
  }

  void update_distant_emitter_path_pdfs(PathVertex& curr, PathVertex& prev, const EmitterSample& em) const {
    const auto& scene = rt.scene();

    const float pdf = emitter_pdf_in_dist(scene.emitters[em.emitter_index], -em.direction, scene);
    const float p_discrete = emitter_discrete_pdf(scene.emitters[em.emitter_index], scene.emitters_distribution);

    prev.pdf.from_prev = p_discrete * pdf;
    ETX_VALIDATE(prev.pdf.from_prev);

    if (mode == Mode::BDPTFast) {
      prev.pdf.accumulated = safe_div(1.0f, prev.pdf.from_prev);
    }

    curr.pdf.from_prev = em.pdf_area;
    if (curr.is_surface_interaction()) {
      const auto& tri = scene.triangles[curr.intersection.triangle_index];
      curr.pdf.from_prev *= fabsf(dot(em.direction, tri.geo_n));
      ETX_VALIDATE(curr.pdf.from_prev);
    }
  }

  SpectralResponse connect_camera_to_light_path(const PathVertex& z_i, const PathVertex& z_prev, Sampler& smp, SpectralQuery spect, PathData& path_data) const {
    const auto& scene = rt.scene();

    SpectralResponse result = {spect, 0.0f};
    if ((mode != Mode::BDPTFull) || (enable_connect_vertices == false) || (z_i.connectible == false)) {
      return result;
    }

    const uint64_t camera_path_length = path_data.camera_path_length();
    for (uint64_t light_s = 1, light_s_e = path_data.emitter_path.size(); running() && (light_s < light_s_e); ++light_s) {
      const uint64_t target_path_length = camera_path_length + light_s + 1;
      if (target_path_length < scene.min_path_length)
        continue;

      if (target_path_length > scene.max_path_length)
        break;

      const auto& y_i = path_data.emitter_path[light_s];
      if (y_i.connectible == false) {
        continue;
      }

      auto dw = z_i.intersection.pos - y_i.intersection.pos;
      float dwl = dot(dw, dw);
      if (dwl <= kInvMaxHalf) {
        continue;
      }
      dw *= 1.0f / std::sqrt(dwl);

      float g_term = 1.0f / dwl;

      // G term = abs(cos(dw, y_i.nrm) * cos(dw, z_i.nrm)) / dwl;
      // cosines already accounted in "bsdf", 1.0 / dwl multiplied below
      auto bsdf_y = y_i.bsdf_in_direction(spect, PathSource::Light, dw, rt.scene(), smp).bsdf;
      ETX_VALIDATE(bsdf_y);

      auto bsdf_z = z_i.bsdf_in_direction(spect, PathSource::Camera, -dw, rt.scene(), smp).bsdf;
      ETX_VALIDATE(bsdf_z);

      SpectralResponse connect_result = y_i.throughput * bsdf_y * bsdf_z;
      ETX_VALIDATE(connect_result);

      if (connect_result.is_zero())
        continue;

      SpectralResponse tr = local_transmittance(spect, smp, y_i, z_i.intersection.pos);
      ETX_VALIDATE(connect_result);

      float weight = mis_weight_camera_to_light_path(z_i, z_prev, path_data, spect, light_s, smp);
      ETX_VALIDATE(weight);

      result += connect_result * tr * (weight * g_term);
      ETX_VALIDATE(result);
    }

    return result * z_i.throughput;
  }

  void update_mis(const EmitterSample& emitter_sample, const bool first_interaction, Payload& payload, Sampler& smp, PathData& path_data, PathVertex& curr,
    PathVertex& prev) const {
    if (payload.mode == PathSource::Light) {
      if (first_interaction && emitter_sample.is_distant) {
        update_distant_emitter_path_pdfs(curr, prev, emitter_sample);
      }
      precompute_light_mis(curr, prev, path_data);
      path_data.emitter_path.back() = prev;
      path_data.emitter_path.emplace_back(curr);
    } else if (payload.mode == PathSource::Camera) {
      precompute_camera_mis(curr, prev, path_data);
    }
  }

  void connect(Payload& payload, Sampler& smp, const float3& smp_fixed, PathData& path_data, PathVertex& curr, PathVertex& prev) const {
    if (payload.mode == PathSource::Light) {
      CameraSample camera_sample = {};
      auto splat = connect_light_to_camera(smp, path_data, curr, prev, payload.spect, camera_sample);
      rt.film().atomic_add(Film::LightIteration, splat.to_rgb(), camera_sample.uv);
    } else if (payload.mode == PathSource::Camera) {
      smp.push_fixed(smp_fixed.x, smp_fixed.y, smp_fixed.z);
      payload.result += connect_camera_to_light(curr, prev, smp, path_data, payload.spect);
      payload.result += direct_hit_area_emitter(curr, prev, path_data, payload.spect, smp, false);
      smp.pop_fixed();  // Pop fixed samples before vertex connections to sync with VCM
      payload.result += connect_camera_to_light_path(curr, prev, smp, payload.spect, path_data);
    }
  }

  void handle_medium(const EmitterSample& emitter_sample, const bool first_interaction, const bool explicit_connections, const float3& medium_sample_pos,
    const Medium::Instance& medium_instance, Payload& payload, Ray& ray, Sampler& smp, PathData& path_data, PathVertex& curr, PathVertex& prev) const {
    const auto& scene = rt.scene();

    ETX_ASSERT(medium_instance.valid());

    float2 rnd_bsdf = smp.next_2d();
    float2 rnd_em_sample = smp.next_2d();
    float2 rnd_support = smp.next_2d();
    if (enable_blue_noise && (payload.mode == PathSource::Camera) && first_interaction && (payload.iteration < 256u)) {
      rnd_bsdf = sample_blue_noise(payload.pixel, rt.scene().samples, payload.iteration, 0);
      rnd_em_sample = sample_blue_noise(payload.pixel, rt.scene().samples, payload.iteration, 2);
      rnd_support = sample_blue_noise(payload.pixel, rt.scene().samples, payload.iteration, 4);
    }

    float3 w_o = medium::sample_phase_function(ray.d, medium_instance.anisotropy, rnd_bsdf);
    float pdf_fwd = medium::phase_function(ray.d, w_o, medium_instance.anisotropy);
    float pdf_bck = medium::phase_function(w_o, ray.d, medium_instance.anisotropy);

    path_data.camera_path_size += uint32_t(payload.mode == PathSource::Camera);
    path_data.emitter_path_size += uint32_t(payload.mode == PathSource::Light);

    curr = PathVertex{medium_sample_pos, ray.d, medium_instance};
    curr.connectible = true;  // phase function is non-delta
    curr.throughput = payload.throughput;
    curr.pdf.next = pdf_fwd;
    curr.pdf.from_prev = PathVertex::convert_solid_angle_pdf_to_area(payload.pdf_dir, prev, curr);
    prev.pdf.from_next = PathVertex::convert_solid_angle_pdf_to_area(pdf_bck, curr, prev);

    ray.o = medium_sample_pos;
    ray.d = w_o;
    ray.min_t = kRayEpsilon;
    ray.max_t = kMaxFloat;

    payload.pdf_dir = pdf_fwd;

    update_mis(emitter_sample, first_interaction, payload, smp, path_data, curr, prev);

    if (explicit_connections) {
      connect(payload, smp, {rnd_em_sample.x, rnd_em_sample.y, rnd_support.y}, path_data, curr, prev);
    }
  }

  InteractionResult handle_surface(const Intersection& a_intersection, const EmitterSample& emitter_sample, const bool first_interaction, Payload& payload, Ray& ray, Sampler& smp,
    PathData& path_data, PathVertex& curr, PathVertex& prev, GBuffer& gbuffer, bool subsurface_exit) const {
    const auto& scene = rt.scene();

    float2 rnd_bsdf = smp.next_2d();
    float2 rnd_em_sample = smp.next_2d();
    float2 rnd_support = smp.next_2d();
    if ((payload.mode == PathSource::Camera) && first_interaction && enable_blue_noise) {
      rnd_bsdf = sample_blue_noise(payload.pixel, rt.scene().samples, payload.iteration, 0);
      rnd_em_sample = sample_blue_noise(payload.pixel, rt.scene().samples, payload.iteration, 2);
      rnd_support = sample_blue_noise(payload.pixel, rt.scene().samples, payload.iteration, 4);
    }

    if (scene.materials[a_intersection.material_index].cls == Material::Class::Boundary) {
      const auto& m = scene.materials[a_intersection.material_index];
      payload.medium_index = (dot(a_intersection.nrm, ray.d) < 0.0f) ? m.int_medium : m.ext_medium;
      ray.o = shading_pos(scene.vertices, scene.triangles[a_intersection.triangle_index], a_intersection.barycentric, ray.d);
      ray.min_t = kRayEpsilon;
      ray.max_t = kMaxFloat;
      return InteractionResult::Continue;
    }

    BSDFData bsdf_data = {payload.spect, payload.medium_index, payload.mode, a_intersection, a_intersection.w_i};

    if (gbuffer.recorded == false) {
      gbuffer.normal = a_intersection.nrm;
      gbuffer.albedo = bsdf::albedo(bsdf_data, scene.materials[a_intersection.material_index], scene, smp);
      gbuffer.recorded = true;
    }

    smp.push_fixed(rnd_bsdf.x, rnd_bsdf.y, rnd_support.x);
    auto bsdf_sample = bsdf::sample(bsdf_data, scene.materials[a_intersection.material_index], scene, smp);
    smp.pop_fixed();

    ETX_VALIDATE(bsdf_sample.weight);

    bool subsurface_path = (subsurface_exit == false) &&                                                                              //
                           (scene.materials[a_intersection.material_index].subsurface.cls != SubsurfaceMaterial::Class::Disabled) &&  //
                           (bsdf_sample.properties & BSDFSample::Reflection) && (bsdf_sample.properties & BSDFSample::Diffuse);

    uint32_t material_index = a_intersection.material_index;

    Medium::Instance medium_instance = {
      .index = (bsdf_sample.properties & BSDFSample::MediumChanged) ? bsdf_sample.medium_index : payload.medium_index,
    };

    if (subsurface_path) {
      const auto& sss_material = scene.materials[a_intersection.material_index];
      material_index = scene.subsurface_scatter_material;
      medium_instance.index = sss_material.int_medium;

      if (medium_instance.index == kInvalidIndex) {
        medium_instance = subsurface_to_medium_instance(material_index, payload, a_intersection);
      }

      const bool diffuse_transmission = sss_material.subsurface.path == SubsurfaceMaterial::Path::Diffuse;
      auto w_o = diffuse_transmission ? sample_cosine_distribution(smp.next_2d(), -a_intersection.nrm, 1.0f) : a_intersection.w_i;

      bsdf_sample.w_o = w_o;
      bsdf_sample.weight = {payload.spect, 1.0f};
      bsdf_sample.pdf = fabsf(dot(w_o, a_intersection.nrm)) / kPi;
      bsdf_sample.eta = 1.0f;
      bsdf_sample.medium_index = medium_instance.index;
      bsdf_sample.properties = BSDFSample::Transmission | BSDFSample::Diffuse | BSDFSample::MediumChanged;
    }

    path_data.camera_path_size += uint32_t(payload.mode == PathSource::Camera);
    path_data.emitter_path_size += uint32_t(payload.mode == PathSource::Light);

    curr = PathVertex{PathVertex::Class::Surface, a_intersection};
    curr.throughput = payload.throughput;
    curr.intersection.material_index = material_index;
    curr.medium = medium_instance;
    curr.pdf.next = bsdf_sample.pdf;
    curr.connectible = (bsdf_sample.properties & BSDFSample::Delta) == 0;  // non-delta events are connectible

    curr.pdf.from_prev = PathVertex::convert_solid_angle_pdf_to_area(payload.pdf_dir, prev, curr);
    ETX_VALIDATE(curr.pdf.from_prev);

    float rev_bsdf_pdf = bsdf::reverse_pdf(bsdf_data, bsdf_sample.w_o, scene.materials[material_index], scene, smp);
    prev.pdf.from_next = PathVertex::convert_solid_angle_pdf_to_area(rev_bsdf_pdf, curr, prev);
    ETX_VALIDATE(prev.pdf.from_next);

    payload.medium_index = medium_instance.index;

    bool terminate_path = false;
    if (bsdf_sample.valid()) {
      payload.eta *= (payload.mode == PathSource::Camera) ? bsdf_sample.eta : 1.0f;
      ETX_VALIDATE(payload.eta);

      payload.pdf_dir = bsdf_sample.pdf;
      ETX_VALIDATE(payload.pdf_dir);

      payload.throughput *= bsdf_sample.weight;
      ETX_VALIDATE(payload.throughput);

      const auto& tri = scene.triangles[a_intersection.triangle_index];

      ray.o = shading_pos(scene.vertices, tri, curr.intersection.barycentric, bsdf_sample.w_o);
      ray.d = bsdf_sample.w_o;
      ray.min_t = kRayEpsilon;
      ray.max_t = kMaxFloat;

      if (payload.mode == PathSource::Light) {
        payload.throughput *= fix_shading_normal(tri.geo_n, curr.intersection.nrm, curr.intersection.w_i, bsdf_sample.w_o);
        ETX_VALIDATE(payload.throughput);
      }
    } else {
      terminate_path = true;
    }

    update_mis(emitter_sample, first_interaction, payload, smp, path_data, curr, prev);
    connect(payload, smp, {rnd_em_sample.x, rnd_em_sample.y, rnd_support.y}, path_data, curr, prev);

    return terminate_path ? InteractionResult::Break : (subsurface_path ? InteractionResult::SampleSubsurface : InteractionResult::NextIteration);
  }

  enum class StepResult {
    Nothing = 0,
    SampledMedium,
    IntersectionFound,
    Continue,
    Break,
  };

  StepResult regular_step(const Ray& ray, Sampler& smp, Intersection& intersection, Medium::Sample& medium_sample, Payload& payload) const {
    const auto& scene = rt.scene();
    bool found_intersection = rt.trace(scene, ray, intersection, smp);

    if (payload.medium_index != kInvalidIndex) {
      const auto& m = scene.mediums[payload.medium_index];
      medium_sample = m.sample(payload.spect, payload.throughput, smp, ray.o, ray.d, found_intersection ? intersection.t : kMaxFloat);
      payload.throughput *= medium_sample.weight;
      ETX_VALIDATE(payload.throughput);
    }

    if (medium_sample.sampled_medium())
      return StepResult::SampledMedium;

    return found_intersection ? StepResult::IntersectionFound : StepResult::Nothing;
  }

  Medium::Instance subsurface_to_medium_instance(const uint32_t subsurface_material, const Payload& payload, const Intersection& intersection) const {
    const auto& scene = rt.scene();
    const auto& mat = scene.materials[subsurface_material];
    auto color = apply_image(payload.spect, mat.transmittance, intersection.tex, scene, nullptr);
    auto distances = mat.subsurface.scale * apply_image(payload.spect, mat.subsurface, intersection.tex, scene, nullptr);

    SpectralResponse extinction = {payload.spect};
    SpectralResponse scattering = {payload.spect};
    SpectralResponse albedo = {payload.spect};
    subsurface::remap(color.integrated, distances.integrated, albedo.integrated, extinction.integrated, scattering.integrated);
    subsurface::remap_channel(color.value, distances.value, albedo.value, extinction.value, scattering.value);

    return {
      .extinction = extinction,
      .index = kInvalidIndex,
    };
  }

  StepResult subsurface_step(const uint32_t subsurface_material, Ray& ray, Sampler& smp, Intersection& intersection, Payload& payload, PathData& path_data, PathVertex& curr,
    PathVertex& prev) const {
    const auto& scene = rt.scene();

    SpectralResponse extinction = {payload.spect};
    SpectralResponse scattering = {payload.spect};
    SpectralResponse albedo = {payload.spect};

    const auto& mat = scene.materials[subsurface_material];

    Medium::Instance medium_instance = {
      .index = mat.int_medium,
    };

    if (mat.int_medium == kInvalidIndex) {
      auto color = apply_image(payload.spect, mat.transmittance, intersection.tex, scene, nullptr);
      auto distances = mat.subsurface.scale * apply_image(payload.spect, mat.subsurface, intersection.tex, scene, nullptr);
      subsurface::remap(color.integrated, distances.integrated, albedo.integrated, extinction.integrated, scattering.integrated);
      subsurface::remap_channel(color.value, distances.value, albedo.value, extinction.value, scattering.value);
      medium_instance = {.extinction = extinction, .index = kInvalidIndex};
    } else {
      const Medium& medium = scene.mediums[mat.int_medium];
      scattering = medium.s_scattering(payload.spect);
      extinction = scattering + medium.s_absorption(payload.spect);
      albedo = medium::calculate_albedo(payload.spect, scattering, extinction);
    }

    for (uint32_t counter = 0; counter < 1024u; ++counter) {
      prev = curr;

      SpectralResponse pdf = {};

      ray.max_t = 0.0f;
      while (ray.max_t < kRayEpsilon) {
        uint32_t channel = medium::sample_spectrum_component(payload.spect, albedo, payload.throughput, smp.next(), pdf);
        float sample_t = extinction.component(channel);
        ray.max_t = (sample_t > 0.0f) ? -logf(1.0f - smp.next()) / sample_t : kMaxFloat;
        ETX_VALIDATE(ray.max_t);
      }

      bool found_intersection = rt.trace_material(scene, ray, subsurface_material, intersection, smp);

      if (found_intersection) {
        ray.max_t = intersection.t;
      }

      ETX_VALIDATE(ray.max_t);

      SpectralResponse tr = exp(-ray.max_t * extinction);

      pdf *= found_intersection ? tr : tr * extinction;
      if (pdf.is_zero())
        return StepResult::Break;

      auto weight = (found_intersection ? tr : tr * scattering) / pdf.sum();
      ETX_VALIDATE(weight);

      payload.throughput *= weight;
      ETX_VALIDATE(payload.throughput);

      if (found_intersection) {
        return StepResult::IntersectionFound;
      }

      const float3 medium_sample_pos = ray.o + ray.d * ray.max_t;
      handle_medium({}, false, false, medium_sample_pos, medium_instance, payload, ray, smp, path_data, curr, prev);
    }

    return StepResult::Nothing;
  }

  SpectralResponse build_path(Sampler& smp, const float3& ray_o, const float3& ray_d, PathData& path_data, Payload& payload, const EmitterSample& emitter_sample, GBuffer& gbuffer,
    PathVertex& curr, PathVertex& prev) const {
    ETX_VALIDATE(payload.throughput);

    const auto& scene = rt.scene();

    uint32_t subsurface_material = kInvalidIndex;

    Ray ray = {ray_o, ray_d, kRayEpsilon, kMaxFloat};

    Intersection intersection = {};
    Medium::Sample medium_sample = {};
    for (uint32_t path_length = 0; running() && (path_length < scene.max_path_length);) {
      prev = curr;

      auto step = StepResult::Nothing;

      if (subsurface_material == kInvalidIndex) {
        step = regular_step(ray, smp, intersection, medium_sample, payload);
      } else {
        step = subsurface_step(subsurface_material, ray, smp, intersection, payload, path_data, curr, prev);
      }

      if (step == StepResult::Continue) {
        continue;
      } else if (step == StepResult::Break) {
        break;
      }

      bool should_break = true;
      bool first_interaction = path_length == 0;

      if (step == StepResult::SampledMedium) {
        ETX_CRITICAL(payload.medium_index != kInvalidIndex);
        const auto& medium = scene.mediums[payload.medium_index];
        const auto medium_instance = medium.instance(payload.spect, payload.medium_index);
        handle_medium(emitter_sample, first_interaction, medium.enable_explicit_connections, medium_sample.pos, medium_instance, payload, ray, smp, path_data, curr, prev);
        should_break = false;
      } else if (step == StepResult::IntersectionFound) {
        bool from_subsurface = subsurface_material != kInvalidIndex;
        if (from_subsurface) {
          intersection.material_index = scene.subsurface_scatter_material;
          subsurface_material = kInvalidIndex;
        }

        auto result = handle_surface(intersection, emitter_sample, first_interaction, payload, ray, smp, path_data, curr, prev, gbuffer, from_subsurface);

        if (result == InteractionResult::SampleSubsurface) {
          subsurface_material = intersection.material_index;
        }

        if (result == InteractionResult::Continue) {
          continue;
        }

        should_break = result == InteractionResult::Break;
      } else if (enable_direct_hit && (mode != Mode::LightTracing) && (payload.mode == PathSource::Camera)) {
        curr = PathVertex{PathVertex::Class::Emitter};
        curr.medium = {.index = payload.medium_index};
        curr.throughput = payload.throughput;
        curr.pdf.from_prev = payload.pdf_dir;
        curr.intersection.w_i = ray.d;
        curr.intersection.pos = ray.o;  // Store ray origin, direction is in w_i
        path_data.camera_path_size += 1u;
        precompute_camera_mis(curr, prev, path_data);
        payload.result += direct_hit_environment_emitter(curr, prev, path_data, payload.spect, smp, path_length == 0);
      }

      if (should_break || random_continue(path_length, scene.random_path_termination, payload.eta, smp, payload.throughput) == false) {
        break;
      }

      path_length += 1;
    }

    return payload.result;
  }

  SpectralResponse build_camera_path(Sampler& smp, SpectralQuery spect, const float2& uv, PathData& path_data, GBuffer& gbuffer, const uint2& pixel, uint32_t iteration) const {
    auto ray = generate_ray(rt.scene(), rt.camera(), uv, smp.next_2d());
    auto eval = film_evaluate_out(spect, rt.camera(), ray);

    PathVertex prev = {PathVertex::Class::Camera};
    prev.throughput = {spect, 1.0f};

    PathVertex curr = {PathVertex::Class::Camera};
    curr.medium = {.index = rt.camera().medium_index};
    curr.throughput = {spect, 1.0f};
    curr.connectible = true;  // camera vertex is connectible
    curr.intersection.pos = ray.o;
    curr.intersection.nrm = eval.normal;
    curr.intersection.w_i = ray.d;
    curr.pdf.next = eval.pdf_dir;
    curr.pdf.from_prev = 1.0f;
    if (mode == Mode::BDPTFast) {
      curr.pdf.accumulated = 1.0f;
    }
    path_data.camera_path_size = 1u;

    Payload payload = {
      .spect = spect,
      .result = {spect, 0.0f},
      .throughput = curr.throughput,
      .eta = 1.0f,
      .pdf_dir = eval.pdf_dir,
      .medium_index = curr.medium.index,
      .mode = PathSource::Camera,
      .pixel = pixel,
      .iteration = iteration,
      .use_blue_noise = enable_blue_noise,
    };

    return build_path(smp, ray.o, ray.d, path_data, payload, {}, gbuffer, curr, prev);
  }

  SpectralResponse build_emitter_path(Sampler& smp, SpectralQuery spect, PathData& path_data) const {
    path_data.emitter_path.clear();
    const auto& emitter_sample = sample_emission(rt.scene(), spect, smp);
    if ((emitter_sample.pdf_area == 0.0f) || (emitter_sample.pdf_dir == 0.0f) || (emitter_sample.value.is_zero())) {
      return {spect, 0.0f};
    }

    PathVertex prev = {PathVertex::Class::Emitter};
    prev.throughput = {spect, 1.0f};
    prev.connectible = true;
    prev.delta_emitter = emitter_sample.is_delta;

    PathVertex curr = {PathVertex::Class::Emitter};
    curr.intersection.triangle_index = emitter_sample.triangle_index;
    curr.intersection.barycentric = emitter_sample.barycentric;
    curr.intersection.pos = emitter_sample.origin;
    curr.intersection.nrm = emitter_sample.normal;
    curr.intersection.w_i = emitter_sample.direction;
    curr.intersection.emitter_index = emitter_sample.emitter_index;
    curr.medium = {.index = emitter_sample.medium_index};
    curr.throughput = emitter_sample.value;
    curr.pdf.next = emitter_sample.pdf_dir;
    curr.pdf.from_prev = emitter_sample.pdf_area * emitter_sample.pdf_sample;
    curr.connectible = true;  // directional/local emission connection handled via MIS; vertex itself is connectible if subsequent BSDF is
    curr.delta_emitter = emitter_sample.is_delta;
    if (mode == Mode::BDPTFast) {
      curr.pdf.accumulated = safe_div(1.0f, curr.pdf.from_prev);
    }
    path_data.emitter_path.emplace_back(curr);
    path_data.emitter_path_size = 1u;

    float3 o = offset_ray(emitter_sample.origin, curr.intersection.nrm);
    GBuffer gbuffer = {};

    Payload payload = {
      .spect = spect,
      .result = {spect, 0.0f},
      .throughput = curr.throughput * dot(emitter_sample.direction, curr.intersection.nrm) / (emitter_sample.pdf_dir * emitter_sample.pdf_area * emitter_sample.pdf_sample),
      .eta = 1.0f,
      .pdf_dir = emitter_sample.pdf_dir,
      .medium_index = curr.medium.index,
      .mode = PathSource::Light,
    };

    return build_path(smp, o, emitter_sample.direction, path_data, payload, emitter_sample, gbuffer, curr, prev);
  }

  void precompute_camera_mis(PathVertex& curr, PathVertex& prev, PathData& path_data) const {
    if ((mode == Mode::PathTracing) || (mode == Mode::LightTracing)) {
      return;
    }

    const bool enough_length = path_data.camera_path_length() > (mode == Mode::BDPTFast ? 1u : 2u);
    if (mode == Mode::BDPTFast) {
      curr.pdf.accumulated = prev.pdf.accumulated * (enough_length ? safe_div(prev.pdf.from_next, prev.pdf.from_prev) : 1.0f);
      return;
    }

    auto& history = path_data.camera_history;

    history[2] = history[1];
    history[1] = history[0];
    ETX_VALIDATE(history[1].pdf_ratio);

    float ratio = strict_div(prev.pdf.from_next, prev.pdf.from_prev);
    ETX_VALIDATE(ratio);

    history[0] = {
      .pdf_forward = prev.pdf.from_prev,
      .pdf_ratio = ratio,
      .mis_accumulated = enough_length ? history[1].pdf_ratio * (1.0f + history[1].mis_accumulated) : 0.0f,
    };
    ETX_VALIDATE(history[0].mis_accumulated);
    prev.pdf.accumulated = history[0].mis_accumulated;
  }

  void precompute_light_mis(PathVertex& curr, PathVertex& prev, PathData& path_data) const {
    if ((mode == Mode::PathTracing) || (mode == Mode::LightTracing)) {
      return;
    }

    const bool enough_length = path_data.emitter_path_length() > 1;

    if (mode == Mode::BDPTFast) {
      curr.pdf.accumulated = prev.pdf.accumulated * (enough_length ? safe_div(prev.pdf.from_next, prev.pdf.from_prev) : 1.0f);
      return;
    }

    auto& history = path_data.emitter_history;

    history[2] = history[1];
    history[1] = history[0];
    history[0] = {
      .pdf_forward = prev.pdf.from_prev,
      .pdf_ratio = strict_div(prev.pdf.from_next, prev.pdf.from_prev),
      .mis_accumulated = enough_length ? history[1].pdf_ratio * (1.0f + history[1].mis_accumulated) : 0.0f,
    };
    ETX_VALIDATE(history[0].pdf_ratio);
    ETX_VALIDATE(history[0].mis_accumulated);
    prev.pdf.accumulated = history[0].mis_accumulated;
  }

  float mis_camera(const PathData& path_data, const float z_curr_backward, const PathVertex& z_curr, const float z_prev_backward, const PathVertex& z_prev) const {
    float r1 = strict_div(z_prev_backward, z_prev.pdf.from_prev);
    ETX_VALIDATE(r1);

    bool can_connect1 = (path_data.camera_path_length() > 1);
    float result_accumulated = r1 * (float(can_connect1) + z_prev.pdf.accumulated);
    ETX_VALIDATE(result_accumulated);

    float r0 = strict_div(z_curr_backward, z_curr.pdf.from_prev);
    ETX_VALIDATE(r0);

    result_accumulated = r0 * (1.0f + result_accumulated);
    ETX_VALIDATE(result_accumulated);

    return result_accumulated;
  }

  float mis_light(const PathData& path_data, const float y_curr_backward, const PathVertex& y_curr, const float y_prev_backward, const PathVertex& y_prev,
    const uint64_t light_s) const {
    float result = 0.0f;

    if (light_s >= 1) {
      float r1 = strict_div(y_prev_backward, y_prev.pdf.from_prev);
      result = r1 * (1.0f + y_prev.pdf.accumulated);
    }

    float r0 = strict_div(y_curr_backward, y_curr.pdf.from_prev);
    result = r0 * (1.0f + result);

    return result;
  }

  float mis_weight_camera_to_light(const PathVertex& z_curr, const PathVertex& z_prev, const PathData& path_data, SpectralQuery spect, const PathVertex& sampled_light_vertex,
    const bool sampled_light_vertex_is_delta, Sampler& smp) const {
    if (enable_mis == false) {
      return 1.0f;
    }

    const auto& scene = rt.scene();

    if (mode == Mode::BDPTFull) {
      float z_curr_backward_pdf = PathVertex::pdf_from_emitter(spect, sampled_light_vertex, z_curr, scene);
      ETX_VALIDATE(z_curr_backward_pdf);
      float z_prev_backward_pdf = PathVertex::pdf_area(spect, PathSource::Camera, sampled_light_vertex, z_curr, z_prev, scene, smp);
      ETX_VALIDATE(z_prev_backward_pdf);

      float w_camera = mis_camera(path_data, z_curr_backward_pdf, z_curr, z_prev_backward_pdf, z_prev);
      float w_light = 0.0f;

      if (sampled_light_vertex_is_delta == false) {
        float y_curr_pdf_backward = PathVertex::pdf_area(spect, PathSource::Light, z_prev, z_curr, sampled_light_vertex, scene, smp);
        w_light = strict_div(y_curr_pdf_backward, sampled_light_vertex.pdf.from_prev);
        ETX_VALIDATE(w_light);
      }
      float result = 1.0f / (w_camera + 1.0f + w_light);
      ETX_VALIDATE(result);
      return result;
    }

    if (mode == Mode::PathTracing) {
      float p_connect = sampled_light_vertex.pdf.next;
      ETX_VALIDATE(p_connect);
      float p_direct = sampled_light_vertex_is_delta ? 0.0f : sampled_light_vertex.pdf.accumulated;
      ETX_VALIDATE(p_direct);
      float result = power_heuristic(p_connect, p_direct);
      ETX_VALIDATE(result);
      return result;
    }

    if (mode == Mode::BDPTFast) {
      float p_ratio = z_curr.pdf.accumulated * safe_div(1.0f, z_curr.pdf.from_prev);
      float p_connection = map0(sampled_light_vertex.pdf.from_prev);
      float p_direct_connection = PathVertex::convert_solid_angle_pdf_to_area(sampled_light_vertex.pdf.accumulated, z_curr, sampled_light_vertex);
      float p_direct = sampled_light_vertex_is_delta ? 0.0f : map0(p_direct_connection);
      float camera_connection_pdf = PathVertex::pdf_from_emitter(spect, sampled_light_vertex, z_curr, scene);
      float p_light_path = map0(camera_connection_pdf) * map0(sampled_light_vertex.pdf.from_prev);
      float result = balance_heuristic(p_connection, p_direct, p_ratio * p_light_path);
      ETX_VALIDATE(result);
      return result;
    }

    return 0.0f;
  }

  float mis_weight_light_to_camera(SpectralQuery spect, const PathData& path_data, const PathVertex& y_curr, const PathVertex& y_prev, const PathVertex& sampled_camera_vertex,
    Sampler& smp) const {
    if (enable_mis == false) {
      return 1.0f;
    }

    const auto& scene = rt.scene();

    if (mode == Mode::BDPTFull) {
      float pdf_dir = film_pdf_out(rt.camera(), y_curr.intersection.pos);
      float y_curr_pdf = PathVertex::convert_solid_angle_pdf_to_area(pdf_dir, sampled_camera_vertex, y_curr);
      ETX_VALIDATE(y_curr_pdf);
      float y_prev_pdf = PathVertex::pdf_area(spect, PathSource::Light, sampled_camera_vertex, y_curr, y_prev, scene, smp);
      ETX_VALIDATE(y_prev_pdf);
      float w_light = mis_light(path_data, y_curr_pdf, y_curr, y_prev_pdf, y_prev, path_data.emitter_path.size());
      return 1.0f / (1.0f + w_light);
    }

    if (mode == Mode::LightTracing) {
      return 1.0f;
    }

    if (mode == Mode::BDPTFast) {
      float p_ratio = y_curr.pdf.accumulated * safe_div(1.0f, y_curr.pdf.from_prev);
      float p_connection = 1.0f;
      float p_camera = film_pdf_out(rt.camera(), y_curr.intersection.pos);
      p_camera = PathVertex::convert_solid_angle_pdf_to_area(p_camera, sampled_camera_vertex, y_curr);
      float p_emitter_direct = map0(path_data.emitter_path[0].pdf.from_next);
      float p_from_camera_direct = path_data.emitter_path[0].delta_emitter ? 0.0f : map0(p_camera) * p_emitter_direct;
      float p_emitter_connect = path_data.emitter_path.size() > 1 ? PathVertex::pdf_to_emitter(spect, path_data.emitter_path[1], path_data.emitter_path[0], scene) : 0.0f;
      float p_from_camera_connect = map0(p_camera) * map0(p_emitter_connect);
      float result = balance_heuristic(p_connection, p_ratio * p_from_camera_connect, p_ratio * p_from_camera_direct);
      ETX_VALIDATE(result);
      return result;
    }

    return 0.0f;
  }

  float mis_weight_camera_to_light_path(const PathVertex& z_curr, const PathVertex& z_prev, PathData& c, SpectralQuery spect, uint64_t light_s, Sampler& smp) const {
    if (enable_mis == false) {
      return 1.0f;
    }

    const auto& scene = rt.scene();
    const PathVertex& y_curr = c.emitter_path[light_s];
    const PathVertex& y_prev = c.emitter_path[light_s - 1];

    float z_curr_pdf = PathVertex::pdf_area(spect, PathSource::Light, y_prev, y_curr, z_curr, scene, smp);
    ETX_VALIDATE(z_curr_pdf);

    float z_prev_pdf = PathVertex::pdf_area(spect, PathSource::Camera, y_curr, z_curr, z_prev, scene, smp);
    ETX_VALIDATE(z_prev_pdf);

    float y_curr_pdf = PathVertex::pdf_area(spect, PathSource::Camera, z_prev, z_curr, y_curr, scene, smp);
    ETX_VALIDATE(y_curr_pdf);

    float y_prev_pdf = PathVertex::pdf_area(spect, PathSource::Light, z_curr, y_curr, y_prev, scene, smp);
    ETX_VALIDATE(y_prev_pdf);

    float w_camera = mis_camera(c, z_curr_pdf, z_curr, z_prev_pdf, z_prev);
    float w_light = mis_light(c, y_curr_pdf, y_curr, y_prev_pdf, y_prev, light_s);

    return 1.0f / (1.0f + w_camera + w_light);
  }

  SpectralResponse direct_hit_area_emitter(const PathVertex& z_curr, const PathVertex& z_prev, const PathData& path_data, SpectralQuery spect, Sampler& smp, bool force) const {
    if ((force == false) && (enable_direct_hit == false))
      return {spect, 0.0f};

    if (z_curr.is_emitter() == false) {
      return {spect, 0.0f};
    }
    ETX_ASSERT(z_curr.is_specific_emitter());

    const auto& scene = rt.scene();

    const uint32_t target_path_length = path_data.camera_path_length();
    if ((target_path_length > scene.max_path_length) || (target_path_length < scene.min_path_length))
      return {spect, 0.0f};

    const auto& emitter = scene.emitters[z_curr.intersection.emitter_index];
    ETX_ASSERT(emitter.is_local());
    EmitterRadianceQuery q = {
      .source_position = z_prev.intersection.pos,
      .target_position = z_curr.intersection.pos,
      .uv = z_curr.intersection.tex,
      .directly_visible = path_data.camera_path_length() <= 1,
    };

    float pdf_dir = 0.0f;
    float pdf_area = 0.0f;
    float pdf_dir_out = 0.0f;
    auto emitter_value = emitter_get_radiance(emitter, spect, q, pdf_area, pdf_dir, pdf_dir_out, scene);

    if (pdf_dir == 0.0f) {
      return {spect, 0.0f};
    }

    pdf_dir *= emitter_discrete_pdf(emitter, scene.emitters_distribution);

    float mis_weight = 1.0f;

    if (enable_mis && (path_data.camera_path_length() > 1u)) {
      switch (mode) {
        case Mode::PathTracing: {
          float p_connect = pdf_dir;
          ETX_VALIDATE(p_connect);
          float result = power_heuristic(z_prev.pdf.next, p_connect);
          ETX_VALIDATE(result);
          mis_weight = result;
          break;
        }

        case Mode::BDPTFull: {
          float z_curr_pdf = PathVertex::pdf_to_emitter(spect, z_prev, z_curr, scene);
          ETX_VALIDATE(z_curr_pdf);
          float z_prev_pdf = PathVertex::pdf_from_emitter(spect, z_curr, z_prev, scene);
          ETX_VALIDATE(z_prev_pdf);
          float result = mis_camera(path_data, z_curr_pdf, z_curr, z_prev_pdf, z_prev);
          mis_weight = 1.0f / (1.0f + result);
          break;
        }

        case Mode::BDPTFast: {
          float p_ratio = z_curr.pdf.accumulated;
          float p_direct = map0(z_curr.pdf.from_prev);
          float p_em = PathVertex::pdf_to_emitter(spect, z_prev, z_curr, scene);
          float p_connection = z_prev.connectible ? map0(p_em) : 0.0f;
          float p_sample = emitter_discrete_pdf(scene.emitters[z_curr.intersection.emitter_index], scene.emitters_distribution);
          float camera_connection_pdf = p_sample * emitter_pdf_area_local(scene.emitters[z_curr.intersection.emitter_index], scene);
          float p_light_path = map0(camera_connection_pdf);
          mis_weight = balance_heuristic(p_direct, p_connection, p_ratio * p_light_path);
          ETX_VALIDATE(mis_weight);
          break;
        }

        default:
          mis_weight = 0.0f;
          break;
      }
    }

    return emitter_value * z_curr.throughput * mis_weight;
  }

  SpectralResponse direct_hit_environment_emitter(const PathVertex& z_curr, const PathVertex& z_prev, const PathData& path_data, SpectralQuery spect, Sampler& smp,
    bool force) const {
    if ((force == false) && (enable_direct_hit == false))
      return {spect, 0.0f};

    ETX_ASSERT(z_curr.is_emitter() && (z_curr.is_specific_emitter() == false));

    const auto& scene = rt.scene();
    if (scene.environment_emitters.count == 0)
      return {spect, 0.0f};

    const uint32_t target_path_length = path_data.camera_path_length();
    if ((target_path_length > scene.max_path_length) || (target_path_length < scene.min_path_length))
      return {spect, 0.0f};

    EmitterRadianceQuery q = {
      .direction = z_curr.intersection.w_i,  // Use stored ray direction for environment emitters
      .directly_visible = path_data.camera_path_length() <= 1,
    };

    SpectralResponse accumulated_emitter_value = {spect, 0.0f};
    for (uint32_t ie = 0; ie < scene.environment_emitters.count; ++ie) {
      const auto& emitter = scene.emitters[rt.scene().environment_emitters.emitters[ie]];

      float local_pdf_area = 0.0f;
      float local_pdf_dir = 0.0f;
      float local_pdf_dir_out = 0.0f;
      auto value = emitter_get_radiance(emitter, spect, q, local_pdf_area, local_pdf_dir, local_pdf_dir_out, scene);

      float this_weight = 1.0f;
      if ((mode == Mode::PathTracing) && (path_data.camera_path_length() > 1u)) {
        float local_pdf_sample = emitter_discrete_pdf(emitter, scene.emitters_distribution);
        float this_p_connect = local_pdf_dir * local_pdf_sample;
        ETX_VALIDATE(this_p_connect);
        this_weight = power_heuristic(z_prev.pdf.next, this_p_connect);
        ETX_VALIDATE(this_weight);
      }
      accumulated_emitter_value += value * this_weight;
      ETX_VALIDATE(accumulated_emitter_value);
    }

    float mis_weight = 1.0f;
    if (enable_mis && (path_data.camera_path_length() > 1u)) {
      auto [pdf_from, pdf_to] = PathVertex::pdf_for_environment_emitter(spect, z_curr, z_prev, scene);

      if (mode == Mode::BDPTFull) {
        float z_curr_pdf = pdf_to;
        ETX_VALIDATE(z_curr_pdf);
        float z_prev_pdf = pdf_from;
        ETX_VALIDATE(z_prev_pdf);
        float result = mis_camera(path_data, z_curr_pdf, z_curr, z_prev_pdf, z_prev);
        mis_weight = 1.0f / (1.0f + result);
      } else if (mode == Mode::BDPTFast) {
        float p_ratio = z_curr.pdf.accumulated;
        float p_direct = map0(z_curr.pdf.from_prev);
        float p_em = pdf_to;
        float p_connection = z_prev.connectible ? map0(p_em) : 0.0f;
        float camera_connection_pdf = pdf_from;
        float p_light_path = map0(p_em) * map0(camera_connection_pdf);
        mis_weight = balance_heuristic(p_direct, p_connection, p_ratio * p_light_path);
        ETX_VALIDATE(mis_weight);
      }
    }

    return accumulated_emitter_value * z_curr.throughput * mis_weight;
  }

  SpectralResponse connect_camera_to_light(const PathVertex& z_curr, const PathVertex& z_prev, Sampler& smp, PathData& path_data, SpectralQuery spect) const {
    const auto& scene = rt.scene();

    uint32_t connection_len = path_data.camera_path_length() + 1u;
    bool invalid_path_length = (connection_len > scene.max_path_length) || (connection_len < scene.min_path_length);
    if (invalid_path_length || (enable_connect_to_light == false) || (mode == Mode::LightTracing))
      return {spect, 0.0f};

    uint32_t emitter_index = sample_emitter_index(scene, smp.fixed_w);
    auto emitter_sample = sample_emitter(spect, emitter_index, {smp.fixed_u, smp.fixed_v}, z_curr.intersection.pos, rt.scene());
    if (emitter_sample.value.is_zero() || (emitter_sample.pdf_dir == 0.0f)) {
      return {spect, 0.0f};
    }

    auto dp = emitter_sample.origin - z_curr.intersection.pos;
    if (dot(dp, dp) <= kEpsilon) {
      return {spect, 0.0f};
    }

    auto bsdf_eval = z_curr.bsdf_in_direction(spect, PathSource::Camera, emitter_sample.direction, rt.scene(), smp);
    if (bsdf_eval.bsdf.is_zero()) {
      return {spect, 0.0f};
    }

    PathVertex sampled_vertex = {PathVertex::Class::Emitter};
    sampled_vertex.intersection.w_i = normalize(dp);
    sampled_vertex.intersection.pos = emitter_sample.origin;
    sampled_vertex.intersection.nrm = emitter_sample.normal;
    sampled_vertex.intersection.triangle_index = emitter_sample.triangle_index;
    sampled_vertex.intersection.emitter_index = emitter_sample.emitter_index;

    sampled_vertex.pdf.accumulated = bsdf_eval.pdf;
    sampled_vertex.pdf.from_prev = PathVertex::pdf_to_emitter(spect, z_curr, sampled_vertex, rt.scene());
    sampled_vertex.pdf.next = emitter_sample.pdf_dir * emitter_sample.pdf_sample;

    SpectralResponse emitter_throughput = emitter_sample.value / sampled_vertex.pdf.next;
    ETX_VALIDATE(emitter_throughput);

    SpectralResponse tr = local_transmittance(spect, smp, z_curr, sampled_vertex.intersection.pos);
    float weight = mis_weight_camera_to_light(z_curr, z_prev, path_data, spect, sampled_vertex, emitter_sample.is_delta, smp);

    return z_curr.throughput * bsdf_eval.bsdf * emitter_throughput * tr * weight;
  }

  SpectralResponse connect_light_to_camera(Sampler& smp, PathData& path_data, const PathVertex& y_curr, const PathVertex& y_prev, SpectralQuery spect,
    CameraSample& camera_sample) const {
    const auto& scene = rt.scene();

    const uint32_t target_path_length = path_data.emitter_path_length() + 1u;
    if ((mode == Mode::PathTracing) || (enable_connect_to_camera == false) || (target_path_length > scene.max_path_length) || (target_path_length < scene.min_path_length))
      return {spect, 0.0f};

    camera_sample = sample_film(smp, scene, rt.camera(), y_curr.intersection.pos);
    if (camera_sample.valid() == false) {
      return {spect, 0.0f};
    }

    ETX_VALIDATE(camera_sample.weight);

    PathVertex sampled_vertex = {PathVertex::Class::Camera};
    sampled_vertex.intersection.pos = camera_sample.position;
    sampled_vertex.intersection.nrm = camera_sample.normal;
    sampled_vertex.intersection.w_i = camera_sample.direction;
    sampled_vertex.pdf.next = camera_sample.pdf_dir;

    auto bsdf = y_curr.bsdf_in_direction(spect, PathSource::Light, camera_sample.direction, scene, smp).bsdf;
    if (bsdf.is_zero()) {
      return {spect, 0.0f};
    }

    float weight = mis_weight_light_to_camera(spect, path_data, y_curr, y_prev, sampled_vertex, smp);

    SpectralResponse splat = y_curr.throughput * bsdf * (camera_sample.weight * weight / spect.sampling_pdf());
    ETX_VALIDATE(splat);

    if (splat.is_zero() == false) {
      splat *= local_transmittance(spect, smp, y_curr, sampled_vertex.intersection.pos);
    }

    return splat;
  }

  SpectralResponse local_transmittance(SpectralQuery spect, Sampler& smp, const PathVertex& p0, const float3& p1) const {
    auto& scene = rt.scene();
    float3 origin = p0.intersection.pos;
    if (p0.is_surface_interaction()) {
      const auto& tri = scene.triangles[p0.intersection.triangle_index];
      origin = shading_pos(scene.vertices, tri, p0.intersection.barycentric, normalize(p1 - p0.intersection.pos));
    }
    return rt.trace_transmittance(spect, scene, origin, p1, p0.medium, smp);
  }

  void start(const Options& opt) {
    mode = opt.get("bdpt-mode", uint32_t(mode)).to_enum<Mode>();

    enable_direct_hit = opt.get("bdpt-conn_direct_hit", enable_direct_hit).to_bool();
    enable_connect_to_camera = opt.get("bdpt-conn_connect_to_camera", enable_connect_to_camera).to_bool();
    enable_connect_to_light = opt.get("bdpt-conn_connect_to_light", enable_connect_to_light).to_bool();
    enable_connect_vertices = opt.get("bdpt-conn_connect_vertices", enable_connect_vertices).to_bool();
    enable_mis = opt.get("bdpt-conn_mis", enable_mis).to_bool();
    enable_blue_noise = opt.get("bdpt-blue_noise", enable_blue_noise).to_bool();

    for (auto& path_data : per_thread_path_data) {
      path_data.emitter_path.reserve(2llu + rt.scene().max_path_length);
    }

    status = {};
    iteration_time = {};
    rt.film().clear({Film::Internal, Film::LightImage, Film::LightIteration});
    current_task = rt.scheduler().schedule(rt.film().pixel_count(), this);
  }
};

CPUBidirectional::CPUBidirectional(Raytracing& rt)
  : Integrator(rt) {
  ETX_PIMPL_INIT(CPUBidirectional, rt, &current_state);
}

CPUBidirectional::~CPUBidirectional() {
  if (current_state != State::Stopped) {
    stop(Stop::Immediate);
  }
  ETX_PIMPL_CLEANUP(CPUBidirectional);
}

void CPUBidirectional::run(const Options& opt) {
  stop(Stop::Immediate);

  if (can_run()) {
    current_state = State::Running;
    _private->start(opt);
  }
}

void CPUBidirectional::update() {
  if ((current_state == State::Stopped) || (rt.scheduler().completed(_private->current_task) == false)) {
    return;
  }

  rt.film().commit_light_iteration(_private->status.current_iteration);

  if (current_state == State::WaitingForCompletion) {
    rt.scheduler().wait(_private->current_task);
    _private->current_task = {};
    current_state = Integrator::State::Stopped;
  } else if (_private->status.current_iteration + 1u < rt.scene().samples) {
    _private->completed();
    rt.scheduler().restart(_private->current_task);
  } else {
    current_state = Integrator::State::Stopped;
  }
}

void CPUBidirectional::stop(Stop st) {
  if (current_state == State::Stopped) {
    return;
  }

  if (st == Stop::Immediate) {
    current_state = State::Stopped;
    rt.scheduler().wait(_private->current_task);
    _private->current_task = {};
  } else {
    current_state = State::WaitingForCompletion;
  }
}

Options CPUBidirectional::options() const {
  auto mode = OptionalValue(
    _private->mode, CPUBidirectionalImpl::Mode::Count,
    [](uint32_t index) -> std::string {
      switch (CPUBidirectionalImpl::Mode(index)) {
        case CPUBidirectionalImpl::Mode::PathTracing:
          return "Path Tracing";
        case CPUBidirectionalImpl::Mode::LightTracing:
          return "Light Tracing";
        case CPUBidirectionalImpl::Mode::BDPTFast:
          return "BDPT Fast";
        case CPUBidirectionalImpl::Mode::BDPTFull:
          return "BDPT Full";
        default:
          return "Unknown";
      }
    },
    "bdpt-mode", "Mode");

  Options result = {};
  if (_private->mode != CPUBidirectionalImpl::Mode::LightTracing) {
    result.add("bdpt-conn", "Connections:");
    result.add(_private->enable_direct_hit, "bdpt-conn_direct_hit", "Direct Hits");
    if (_private->mode != CPUBidirectionalImpl::Mode::PathTracing) {
      result.add(_private->enable_connect_to_camera, "bdpt-conn_connect_to_camera", "Light Path to Camera");
    }
    result.add(_private->enable_connect_to_light, "bdpt-conn_connect_to_light", "Camera Path to Light");
    if (_private->mode == CPUBidirectionalImpl::Mode::BDPTFull) {
      result.add(_private->enable_connect_vertices, "bdpt-conn_connect_vertices", "Camera Path to Light Path");
    }
    result.add("bdpt-opt", "Bidirectional Path Tracing Options");
    result.add(_private->enable_mis, "bdpt-conn_mis", "Multiple Importance Sampling");
    result.add(_private->enable_blue_noise, "bdpt-blue_noise", "Enable Blue Noise");
  } else {
    result.add("bdpt-opt", "Light Tracing Mode Has No Options");
  }
  result.add(mode);
  return result;
}

void CPUBidirectional::update_options(const Options& opt) {
  if (current_state == State::Running) {
    run(opt);
  }
}

const Integrator::Status& CPUBidirectional::status() const {
  return _private->status;
}

}  // namespace etx
