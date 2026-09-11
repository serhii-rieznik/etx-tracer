#include <etx/core/core.hxx>
#include <etx/render/host/film.hxx>
#include <etx/rt/integrators/bidirectional.hxx>
#include <etx/rt/shared/bdpt_mode.hxx>
#include <etx/rt/shared/path_tracing_shared.hxx>

namespace etx {

namespace {

#define ETX_INCLUDE_CAMERA_PATH 0

struct PathVertex {
  enum class Class : uint16_t {
    Invalid,
    Camera,
    Emitter,
    Surface,
    Medium,
  };

  Intersection intersection = {};
  MediumInstance medium = {};
  SpectralResponse throughput = {};

  struct {
    float bsdf_sample_next = 0.0f;
    float d_vcm = 0.0f;
    float d_vc = 0.0f;
  } pdf;

  uint32_t material = MaterialClass::Undefined;
  Class cls = Class::Invalid;
  bool connectible = true;

  PathVertex() = default;

  PathVertex(Class c, const Intersection& i)
    : intersection(i)
    , cls(c) {
  }

  PathVertex(const float3& medium_sample_pos, const float3& a_w_i, const MediumInstance m)
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
    return (cls == Class::Medium) && medium_instance_valid(medium);
  }

  static bool safe_normalize(const float3& to_vertex, const float3& from_vertex, float3& n) {
    n = to_vertex - from_vertex;
    float len = dot(n, n);
    if (len == 0.0f)
      return false;

    n *= 1.0f / sqrtf(len);
    return true;
  }

  static float pdf_area(SpectralQuery spect, PathSource path_source, const PathVertex& prev, const PathVertex& curr, const PathVertex& next, const Material* material,
    Sampler& smp) {
    ETX_CRITICAL(curr.is_surface_interaction() || curr.is_medium_interaction());

    float3 w_i = {};
    float3 w_o = {};
    if (safe_normalize(curr.intersection.pos, prev.intersection.pos, w_i) == false)
      return 0.0f;

    if (safe_normalize(next.intersection.pos, curr.intersection.pos, w_o) == false)
      return 0.0f;

    float eval_pdf = 0.0f;
    if (curr.is_surface_interaction()) {
      ETX_ASSERT(material != nullptr);
      eval_pdf = bsdf::pdf({spect, kInvalidIndex, path_source, curr.intersection, w_i}, w_o, *material, smp);
      ETX_VALIDATE(eval_pdf);
    } else if (curr.is_medium_interaction()) {
      eval_pdf = phase_function(w_i, w_o, curr.medium.anisotropy);
      ETX_VALIDATE(eval_pdf);
    }

    return convert_solid_angle_pdf_to_area(eval_pdf, curr, next);
  }

  static float2 pdf_for_environment_emitter(SpectralQuery spect, const float3& w_i, const PathVertex& target_vertex) {
    (void)spect;
    return ::etx::emitter_environment_pdf(w_i, target_vertex.is_surface_interaction(), target_vertex.intersection.triangle_index, target_vertex.intersection.instance_index);
  }

  static float convert_solid_angle_pdf_to_area(float pdf_dir, const PathVertex& from_vertex, const PathVertex& to_vertex) {
    if ((pdf_dir == 0.0f) || to_vertex.is_environment_emitter()) {
      return pdf_dir;
    }

    auto w_o = to_vertex.intersection.pos - from_vertex.intersection.pos;
    float d_squared = fmaxf(dot(w_o, w_o), kRayEpsilon * kRayEpsilon);

    float inv_d_squared = 1.0f / d_squared;
    w_o *= sqrtf(inv_d_squared);

    float cos_t = (to_vertex.is_surface_interaction() ? fabsf(dot(w_o, to_vertex.intersection.nrm)) : 1.0f);

    float result = cos_t * pdf_dir * inv_d_squared;
    ETX_VALIDATE(result);
    return result;
  }

  auto bsdf_in_direction(SpectralQuery spect, PathSource mode, const float3& w_o, const Material* material, const float3& geo_n, Sampler& smp) const {
    ETX_ASSERT(is_surface_interaction() || is_medium_interaction());

    struct Result {
      SpectralResponse bsdf = {};
      float pdf = {};
    };

    if (is_surface_interaction()) {
      ETX_ASSERT(material != nullptr);
      BSDFEval eval = bsdf::evaluate({spect, kInvalidIndex, mode, intersection, intersection.w_i}, w_o, *material, smp);
      ETX_VALIDATE(eval.bsdf);
      if (mode == PathSource::Light) {
        eval.bsdf *= fix_shading_normal(geo_n, intersection.nrm, intersection.w_i, w_o);
        ETX_VALIDATE(eval.bsdf);
      }

      return Result{eval.bsdf, eval.pdf};
    }

    if (is_medium_interaction()) {
      float eval_pdf = phase_function(intersection.w_i, w_o, medium.anisotropy);
      return Result{{spect, eval_pdf}, eval_pdf};
    }

    ETX_FAIL("Invalid path vertex");
    return Result{{spect, 0.0f}, 0.0f};
  }
};

struct PathData {
  std::vector<PathVertex> emitter_path;

#if (ETX_INCLUDE_CAMERA_PATH)
  std::vector<PathVertex> camera_path;
#endif

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

inline float safe_div(float a, float b) {
  if (b == 0.0f) {
    return 0.0f;
  }
  float result = a / b;
  ETX_VALIDATE(result);
  return result;
}

}  // namespace

struct CPUBidirectionalImpl : public Task {
  Raytracing& rt;
  std::vector<PathData> per_thread_path_data;
  std::atomic<Integrator::State>* state = {};
  TimeMeasure iteration_time = {};
  Integrator::Status status = {};
  Handle current_task = {};
  bool enable_direct_hit = true;
  bool enable_connect_to_camera = true;
  bool enable_connect_to_light = true;
  bool enable_connect_vertices = true;
  bool enable_mis = true;
  bool enable_blue_noise = true;
  bool mode_locked = false;

  using Mode = BDPTMode;
  Mode mode = Mode::BDPTFast;

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
    float d_vcm = 0.0f;
    float d_vc = 0.0f;
    float path_distance = 0.0f;
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
    ETX_PROFILER_SCOPE();
    auto& path_data = per_thread_path_data[thread_id];
    auto& film = rt.film();
    auto& scene = rt.scene();

    for (uint32_t i = begin; (state->load() != Integrator::State::Stopped) && (i < end); ++i) {
      const uint2 pixel = film.pixel_location(i);

      const uint2 film_size = film.base_dimensions();
      const uint32_t pixel_index = pixel.x + pixel.y * film_size.x;
      auto camera_smp = Sampler(scene.sampler_seed(pixel_index, status.current_iteration, kSamplerRandomDomainCameraPathRoot));
      auto light_smp = Sampler(scene.sampler_seed(pixel_index, status.current_iteration, kSamplerRandomDomainLightPathRoot));

      SpectralQuery spect = SpectralQuery::sample();
      if (mode != Mode::PathTracing) {
        if (scene.spectral()) {
          spect = SpectralQuery::spectral_sample(light_smp.next());
          camera_smp.next();
        }
        build_emitter_path(light_smp, spect, path_data);
      } else {
        if (scene.spectral()) {
          spect = SpectralQuery::spectral_sample(camera_smp.next());
        }
      }

      GBuffer gbuffer = {};
      SpectralResponse result = {spect, 0.0f};

      if (mode != Mode::LightTracing) {
        const float2 pixel_sample = camera_smp.next_2d();
        float2 uv = film.sample(rt.scene().pixel_sampler, pixel, pixel_sample, camera_smp.next_2d());
        result = build_camera_path(camera_smp, spect, uv, path_data, gbuffer, pixel, status.current_iteration);
      }

      if (running() == false) {
        break;
      }

      auto xyz = result.to_rgb_estimate();
      auto albedo = gbuffer.albedo.to_rgb_estimate();
      film.submit(xyz, gbuffer.normal, albedo, pixel);
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

  SpectralResponse connect_camera_to_light_path(const PathVertex& z_i, const PathVertex& z_prev, Sampler& smp, SpectralQuery spect, PathData& path_data) const {
    const auto& scene = rt.scene();

    SpectralResponse result = {spect, 0.0f};
    if ((mode != Mode::BDPTFull) || (enable_connect_vertices == false) || (z_i.connectible == false)) {
      return result;
    }

    const uint32_t camera_path_length = path_data.camera_path_length();
    for (uint32_t light_s = 1, light_s_e = static_cast<uint32_t>(path_data.emitter_path.size()); running() && (light_s < light_s_e); ++light_s) {
      const uint32_t target_path_length = camera_path_length + light_s + 1;
      if (target_path_length > scene.options.max_path_length)
        break;
      if (target_path_length < scene.options.min_path_length)
        continue;

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
      const Material* y_i_material = y_i.is_surface_interaction() ? &rt.scene().materials[y_i.intersection.material_index] : nullptr;
      float3 y_i_geo_n = y_i.is_surface_interaction()
                           ? scene_triangle_world_geometric_normal(rt.scene(), rt.scene().triangles[y_i.intersection.triangle_index], y_i.intersection.instance_index)
                           : float3{};
      auto bsdf_y = y_i.bsdf_in_direction(spect, PathSource::Light, dw, y_i_material, y_i_geo_n, smp).bsdf;
      ETX_VALIDATE(bsdf_y);

      const Material* z_i_material = z_i.is_surface_interaction() ? &rt.scene().materials[z_i.intersection.material_index] : nullptr;
      float3 z_i_geo_n = z_i.is_surface_interaction()
                           ? scene_triangle_world_geometric_normal(rt.scene(), rt.scene().triangles[z_i.intersection.triangle_index], z_i.intersection.instance_index)
                           : float3{};
      auto bsdf_z = z_i.bsdf_in_direction(spect, PathSource::Camera, -dw, z_i_material, z_i_geo_n, smp).bsdf;
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

  void record_path_vertex(const Payload& payload, PathData& path_data, const PathVertex& curr, const PathVertex& prev) const {
    if (payload.mode == PathSource::Light) {
      path_data.emitter_path.back() = prev;
      path_data.emitter_path.emplace_back(curr);
    } else if (payload.mode == PathSource::Camera) {
#if (ETX_INCLUDE_CAMERA_PATH)
      path_data.camera_path.back() = prev;
      path_data.camera_path.emplace_back(curr);
#endif
    }
  }

  void update_vertex_mis_state(const EmitterSample& emitter_sample, const bool first_interaction, const float segment_distance, Payload& payload, PathVertex& curr) const {
    if ((mode == Mode::PathTracing) || (mode == Mode::LightTracing)) {
      return;
    }

    if (curr.is_surface_interaction()) {
      const float cos_to_prev = fabsf(dot(curr.intersection.nrm, -curr.intersection.w_i));
      if (cos_to_prev > 0.0f) {
        const bool finite_segment = (payload.mode == PathSource::Camera) || (first_interaction == false) || (emitter_sample.is_distant == false);
        if (finite_segment) {
          payload.d_vcm *= sqr(segment_distance);
        }
        payload.d_vcm /= cos_to_prev;
        payload.d_vc /= cos_to_prev;
      }
    } else {
      payload.d_vcm *= sqr(segment_distance);
    }

    payload.path_distance = 0.0f;
    curr.pdf.d_vcm = payload.d_vcm;
    curr.pdf.d_vc = payload.d_vc;
    ETX_VALIDATE(curr.pdf.d_vcm);
    ETX_VALIDATE(curr.pdf.d_vc);
  }

  void update_scatter_mis_state(const bool first_interaction, const float reverse_pdf, const BSDFSample& sample, const PathVertex& curr, Payload& payload) const {
    if ((mode == Mode::PathTracing) || (mode == Mode::LightTracing)) {
      return;
    }

    const float cos_theta = curr.is_surface_interaction() ? fabsf(dot(curr.intersection.nrm, sample.w_o)) : 1.0f;
    if (sample.properties & BSDFSample::Delta) {
      payload.d_vcm = 0.0f;
      payload.d_vc *= cos_theta;
    } else {
      const float endpoint_source = ((mode == Mode::BDPTFast) && first_interaction) ? payload.d_vcm : 0.0f;
      const float connection_source = (mode == Mode::BDPTFull) ? payload.d_vcm : endpoint_source;
      payload.d_vc = safe_div(cos_theta * (payload.d_vc * reverse_pdf + connection_source), sample.pdf);
      payload.d_vcm = safe_div(1.0f, sample.pdf);
    }

    ETX_VALIDATE(payload.d_vcm);
    ETX_VALIDATE(payload.d_vc);
  }

  void connect(Payload& payload, Sampler& smp, const float3& smp_fixed, PathData& path_data, PathVertex& curr, PathVertex& prev) const {
    if (payload.mode == PathSource::Light) {
      if (curr.connectible) {
        CameraSample camera_sample = {};
        auto splat = connect_light_to_camera(smp, path_data, curr, prev, payload.spect, camera_sample);
        rt.film().submit(splat.to_rgb_estimate(), camera_sample.uv);
      }
    } else if (payload.mode == PathSource::Camera) {
      smp.push_fixed(smp_fixed.x, smp_fixed.y, smp_fixed.z);
      if (curr.connectible) {
        payload.result += connect_camera_to_light(curr, prev, smp, path_data, payload.spect);
      }
      payload.result += direct_hit_area_emitter(curr, prev, path_data, payload.spect, smp, false);
      smp.pop_fixed();
      if (curr.connectible) {
        payload.result += connect_camera_to_light_path(curr, prev, smp, payload.spect, path_data);
      }
    }
  }

  void handle_medium(const EmitterSample& emitter_sample, const bool first_interaction, const bool explicit_connections, const float3& medium_sample_pos,
    const MediumInstance& medium_instance, Payload& payload, Ray& ray, Sampler& smp, PathData& path_data, PathVertex& curr, PathVertex& prev) const {
    const auto& scene = rt.scene();

    ETX_ASSERT(medium_instance_valid(medium_instance));

    float2 rnd_bsdf = smp.next_2d();
    float2 rnd_em_sample = smp.next_2d();
    float2 rnd_support = smp.next_2d();
    if (enable_blue_noise && (payload.mode == PathSource::Camera) && first_interaction && (payload.iteration < kSamplerBlueNoiseSampleCount)) {
      const uint2 sample_pixel = sampler_blue_noise_pixel(payload.pixel, rt.scene().options.random_seed);
      rnd_bsdf = sample_blue_noise_at_translated_pixel(sample_pixel, rt.scene().options.samples, payload.iteration, 0u);
      rnd_em_sample = sample_blue_noise_at_translated_pixel(sample_pixel, rt.scene().options.samples, payload.iteration, 2u);
      rnd_support = sample_blue_noise_at_translated_pixel(sample_pixel, rt.scene().options.samples, payload.iteration, 4u);
    }

    float3 w_o = sample_phase_function(ray.d, medium_instance.anisotropy, rnd_bsdf);
    float pdf_fwd = phase_function(ray.d, w_o, medium_instance.anisotropy);
    float pdf_bck = phase_function(w_o, ray.d, medium_instance.anisotropy);

    path_data.camera_path_size += uint32_t(payload.mode == PathSource::Camera);
    path_data.emitter_path_size += uint32_t(payload.mode == PathSource::Light);

    curr = PathVertex{medium_sample_pos, ray.d, medium_instance};
    curr.material = MaterialClass::Undefined;
    curr.connectible = true;
    curr.throughput = payload.throughput;
    curr.pdf.bsdf_sample_next = pdf_fwd;

    const float segment_distance = payload.path_distance + length(medium_sample_pos - ray.o);
    update_vertex_mis_state(emitter_sample, first_interaction, segment_distance, payload, curr);

    ray.o = medium_sample_pos;
    ray.d = w_o;
    ray.min_t = kRayEpsilon;
    ray.max_t = kMaxFloat;

    record_path_vertex(payload, path_data, curr, prev);

    if (explicit_connections) {
      connect(payload, smp, {rnd_em_sample.x, rnd_em_sample.y, rnd_support.y}, path_data, curr, prev);
    }

    BSDFSample phase_sample = {};
    phase_sample.w_o = w_o;
    phase_sample.pdf = pdf_fwd;
    update_scatter_mis_state(first_interaction, pdf_bck, phase_sample, curr, payload);
  }

  InteractionResult handle_surface(const Intersection& a_intersection, const EmitterSample& emitter_sample, const bool first_interaction, Payload& payload, Ray& ray, Sampler& smp,
    PathData& path_data, PathVertex& curr, PathVertex& prev, GBuffer& gbuffer, bool subsurface_exit) const {
    const auto& scene = rt.scene();

    float2 rnd_bsdf = smp.next_2d();
    float2 rnd_em_sample = smp.next_2d();
    float2 rnd_support = smp.next_2d();
    if (enable_blue_noise && (payload.mode == PathSource::Camera) && first_interaction && (payload.iteration < kSamplerBlueNoiseSampleCount)) {
      const uint2 sample_pixel = sampler_blue_noise_pixel(payload.pixel, rt.scene().options.random_seed);
      rnd_bsdf = sample_blue_noise_at_translated_pixel(sample_pixel, rt.scene().options.samples, payload.iteration, 0u);
      rnd_em_sample = sample_blue_noise_at_translated_pixel(sample_pixel, rt.scene().options.samples, payload.iteration, 2u);
      rnd_support = sample_blue_noise_at_translated_pixel(sample_pixel, rt.scene().options.samples, payload.iteration, 4u);
    }

    if (scene.materials[a_intersection.material_index].cls == MaterialClass::Boundary) {
      const auto& m = scene.materials[a_intersection.material_index];
      const auto& tri = scene.triangles[a_intersection.triangle_index];
      const float3 geo_normal = scene_triangle_world_geometric_normal(scene, tri, a_intersection.instance_index);
      payload.medium_index = (dot(geo_normal, ray.d) < 0.0f) ? m.int_medium : m.ext_medium;
      payload.path_distance += a_intersection.t;
      ray.o = shading_pos(scene, tri, a_intersection.barycentric, ray.d, a_intersection.instance_index);
      ray.min_t = kRayEpsilon;
      ray.max_t = kMaxFloat;
      return InteractionResult::Continue;
    }

    BSDFData bsdf_data = {payload.spect, payload.medium_index, payload.mode, a_intersection, a_intersection.w_i};

    if (gbuffer.recorded == false) {
      gbuffer.normal = a_intersection.nrm;
      gbuffer.albedo = bsdf::albedo(bsdf_data, scene.materials[a_intersection.material_index], smp);
      gbuffer.recorded = true;
    }

    smp.push_fixed(rnd_bsdf.x, rnd_bsdf.y, rnd_support.x);
    auto bsdf_sample = bsdf::sample(bsdf_data, scene.materials[a_intersection.material_index], smp);
    smp.pop_fixed();

    ETX_VALIDATE(bsdf_sample.weight);

    bool subsurface_path = (subsurface_exit == false) &&                                                                       //
                           (scene.materials[a_intersection.material_index].subsurface_cls != SubsurfaceMaterial::Disabled) &&  //
                           (bsdf_sample.properties & BSDFSample::Reflection) && (bsdf_sample.properties & BSDFSample::Diffuse);

    uint32_t material_index = a_intersection.material_index;

    MediumInstance medium_instance = {
      .index = (bsdf_sample.properties & BSDFSample::MediumChanged) ? bsdf_sample.medium_index : payload.medium_index,
    };

    if (subsurface_path) {
      const auto& sss_material = scene.materials[a_intersection.material_index];
      material_index = scene.defaults.subsurface_scatter_material;
      medium_instance.index = sss_material.int_medium;

      if (medium_instance.index == kInvalidIndex) {
        medium_instance = subsurface_to_medium_instance(material_index, payload, a_intersection);
      }

      const bool diffuse_transmission = sss_material.subsurface_path == SubsurfaceMaterial::DiffusePath;
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
    curr.material = scene.materials[material_index].cls;
    curr.throughput = payload.throughput;
    curr.intersection.material_index = material_index;
    curr.medium = medium_instance;
    curr.pdf.bsdf_sample_next = bsdf_sample.pdf;
    curr.connectible = (bsdf_sample.properties & BSDFSample::Delta) == 0;

    const float segment_distance = payload.path_distance + a_intersection.t;
    update_vertex_mis_state(emitter_sample, first_interaction, segment_distance, payload, curr);

    float rev_bsdf_pdf = bsdf::reverse_pdf(bsdf_data, bsdf_sample.w_o, scene.materials[material_index], smp);

    payload.medium_index = medium_instance.index;

    bool terminate_path = false;
    if (bsdf_sample.valid()) {
      payload.eta *= (payload.mode == PathSource::Camera) ? bsdf_sample.eta : 1.0f;
      ETX_VALIDATE(payload.eta);

      payload.throughput *= bsdf_sample.weight;
      ETX_VALIDATE(payload.throughput);

      const auto& tri = scene.triangles[a_intersection.triangle_index];

      ray.o = shading_pos(scene, tri, curr.intersection.barycentric, bsdf_sample.w_o, curr.intersection.instance_index);
      ray.d = bsdf_sample.w_o;
      ray.min_t = kRayEpsilon;
      ray.max_t = kMaxFloat;

      if (payload.mode == PathSource::Light) {
        const float3 geo_normal = scene_triangle_world_geometric_normal(scene, tri, curr.intersection.instance_index);
        payload.throughput *= fix_shading_normal(geo_normal, curr.intersection.nrm, curr.intersection.w_i, bsdf_sample.w_o);
        ETX_VALIDATE(payload.throughput);
      }
    } else {
      terminate_path = true;
    }

    record_path_vertex(payload, path_data, curr, prev);
    connect(payload, smp, {rnd_em_sample.x, rnd_em_sample.y, rnd_support.y}, path_data, curr, prev);
    update_scatter_mis_state(first_interaction, rev_bsdf_pdf, bsdf_sample, curr, payload);

    return terminate_path ? InteractionResult::Break : (subsurface_path ? InteractionResult::SampleSubsurface : InteractionResult::NextIteration);
  }

  enum class StepResult {
    Nothing = 0,
    SampledMedium,
    IntersectionFound,
    Continue,
    Break,
  };

  StepResult regular_step(const Ray& ray, Sampler& smp, Intersection& intersection, MediumSample& medium_sample, Payload& payload) const {
    const auto& scene = rt.scene();
    bool found_intersection = rt.trace(scene, ray, intersection, smp);

    if (payload.medium_index != kInvalidIndex) {
      const auto& m = scene.mediums[payload.medium_index];
      medium_sample = sample_medium(m, payload.spect, payload.throughput, smp, ray.o, ray.d, found_intersection ? intersection.t : kMaxFloat);
      spectral_response_mul_assign(payload.throughput, medium_sample.weight);
      ETX_VALIDATE(payload.throughput);
    }

    if (medium_sample_sampled_medium(medium_sample))
      return StepResult::SampledMedium;

    return found_intersection ? StepResult::IntersectionFound : StepResult::Nothing;
  }

  MediumInstance subsurface_to_medium_instance(const uint32_t subsurface_material, const Payload& payload, const Intersection& intersection) const {
    const auto& scene = rt.scene();
    const auto& mat = scene.materials[subsurface_material];
    auto color = apply_image(payload.spect, mat.scattering, intersection.tex);
    auto distances = apply_image(payload.spect, mat.subsurface, intersection.tex);

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

    MediumInstance medium_instance = {
      .index = mat.int_medium,
    };

    if (mat.int_medium == kInvalidIndex) {
      auto color = apply_image(payload.spect, mat.scattering, intersection.tex);
      auto distances = apply_image(payload.spect, mat.subsurface, intersection.tex);
      subsurface::remap(color.integrated, distances.integrated, albedo.integrated, extinction.integrated, scattering.integrated);
      subsurface::remap_channel(color.value, distances.value, albedo.value, extinction.value, scattering.value);
      medium_instance = {.extinction = extinction, .index = kInvalidIndex};
    } else {
      const Medium& medium = scene.mediums[mat.int_medium];
      medium_instance = make_medium_instance(medium, payload.spect, mat.int_medium);
      scattering = medium_scattering(medium, payload.spect);
      auto absorption = medium_absorption(medium, payload.spect);
      extinction = scattering + absorption;
      albedo = calculate_albedo(payload.spect, scattering, extinction);
    }

    for (uint32_t counter = 0; running() && (counter < 1024u); ++counter) {
      prev = curr;

      SpectralResponse pdf = {};

      ray.max_t = 0.0f;
      while (running() && (ray.max_t < kRayEpsilon)) {
        uint32_t channel = sample_spectrum_component(payload.spect, albedo, payload.throughput, smp.next(), pdf);
        float sample_t = extinction.component(channel);
        ray.max_t = (sample_t > 0.0f) ? -logf(1.0f - smp.next()) / sample_t : kMaxFloat;
        ETX_VALIDATE(ray.max_t);
      }

      if (running() == false) {
        return StepResult::Break;
      }

      bool found_intersection = rt.trace_material(scene, ray, subsurface_material, intersection, smp);

      if (found_intersection) {
        ray.max_t = intersection.t;
      }

      ETX_VALIDATE(ray.max_t);

      SpectralResponse tr = spectrum_exp(-ray.max_t * extinction);

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

  SpectralResponse build_path(Sampler& smp, Ray ray, PathData& path_data, Payload& payload, const EmitterSample& emitter_sample, GBuffer& gbuffer, PathVertex& curr,
    PathVertex& prev) const {
    ETX_VALIDATE(payload.throughput);

    const auto& scene = rt.scene();

    uint32_t subsurface_material = kInvalidIndex;

    Intersection intersection = {};
    MediumSample medium_sample = {};
    for (uint32_t path_length = 0; running() && (path_length < scene.options.max_path_length);) {
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
        const MediumInstance medium_inst = make_medium_instance(medium, payload.spect, payload.medium_index);
        handle_medium(emitter_sample, first_interaction, medium.enable_explicit_connections, medium_sample.pos, medium_inst, payload, ray, smp, path_data, curr, prev);
        should_break = false;
      } else if (step == StepResult::IntersectionFound) {
        bool from_subsurface = subsurface_material != kInvalidIndex;
        if (from_subsurface) {
          intersection.material_index = scene.defaults.subsurface_scatter_material;
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
        curr.intersection.w_i = ray.d;
        curr.intersection.pos = ray.o;  // Store ray origin, direction is in w_i
        if (payload.path_distance > 0.0f) {
          payload.d_vcm *= sqr(payload.path_distance);
          payload.path_distance = 0.0f;
        }
        curr.pdf.d_vcm = payload.d_vcm;
        curr.pdf.d_vc = payload.d_vc;
        path_data.camera_path_size += 1u;
#if (ETX_INCLUDE_CAMERA_PATH)
        path_data.camera_path.back() = prev;
        path_data.camera_path.emplace_back(curr);
#endif
        payload.result += direct_hit_environment_emitter(curr, prev, path_data, payload.spect, smp, path_length == 0);
      }

      if (should_break || random_continue(path_length, scene.options.random_path_termination, payload.eta, smp, payload.throughput) == false) {
        break;
      }

      path_length += 1;
    }

    return payload.result;
  }

  SpectralResponse build_camera_path(Sampler& smp, SpectralQuery spect, const float2& uv, PathData& path_data, GBuffer& gbuffer, const uint2& pixel, uint32_t iteration) const {
#if (ETX_INCLUDE_CAMERA_PATH)
    path_data.camera_path.clear();
#endif

    auto ray = generate_ray(rt.camera(), uv, smp.next_2d());
    auto eval = film_evaluate_out(spect, rt.camera(), ray);

    PathVertex prev = {PathVertex::Class::Camera};
    prev.throughput = {spect, 1.0f};
    prev.connectible = true;

    PathVertex curr = {PathVertex::Class::Camera};
    curr.medium = {.index = rt.camera().medium_index};
    curr.throughput = {spect, 1.0f};
    curr.connectible = true;
    curr.intersection.pos = ray.o;
    curr.intersection.nrm = eval.normal;
    curr.intersection.w_i = ray.d;
    curr.pdf.bsdf_sample_next = eval.pdf_dir;

    path_data.camera_path_size = 1u;

#if (ETX_INCLUDE_CAMERA_PATH)
    path_data.camera_path.emplace_back(curr);
#endif

    Payload payload = {
      .spect = spect,
      .result = {spect, 0.0f},
      .throughput = curr.throughput,
      .eta = 1.0f,
      .d_vcm = safe_div(1.0f, eval.pdf_dir),
      .d_vc = 0.0f,
      .medium_index = curr.medium.index,
      .mode = PathSource::Camera,
      .pixel = pixel,
      .iteration = iteration,
      .use_blue_noise = enable_blue_noise,
    };

    return build_path(smp, ray, path_data, payload, {}, gbuffer, curr, prev);
  }

  SpectralResponse build_emitter_path(Sampler& smp, SpectralQuery spect, PathData& path_data) const {
    path_data.emitter_path.clear();
    path_data.emitter_path_size = 0;

    const auto& emitter_sample = sample_emission(spect, smp);
    if ((emitter_sample.pdf_area == 0.0f) || (emitter_sample.pdf_dir == 0.0f) || (emitter_sample.value.is_zero())) {
      return {spect, 0.0f};
    }

    PathVertex prev = {PathVertex::Class::Emitter};
    prev.throughput = {spect, 1.0f};
    prev.connectible = true;

    PathVertex curr = {PathVertex::Class::Emitter};
    curr.intersection.triangle_index = emitter_sample.triangle_index;
    curr.intersection.barycentric = emitter_sample.barycentric;
    curr.intersection.pos = emitter_sample.origin;
    curr.intersection.nrm = emitter_sample.normal;
    curr.intersection.w_i = emitter_sample.direction;
    curr.intersection.emitter_index = emitter_sample.emitter_index;
    curr.intersection.instance_index = emitter_sample.instance_index;
    curr.medium = {.index = emitter_sample.medium_index};
    curr.throughput = emitter_sample.value;
    curr.pdf.bsdf_sample_next = emitter_sample.pdf_dir;
    curr.connectible = true;
    path_data.emitter_path.emplace_back(curr);
    path_data.emitter_path_size = 1u;

    GBuffer gbuffer = {};

    Payload payload = {
      .spect = spect,
      .result = {spect, 0.0f},
      .throughput = curr.throughput * dot(emitter_sample.direction, curr.intersection.nrm) / (emitter_sample.pdf_dir * emitter_sample.pdf_area * emitter_sample.pdf_sample),
      .eta = 1.0f,
      .d_vcm = emitter_sample.is_distant ? safe_div(1.0f, emitter_sample.pdf_area) : safe_div(1.0f, emitter_sample.pdf_dir),
      .d_vc = emitter_sample.is_delta ? 0.0f
                                      : safe_div(emitter_sample.is_distant ? 1.0f : dot(emitter_sample.direction, curr.intersection.nrm),
                                          emitter_sample.pdf_dir * emitter_sample.pdf_area * emitter_sample.pdf_sample),
      .medium_index = curr.medium.index,
      .mode = PathSource::Light,
    };

    Ray ray = {
      offset_ray(emitter_sample.origin, curr.intersection.nrm),
      emitter_sample.direction,
      kRayEpsilon,
      kMaxFloat,
    };
    return build_path(smp, ray, path_data, payload, emitter_sample, gbuffer, curr, prev);
  }

  float mis_weight_camera_to_light(const PathVertex& z_curr, const PathVertex& z_prev, PathData& path_data, SpectralQuery spect, const EmitterSample& emitter_sample,
    const float sampling_pdf, const float bsdf_eval_pdf, Sampler& smp) const {
    if (enable_mis == false) {
      return 1.0f;
    }

    if (mode == Mode::PathTracing) {
      float p_connect = sampling_pdf;
      ETX_VALIDATE(p_connect);
      float p_direct = emitter_sample.is_delta ? 0.0f : bsdf_eval_pdf;
      ETX_VALIDATE(p_direct);
      float result = power_heuristic(p_connect, p_direct);
      ETX_VALIDATE(result);
      return result;
    }

    const auto& scene = rt.scene();

    float3 z_curr_geo_n = z_curr.is_surface_interaction()
                            ? scene_triangle_world_geometric_normal(scene, scene.triangles[z_curr.intersection.triangle_index], z_curr.intersection.instance_index)
                            : float3{};
    float reverse_pdf = 0.0f;
    if (z_curr.is_surface_interaction()) {
      const auto& material = scene.materials[z_curr.intersection.material_index];
      const BSDFData data = {spect, z_prev.medium.index, PathSource::Camera, z_curr.intersection, z_curr.intersection.w_i};
      reverse_pdf = bsdf::reverse_pdf(data, emitter_sample.direction, material, smp);
    } else if (z_curr.is_medium_interaction()) {
      reverse_pdf = phase_function(emitter_sample.direction, z_curr.intersection.w_i, z_curr.medium.anisotropy);
    }
    ETX_VALIDATE(reverse_pdf);

    const float w_light = emitter_sample.is_delta ? 0.0f : safe_div(bsdf_eval_pdf, sampling_pdf);
    const float camera_factor = z_curr.is_surface_interaction() ? fabsf(dot(emitter_sample.direction, z_curr_geo_n)) : 1.0f;
    const float emitter_cosine = fabsf(dot(emitter_sample.direction, emitter_sample.normal));
    const float density_ratio = safe_div(emitter_sample.pdf_dir * emitter_cosine, emitter_sample.pdf_dir_out * camera_factor);
    const float adjacent_connection = ((mode == Mode::BDPTFull) || (path_data.camera_path_length() == 1u)) ? z_curr.pdf.d_vcm : 0.0f;
    const float w_camera = safe_div(adjacent_connection + z_curr.pdf.d_vc * reverse_pdf, density_ratio);
    const float result = 1.0f / (1.0f + w_light + w_camera);
    ETX_VALIDATE(result);
    return result;
  }

  float mis_weight_light_to_camera(SpectralQuery spect, const PathData& path_data, const PathVertex& y_curr, const PathVertex& y_prev, const PathVertex& sampled_camera_vertex,
    Sampler& smp) const {
    if ((enable_mis == false) || (mode == Mode::LightTracing)) {
      return 1.0f;
    }

    const auto& scene = rt.scene();

    float curr_from_camera = film_pdf_out(rt.camera(), sampled_camera_vertex.intersection.pos, y_curr.intersection.pos);
    curr_from_camera = PathVertex::convert_solid_angle_pdf_to_area(curr_from_camera, sampled_camera_vertex, y_curr);

    float reverse_pdf = 0.0f;
    if (y_curr.is_surface_interaction()) {
      const auto& material = scene.materials[y_curr.intersection.material_index];
      const BSDFData data = {spect, y_prev.medium.index, PathSource::Light, y_curr.intersection, y_curr.intersection.w_i};
      reverse_pdf = bsdf::reverse_pdf(data, sampled_camera_vertex.intersection.w_i, material, smp);
    } else if (y_curr.is_medium_interaction()) {
      reverse_pdf = phase_function(sampled_camera_vertex.intersection.w_i, y_curr.intersection.w_i, y_curr.medium.anisotropy);
    }
    ETX_VALIDATE(reverse_pdf);

    const float adjacent_connection = ((mode == Mode::BDPTFull) || (path_data.emitter_path_length() == 1u)) ? y_curr.pdf.d_vcm : 0.0f;
    const float competing_density = curr_from_camera * (adjacent_connection + y_curr.pdf.d_vc * reverse_pdf);
    ETX_VALIDATE(competing_density);
    return 1.0f / (1.0f + competing_density);
  }

  float mis_weight_camera_to_light_path(const PathVertex& z_curr, const PathVertex& z_prev, PathData& c, SpectralQuery spect, uint32_t light_s, Sampler& smp) const {
    if (enable_mis == false) {
      return 1.0f;
    }

    const auto& scene = rt.scene();
    const PathVertex& y_curr = c.emitter_path[light_s];
    const PathVertex& y_prev = c.emitter_path[light_s - 1];

    float3 camera_to_light = {};
    if (PathVertex::safe_normalize(y_curr.intersection.pos, z_curr.intersection.pos, camera_to_light) == false) {
      return 0.0f;
    }

    const Material* y_curr_material = y_curr.is_surface_interaction() ? &scene.materials[y_curr.intersection.material_index] : nullptr;
    const float light_area_pdf = PathVertex::pdf_area(spect, PathSource::Light, y_prev, y_curr, z_curr, y_curr_material, smp);
    ETX_VALIDATE(light_area_pdf);

    const Material* z_curr_material = z_curr.is_surface_interaction() ? &scene.materials[z_curr.intersection.material_index] : nullptr;
    const float camera_area_pdf = PathVertex::pdf_area(spect, PathSource::Camera, z_prev, z_curr, y_curr, z_curr_material, smp);
    ETX_VALIDATE(camera_area_pdf);

    float camera_reverse_pdf = 0.0f;
    if (z_curr.is_surface_interaction()) {
      const BSDFData data = {spect, z_prev.medium.index, PathSource::Camera, z_curr.intersection, z_curr.intersection.w_i};
      camera_reverse_pdf = bsdf::reverse_pdf(data, camera_to_light, *z_curr_material, smp);
    } else if (z_curr.is_medium_interaction()) {
      camera_reverse_pdf = phase_function(camera_to_light, z_curr.intersection.w_i, z_curr.medium.anisotropy);
    }
    ETX_VALIDATE(camera_reverse_pdf);

    float light_reverse_pdf = 0.0f;
    if (y_curr.is_surface_interaction()) {
      const BSDFData data = {spect, y_prev.medium.index, PathSource::Light, y_curr.intersection, y_curr.intersection.w_i};
      light_reverse_pdf = bsdf::reverse_pdf(data, -camera_to_light, *y_curr_material, smp);
    } else if (y_curr.is_medium_interaction()) {
      light_reverse_pdf = phase_function(-camera_to_light, y_curr.intersection.w_i, y_curr.medium.anisotropy);
    }
    ETX_VALIDATE(light_reverse_pdf);

    const float w_light = camera_area_pdf * (y_curr.pdf.d_vcm + y_curr.pdf.d_vc * light_reverse_pdf);
    const float w_camera = light_area_pdf * (z_curr.pdf.d_vcm + z_curr.pdf.d_vc * camera_reverse_pdf);
    ETX_VALIDATE(w_light);
    ETX_VALIDATE(w_camera);

    return 1.0f / (1.0f + w_camera + w_light);
  }

  float mis_weight_direct_hit(const PathVertex& z_curr, const float emitter_position_pdf, const float emitter_direction_pdf) const {
    const float competing_density = z_curr.pdf.d_vcm * emitter_position_pdf + z_curr.pdf.d_vc * emitter_direction_pdf;
    ETX_VALIDATE(competing_density);
    return 1.0f / (1.0f + competing_density);
  }

  SpectralResponse direct_hit_area_emitter(const PathVertex& z_curr, const PathVertex& z_prev, PathData& path_data, SpectralQuery spect, Sampler& smp, bool force) const {
    if ((force == false) && (enable_direct_hit == false))
      return {spect, 0.0f};

    if (z_curr.is_emitter() == false) {
      return {spect, 0.0f};
    }
    ETX_ASSERT(z_curr.is_specific_emitter());

    const auto& scene = rt.scene();

    const uint32_t target_path_length = path_data.camera_path_length();
    if ((target_path_length > scene.options.max_path_length) || (target_path_length < scene.options.min_path_length))
      return {spect, 0.0f};

    const auto& emitter_instance = scene.emitter_instances[z_curr.intersection.emitter_index];
    ETX_ASSERT(emitter_instance.is_local());
    const bool directly_visible = path_data.camera_path_length() <= 1u;
    EmitterRadianceQuery q = {
      .source_position = z_prev.intersection.pos,
      .target_position = z_curr.intersection.pos,
      .uv = z_curr.intersection.tex,
      .directly_visible = directly_visible,
    };

    float pdf_dir = 0.0f;
    float pdf_area = 0.0f;
    float pdf_dir_out = 0.0f;
    auto emitter_value = emitter_get_radiance(emitter_instance, spect, q, pdf_area, pdf_dir, pdf_dir_out);

    if (pdf_dir == 0.0f) {
      return {spect, 0.0f};
    }

    float mis_weight = 1.0f;

    if (enable_mis && (path_data.camera_path_size > 2u)) {
      if (mode == Mode::PathTracing) {
        float p_sample = emitter_discrete_pdf(emitter_instance);
        float p_connect = pdf_dir * p_sample;
        ETX_VALIDATE(p_connect);
        mis_weight = z_prev.connectible ? power_heuristic(z_prev.pdf.bsdf_sample_next, p_connect) : 1.0f;
        ETX_VALIDATE(mis_weight);
      } else {
        const float emitter_selection_pdf = emitter_discrete_pdf(emitter_instance);
        const float emitter_position_pdf = pdf_area * emitter_selection_pdf;
        const float emitter_direction_pdf = pdf_dir_out * emitter_selection_pdf;
        mis_weight = mis_weight_direct_hit(z_curr, emitter_position_pdf, emitter_direction_pdf);
      }
    }

    return emitter_value * z_curr.throughput * mis_weight;
  }

  SpectralResponse direct_hit_environment_emitter(const PathVertex& z_curr, const PathVertex& z_prev, PathData& path_data, SpectralQuery spect, Sampler& smp, bool force) const {
    if ((force == false) && (enable_direct_hit == false))
      return {spect, 0.0f};

    ETX_ASSERT(z_curr.is_emitter() && (z_curr.is_specific_emitter() == false));

    const auto& scene = rt.scene();
    uint32_t environment_emitter_count = environment_emitter_shared_count();
    if (environment_emitter_count == 0u)
      return {spect, 0.0f};

    const uint32_t target_path_length = path_data.camera_path_length();
    if ((target_path_length > scene.options.max_path_length) || (target_path_length < scene.options.min_path_length))
      return {spect, 0.0f};

    EmitterRadianceQuery q = {
      .direction = z_curr.intersection.w_i,  // Use stored ray direction for environment emitters
      .directly_visible = path_data.camera_path_length() <= 1,
    };

    SpectralResponse accumulated_emitter_value = {spect, 0.0f};
    for (uint32_t ie = 0u; ie < environment_emitter_count; ++ie) {
      uint32_t emitter_index = kInvalidIndex;
      if (environment_emitter_shared_try_load_index(ie, emitter_index) == false) {
        continue;
      }

      Emitter emitter_instance = {};
      if (try_load_emitter_instance(emitter_index, emitter_instance) == false) {
        continue;
      }

      float local_pdf_area = 0.0f;
      float local_pdf_dir = 0.0f;
      float local_pdf_dir_out = 0.0f;
      auto value = emitter_get_radiance(emitter_instance, spect, q, local_pdf_area, local_pdf_dir, local_pdf_dir_out);

      float this_weight = 1.0f;
      if ((mode == Mode::PathTracing) && z_prev.connectible && (path_data.camera_path_length() > 1u)) {
        float local_pdf_sample = emitter_discrete_pdf(emitter_instance);
        float this_p_connect = local_pdf_dir * local_pdf_sample;
        ETX_VALIDATE(this_p_connect);
        this_weight = power_heuristic(z_prev.pdf.bsdf_sample_next, this_p_connect);
        ETX_VALIDATE(this_weight);
      }
      accumulated_emitter_value += value * this_weight;
      ETX_VALIDATE(accumulated_emitter_value);
    }

    if (accumulated_emitter_value.is_zero())
      return accumulated_emitter_value;

    float mis_weight = 1.0f;
    if (enable_mis && (path_data.camera_path_length() > 1u) && (mode != Mode::PathTracing)) {
      auto [p_from, p_sample] = PathVertex::pdf_for_environment_emitter(spect, z_curr.intersection.w_i, z_prev);
      mis_weight = mis_weight_direct_hit(z_curr, p_sample, p_from * p_sample);
    }

    return accumulated_emitter_value * z_curr.throughput * mis_weight;
  }

  SpectralResponse connect_camera_to_light(const PathVertex& z_curr, const PathVertex& z_prev, Sampler& smp, PathData& path_data, SpectralQuery spect) const {
    const auto& scene = rt.scene();

    const uint32_t connection_len = path_data.camera_path_length() + 1u;
    const bool invalid_path_length = (connection_len < scene.options.min_path_length) || (connection_len > scene.options.max_path_length);
    if (invalid_path_length || (enable_connect_to_light == false) || (mode == Mode::LightTracing))
      return {spect, 0.0f};

    const bool source_is_surface = z_curr.is_surface_interaction();
    EmitterSampleQuery query = {
      .spect = spect,
      .source_type = source_is_surface ? InteractionType::Surface : InteractionType::Medium,
      .source_position = z_curr.intersection.pos,
      .source_normal = source_is_surface ? z_curr.intersection.nrm : float3{},
    };
    auto emitter_sample = sample_emitter(scene.light_sampling_method(), query, smp);
    if (emitter_sample.value.is_zero() || (emitter_sample.pdf_dir == 0.0f)) {
      return {spect, 0.0f};
    }

    auto dp = emitter_sample.origin - z_curr.intersection.pos;
    if (dot(dp, dp) <= kEpsilon) {
      return {spect, 0.0f};
    }

    const Material* z_curr_material = z_curr.is_surface_interaction() ? &rt.scene().materials[z_curr.intersection.material_index] : nullptr;
    float3 z_curr_geo_n = z_curr.is_surface_interaction()
                            ? scene_triangle_world_geometric_normal(rt.scene(), rt.scene().triangles[z_curr.intersection.triangle_index], z_curr.intersection.instance_index)
                            : float3{};
    auto bsdf_eval = z_curr.bsdf_in_direction(spect, PathSource::Camera, emitter_sample.direction, z_curr_material, z_curr_geo_n, smp);
    if (bsdf_eval.bsdf.is_zero()) {
      return {spect, 0.0f};
    }

    PathVertex sampled_vertex = {PathVertex::Class::Emitter};
    sampled_vertex.intersection.w_i = normalize(dp);
    sampled_vertex.intersection.pos = emitter_sample.origin;
    sampled_vertex.intersection.nrm = emitter_sample.normal;
    sampled_vertex.intersection.triangle_index = emitter_sample.triangle_index;
    sampled_vertex.intersection.emitter_index = emitter_sample.emitter_index;
    sampled_vertex.intersection.instance_index = emitter_sample.instance_index;

    float sampling_pdf = emitter_sample.pdf_dir * emitter_sample.pdf_sample;
    SpectralResponse emitter_throughput = emitter_sample.value / sampling_pdf;
    ETX_VALIDATE(emitter_throughput);

    float3 shadow_origin = z_curr.intersection.pos;
    if (z_curr.is_surface_interaction()) {
      const auto& tri = scene.triangles[z_curr.intersection.triangle_index];
      shadow_origin =
        shading_pos(scene, tri, z_curr.intersection.barycentric, normalize(sampled_vertex.intersection.pos - z_curr.intersection.pos), z_curr.intersection.instance_index);
    }

    SpectralResponse tr = rt.trace_transmittance(spect, scene, shadow_origin, sampled_vertex.intersection.pos, z_curr.medium, smp);
    float weight = mis_weight_camera_to_light(z_curr, z_prev, path_data, spect, emitter_sample, sampling_pdf, bsdf_eval.pdf, smp);
    return z_curr.throughput * bsdf_eval.bsdf * emitter_throughput * tr * weight;
  }

  SpectralResponse connect_light_to_camera(Sampler& smp, PathData& path_data, const PathVertex& y_curr, const PathVertex& y_prev, SpectralQuery spect,
    CameraSample& camera_sample) const {
    const auto& scene = rt.scene();

    const uint32_t target_path_length = path_data.emitter_path_length() + 1u;
    if ((mode == Mode::PathTracing) || (enable_connect_to_camera == false) || (target_path_length < scene.options.min_path_length) ||
        (target_path_length > scene.options.max_path_length))
      return {spect, 0.0f};

    const auto& camera = rt.camera();
    camera_sample = sample_film(smp, camera, y_curr.intersection.pos);
    if (camera_sample.valid() == false) {
      return {spect, 0.0f};
    }
    const float2 splat_uv = pixel_filter_splat_uv(camera_sample.uv, camera.film_size, sample_pixel_filter_offset(scene.pixel_sampler, smp.next_2d()));
    if (pixel_filter_contains_uv(splat_uv) == false) {
      return {spect, 0.0f};
    }

    float len = length(camera_sample.position - y_curr.intersection.pos);
    float direction_scale = camera_clip_direction_scale(camera, camera_sample.direction);
    float near_extent = (camera.clip_near > 0.0f) ? camera.clip_near / direction_scale : 0.0f;
    float far_extent = (camera.clip_far > 0.0f) ? camera.clip_far / direction_scale : kMaxFloat;
    if ((len < near_extent) || (len > far_extent)) {
      return {spect, 0.0f};
    }

    ETX_VALIDATE(camera_sample.weight);

    PathVertex sampled_vertex = {PathVertex::Class::Camera};
    sampled_vertex.intersection.pos = camera_sample.position;
    sampled_vertex.intersection.nrm = camera_sample.normal;
    sampled_vertex.intersection.w_i = camera_sample.direction;

    const Material* y_curr_material = y_curr.is_surface_interaction() ? &scene.materials[y_curr.intersection.material_index] : nullptr;
    float3 y_curr_geo_n = y_curr.is_surface_interaction()
                            ? scene_triangle_world_geometric_normal(scene, scene.triangles[y_curr.intersection.triangle_index], y_curr.intersection.instance_index)
                            : float3{};
    auto bsdf = y_curr.bsdf_in_direction(spect, PathSource::Light, camera_sample.direction, y_curr_material, y_curr_geo_n, smp).bsdf;
    if (bsdf.is_zero()) {
      return {spect, 0.0f};
    }
    float weight = mis_weight_light_to_camera(spect, path_data, y_curr, y_prev, sampled_vertex, smp);

    SpectralResponse splat = y_curr.throughput * bsdf * (camera_sample.weight * weight);
    ETX_VALIDATE(splat);

    if (splat.is_zero() == false) {
      float3 clip_pos = y_curr.intersection.pos + camera_sample.direction * fmaxf(0.0f, len - near_extent);
      splat *= local_transmittance(spect, smp, y_curr, clip_pos);
    }

    camera_sample.uv = splat_uv;
    return splat;
  }

  SpectralResponse local_transmittance(SpectralQuery spect, Sampler& smp, const PathVertex& p0, const float3& p1) const {
    auto& scene = rt.scene();
    float3 origin = p0.intersection.pos;
    if (p0.is_surface_interaction()) {
      const auto& tri = scene.triangles[p0.intersection.triangle_index];
      origin = shading_pos(scene, tri, p0.intersection.barycentric, normalize(p1 - p0.intersection.pos), p0.intersection.instance_index);
    }
    return rt.trace_transmittance(spect, scene, origin, p1, p0.medium, smp);
  }

  void build_options(Options& options) const {
    options.options.clear();

    if (mode_locked) {
      return;
    }

    options.set_integral("bdpt-mode", mode, "Mode", Option::Meta::EnumValue, {BDPTMode::PathTracing, BDPTMode::BDPTFull}).name_getter = [](uint32_t index) -> std::string {
      return bdpt_mode_display_name(static_cast<BDPTMode>(index));
    };
  }

  void start(const Options& opt) {
    if (mode_locked == false) {
      mode = opt.get_integral("bdpt-mode", mode);
    }

    const auto& scene = rt.scene();
    enable_direct_hit = scene.strategy_enabled(Scene::Strategy::DirectHit);
    enable_connect_to_camera = scene.strategy_enabled(Scene::Strategy::ConnectToCamera);
    enable_connect_to_light = scene.strategy_enabled(Scene::Strategy::ConnectToLight);
    enable_connect_vertices = scene.strategy_enabled(Scene::Strategy::ConnectVertices);
    enable_mis = scene.multiple_importance_sampling();
    enable_blue_noise = scene.blue_noise();
    for (auto& path_data : per_thread_path_data) {
      path_data.emitter_path.reserve(2llu + rt.scene().options.max_path_length);
    }

    status = {};
    iteration_time = {};
    rt.film().clear(Film::ClearEverything);
    current_task = rt.scheduler().schedule(rt.film().current_pixel_count(), this);
  }
};

CPUBidirectional::CPUBidirectional(Raytracing& rt)
  : CPUBidirectional(rt, BDPTMode::BDPTFast, Integrator::Type::Bidirectional) {
}

CPUBidirectional::CPUBidirectional(Raytracing& rt, const BDPTMode initial_mode, const Integrator::Type exposed_type)
  : Integrator(rt)
  , _type(exposed_type) {
  ETX_CRITICAL((exposed_type == Integrator::Type::PathTracing) || (exposed_type == Integrator::Type::Bidirectional));
  ETX_CRITICAL(bdpt_mode_valid(initial_mode));
  ETX_CRITICAL((exposed_type == Integrator::Type::Bidirectional) || (initial_mode == BDPTMode::PathTracing));
  ETX_PIMPL_INIT(CPUBidirectional, rt, &current_state);
  _private->mode = initial_mode;
  _private->mode_locked = exposed_type == Integrator::Type::PathTracing;
  _private->build_options(integrator_options);
}

CPUBidirectional::~CPUBidirectional() {
  if (current_state != State::Stopped) {
    stop(Stop::Immediate);
  }
  ETX_PIMPL_CLEANUP(CPUBidirectional);
}

void CPUBidirectional::run() {
  stop(Stop::Immediate);

  if (can_run()) {
    current_state = State::Running;
    _private->start(integrator_options);
  }
}

void CPUBidirectional::update() {
  ETX_PROFILER_SCOPE();
  if ((current_state == State::Stopped) || (rt.scheduler().completed(_private->current_task) == false)) {
    return;
  }

  rt.scheduler().wait_task(_private->current_task);
  const auto& scene = rt.scene();
  rt.film().commit_iteration(scene.options.radiance_clamp);
  _private->completed();

  if (current_state == State::WaitingForCompletion) {
    rt.scheduler().release(_private->current_task);
    current_state = Integrator::State::Stopped;
  } else if (_private->status.current_iteration + 1u <= rt.sample_limit()) {
    rt.scheduler().restart(_private->current_task);
  } else {
    rt.scheduler().release(_private->current_task);
    current_state = Integrator::State::Stopped;
  }
}

void CPUBidirectional::stop(Stop st) {
  if (current_state == State::Stopped) {
    return;
  }

  if (st == Stop::Immediate) {
    current_state = State::Stopped;
    rt.scheduler().wait_and_release(_private->current_task);
  } else {
    current_state = State::WaitingForCompletion;
  }
}

void CPUBidirectional::update_options() {
  if (current_state == State::Running) {
    run();
  }
}

void CPUBidirectional::sync_from_options(const Options& options) {
  if (_private->mode_locked == false) {
    _private->mode = options.get_integral("bdpt-mode", _private->mode);
  }
  _private->build_options(integrator_options);
}

uint32_t CPUBidirectional::supported_strategies() const {
  if (_type == Integrator::Type::PathTracing) {
    return Scene::Strategy::DirectHit | Scene::Strategy::ConnectToLight;
  }
  return Scene::Strategy::DirectHit | Scene::Strategy::ConnectToLight | Scene::Strategy::ConnectToCamera | Scene::Strategy::ConnectVertices;
}

const Integrator::Status& CPUBidirectional::status() const {
  return _private->status;
}

}  // namespace etx
