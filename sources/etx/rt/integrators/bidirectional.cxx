#include <etx/core/core.hxx>
#include <etx/render/host/film.hxx>
#include <etx/rt/integrators/bidirectional.hxx>
#include <etx/rt/integrators/bidirectional_shared.hxx>
#include <atomic>

#include <etx/rt/shared/path_tracing_shared.hxx>

namespace etx {

namespace {

inline float safe_div(float a, float b) {
  float result = ((a == 0.0f) ? 1.0f : a) / ((b == 0.0f) ? 1.0f : b);
  ETX_VALIDATE(result);
  return result;
}

}  // namespace

struct CPUBidirectionalImpl : public Task {
  Raytracing& rt;
  std::vector<PathData> per_thread_path_data;
  std::atomic<Integrator::State>* state = {};
  TimeMeasure iteration_time = {};
  Handle current_task = {};
  Integrator::Status status = {};

  bool conn_direct_hit = true;
  bool conn_connect_to_light = true;
  bool conn_connect_to_camera = true;
  bool conn_connect_vertices = true;
  bool conn_mis = true;

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
      auto smp = Sampler(i, status.current_iteration);

      uint2 pixel = {};
      if (film.active_pixel(i, pixel) == false)
        return;

      auto spect = scene.spectral ? SpectralQuery::spectral_sample(smp.next()) : SpectralQuery::sample();

      build_emitter_path(smp, spect, path_data);

      float2 uv = film.sample(rt.scene(), status.current_iteration == 0u ? PixelFilter::empty() : rt.scene().pixel_sampler, pixel, smp.next_2d());
      GBuffer gbuffer = {};
      SpectralResponse result = build_camera_path(smp, spect, uv, path_data, gbuffer);

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

  void update_emitter_path_pdfs(PathVertex& curr, float& prev_pdf_forward, const EmitterSample& em) const {
    const auto& scene = rt.scene();

    float total_pdf = 0.0f;
    float total_weight = 0.0f;
    for (uint32_t ei = 0, ee = scene.environment_emitters.count; ei < ee; ++ei) {
      float weight = scene.emitters_distribution.values[ei].value;
      total_weight += weight;
      float pdf = emitter_pdf_in_dist(scene.emitters[em.emitter_index], em.direction, scene);
      total_pdf += pdf * weight;
    }
    prev_pdf_forward = total_pdf / (total_weight * float(scene.environment_emitters.count));
    ETX_VALIDATE(prev_pdf_forward);

    curr.pdf.forward = em.pdf_area;
    if (curr.is_surface_interaction()) {
      const auto& tri = scene.triangles[curr.intersection.triangle_index];
      curr.pdf.forward *= fabsf(dot(em.direction, tri.geo_n));
      ETX_VALIDATE(curr.pdf.forward);
    }
  }

  SpectralResponse connect_to_light_path(const PathVertex& z_i, const PathVertex& z_prev, Sampler& smp, SpectralQuery spect, PathData& path_data) const {
    const auto& scene = rt.scene();

    SpectralResponse result = {spect, 0.0f};
    if ((conn_connect_vertices == false) || (z_i.connectible == false) || (path_data.camera_path_length() + 1u >= scene.max_path_length))
      return result;

    bool is_medium = z_i.is_medium_interaction();

    for (uint64_t light_s = 2, light_s_e = path_data.emitter_path.size(); running() && (light_s < light_s_e); ++light_s) {
      if (path_data.camera_path_length() + light_s > scene.max_path_length + 1u)
        break;

      const auto& y_i = path_data.emitter_path[light_s];

      if (y_i.connectible == false)
        continue;

      auto dw = z_i.intersection.pos - y_i.intersection.pos;
      float dwl = dot(dw, dw);
      dw *= 1.0f / std::sqrt(dwl);

      float g_term = 1.0f / dwl;

      // G term = abs(cos(dw, y_i.nrm) * cos(dw, z_i.nrm)) / dwl;
      // cosines already accounted in "bsdf", 1.0 / dwl multiplied below
      SpectralResponse connect_result = y_i.throughput * y_i.bsdf_in_direction(spect, PathSource::Light, dw, rt.scene(), smp) *   //
                                        z_i.throughput * z_i.bsdf_in_direction(spect, PathSource::Camera, -dw, rt.scene(), smp);  //
      ETX_VALIDATE(connect_result);

      if (connect_result.is_zero())
        continue;

      SpectralResponse tr = local_transmittance(spect, smp, y_i, z_i);
      ETX_VALIDATE(connect_result);

      float weight = mis_weight_connect(z_i, z_prev, path_data, spect, light_s, smp);
      ETX_VALIDATE(weight);

      result += connect_result * tr * (weight * g_term);
      ETX_VALIDATE(result);
    }

    return result;
  }

  void handle_medium(const EmitterSample& emitter_sample, const uint32_t path_length, const bool explicit_connections, const float anisotropy,  //
    const float3& medium_sample_pos, Payload& payload, Ray& ray, Sampler& smp, PathData& path_data, PathVertex& curr, PathVertex& prev) {
    const auto& scene = rt.scene();

    float3 w_o = medium::sample_phase_function(ray.d, anisotropy, smp);
    float pdf_fwd = medium::phase_function(ray.d, w_o, anisotropy);
    float pdf_bck = medium::phase_function(w_o, ray.d, anisotropy);

    curr = PathVertex{medium_sample_pos, ray.d, payload.medium_index, anisotropy};

    path_data.camera_path_size += uint32_t(payload.mode == PathSource::Camera);
    path_data.emitter_path_size += uint32_t(payload.mode == PathSource::Light);

    curr.delta_emitter = prev.delta_emitter;
    curr.throughput = payload.throughput;
    curr.connectible = explicit_connections;
    curr.pdf.forward = prev.pdf_solid_angle_to_area(payload.pdf_dir, curr);
    prev.pdf.backward = curr.pdf_solid_angle_to_area(pdf_bck, prev);

    ray.o = medium_sample_pos;
    ray.d = w_o;
    ray.max_t = kMaxFloat;

    payload.pdf_dir = pdf_fwd;

    if (payload.mode == PathSource::Camera) {
      precompute_camera_mis(prev, path_data.camera_path_size, path_data.camera_history);
    } else if (payload.mode == PathSource::Light) {
      if (emitter_sample.is_distant && (path_length == 0)) {
        update_emitter_path_pdfs(curr, prev.pdf.forward, emitter_sample);
      }
      precompute_light_mis(prev, path_data.emitter_path_size, path_data.emitter_history);
      if (explicit_connections) {
        path_data.emitter_path.back() = prev;
        path_data.emitter_path.emplace_back(curr);
      }
    }

    if (payload.mode == PathSource::Camera) {
      payload.result += direct_hit(curr, prev, path_data, payload.spect, smp, false);
      payload.result += connect_to_light(curr, prev, smp, path_data, payload.spect);
      payload.result += connect_to_light_path(curr, prev, smp, payload.spect, path_data);
    } else {
      CameraSample camera_sample = {};
      auto splat = connect_to_camera(smp, path_data, curr, prev, payload.spect, camera_sample);
      auto xyz = splat.to_rgb();
      rt.film().atomic_add(Film::LightIteration, xyz, camera_sample.uv);
    }
  }

  InteractionResult handle_surface(const Intersection& a_intersection, const EmitterSample& emitter_sample, const uint32_t path_length, Payload& payload, Ray& ray, Sampler& smp,
    PathData& path_data, PathVertex& curr, PathVertex& prev, GBuffer& gbuffer, bool is_subsurface_interaction) const {
    const auto& scene = rt.scene();

    if (scene.materials[a_intersection.material_index].cls == Material::Class::Boundary) {
      const auto& m = scene.materials[a_intersection.material_index];
      payload.medium_index = (dot(a_intersection.nrm, ray.d) < 0.0f) ? m.int_medium : m.ext_medium;
      ray.o = shading_pos(scene.vertices, scene.triangles[a_intersection.triangle_index], a_intersection.barycentric, ray.d);
      ray.max_t = kMaxFloat;
      return InteractionResult::Continue;
    }

    BSDFData bsdf_data = {payload.spect, payload.medium_index, payload.mode, a_intersection, a_intersection.w_i};

    if (gbuffer.recorded == false) {
      gbuffer.normal = a_intersection.nrm;
      gbuffer.albedo = bsdf::albedo(bsdf_data, scene.materials[a_intersection.material_index], scene, smp);
      gbuffer.recorded = true;
    }

    auto bsdf_sample = bsdf::sample(bsdf_data, scene.materials[a_intersection.material_index], scene, smp);
    ETX_VALIDATE(bsdf_sample.weight);

    bool subsurface_path = (is_subsurface_interaction == false) &&                                                                    //
                           (scene.materials[a_intersection.material_index].subsurface.cls != SubsurfaceMaterial::Class::Disabled) &&  //
                           (bsdf_sample.properties & BSDFSample::Reflection) && (bsdf_sample.properties & BSDFSample::Diffuse);

    uint32_t material_index = a_intersection.material_index;
    uint32_t medium_index = (bsdf_sample.properties & BSDFSample::MediumChanged) ? bsdf_sample.medium_index : payload.medium_index;

    if (subsurface_path) {
      const auto& sss_material = scene.materials[a_intersection.material_index];
      medium_index = sss_material.int_medium;
      material_index = scene.subsurface_scatter_material;

      const bool diffuse_transmission = sss_material.subsurface.path == SubsurfaceMaterial::Path::Diffuse;
      auto w_o = diffuse_transmission ? sample_cosine_distribution(smp.next_2d(), -a_intersection.nrm, 1.0f) : a_intersection.w_i;

      bsdf_sample.w_o = w_o;
      bsdf_sample.weight = {payload.spect, 1.0f};
      bsdf_sample.pdf = fabsf(dot(w_o, a_intersection.nrm)) / kPi;
      bsdf_sample.eta = 1.0f;
      bsdf_sample.medium_index = medium_index;
      bsdf_sample.properties = BSDFSample::Transmission | BSDFSample::Diffuse | BSDFSample::MediumChanged;
    }

    path_data.camera_path_size += uint32_t(payload.mode == PathSource::Camera);
    path_data.emitter_path_size += uint32_t(payload.mode == PathSource::Light);

    curr = PathVertex{PathVertex::Class::Surface, a_intersection};
    curr.throughput = payload.throughput;
    curr.intersection.material_index = material_index;
    curr.delta_emitter = prev.delta_emitter;
    curr.delta_connection = bsdf_sample.is_delta();
    curr.scene_medium_index = medium_index;
    curr.pdf.forward = prev.pdf_solid_angle_to_area(payload.pdf_dir, curr);
    ETX_VALIDATE(curr.pdf.forward);

    payload.medium_index = medium_index;

    bool terminate_path = false;
    if (bsdf_sample.valid()) {
      float rev_bsdf_pdf = bsdf::reverse_pdf(bsdf_data, bsdf_sample.w_o, scene.materials[material_index], scene, smp);
      prev.pdf.backward = curr.pdf_solid_angle_to_area(rev_bsdf_pdf, prev);
      ETX_VALIDATE(prev.pdf.backward);

      payload.eta *= (payload.mode == PathSource::Camera) ? bsdf_sample.eta : 1.0f;
      ETX_VALIDATE(payload.eta);

      payload.pdf_dir = curr.delta_connection ? 0.0f : bsdf_sample.pdf;
      ETX_VALIDATE(payload.pdf_dir);

      payload.throughput *= bsdf_sample.weight;
      ETX_VALIDATE(payload.throughput);

      const auto& tri = scene.triangles[a_intersection.triangle_index];

      ray.o = shading_pos(scene.vertices, tri, curr.intersection.barycentric, bsdf_sample.w_o);
      ray.d = bsdf_sample.w_o;
      ray.max_t = kMaxFloat;

      if (payload.mode == PathSource::Light) {
        payload.throughput *= fix_shading_normal(tri.geo_n, curr.intersection.nrm, curr.intersection.w_i, bsdf_sample.w_o);
        ETX_VALIDATE(payload.throughput);
      }
    } else {
      terminate_path = true;
    }

    if (payload.mode == PathSource::Light) {
      if (emitter_sample.is_distant && (path_length == 0)) {
        update_emitter_path_pdfs(curr, prev.pdf.forward, emitter_sample);
      }
      precompute_light_mis(prev, path_data.emitter_path_size, path_data.emitter_history);

      path_data.emitter_path.back() = prev;
      path_data.emitter_path.emplace_back(curr);

      CameraSample camera_sample = {};
      auto splat = connect_to_camera(smp, path_data, curr, prev, payload.spect, camera_sample);
      if (splat.is_zero() == false) {
        rt.film().atomic_add(Film::LightIteration, splat.to_rgb(), camera_sample.uv);
      }
    } else if (payload.mode == PathSource::Camera) {
      precompute_camera_mis(prev, path_data.camera_path_size, path_data.camera_history);

      auto sampled_value = direct_hit(curr, prev, path_data, payload.spect, smp, false);
      sampled_value += connect_to_light(curr, prev, smp, path_data, payload.spect);
      sampled_value += connect_to_light_path(curr, prev, smp, payload.spect, path_data);
      payload.result += sampled_value;
    }

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

  StepResult subsurface_step(const uint32_t subsurface_material, Ray& ray, Sampler& smp, Intersection& intersection, Payload& payload, PathData& path_data, PathVertex& curr,
    PathVertex& prev) {
    const auto& scene = rt.scene();

    float anisotropy = 0.0f;
    SpectralResponse extinction = {payload.spect};
    SpectralResponse scattering = {payload.spect};
    SpectralResponse albedo = {payload.spect};

    const auto& mat = scene.materials[subsurface_material];

    if (mat.int_medium == kInvalidIndex) {
      auto color = apply_image(payload.spect, mat.transmittance, intersection.tex, scene, nullptr);
      auto distances = mat.subsurface.scale * apply_image(payload.spect, mat.subsurface, intersection.tex, scene, nullptr);
      subsurface::remap(color.integrated, distances.integrated, albedo.integrated, extinction.integrated, scattering.integrated);
      subsurface::remap_channel(color.value, distances.value, albedo.value, extinction.value, scattering.value);
    } else {
      const Medium& medium = scene.mediums[payload.medium_index];
      anisotropy = medium.phase_function_g;
      scattering = medium.s_scattering(payload.spect);
      extinction = scattering + medium.s_absorption(payload.spect);
      albedo = medium::calculate_albedo(payload.spect, scattering, extinction);
    }

    for (uint32_t counter = 0; counter < 1024 * 4; ++counter) {
      prev = curr;

      SpectralResponse pdf = {};
      uint32_t channel = medium::sample_spectrum_component(payload.spect, albedo, payload.throughput, smp, pdf);
      float sample_t = extinction.component(channel);

      ray.max_t = (sample_t > 0.0f) ? -logf(1.0f - smp.next()) / sample_t : kMaxFloat;
      ETX_VALIDATE(ray.max_t);

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
        if ((payload.mode == PathSource::Light) && (counter > 0)) {
          path_data.emitter_path.emplace_back(curr);
          prev = curr;
        }
        return StepResult::IntersectionFound;
      }

      const float3 medium_sample_pos = ray.o + ray.d * ray.max_t;
      handle_medium({}, 65536, false, anisotropy, medium_sample_pos, payload, ray, smp, path_data, curr, prev);
      if ((payload.mode == PathSource::Light) && (counter == 0)) {
        path_data.emitter_path.back() = prev;
      }
    }

    return StepResult::Nothing;
  }

  SpectralResponse build_path(Sampler& smp, Ray ray, PathData& path_data, Payload& payload, const EmitterSample& emitter_sample, GBuffer& gbuffer, PathVertex& curr,
    PathVertex& prev) {
    ETX_VALIDATE(payload.throughput);

    const auto& scene = rt.scene();

    uint32_t subsurface_material = kInvalidIndex;

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

      if (step == StepResult::SampledMedium) {
        ETX_CRITICAL(payload.medium_index != kInvalidIndex);
        const auto& medium = scene.mediums[payload.medium_index];
        handle_medium(emitter_sample, path_length, medium.enable_explicit_connections, medium.phase_function_g, medium_sample.pos, payload, ray, smp, path_data, curr, prev);
        should_break = false;
      } else if (step == StepResult::IntersectionFound) {
        bool from_subsurface = subsurface_material != kInvalidIndex;
        if (from_subsurface) {
          intersection.material_index = scene.subsurface_scatter_material;
          subsurface_material = kInvalidIndex;
        }

        auto result = handle_surface(intersection, emitter_sample, path_length, payload, ray, smp, path_data, curr, prev, gbuffer, from_subsurface);

        if (result == InteractionResult::SampleSubsurface) {
          subsurface_material = intersection.material_index;
        }

        if (result == InteractionResult::Continue) {
          continue;
        }

        should_break = result == InteractionResult::Break;
      } else if (payload.mode == PathSource::Camera) {
        curr = PathVertex{PathVertex::Class::Emitter};
        curr.scene_medium_index = payload.medium_index;
        curr.throughput = payload.throughput;
        curr.pdf.forward = payload.pdf_dir;
        curr.intersection.w_i = ray.d;
        curr.intersection.pos = ray.o + scene.bounding_sphere_radius * curr.intersection.w_i;
        curr.intersection.nrm = -curr.intersection.w_i;
        path_data.camera_path_size += 1u;
        precompute_camera_mis(prev, path_data.camera_path_size, path_data.camera_history);
        payload.result += direct_hit(curr, prev, path_data, payload.spect, smp, path_length == 0);
      }

      if (should_break || random_continue(path_length, scene.random_path_termination, payload.eta, smp, payload.throughput) == false) {
        break;
      }

      path_length += 1;
    }

    return payload.result;
  }

  SpectralResponse build_camera_path(Sampler& smp, SpectralQuery spect, const float2& uv, PathData& path_data, GBuffer& gbuffer) {
    auto ray = generate_ray(rt.scene(), rt.camera(), uv, smp.next_2d());
    auto eval = film_evaluate_out(spect, rt.camera(), ray);

    PathVertex prev = {PathVertex::Class::Camera};
    prev.throughput = {spect, 1.0f};

    PathVertex curr = {PathVertex::Class::Camera};
    curr.scene_medium_index = rt.camera().medium_index;
    curr.throughput = {spect, 1.0f};
    curr.intersection.pos = ray.o;
    curr.intersection.nrm = eval.normal;
    curr.intersection.w_i = ray.d;

    path_data.camera_path_size = 2u;

    Payload payload = {
      .spect = spect,
      .result = {spect, 0.0f},
      .throughput = curr.throughput,
      .eta = 1.0f,
      .pdf_dir = eval.pdf_dir,
      .medium_index = curr.scene_medium_index,
      .mode = PathSource::Camera,
    };

    return build_path(smp, ray, path_data, payload, {}, gbuffer, curr, prev);
  }

  SpectralResponse build_emitter_path(Sampler& smp, SpectralQuery spect, PathData& path_data) {
    path_data.emitter_path.clear();
    const auto& emitter_sample = sample_emission(rt.scene(), spect, smp);
    if ((emitter_sample.pdf_area == 0.0f) || (emitter_sample.pdf_dir == 0.0f) || (emitter_sample.value.is_zero())) {
      return {spect, 0.0f};
    }

    PathVertex prev = {PathVertex::Class::Emitter};
    prev.throughput = {spect, 1.0f};
    prev.delta_emitter = emitter_sample.is_delta;
    path_data.emitter_path.emplace_back(prev);

    PathVertex curr = {PathVertex::Class::Emitter};
    curr.intersection.triangle_index = emitter_sample.triangle_index;
    curr.intersection.barycentric = emitter_sample.barycentric;
    curr.intersection.pos = emitter_sample.origin;
    curr.intersection.nrm = emitter_sample.normal;
    curr.intersection.w_i = emitter_sample.direction;
    curr.intersection.emitter_index = emitter_sample.emitter_index;
    curr.scene_medium_index = emitter_sample.medium_index;
    curr.throughput = emitter_sample.value;
    curr.pdf.forward = emitter_sample.pdf_area * emitter_sample.pdf_sample;
    curr.delta_emitter = emitter_sample.is_delta;
    path_data.emitter_path.emplace_back(curr);

    path_data.emitter_path_size = 2u;

    float3 o = offset_ray(emitter_sample.origin, curr.intersection.nrm);
    GBuffer gbuffer = {};

    Payload payload = {
      .spect = spect,
      .result = {spect, 0.0f},
      .throughput = curr.throughput * dot(emitter_sample.direction, curr.intersection.nrm) / (emitter_sample.pdf_dir * emitter_sample.pdf_area * emitter_sample.pdf_sample),
      .eta = 1.0f,
      .pdf_dir = emitter_sample.pdf_dir,
      .medium_index = curr.scene_medium_index,
      .mode = PathSource::Light,
    };

    return build_path(smp, {o, emitter_sample.direction}, path_data, payload, emitter_sample, gbuffer, curr, prev);
  }

  void precompute_camera_mis(PathVertex& prev, const uint32_t path_size, PathData::History* history) const {
    const bool enough_length = path_size > 4;
    const bool can_connect = (history[1].delta == false) && (history[2].delta == false);

    history[2] = history[1];
    history[1] = history[0];
    history[0] = {
      .pdf_forward = prev.pdf.forward,
      .pdf_ratio = safe_div(prev.pdf.backward, prev.pdf.forward),
      .mis_accumulated = float(enough_length) * history[1].pdf_ratio * (float(can_connect) + history[1].mis_accumulated),
      .delta = prev.delta_connection,
    };
    prev.pdf.accumulated = history[0].mis_accumulated;
  }

  void precompute_light_mis(PathVertex& prev, const uint32_t path_size, PathData::History* history) const {
    const bool enough_length = path_size > 3;
    const bool is_delta = (path_size > 4u) ? history[1u].delta : prev.delta_emitter;
    const bool can_connect = (is_delta == false) * (prev.delta_connection == false);

    history[2] = history[1];
    history[1] = history[0];
    history[0] = {
      .pdf_forward = prev.pdf.forward,
      .pdf_ratio = safe_div(prev.pdf.backward, prev.pdf.forward),
      .mis_accumulated = float(enough_length) * history[1].pdf_ratio * (float(can_connect) + history[1].mis_accumulated),
      .delta = prev.delta_connection,
    };
    ETX_VALIDATE(history[0].pdf_ratio);
    ETX_VALIDATE(history[0].mis_accumulated);
    prev.pdf.accumulated = history[0].mis_accumulated;
  }

  float mis_camera(const PathData& path_data, const float z_curr_backward, const PathVertex& z_curr, const float z_prev_backward) const {
    float r1 = safe_div(z_prev_backward, path_data.camera_history[0].pdf_forward);
    ETX_VALIDATE(r1);

    bool can_connect1 = (path_data.camera_path_size > 3) && (path_data.camera_history[0].delta == false) && (path_data.camera_history[2].delta == false);
    float result_accumulated = r1 * (float(can_connect1) + path_data.camera_history[0].mis_accumulated);
    ETX_VALIDATE(result_accumulated);

    float r0 = safe_div(z_curr_backward, z_curr.pdf.forward);
    ETX_VALIDATE(r0);

    bool can_connect0 = path_data.camera_history[0].delta == false;
    result_accumulated = r0 * (float(can_connect0) + result_accumulated);
    ETX_VALIDATE(result_accumulated);

    return result_accumulated;
  }

  float mis_light(const PathData& path_data, const float y_curr_backward, const PathVertex& y_curr, const float y_prev_backward, const PathVertex& y_prev,
    const uint64_t light_s) const {
    float result = 0.0f;

    if (light_s >= 2) {
      bool delta1 = (light_s > 2u) ? path_data.emitter_path[light_s - 2u].delta_connection : y_prev.delta_emitter;
      bool can_connect1 = (delta1 == false) && (y_prev.delta_connection == false);
      float r1 = safe_div(y_prev_backward, y_prev.pdf.forward);
      result = r1 * (float(can_connect1) + y_prev.pdf.accumulated);
    }

    bool can_connect0 = y_prev.delta_connection == false;
    float r0 = safe_div(y_curr_backward, y_curr.pdf.forward);
    result = r0 * (float(can_connect0) + result);

    return result;
  }

  float mis_weight_light(const PathVertex& z_curr, const PathVertex& z_prev, const PathData& path_data, SpectralQuery spect, PathVertex y_curr, Sampler& smp) const {
    if (conn_mis == false) {
      return 1.0f;
    }

    float z_curr_pdf = y_curr.pdf_area(spect, PathSource::Light, nullptr, &z_curr, rt.scene(), smp);
    ETX_VALIDATE(z_curr_pdf);
    float z_prev_pdf = z_curr.pdf_area(spect, PathSource::Camera, &y_curr, &z_prev, rt.scene(), smp);
    ETX_VALIDATE(z_prev_pdf);

    float result = mis_camera(path_data, z_curr_pdf, z_curr, z_prev_pdf);

    if (y_curr.delta_emitter == false) {
      y_curr.pdf.backward = z_curr.pdf_area(spect, PathSource::Camera, &z_prev, &y_curr, rt.scene(), smp);
      float r = safe_div(y_curr.pdf.backward, y_curr.pdf.forward);
      result += r;
      ETX_VALIDATE(result);
    }

    return 1.0f / (1.0f + result);
  }

  float mis_weight_direct(const PathVertex& z_curr, const PathVertex& z_prev, const PathData& path_data, SpectralQuery spect, Sampler& smp) const {
    if (conn_mis == false) {
      return 1.0f;
    }

    if (path_data.camera_path_size == 3u) {
      return 1.0f;
    }

    float z_curr_pdf = z_curr.pdf_to_light_in(spect, &z_prev, rt.scene());
    ETX_VALIDATE(z_curr_pdf);

    float z_prev_pdf = z_curr.pdf_to_light_out(spect, &z_prev, rt.scene());
    ETX_VALIDATE(z_prev_pdf);

    float result = mis_camera(path_data, z_curr_pdf, z_curr, z_prev_pdf);
    return 1.0f / (1.0f + result);
  }

  float mis_weight_camera(SpectralQuery spect, const PathData& path_data, const PathVertex& y_curr, const PathVertex& y_prev, const PathVertex& z_curr, Sampler& smp) const {
    if (conn_mis == false) {
      return 1.0f;
    }

    float pdf_dir = film_pdf_out(rt.camera(), y_curr.intersection.pos);
    float y_curr_pdf = z_curr.pdf_solid_angle_to_area(pdf_dir, y_curr);
    ETX_VALIDATE(y_curr_pdf);

    float y_prev_pdf = y_curr.pdf_area(spect, PathSource::Light, &z_curr, &y_prev, rt.scene(), smp);
    ETX_VALIDATE(y_prev_pdf);

    float w_light = mis_light(path_data, y_curr_pdf, y_curr, y_prev_pdf, y_prev, path_data.emitter_path.size() - 1u);

    return 1.0f / (1.0f + w_light);
  }

  float mis_weight_connect(const PathVertex& z_curr, const PathVertex& z_prev, PathData& c, SpectralQuery spect, uint64_t light_s, Sampler& smp) const {
    if (conn_mis == false) {
      return 1.0f;
    }

    const PathVertex& y_curr = c.emitter_path[light_s];
    const PathVertex& y_prev = c.emitter_path[light_s - 1];

    float z_curr_pdf = y_curr.pdf_area(spect, PathSource::Light, &y_prev, &z_curr, rt.scene(), smp);
    ETX_VALIDATE(z_curr_pdf);

    float z_prev_pdf = z_curr.pdf_area(spect, PathSource::Camera, &y_curr, &z_prev, rt.scene(), smp);
    ETX_VALIDATE(z_prev_pdf);

    float y_curr_pdf = z_curr.pdf_area(spect, PathSource::Camera, &z_prev, &y_curr, rt.scene(), smp);
    ETX_VALIDATE(y_curr_pdf);

    float y_prev_pdf = y_curr.pdf_area(spect, PathSource::Light, &z_curr, &y_prev, rt.scene(), smp);
    ETX_VALIDATE(y_prev_pdf);

    float w_camera = mis_camera(c, z_curr_pdf, z_curr, z_prev_pdf);
    float w_light = mis_light(c, y_curr_pdf, y_curr, y_prev_pdf, y_prev, light_s);

    return 1.0f / (1.0f + w_camera + w_light);
  }

  SpectralResponse direct_hit(const PathVertex& z_curr, const PathVertex& z_prev, const PathData& path_data, SpectralQuery spect, Sampler& smp, bool force) const {
    if ((force == false) && (conn_direct_hit == false))
      return {spect, 0.0f};

    if (z_curr.is_emitter() == false) {
      return {spect, 0.0f};
    }

    float pdf_area = 0.0f;
    float pdf_dir = 0.0f;
    float pdf_dir_out = 0.0f;

    SpectralResponse emitter_value = {spect, 0.0f};

    if (z_curr.is_specific_emitter()) {
      const auto& emitter = rt.scene().emitters[z_curr.intersection.emitter_index];
      ETX_ASSERT(emitter.is_local());
      EmitterRadianceQuery q = {
        .source_position = z_prev.intersection.pos,
        .target_position = z_curr.intersection.pos,
        .uv = z_curr.intersection.tex,
        .directly_visible = path_data.camera_path_size <= 3,
      };
      emitter_value = emitter_get_radiance(emitter, spect, q, pdf_area, pdf_dir, pdf_dir_out, rt.scene());
    } else if (rt.scene().environment_emitters.count > 0) {
      EmitterRadianceQuery q = {
        .direction = normalize(z_curr.intersection.pos - z_prev.intersection.pos),
        .directly_visible = path_data.camera_path_size <= 3,
      };
      for (uint32_t ie = 0; ie < rt.scene().environment_emitters.count; ++ie) {
        const auto& emitter = rt.scene().emitters[rt.scene().environment_emitters.emitters[ie]];
        float local_pdf_dir = 0.0f;
        float local_pdf_dir_out = 0.0f;
        emitter_value += emitter_get_radiance(emitter, spect, q, pdf_area, local_pdf_dir, local_pdf_dir_out, rt.scene());
        pdf_dir += local_pdf_dir;
      }
    }

    if (pdf_dir == 0.0f) {
      return {spect, 0.0f};
    }

    ETX_VALIDATE(emitter_value);
    float weight = mis_weight_direct(z_curr, z_prev, path_data, spect, smp);
    return emitter_value * z_curr.throughput * weight;
  }

  SpectralResponse connect_to_light(const PathVertex& z_curr, const PathVertex& z_prev, Sampler& smp, PathData& path_data, SpectralQuery spect) const {
    const auto& scene = rt.scene();

    if ((conn_connect_to_light == false) || (path_data.camera_path_length() + 1u > scene.max_path_length))
      return {spect, 0.0f};

    uint32_t emitter_index = sample_emitter_index(scene, smp.next());
    auto emitter_sample = sample_emitter(spect, emitter_index, smp.next_2d(), z_curr.intersection.pos, rt.scene());
    if (emitter_sample.value.is_zero() || (emitter_sample.pdf_dir == 0.0f)) {
      return {spect, 0.0f};
    }

    auto dp = emitter_sample.origin - z_curr.intersection.pos;
    if (dot(dp, dp) <= kEpsilon) {
      return {spect, 0.0f};
    }

    SpectralResponse bsdf = z_curr.bsdf_in_direction(spect, PathSource::Camera, emitter_sample.direction, rt.scene(), smp);
    if (bsdf.is_zero()) {
      return {spect, 0.0f};
    }

    PathVertex sampled_vertex = {PathVertex::Class::Emitter};
    sampled_vertex.intersection.w_i = normalize(dp);
    sampled_vertex.intersection.triangle_index = emitter_sample.triangle_index;
    sampled_vertex.intersection.pos = emitter_sample.origin;
    sampled_vertex.intersection.nrm = emitter_sample.normal;
    sampled_vertex.intersection.emitter_index = emitter_sample.emitter_index;
    sampled_vertex.pdf.forward = sampled_vertex.pdf_to_light_in(spect, &z_curr, rt.scene());
    sampled_vertex.delta_emitter = emitter_sample.is_delta;

    SpectralResponse emitter_throughput = emitter_sample.value / (emitter_sample.pdf_dir * emitter_sample.pdf_sample);
    ETX_VALIDATE(emitter_throughput);

    SpectralResponse tr = local_transmittance(spect, smp, z_curr, sampled_vertex);
    float weight = mis_weight_light(z_curr, z_prev, path_data, spect, sampled_vertex, smp);
    return z_curr.throughput * bsdf * emitter_throughput * tr * weight;
  }

  SpectralResponse connect_to_camera(Sampler& smp, PathData& path_data, const PathVertex& y_curr, const PathVertex& y_prev, SpectralQuery spect,
    CameraSample& camera_sample) const {
    const auto& scene = rt.scene();

    if ((conn_connect_to_camera == false) || (path_data.emitter_path_length() + 1u > scene.max_path_length))
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

    SpectralResponse bsdf = y_curr.bsdf_in_direction(spect, PathSource::Light, camera_sample.direction, scene, smp);
    if (bsdf.is_zero()) {
      return {spect, 0.0f};
    }

    float weight = mis_weight_camera(spect, path_data, y_curr, y_prev, sampled_vertex, smp);

    SpectralResponse splat = y_curr.throughput * bsdf * camera_sample.weight * (weight / spect.sampling_pdf());
    ETX_VALIDATE(splat);

    if (splat.is_zero() == false) {
      splat *= local_transmittance(spect, smp, y_curr, sampled_vertex);
    }

    return splat;
  }

  SpectralResponse local_transmittance(SpectralQuery spect, Sampler& smp, const PathVertex& p0, const PathVertex& p1) const {
    auto& scene = rt.scene();
    float3 origin = p0.intersection.pos;
    if (p0.is_surface_interaction()) {
      const auto& tri = scene.triangles[p0.intersection.triangle_index];
      origin = shading_pos(scene.vertices, tri, p0.intersection.barycentric, normalize(p1.intersection.pos - p0.intersection.pos));
    }

    return rt.trace_transmittance(spect, scene, origin, p1.intersection.pos, p0.scene_medium_index, smp);
  }

  void start(const Options& opt) {
    conn_direct_hit = opt.get("conn_direct_hit", conn_direct_hit).to_bool();
    conn_connect_to_camera = opt.get("conn_connect_to_camera", conn_connect_to_camera).to_bool();
    conn_connect_to_light = opt.get("conn_connect_to_light", conn_connect_to_light).to_bool();
    conn_connect_vertices = opt.get("conn_connect_vertices", conn_connect_vertices).to_bool();
    conn_mis = opt.get("conn_mis", conn_mis).to_bool();

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

  if (rt.has_scene()) {
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
  Options result = {};
  result.add(_private->conn_direct_hit, "conn_direct_hit", "Direct Hits");
  result.add(_private->conn_connect_to_camera, "conn_connect_to_camera", "Connect to Camera");
  result.add(_private->conn_connect_to_light, "conn_connect_to_light", "Connect to Light");
  result.add(_private->conn_connect_vertices, "conn_connect_vertices", "Connect Vertices");
  result.add(_private->conn_mis, "conn_mis", "Multiple Importance Sampling");
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
