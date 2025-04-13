#include <etx/core/core.hxx>
#include <etx/render/host/film.hxx>
#include <etx/rt/integrators/bidirectional.hxx>
#include <etx/rt/integrators/bidirectional_shared.hxx>
#include <atomic>

namespace etx {

namespace {

inline float safe_div(float a, float b) {
  return ((a == 0.0f) ? 1.0f : a) / ((b == 0.0f) ? 1.0f : b);
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
      float3 surface_normal = {};
      SpectralResponse surface_albedo = {};
      SpectralResponse result = build_camera_path(smp, spect, uv, path_data, surface_normal, surface_albedo);

      auto xyz = (result / spect.sampling_pdf()).to_rgb();
      auto albedo = (surface_albedo / spect.sampling_pdf()).to_rgb();
      film.accumulate(pixel, {{xyz, Film::CameraImage}, {surface_normal, Film::Normals}, {albedo, Film::Albedo}});
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

  void update_emitter_path_pdfs(std::vector<PathVertex>& path, const EmitterSample& em, const Triangle* tri) const {
    const auto& scene = rt.scene();

    float total_pdf = 0.0f;
    float total_weight = 0.0f;
    for (uint32_t ei = 0, ee = scene.environment_emitters.count; ei < ee; ++ei) {
      float weight = scene.emitters_distribution.values[ei].value;
      total_weight += weight;
      total_pdf += weight * emitter_pdf_in_dist(scene.emitters[em.emitter_index], em.direction, scene);
    }
    path[1].pdf.forward = total_pdf / (total_weight * float(scene.environment_emitters.count));
    ETX_VALIDATE(path[1].pdf.forward);

    path[2].pdf.forward = em.pdf_area;
    if (tri != nullptr) {
      path[2].pdf.forward *= fabsf(dot(em.direction, tri->geo_n));
      ETX_VALIDATE(path[2].pdf.forward);
    }
  }

  SpectralResponse connect_to_light_path(Sampler& smp, SpectralQuery spect, PathData& path_data) const {
    if ((conn_connect_vertices == false) || (path_data.camera_path.size() <= 2))
      return {spect, 0.0f};

    SpectralResponse result = {spect, 0.0f};

    const uint64_t eye_t = path_data.camera_path.size() - 1u;
    const auto& z_i = path_data.camera_path.back();

    const uint32_t max_connect_len = 65536u;  // scene.max_camera_path_length + scene.max_light_path_length + 2u;

    for (uint64_t light_s = 2, light_s_e = path_data.emitter_path.size(); running() && (light_s < light_s_e); ++light_s) {
      if (eye_t + light_s >= max_connect_len)
        break;

      const auto& y_i = path_data.emitter_path[light_s];

      auto dw = z_i.pos - y_i.pos;
      float dwl = dot(dw, dw);
      dw *= 1.0f / std::sqrt(dwl);

      SpectralResponse connect_result = y_i.throughput * y_i.bsdf_in_direction(spect, PathSource::Light, dw, rt.scene(), smp) *    //
                                        z_i.throughput * z_i.bsdf_in_direction(spect, PathSource::Camera, -dw, rt.scene(), smp) *  //
                                        (1.0f / dwl);  // G term = abs(cos(dw, y_i.nrm) * cos(dw, z_i.nrm)) / dwl; cosines already accounted in "bsdf"
      ETX_VALIDATE(connect_result);

      if (connect_result.is_zero())
        continue;

      SpectralResponse tr = local_transmittance(spect, smp, y_i, z_i);
      ETX_VALIDATE(connect_result);

      float weight = mis_weight_connect(path_data, spect, light_s, smp);
      ETX_VALIDATE(weight);

      result += connect_result * tr * weight;
    }

    return result;
  }

  SpectralResponse build_path(Sampler& smp, SpectralQuery spect, Ray ray, PathData& path_data, PathSource mode, SpectralResponse throughput, float pdf_dir, uint32_t medium_index,
    const EmitterSample& emitter_sample, float3& surface_normal, SpectralResponse& surface_albedo) {
    ETX_VALIDATE(throughput);

    std::vector<PathVertex>& path = mode == PathSource::Camera ? path_data.camera_path : path_data.emitter_path;

    const auto& scene = rt.scene();
    SpectralResponse result = {spect, 0.0f};

    bool gbuffer_recorded = false;
    float eta = 1.0f;
    uint32_t max_path_len = mode == PathSource::Camera ? scene.max_camera_path_length : scene.max_light_path_length;

    path_data.camera_mis = 0.0f;
    for (uint32_t path_length = 0; running() && (path_length < max_path_len);) {
      Intersection intersection = {};
      bool found_intersection = rt.trace(scene, ray, intersection, smp);

      Medium::Sample medium_sample = {};
      if (medium_index != kInvalidIndex) {
        medium_sample = scene.mediums[medium_index].sample(spect, throughput, smp, ray.o, ray.d, found_intersection ? intersection.t : kMaxFloat);
        throughput *= medium_sample.weight;
        ETX_VALIDATE(throughput);
      }

      if (medium_sample.sampled_medium()) {
        const auto& medium = scene.mediums[medium_index];

        float3 w_i = ray.d;
        float3 w_o = medium.sample_phase_function(spect, smp, w_i);

        auto& v = path.emplace_back(medium_sample, w_i);
        auto& w = path[path.size() - 2];
        v.medium_index = medium_index;
        v.throughput = throughput;
        v.pdf.forward = w.pdf_solid_angle_to_area(pdf_dir, v);

        float rev_pdf = medium.phase_function(spect, medium_sample.pos, w_o, w_i);
        w.pdf.backward = v.pdf_solid_angle_to_area(rev_pdf, w);

        pdf_dir = medium.phase_function(spect, medium_sample.pos, w_i, w_o);
        ray.o = medium_sample.pos;
        ray.d = w_o;

        if ((mode == PathSource::Light) && emitter_sample.is_distant && (path.size() == 3)) {
          update_emitter_path_pdfs(path, emitter_sample, nullptr);
        }

        if ((mode == PathSource::Camera) && (path_length > 1)) {
          precompute_camera_mis(path_data);
        }

        bool can_connect = (path.size() <= 1llu + max_path_len) && medium.enable_explicit_connections;

        if (mode == PathSource::Camera) {
          result += direct_hit(path_data, spect, smp);
          if (can_connect) {
            result += connect_to_light(smp, path_data, spect);
            result += connect_to_light_path(smp, spect, path_data);
          }
        } else if (conn_connect_to_camera && can_connect) {
          CameraSample camera_sample = {};
          auto splat = connect_to_camera(smp, path, spect, camera_sample);
          auto xyz = splat.to_rgb();
          rt.film().atomic_add(Film::LightIteration, xyz, camera_sample.uv);
        }

      } else if (found_intersection) {
        const auto& tri = scene.triangles[intersection.triangle_index];
        const auto& mat = scene.materials[intersection.material_index];

        if (mat.cls == Material::Class::Boundary) {
          medium_index = (dot(intersection.nrm, ray.d) < 0.0f) ? mat.int_medium : mat.ext_medium;
          ray.o = shading_pos(scene.vertices, tri, intersection.barycentric, ray.d);
          continue;
        }

        auto& v = path.emplace_back(PathVertex::Class::Surface, intersection);
        auto& w = path[path.size() - 2];
        v.emitter_index = intersection.emitter_index;
        v.throughput = throughput;
        v.pdf.forward = w.pdf_solid_angle_to_area(pdf_dir, v);
        ETX_VALIDATE(v.pdf.forward);

        auto bsdf_data = BSDFData(spect, medium_index, mode, v, v.w_i);

        if (gbuffer_recorded == false) {
          surface_normal = intersection.nrm;
          surface_albedo = bsdf::albedo(bsdf_data, mat, scene, smp);
          gbuffer_recorded = true;
        }

        auto bsdf_sample = bsdf::sample(bsdf_data, mat, scene, smp);
        ETX_VALIDATE(bsdf_sample.weight);

        v.delta_connection = bsdf_sample.is_delta();
        v.medium_index = (bsdf_sample.properties & BSDFSample::MediumChanged) ? bsdf_sample.medium_index : medium_index;
        medium_index = v.medium_index;

        bool terminate_path = false;

        if (bsdf_sample.valid()) {
          auto rev_bsdf_pdf = bsdf::reverse_pdf(bsdf_data, -v.w_i, mat, scene, smp);
          ETX_VALIDATE(rev_bsdf_pdf);

          w.pdf.backward = v.pdf_solid_angle_to_area(rev_bsdf_pdf, w);
          ETX_VALIDATE(w.pdf.backward);

          if (mode == PathSource::Camera) {
            eta *= bsdf_sample.eta;
          }

          pdf_dir = v.delta_connection ? 0.0f : bsdf_sample.pdf;
          ETX_VALIDATE(pdf_dir);

          throughput *= bsdf_sample.weight;
          ETX_VALIDATE(throughput);

          if (mode == PathSource::Light) {
            throughput *= fix_shading_normal(tri.geo_n, bsdf_data.nrm, bsdf_data.w_i, bsdf_sample.w_o);
            ETX_VALIDATE(throughput);
          }

          ray.o = shading_pos(scene.vertices, tri, intersection.barycentric, bsdf_sample.w_o);
          ray.d = bsdf_sample.w_o;
        } else {
          terminate_path = true;
        }

        if ((mode == PathSource::Light) && emitter_sample.is_distant && (path.size() == 3)) {
          update_emitter_path_pdfs(path, emitter_sample, &tri);
        }

        if ((mode == PathSource::Camera) && (path_length > 1)) {
          precompute_camera_mis(path_data);
        }

        bool can_connect = path.size() <= 1llu + max_path_len;

        if (mode == PathSource::Camera) {
          result += direct_hit(path_data, spect, smp);
          if (can_connect) {
            result += connect_to_light(smp, path_data, spect);
            result += connect_to_light_path(smp, spect, path_data);
          }
        } else if (conn_connect_to_camera && can_connect) {
          CameraSample camera_sample = {};
          auto splat = connect_to_camera(smp, path, spect, camera_sample);
          if (splat.is_zero() == false) {
            auto xyz = splat.to_rgb();
            rt.film().atomic_add(Film::LightIteration, xyz, camera_sample.uv);
          }
        }

        if (terminate_path) {
          break;
        }

      } else if (mode == PathSource::Camera) {
        auto& v = path.emplace_back(PathVertex::Class::Emitter);
        v.medium_index = medium_index;
        v.throughput = throughput;
        v.pdf.forward = pdf_dir;
        v.w_i = ray.d;
        v.pos = ray.o + scene.bounding_sphere_radius * v.w_i;
        v.nrm = -v.w_i;
        result += direct_hit(path_data, spect, smp);
        path.pop_back();
        break;
      } else {
        break;
      }

      if (random_continue(path_length, scene.random_path_termination, eta, smp, throughput) == false) {
        break;
      }

      path_length += 1;
    }

    return result;
  }

  SpectralResponse build_camera_path(Sampler& smp, SpectralQuery spect, const float2& uv, PathData& path_data, float3& surface_normal, SpectralResponse& surface_albedo) {
    path_data.camera_path.clear();
    auto& z0 = path_data.camera_path.emplace_back(PathVertex::Class::Camera);
    z0.throughput = {spect, 1.0f};

    auto ray = generate_ray(rt.scene(), rt.camera(), uv, smp.next_2d());
    auto eval = film_evaluate_out(spect, rt.camera(), ray);

    auto& z1 = path_data.camera_path.emplace_back(PathVertex::Class::Camera);
    z1.medium_index = rt.camera().medium_index;
    z1.throughput = {spect, 1.0f};
    z1.pos = ray.o;
    z1.nrm = eval.normal;
    z1.w_i = ray.d;

    return build_path(smp, spect, ray, path_data, PathSource::Camera, z1.throughput, eval.pdf_dir, z1.medium_index, {}, surface_normal, surface_albedo);
  }

  SpectralResponse build_emitter_path(Sampler& smp, SpectralQuery spect, PathData& path_data) {
    path_data.emitter_path.clear();
    const auto& emitter_sample = sample_emission(rt.scene(), spect, smp);
    if ((emitter_sample.pdf_area == 0.0f) || (emitter_sample.pdf_dir == 0.0f) || (emitter_sample.value.is_zero())) {
      return {spect, 0.0f};
    }

    auto& y0 = path_data.emitter_path.emplace_back(PathVertex::Class::Emitter);
    y0.throughput = {spect, 1.0f};
    y0.delta_emitter = emitter_sample.is_delta;

    auto& y1 = path_data.emitter_path.emplace_back(PathVertex::Class::Emitter);
    y1.triangle_index = emitter_sample.triangle_index;
    y1.medium_index = emitter_sample.medium_index;
    y1.emitter_index = emitter_sample.emitter_index;
    y1.throughput = emitter_sample.value;
    y1.barycentric = emitter_sample.barycentric;
    y1.pos = emitter_sample.origin;
    y1.nrm = emitter_sample.normal;
    y1.pdf.forward = emitter_sample.pdf_area * emitter_sample.pdf_sample;
    y1.w_i = emitter_sample.direction;
    y1.delta_emitter = emitter_sample.is_delta;

    float3 o = offset_ray(emitter_sample.origin, y1.nrm);
    SpectralResponse throughput = y1.throughput * dot(emitter_sample.direction, y1.nrm) / (emitter_sample.pdf_dir * emitter_sample.pdf_area * emitter_sample.pdf_sample);
    float3 surface_normal = {};
    SpectralResponse surface_albedo = {};
    return build_path(smp, spect, {o, emitter_sample.direction}, path_data, PathSource::Light, throughput, emitter_sample.pdf_dir, y1.medium_index, emitter_sample, surface_normal,
      surface_albedo);
  }

  void precompute_camera_mis(PathData& path_data) const {
    const uint64_t eye_t = path_data.camera_path.size() - 1u;

    float accum = 1.0f;
    path_data.camera_mis = 0.0f;
    for (uint64_t ti = eye_t - 2u; ti > 1u; --ti) {
      const bool can_connect = (path_data.camera_path[ti].delta_connection == false) && (path_data.camera_path[ti - 1].delta_connection == false);
      accum *= safe_div(path_data.camera_path[ti].pdf.backward, path_data.camera_path[ti].pdf.forward);
      ETX_VALIDATE(accum);
      path_data.camera_mis += float(can_connect) * accum;
      ETX_VALIDATE(path_data.camera_mis);
    }
  }

  float mis_camera(const PathData& path_data, const float z_curr_backward, const PathVertex& z_curr, const float z_prev_backward, const PathVertex& z_prev) const {
    float result = 0.0f;

    float r = safe_div(z_curr_backward, z_curr.pdf.forward);
    ETX_VALIDATE(r);
    if (z_prev.delta_connection == false) {
      result += r;
      ETX_VALIDATE(result);
    }

    const uint64_t eye_t = path_data.camera_path.size() - 1u;
    if (eye_t <= 2)
      return result;

    r *= safe_div(z_prev_backward, z_prev.pdf.forward);
    ETX_VALIDATE(r);
    if ((z_prev.delta_connection == false) && (path_data.camera_path[eye_t - 2].delta_connection == false)) {
      result += r;
      ETX_VALIDATE(result);
    }

    return result + r * path_data.camera_mis;
  }

  float mis_weight_light(PathData& path_data, SpectralQuery spect, PathVertex y_curr, Sampler& smp) const {
    if (conn_mis == false) {
      return 1.0f;
    }

    const PathVertex& z_curr = path_data.camera_path.back();
    const PathVertex& z_prev = path_data.camera_path[path_data.camera_path.size() - 2u];

    float z_curr_pdf = y_curr.pdf_area(spect, PathSource::Light, nullptr, &z_curr, rt.scene(), smp);
    ETX_VALIDATE(z_curr_pdf);
    float z_prev_pdf = z_curr.pdf_area(spect, PathSource::Camera, &y_curr, &z_prev, rt.scene(), smp);
    ETX_VALIDATE(z_prev_pdf);

    float result = mis_camera(path_data, z_curr_pdf, z_curr, z_prev_pdf, z_prev);

    if (y_curr.delta_emitter == false) {
      y_curr.pdf.backward = z_curr.pdf_area(spect, PathSource::Camera, &z_prev, &y_curr, rt.scene(), smp);
      float r = safe_div(y_curr.pdf.backward, y_curr.pdf.forward);
      ETX_VALIDATE(r);
      result += r;
      ETX_VALIDATE(result);
    }

    return 1.0f / (1.0f + result);
  }

  float mis_weight_direct(const PathData& path_data, SpectralQuery spect, Sampler& smp) const {
    if (conn_mis == false) {
      return 1.0f;
    }

    const uint64_t eye_t = path_data.camera_path.size() - 1u;
    if (eye_t == 2) {
      return 1.0f;
    }

    const PathVertex& z_curr = path_data.camera_path[eye_t];
    const PathVertex& z_prev = path_data.camera_path[eye_t - 1];

    float z_curr_pdf = z_curr.pdf_to_light_in(spect, &z_prev, rt.scene());
    ETX_VALIDATE(z_curr_pdf);
    float z_prev_pdf = z_curr.pdf_to_light_out(spect, &z_prev, rt.scene());
    ETX_VALIDATE(z_prev_pdf);

    float result = mis_camera(path_data, z_curr_pdf, z_curr, z_prev_pdf, z_prev);
    return 1.0f / (1.0f + result);
  }

  float mis_weight_camera(std::vector<PathVertex>& emitter_path, SpectralQuery spect, const PathVertex& sampled, Sampler& smp) const {
    if (conn_mis == false) {
      return 1.0f;
    }

    const uint64_t light_s = emitter_path.size() - 1u;
    PathVertex& y_curr = emitter_path[light_s];
    PathVertex& y_prev = emitter_path[light_s - 1];

    ReplaceInScope<bool> y_delta_new;
    ReplaceInScope<float> y_curr_new;
    ReplaceInScope<float> y_prev_new;

    PathVertex z_curr = sampled;
    z_curr.pdf.backward = y_curr.pdf_area(spect, PathSource::Light, &y_prev, &z_curr, rt.scene(), smp);

    {
      y_delta_new = {&y_curr.delta_connection, false};
      float y_curr_pdf = 0.0f;
      ETX_ASSERT(z_curr.cls == PathVertex::Class::Camera);
      float pdf_dir = film_pdf_out(rt.camera(), y_curr.pos);
      y_curr_pdf = z_curr.pdf_solid_angle_to_area(pdf_dir, y_curr);
      ETX_VALIDATE(y_curr_pdf);
      y_curr_new = {&y_curr.pdf.backward, y_curr_pdf};
      ETX_VALIDATE(y_curr.pdf.backward);
    }

    {
      float y_prev_pdf = y_curr.pdf_area(spect, PathSource::Light, &z_curr, &y_prev, rt.scene(), smp);
      ETX_VALIDATE(y_prev_pdf);
      y_prev_new = {&y_prev.pdf.backward, y_prev_pdf};
      ETX_VALIDATE(y_prev.pdf.backward);
    }

    float result = 0.0f;

    float r = 1.0f;
    for (uint64_t si = light_s; si > 0; --si) {
      r *= safe_div(emitter_path[si].pdf.backward, emitter_path[si].pdf.forward);
      ETX_VALIDATE(r);

      bool delta_emitter = (si > 1) ? emitter_path[si - 1u].delta_connection : emitter_path[1u].delta_emitter;
      if ((emitter_path[si].delta_connection == false) && (delta_emitter == false)) {
        result += r;
        ETX_VALIDATE(result);
      }
    }

    return 1.0f / (1.0f + result);
  }

  float mis_weight_connect(PathData& c, SpectralQuery spect, uint64_t light_s, Sampler& smp) const {
    if (conn_mis == false) {
      return 1.0f;
    }

    const PathVertex& z_curr = c.camera_path.back();
    const PathVertex& z_prev = c.camera_path[c.camera_path.size() - 2u];

    PathVertex& y_curr = c.emitter_path[light_s];
    PathVertex& y_prev = c.emitter_path[light_s - 1];

    ReplaceInScope<bool> y_delta_new;
    ReplaceInScope<float> y_curr_new;
    ReplaceInScope<float> y_prev_new;

    float z_curr_pdf = 0.0f;
    if (light_s > 0) {
      z_curr_pdf = y_curr.pdf_area(spect, PathSource::Light, &y_prev, &z_curr, rt.scene(), smp);
    } else {
      z_curr_pdf = z_curr.pdf_to_light_in(spect, &z_prev, rt.scene());
    }
    ETX_VALIDATE(z_curr_pdf);

    float z_prev_pdf = z_prev_pdf = z_curr.pdf_area(spect, PathSource::Camera, &y_curr, &z_prev, rt.scene(), smp);
    ETX_VALIDATE(z_prev_pdf);

    {
      y_delta_new = {&y_curr.delta_connection, false};
      float y_curr_pdf = 0.0f;
      y_curr_pdf = z_curr.pdf_area(spect, PathSource::Camera, &z_prev, &y_curr, rt.scene(), smp);
      ETX_VALIDATE(y_curr_pdf);
      y_curr_new = {&y_curr.pdf.backward, y_curr_pdf};
      ETX_VALIDATE(y_curr.pdf.backward);
    }

    {
      float y_prev_pdf = y_curr.pdf_area(spect, PathSource::Light, &z_curr, &y_prev, rt.scene(), smp);
      ETX_VALIDATE(y_prev_pdf);
      y_prev_new = {&y_prev.pdf.backward, y_prev_pdf};
      ETX_VALIDATE(y_prev.pdf.backward);
    }

    float result = mis_camera(c, z_curr_pdf, z_curr, z_prev_pdf, z_prev);

    float r = 1.0f;
    for (uint64_t si = light_s; si > 0; --si) {
      r *= safe_div(c.emitter_path[si].pdf.backward, c.emitter_path[si].pdf.forward);
      ETX_VALIDATE(r);

      bool delta_emitter = (si > 1) ? c.emitter_path[si - 1u].delta_connection : c.emitter_path[1u].delta_emitter;
      if ((c.emitter_path[si].delta_connection == false) && (delta_emitter == false)) {
        result += r;
        ETX_VALIDATE(result);
      }
    }

    return 1.0f / (1.0f + result);
  }

  SpectralResponse direct_hit(const PathData& path_data, SpectralQuery spect, Sampler& smp) const {
    if (conn_direct_hit == false)
      return {spect, 0.0f};

    const auto& z_i = path_data.camera_path.back();
    if ((conn_direct_hit == false) || (z_i.is_emitter() == false)) {
      return {spect, 0.0f};
    }

    uint64_t eye_t = path_data.camera_path.size() - 1u;
    const auto& z_prev = path_data.camera_path[eye_t - 1];

    float pdf_area = 0.0f;
    float pdf_dir = 0.0f;
    float pdf_dir_out = 0.0f;

    SpectralResponse emitter_value = {spect, 0.0f};

    if (z_i.is_specific_emitter()) {
      const auto& emitter = rt.scene().emitters[z_i.emitter_index];
      ETX_ASSERT(emitter.is_local());
      EmitterRadianceQuery q = {
        .source_position = z_prev.pos,
        .target_position = z_i.pos,
        .uv = z_i.tex,
        .directly_visible = eye_t <= 2,
      };
      emitter_value = emitter_get_radiance(emitter, spect, q, pdf_area, pdf_dir, pdf_dir_out, rt.scene());
    } else if (rt.scene().environment_emitters.count > 0) {
      EmitterRadianceQuery q = {
        .direction = normalize(z_i.pos - z_prev.pos),
        .directly_visible = eye_t <= 2,
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
    float weight = mis_weight_direct(path_data, spect, smp);
    return emitter_value * z_i.throughput * weight;
  }

  SpectralResponse connect_to_light(Sampler& smp, PathData& path_data, SpectralQuery spect) const {
    if (conn_connect_to_light == false)
      return {spect, 0.0f};

    const auto& z_i = path_data.camera_path.back();

    uint32_t emitter_index = sample_emitter_index(rt.scene(), smp.next());
    auto emitter_sample = sample_emitter(spect, emitter_index, smp.next_2d(), z_i.pos, rt.scene());
    if (emitter_sample.value.is_zero() || (emitter_sample.pdf_dir == 0.0f)) {
      return {spect, 0.0f};
    }

    auto dp = emitter_sample.origin - z_i.pos;
    if (dot(dp, dp) <= kEpsilon) {
      return {spect, 0.0f};
    }

    PathVertex sampled_vertex = {PathVertex::Class::Emitter};
    sampled_vertex.w_i = normalize(dp);
    sampled_vertex.triangle_index = emitter_sample.triangle_index;
    sampled_vertex.emitter_index = emitter_sample.emitter_index;
    sampled_vertex.pos = emitter_sample.origin;
    sampled_vertex.nrm = emitter_sample.normal;
    sampled_vertex.pdf.forward = sampled_vertex.pdf_to_light_in(spect, &z_i, rt.scene());
    sampled_vertex.delta_emitter = emitter_sample.is_delta;

    SpectralResponse emitter_throughput = emitter_sample.value / (emitter_sample.pdf_dir * emitter_sample.pdf_sample);
    ETX_VALIDATE(emitter_throughput);

    SpectralResponse bsdf = z_i.bsdf_in_direction(spect, PathSource::Camera, emitter_sample.direction, rt.scene(), smp);
    SpectralResponse tr = local_transmittance(spect, smp, z_i, sampled_vertex);
    float weight = mis_weight_light(path_data, spect, sampled_vertex, smp);
    return z_i.throughput * bsdf * emitter_throughput * tr * weight;
  }

  SpectralResponse connect_to_camera(Sampler& smp, std::vector<PathVertex>& emitter_path, SpectralQuery spect, CameraSample& camera_sample) const {
    if (conn_connect_to_camera == false)
      return {spect, 0.0f};

    const auto& y_i = emitter_path.back();
    camera_sample = sample_film(smp, rt.scene(), rt.camera(), y_i.pos);
    if (camera_sample.valid() == false) {
      return {spect, 0.0f};
    }

    ETX_VALIDATE(camera_sample.weight);

    PathVertex sampled_vertex = {PathVertex::Class::Camera};
    sampled_vertex.pos = camera_sample.position;
    sampled_vertex.nrm = camera_sample.normal;
    sampled_vertex.w_i = camera_sample.direction;

    SpectralResponse bsdf = y_i.bsdf_in_direction(spect, PathSource::Light, camera_sample.direction, rt.scene(), smp);
    if (bsdf.is_zero()) {
      return {spect, 0.0f};
    }

    float weight = mis_weight_camera(emitter_path, spect, sampled_vertex, smp);

    SpectralResponse splat = y_i.throughput * bsdf * camera_sample.weight * (weight / spect.sampling_pdf());
    ETX_VALIDATE(splat);

    if (splat.is_zero() == false) {
      splat *= local_transmittance(spect, smp, y_i, sampled_vertex);
    }

    return splat;
  }

  SpectralResponse connect_vertices(Sampler& smp, PathData& c, SpectralQuery spect, uint64_t eye_t, uint64_t light_s) const {
  }

  SpectralResponse local_transmittance(SpectralQuery spect, Sampler& smp, const PathVertex& p0, const PathVertex& p1) const {
    auto& scene = rt.scene();
    float3 origin = p0.pos;
    if (p0.is_surface_interaction()) {
      const auto& tri = scene.triangles[p0.triangle_index];
      origin = shading_pos(scene.vertices, tri, p0.barycentric, normalize(p1.pos - p0.pos));
    }
    return rt.trace_transmittance(spect, scene, origin, p1.pos, p0.medium_index, smp);
  }

  void start(const Options& opt) {
    conn_direct_hit = opt.get("conn_direct_hit", conn_direct_hit).to_bool();
    conn_connect_to_camera = opt.get("conn_connect_to_camera", conn_connect_to_camera).to_bool();
    conn_connect_to_light = opt.get("conn_connect_to_light", conn_connect_to_light).to_bool();
    conn_connect_vertices = opt.get("conn_connect_vertices", conn_connect_vertices).to_bool();
    conn_mis = opt.get("conn_mis", conn_mis).to_bool();

    for (auto& path_data : per_thread_path_data) {
      path_data.camera_path.reserve(2llu + rt.scene().max_camera_path_length);
      path_data.emitter_path.reserve(2llu + rt.scene().max_light_path_length);
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
