#include <etx/core/core.hxx>

#include <etx/render/host/film.hxx>
#include <etx/render/host/rnd_sampler.hxx>

#include <etx/rt/integrators/bidirectional.hxx>

#include <atomic>

namespace etx {

struct CPUBidirectionalImpl : public Task {
  struct PathVertex : public Intersection {
    enum class Class : uint32_t {
      Invalid,
      Camera,
      Emitter,
      Surface,
      Medium,
    };

    Class cls = Class::Invalid;
    uint32_t emitter_index = kInvalidIndex;
    uint32_t medium_index = kInvalidIndex;
    SpectralResponse throughput = {};
    struct {
      float forward = 0.0f;
      float backward = 0.0f;
    } pdf;

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

    SpectralResponse bsdf_in_direction(SpectralQuery spect, PathSource mode, const float3& w_o, const Scene& scene, Sampler& smp) const;

    float pdf_area(SpectralQuery spect, PathSource mode, const PathVertex* prev, const PathVertex* next, const Scene& scene, Sampler& smp) const;
    float pdf_to_light_out(SpectralQuery spect, const PathVertex* next, const Scene& scene) const;
    float pdf_to_light_in(SpectralQuery spect, const PathVertex* next, const Scene& scene) const;
    float pdf_solid_angle_to_area(float pdf_dir, const PathVertex& to_vertex) const;
  };

  template <class T>
  struct ReplaceInScope {
    ReplaceInScope(const ReplaceInScope&) = delete;
    ReplaceInScope& operator=(const ReplaceInScope&) = delete;

    ReplaceInScope() {
    }

    ReplaceInScope(T* destination, const T& new_value)
      : ptr(destination)
      , old_value(*destination) {
      *destination = new_value;
    }

    ReplaceInScope& operator=(ReplaceInScope&& r) noexcept {
      ptr = r.ptr;
      old_value = r.old_value;
      r.ptr = nullptr;
      return *this;
    }

    ~ReplaceInScope() {
      if (ptr != nullptr) {
        *ptr = old_value;
      }
    }

    T* ptr = nullptr;
    T old_value = {};
  };

  struct PathData {
    std::vector<PathVertex> camera_path;
    std::vector<PathVertex> emitter_path;

    PathData() = default;
    PathData(const PathData&) = delete;
    PathData& operator=(const PathData&) = delete;
  };

  Raytracing& rt;
  std::vector<RNDSampler> samplers;
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
    , samplers(rt.scheduler().max_thread_count())
    , per_thread_path_data(rt.scheduler().max_thread_count())
    , state(st) {
  }

  void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) {
    auto film_dimensions = rt.film().dimensions();
    auto& smp = samplers[thread_id];
    for (uint32_t i = begin; (state->load() != Integrator::State::Stopped) && (i < end); ++i) {
      uint32_t x = i % film_dimensions.x;
      uint32_t y = i / film_dimensions.x;

      auto& path_data = per_thread_path_data[thread_id];
      auto spect = rt.scene().spectral ? SpectralQuery::spectral_sample(smp.next()) : SpectralQuery::sample();

      build_emitter_path(smp, spect, path_data.emitter_path, thread_id);

      float2 uv = get_jittered_uv(smp, {x, y}, film_dimensions);
      SpectralResponse result = build_camera_path(smp, spect, uv, path_data.camera_path, thread_id);

      for (uint64_t light_s = 2, light_s_e = path_data.emitter_path.size(); running() && (light_s < light_s_e); ++light_s) {
        for (uint64_t eye_t = 2, eye_t_e = path_data.camera_path.size(); running() && (eye_t < eye_t_e); ++eye_t) {
          if (eye_t + light_s - 2u < rt.scene().max_path_length) {
            result += connect_vertices(smp, path_data, spect, eye_t, light_s);
          }
        }
      }

      auto xyz = (result / spect.sampling_pdf()).to_xyz();

      rt.film().accumulate(Film::Camera, {xyz.x, xyz.y, xyz.z, 1.0f}, uv, float(status.current_iteration) / float(status.current_iteration + 1));
    }
  }

  void completed() {
    status.last_iteration_time = iteration_time.measure();
    status.total_time += status.last_iteration_time;
    status.completed_iterations += 1u;
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

  SpectralResponse build_path(Sampler& smp, SpectralQuery spect, Ray ray, std::vector<PathVertex>& path, PathSource mode, SpectralResponse throughput, float pdf_dir,
    uint32_t medium_index, uint32_t thread_id, const EmitterSample& em) {
    ETX_VALIDATE(throughput);

    SpectralResponse result = {spect, 0.0f};

    float eta = 1.0f;
    uint32_t max_path_len = rt.scene().max_path_length;

    for (uint32_t path_length = 0; running() && (path_length < max_path_len);) {
      Intersection intersection = {};
      bool found_intersection = rt.trace(rt.scene(), ray, intersection, smp);

      Medium::Sample medium_sample = {};
      if (medium_index != kInvalidIndex) {
        medium_sample = rt.scene().mediums[medium_index].sample(spect, throughput, smp, ray.o, ray.d, found_intersection ? intersection.t : kMaxFloat);
        throughput *= medium_sample.weight;
        ETX_VALIDATE(throughput);
      }

      if (medium_sample.sampled_medium()) {
        const auto& medium = rt.scene().mediums[medium_index];

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

        if ((mode == PathSource::Light) && em.is_distant && (path.size() == 3)) {
          update_emitter_path_pdfs(path, em, nullptr);
        }

        bool can_connect = path.size() <= 1llu + max_path_len;

        if (mode == PathSource::Camera) {
          result += direct_hit(path, spect, smp);
          if (can_connect) {
            result += connect_to_light(smp, path, spect);
          }
        } else if (conn_connect_to_camera && can_connect) {
          CameraSample camera_sample = {};
          auto splat = connect_to_camera(smp, path, spect, camera_sample);
          auto xyz = splat.to_xyz();
          rt.film().atomic_add(Film::LightIteration, {xyz.x, xyz.y, xyz.z, 1.0f}, camera_sample.uv);
        }

      } else if (found_intersection) {
        const auto& tri = rt.scene().triangles[intersection.triangle_index];
        const auto& mat = rt.scene().materials[intersection.material_index];

        if (mat.cls == Material::Class::Boundary) {
          medium_index = (dot(intersection.nrm, ray.d) < 0.0f) ? mat.int_medium : mat.ext_medium;
          ray.o = shading_pos(rt.scene().vertices, tri, intersection.barycentric, ray.d);
          continue;
        }

        auto& v = path.emplace_back(PathVertex::Class::Surface, intersection);
        auto& w = path[path.size() - 2];
        v.medium_index = medium_index;
        v.emitter_index = intersection.emitter_index;
        v.throughput = throughput;
        v.pdf.forward = w.pdf_solid_angle_to_area(pdf_dir, v);
        ETX_VALIDATE(v.pdf.forward);

        auto bsdf_data = BSDFData(spect, medium_index, mode, v, v.w_i);

        auto bsdf_sample = bsdf::sample(bsdf_data, mat, rt.scene(), smp);
        ETX_VALIDATE(bsdf_sample.weight);

        v.delta_connection = bsdf_sample.is_delta();

        if (bsdf_sample.properties & BSDFSample::MediumChanged) {
          medium_index = bsdf_sample.medium_index;
        }

        bool terminate_path = false;

        if (bsdf_sample.valid()) {
          auto rev_bsdf_pdf = bsdf::reverse_pdf(bsdf_data, -v.w_i, mat, rt.scene(), smp);
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

          ray.o = shading_pos(rt.scene().vertices, tri, intersection.barycentric, bsdf_sample.w_o);
          ray.d = bsdf_sample.w_o;
        } else {
          terminate_path = true;
        }

        if ((mode == PathSource::Light) && em.is_distant && (path.size() == 3)) {
          update_emitter_path_pdfs(path, em, &tri);
        }

        bool can_connect = path.size() <= 1llu + max_path_len;

        if (mode == PathSource::Camera) {
          result += direct_hit(path, spect, smp);
          if (can_connect) {
            result += connect_to_light(smp, path, spect);
          }
        } else if (conn_connect_to_camera && can_connect) {
          CameraSample camera_sample = {};
          auto splat = connect_to_camera(smp, path, spect, camera_sample);
          if (splat.is_zero() == false) {
            auto xyz = splat.to_xyz();
            rt.film().atomic_add(Film::LightIteration, {xyz.x, xyz.y, xyz.z, 1.0f}, camera_sample.uv);
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
        v.pos = ray.o + rt.scene().bounding_sphere_radius * v.w_i;
        v.nrm = -v.w_i;
        result += direct_hit(path, spect, smp);
        path.pop_back();
        break;
      } else {
        break;
      }

      if (random_continue(path_length, rt.scene().random_path_termination, eta, smp, throughput) == false) {
        break;
      }

      path_length += 1;
    }

    return result;
  }

  SpectralResponse build_camera_path(Sampler& smp, SpectralQuery spect, const float2& uv, std::vector<PathVertex>& path, uint32_t thread_id) {
    path.clear();
    auto& z0 = path.emplace_back(PathVertex::Class::Camera);
    z0.throughput = {spect, 1.0f};

    auto ray = generate_ray(smp, rt.scene(), uv);
    auto eval = film_evaluate_out(spect, rt.scene().camera, ray);

    auto& z1 = path.emplace_back(PathVertex::Class::Camera);
    z1.medium_index = rt.scene().camera_medium_index;
    z1.throughput = {spect, 1.0f};
    z1.pos = ray.o;
    z1.nrm = eval.normal;
    z1.w_i = ray.d;

    return build_path(smp, spect, ray, path, PathSource::Camera, z1.throughput, eval.pdf_dir, z1.medium_index, thread_id, {});
  }

  SpectralResponse build_emitter_path(Sampler& smp, SpectralQuery spect, std::vector<PathVertex>& path, uint32_t thread_id) {
    path.clear();
    const auto& emitter_sample = sample_emission(rt.scene(), spect, smp);
    if ((emitter_sample.pdf_area == 0.0f) || (emitter_sample.pdf_dir == 0.0f) || (emitter_sample.value.is_zero())) {
      return {spect, 0.0f};
    }

    auto& y0 = path.emplace_back(PathVertex::Class::Emitter);
    y0.throughput = {spect, 1.0f};
    y0.delta_emitter = emitter_sample.is_delta;

    auto& y1 = path.emplace_back(PathVertex::Class::Emitter);
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
    return build_path(smp, spect, {o, emitter_sample.direction}, path, PathSource::Light, throughput, emitter_sample.pdf_dir, y1.medium_index, thread_id, emitter_sample);
  }

  float mis_weight_light(std::vector<PathVertex>& camera_path, SpectralQuery spect, uint64_t eye_t, const PathVertex& sampled, Sampler& smp) const {
    if (conn_mis == false) {
      return 1.0f;
    }

    PathVertex& z_curr = camera_path[eye_t];
    PathVertex& z_prev = camera_path[eye_t - 1];

    PathVertex y_curr = sampled;
    y_curr.pdf.backward = z_curr.pdf_area(spect, PathSource::Camera, &z_prev, &y_curr, rt.scene(), smp);

    ReplaceInScope<bool> z_delta_new;
    ReplaceInScope<float> z_curr_new;
    ReplaceInScope<float> z_prev_new;

    {
      z_delta_new = {&z_curr.delta_connection, false};
      float z_curr_pdf = y_curr.pdf_area(spect, PathSource::Light, nullptr, &z_curr, rt.scene(), smp);
      ETX_VALIDATE(z_curr_pdf);
      z_curr_new = {&z_curr.pdf.backward, z_curr_pdf};
      ETX_VALIDATE(z_curr.pdf.backward);
    }

    {
      float z_prev_pdf = z_curr.pdf_area(spect, PathSource::Camera, &y_curr, &z_prev, rt.scene(), smp);
      ETX_VALIDATE(z_prev_pdf);
      z_prev_new = {&z_prev.pdf.backward, z_prev_pdf};
      ETX_VALIDATE(z_prev.pdf.backward);
    }

    float result = 0.0f;

#define MAP(A) (((A) == 0.0f) ? 1.0f : (A))

    float r = 1.0f;
    for (uint64_t ti = eye_t; ti > 1; --ti) {
      r *= MAP(camera_path[ti].pdf.backward) / MAP(camera_path[ti].pdf.forward);
      ETX_VALIDATE(r);

      if ((camera_path[ti].delta_connection == false) && (camera_path[ti - 1].delta_connection == false)) {
        result += r;
        ETX_VALIDATE(result);
      }
    }

    if (y_curr.delta_emitter == false) {
      r = MAP(y_curr.pdf.backward) / MAP(y_curr.pdf.forward);
      ETX_VALIDATE(r);
      result += r;
      ETX_VALIDATE(result);
    }

    return 1.0f / (1.0f + result);
  }

  float mis_weight_direct(std::vector<PathVertex>& camera_path, SpectralQuery spect, uint64_t eye_t, Sampler& smp) const {
    if (conn_mis == false) {
      return 1.0f;
    }

    if (eye_t == 2) {
      return 1.0f;
    }

    PathVertex& z_curr = camera_path[eye_t];
    PathVertex& z_prev = camera_path[eye_t - 1];

    ReplaceInScope<bool> z_delta_new;
    ReplaceInScope<float> z_curr_new;
    ReplaceInScope<float> z_prev_new;

    {
      z_delta_new = {&z_curr.delta_connection, false};
      float z_curr_pdf = z_curr.pdf_to_light_in(spect, &z_prev, rt.scene());
      ETX_VALIDATE(z_curr_pdf);
      z_curr_new = {&z_curr.pdf.backward, z_curr_pdf};
      ETX_VALIDATE(z_curr.pdf.backward);
    }

    {
      float z_prev_pdf = z_curr.pdf_to_light_out(spect, &z_prev, rt.scene());
      ETX_VALIDATE(z_prev_pdf);
      z_prev_new = {&z_prev.pdf.backward, z_prev_pdf};
      ETX_VALIDATE(z_prev.pdf.backward);
    }

    float result = 0.0f;

#define MAP(A) (((A) == 0.0f) ? 1.0f : (A))

    float r = 1.0f;
    for (uint64_t ti = eye_t; ti > 1; --ti) {
      r *= MAP(camera_path[ti].pdf.backward) / MAP(camera_path[ti].pdf.forward);
      ETX_VALIDATE(r);

      if ((camera_path[ti].delta_connection == false) && (camera_path[ti - 1].delta_connection == false)) {
        result += r;
        ETX_VALIDATE(result);
      }
    }

    return 1.0f / (1.0f + result);
  }

  float mis_weight_camera(std::vector<PathVertex>& emitter_path, SpectralQuery spect, uint64_t light_s, const PathVertex& sampled, Sampler& smp) const {
    if (conn_mis == false) {
      return 1.0f;
    }

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
      float pdf_dir = film_pdf_out(rt.scene().camera, y_curr.pos);
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

#define MAP(A) (((A) == 0.0f) ? 1.0f : (A))

    float r = 1.0f;
    for (uint64_t si = light_s; si > 0; --si) {
      r *= MAP(emitter_path[si].pdf.backward) / MAP(emitter_path[si].pdf.forward);
      ETX_VALIDATE(r);

      bool delta_emitter = (si > 1) ? emitter_path[si - 1u].delta_connection : emitter_path[1u].delta_emitter;
      if ((emitter_path[si].delta_connection == false) && (delta_emitter == false)) {
        result += r;
        ETX_VALIDATE(result);
      }
    }

    return 1.0f / (1.0f + result);
  }

  float mis_weight_connect(PathData& c, SpectralQuery spect, uint64_t eye_t, uint64_t light_s, const PathVertex& sampled, Sampler& smp) const {
    if (conn_mis == false) {
      return 1.0f;
    }

    PathVertex& z_curr = c.camera_path[eye_t];
    PathVertex& z_prev = c.camera_path[eye_t - 1];
    PathVertex& y_curr = c.emitter_path[light_s];
    PathVertex& y_prev = c.emitter_path[light_s - 1];

    ReplaceInScope<bool> z_delta_new;
    ReplaceInScope<float> z_curr_new;
    ReplaceInScope<float> z_prev_new;

    ReplaceInScope<bool> y_delta_new;
    ReplaceInScope<float> y_curr_new;
    ReplaceInScope<float> y_prev_new;

    {
      z_delta_new = {&z_curr.delta_connection, false};
      float z_curr_pdf = 0.0f;
      if (light_s > 0) {
        z_curr_pdf = y_curr.pdf_area(spect, PathSource::Light, &y_prev, &z_curr, rt.scene(), smp);
      } else {
        z_curr_pdf = z_curr.pdf_to_light_in(spect, &z_prev, rt.scene());
      }
      ETX_VALIDATE(z_curr_pdf);
      z_curr_new = {&z_curr.pdf.backward, z_curr_pdf};
      ETX_VALIDATE(z_curr.pdf.backward);
    }

    {
      float z_prev_pdf = z_prev_pdf = z_curr.pdf_area(spect, PathSource::Camera, &y_curr, &z_prev, rt.scene(), smp);
      ETX_VALIDATE(z_prev_pdf);
      z_prev_new = {&z_prev.pdf.backward, z_prev_pdf};
      ETX_VALIDATE(z_prev.pdf.backward);
    }

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

    float result = 0.0f;

#define MAP(A) (((A) == 0.0f) ? 1.0f : (A))

    float r = 1.0f;
    for (uint64_t ti = eye_t; ti > 1; --ti) {
      r *= MAP(c.camera_path[ti].pdf.backward) / MAP(c.camera_path[ti].pdf.forward);
      ETX_VALIDATE(r);

      if ((c.camera_path[ti].delta_connection == false) && (c.camera_path[ti - 1].delta_connection == false)) {
        result += r;
        ETX_VALIDATE(result);
      }
    }

    r = 1.0f;
    for (uint64_t si = light_s; si > 0; --si) {
      r *= MAP(c.emitter_path[si].pdf.backward) / MAP(c.emitter_path[si].pdf.forward);
      ETX_VALIDATE(r);

      bool delta_emitter = (si > 1) ? c.emitter_path[si - 1u].delta_connection : c.emitter_path[1u].delta_emitter;
      if ((c.emitter_path[si].delta_connection == false) && (delta_emitter == false)) {
        result += r;
        ETX_VALIDATE(result);
      }
    }

    return 1.0f / (1.0f + result);
  }

  SpectralResponse direct_hit(std::vector<PathVertex>& camera_path, SpectralQuery spect, Sampler& smp) const {
    if (conn_direct_hit == false)
      return {spect, 0.0f};

    uint64_t eye_t = camera_path.size() - 1u;

    const auto& z_i = camera_path[eye_t];
    if ((conn_direct_hit == false) || (z_i.is_emitter() == false)) {
      return {spect, 0.0f};
    }

    const auto& z_prev = camera_path[eye_t - 1];

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
    float weight = mis_weight_direct(camera_path, spect, eye_t, smp);
    return emitter_value * z_i.throughput * weight;
  }

  SpectralResponse connect_to_light(Sampler& smp, std::vector<PathVertex>& camera_path, SpectralQuery spect) const {
    if (conn_connect_to_light == false)
      return {spect, 0.0f};

    uint64_t eye_t = camera_path.size() - 1u;
    const auto& z_i = camera_path[eye_t];

    uint32_t emitter_index = sample_emitter_index(rt.scene(), smp);
    auto emitter_sample = sample_emitter(spect, emitter_index, smp, z_i.pos, rt.scene());
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
    float weight = mis_weight_light(camera_path, spect, eye_t, sampled_vertex, smp);
    return z_i.throughput * bsdf * emitter_throughput * tr * weight;
  }

  SpectralResponse connect_to_camera(Sampler& smp, std::vector<PathVertex>& emitter_path, SpectralQuery spect, CameraSample& camera_sample) const {
    if (conn_connect_to_camera == false)
      return {spect, 0.0f};

    uint64_t light_s = emitter_path.size() - 1u;

    const auto& y_i = emitter_path[light_s];
    camera_sample = sample_film(smp, rt.scene(), y_i.pos);
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

    float weight = mis_weight_camera(emitter_path, spect, light_s, sampled_vertex, smp);

    SpectralResponse splat = y_i.throughput * bsdf * camera_sample.weight * (weight / spect.sampling_pdf());
    ETX_VALIDATE(splat);

    if (splat.is_zero() == false) {
      splat *= local_transmittance(spect, smp, y_i, sampled_vertex);
    }

    return splat;
  }

  SpectralResponse connect_vertices(Sampler& smp, PathData& c, SpectralQuery spect, uint64_t eye_t, uint64_t light_s) const {
    if (conn_connect_vertices == false)
      return {spect, 0.0f};

    const auto& y_i = c.emitter_path[light_s];
    const auto& z_i = c.camera_path[eye_t];

    auto dw = z_i.pos - y_i.pos;
    float dwl = dot(dw, dw);
    dw *= 1.0f / std::sqrt(dwl);

    SpectralResponse result = y_i.throughput * y_i.bsdf_in_direction(spect, PathSource::Light, dw, rt.scene(), smp) *    //
                              z_i.throughput * z_i.bsdf_in_direction(spect, PathSource::Camera, -dw, rt.scene(), smp) *  //
                              (1.0f / dwl);  // G term = abs(cos(dw, y_i.nrm) * cos(dw, z_i.nrm)) / dwl; cosines already accounted in "bsdf"
    ETX_VALIDATE(result);

    if (result.is_zero()) {
      return {spect, 0.0f};
    }

    SpectralResponse tr = local_transmittance(spect, smp, y_i, z_i);
    ETX_VALIDATE(result);

    float weight = mis_weight_connect(c, spect, eye_t, light_s, {}, smp);
    ETX_VALIDATE(weight);

    return result * tr * weight;
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
      path_data.camera_path.reserve(2llu + rt.scene().max_path_length);
      path_data.emitter_path.reserve(2llu + rt.scene().max_path_length);
    }

    status = {};
    iteration_time = {};
    rt.film().clear({Film::LightIteration});
    current_task = rt.scheduler().schedule(rt.film().count(), this);
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

void CPUBidirectional::preview(const Options& opt) {
  stop(Stop::Immediate);

  if (rt.has_scene()) {
    current_state = State::Preview;
    _private->start(opt);
  }
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
    _private->iteration_time = {};
    _private->status.current_iteration += 1;
    rt.scheduler().restart(_private->current_task, rt.film().count());
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
  if (current_state == State::Preview) {
    preview(opt);
  }
}

Integrator::Status CPUBidirectional::status() const {
  return _private->status;
}

float CPUBidirectionalImpl::PathVertex::pdf_area(SpectralQuery spect, PathSource mode, const PathVertex* prev, const PathVertex* next, const Scene& scene, Sampler& smp) const {
  if (cls == Class::Emitter) {
    return pdf_to_light_out(spect, next, scene);
  }

  ETX_ASSERT(prev != nullptr);
  ETX_ASSERT(next != nullptr);
  ETX_ASSERT(is_surface_interaction() || is_medium_interaction());

  auto w_i = (pos - prev->pos);
  {
    float w_i_len = length(w_i);
    if (w_i_len == 0.0f) {
      return 0.0f;
    }
    w_i *= 1.0f / w_i_len;
  }

  auto w_o = (next->pos - pos);
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

float CPUBidirectionalImpl::PathVertex::pdf_to_light_out(SpectralQuery spect, const PathVertex* next, const Scene& scene) const {
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

float CPUBidirectionalImpl::PathVertex::pdf_to_light_in(SpectralQuery spect, const PathVertex* next, const Scene& scene) const {
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

float CPUBidirectionalImpl::PathVertex::pdf_solid_angle_to_area(float pdf_dir, const PathVertex& to_vertex) const {
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

SpectralResponse CPUBidirectionalImpl::PathVertex::bsdf_in_direction(SpectralQuery spect, PathSource mode, const float3& w_o, const Scene& scene, Sampler& smp) const {
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

}  // namespace etx
