#include <etx/core/core.hxx>
#include <etx/log/log.hxx>
#include <etx/render/host/rnd_sampler.hxx>
#include <etx/render/shared/base.hxx>
#include <etx/render/shared/bsdf.hxx>
#include <etx/rt/integrators/path_tracing.hxx>

namespace etx {

struct CPUPathTracingImpl : public Task {
  Raytracing& rt;
  std::vector<float4> camera_image;
  std::vector<RNDSampler> samplers;
  uint2 dimensions = {};
  char status[2048] = {};

  TimeMeasure total_time = {};
  TimeMeasure iteration_time = {};
  Task::Handle current_task = {};
  uint32_t iteration = 0u;
  uint32_t max_iterations = ~0u;
  uint32_t max_depth = ~0u;
  uint32_t rr_start = ~0u;
  std::atomic<Integrator::State>* state = nullptr;

  CPUPathTracingImpl(Raytracing& a_rt, std::atomic<Integrator::State>* st)
    : rt(a_rt)
    , samplers(rt.scheduler().max_thread_count())
    , state(st) {
  }

  void start() {
    iteration = 0;
    dimensions = {rt.scene().camera.image_size.x, rt.scene().camera.image_size.y};
    snprintf(status, sizeof(status), "[%u] %s ...", iteration, (state->load() == Integrator::State::Running ? "Running" : "Preview"));

    total_time = {};
    iteration_time = {};
    current_task = rt.scheduler().schedule(this, dimensions.x * dimensions.y);
  }

  void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) override {
    auto& smp = samplers[thread_id];
    auto mode = state->load();
    if (mode == Integrator::State::Preview) {
      for (uint32_t i = begin; (state->load() != Integrator::State::Stopped) && (i < end); ++i) {
        uint32_t x = i % dimensions.x;
        uint32_t y = i / dimensions.x;
        float4 xyz = {preview_pixel(smp, x, y), 1.0f};

        uint32_t j = x + (dimensions.y - 1 - y) * dimensions.x;
        camera_image[j] = (iteration == 0) ? xyz : lerp(xyz, camera_image[j], float(iteration) / (float(iteration + 1)));
      }
    } else {
      for (uint32_t i = begin; (state->load() != Integrator::State::Stopped) && (i < end); ++i) {
        uint32_t x = i % dimensions.x;
        uint32_t y = i / dimensions.x;
        float4 xyz = {trace_pixel(smp, x, y), 1.0f};

        uint32_t j = x + (dimensions.y - 1 - y) * dimensions.x;
        camera_image[j] = (iteration == 0) ? xyz : lerp(xyz, camera_image[j], float(iteration) / (float(iteration + 1)));
      }
    }
  }

  float3 trace_pixel(RNDSampler& smp, uint32_t x, uint32_t y) {
    auto& scene = rt.scene();
    auto spect = spectrum::sample(smp.next());
    SpectralResponse result = {spect.wavelength, 0.0f};
    SpectralResponse throughput = {spect.wavelength, 1.0f};

    uint32_t path_length = 1;
    uint32_t medium_index = scene.camera_medium_index;

    float eta = 1.0f;
    float sampled_bsdf_pdf = 0.0f;
    bool sampled_delta_bsdf = false;

    auto ray = generate_ray(smp, rt.scene(), get_jittered_uv(smp, {x, y}, dimensions));
    while ((state->load() != Integrator::State::Stopped) && (path_length <= max_depth)) {
      Intersection intersection = {};
      bool found_intersection = rt.trace(ray, intersection, smp);

      Medium::Sample medium_sample{};
      if (medium_index != kInvalidIndex) {
        medium_sample = scene.mediums[medium_index].sample(spect, smp, ray.o, ray.d, found_intersection ? intersection.t : std::numeric_limits<float>::max());
        throughput *= medium_sample.weight;
      }

      if (medium_sample.sampled_medium) {
        const auto& medium = scene.mediums[medium_index];
        /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
         * direct light sampling from medium
         * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
        if (path_length + 1 <= max_depth) {
          auto emitter_sample = sample_emitter(spect, smp, medium_sample.pos, scene);
          if (emitter_sample.pdf_dir > 0) {
            auto tr = transmittance(spect, smp, medium_sample.pos, emitter_sample.origin, medium_index);
            float phase_function = medium.phase_function(spect, medium_sample.pos, ray.d, emitter_sample.direction);
            auto weight = emitter_sample.is_delta ? 1.0f : power_heuristic(emitter_sample.pdf_dir * emitter_sample.pdf_sample, phase_function);
            result += throughput * emitter_sample.value * tr * (phase_function * weight / (emitter_sample.pdf_dir * emitter_sample.pdf_sample));
            ETX_VALIDATE(result);
          }
        }

        float3 w_o = medium.sample_phase_function(spect, smp, medium_sample.pos, ray.d);
        sampled_bsdf_pdf = medium.phase_function(spect, medium_sample.pos, ray.d, w_o);
        sampled_delta_bsdf = false;

        ray.o = medium_sample.pos;
        ray.d = w_o;

      } else if (found_intersection) {
        const auto& tri = scene.triangles[intersection.triangle_index];
        const auto& mat = scene.materials[tri.material_index];

        if (mat.cls == Material::Class::Boundary) {
          medium_index = (dot(intersection.nrm, ray.d) < 0.0f) ? mat.int_medium : mat.ext_medium;
          ray.o = shading_pos(scene.vertices, tri, intersection.barycentric, ray.d);
          continue;
        }

        /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
         * direct light sampling
         * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
        if (path_length + 1 <= max_depth) {
          auto emitter_sample = sample_emitter(spect, smp, intersection.pos, scene);
          if (emitter_sample.pdf_dir > 0) {
            BSDFEval bsdf_eval = bsdf::evaluate({spect, medium_index, mat, PathSource::Camera, intersection, ray.d, emitter_sample.direction}, scene);
            if (bsdf_eval.valid()) {
              auto pos = shading_pos(scene.vertices, tri, intersection.barycentric, emitter_sample.direction);
              auto tr = transmittance(spect, smp, pos, emitter_sample.origin, medium_index);
              auto weight = emitter_sample.is_delta ? 1.0f : power_heuristic(emitter_sample.pdf_dir * emitter_sample.pdf_sample, bsdf_eval.pdf);
              result += throughput * bsdf_eval.bsdf * emitter_sample.value * tr * (weight / (emitter_sample.pdf_dir * emitter_sample.pdf_sample));
              ETX_VALIDATE(result);
            }
          }
        }

        /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
         * directly visible emitters
         * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
        if (tri.emitter_index != kInvalidIndex) {
          const auto& emitter = scene.emitters[tri.emitter_index];
          float pdf_emitter_area = 0.0f;
          float pdf_emitter_dir = 0.0f;
          float pdf_emitter_dir_out = 0.0f;
          auto e =
            emitter_evaluate_in_local(emitter, spect, intersection.tex, ray.o, intersection.pos, pdf_emitter_area, pdf_emitter_dir, pdf_emitter_dir_out, scene, (path_length == 0));
          if (pdf_emitter_dir > 0.0f) {
            auto tr = transmittance(spect, smp, ray.o, intersection.pos, medium_index);
            float pdf_emitter_discrete = emitter_discrete_pdf(emitter, scene.emitters_distribution);
            auto weight = ((path_length == 1) || sampled_delta_bsdf) ? 1.0f : power_heuristic(sampled_bsdf_pdf, pdf_emitter_discrete * pdf_emitter_dir);
            result += throughput * e * tr * weight;
            ETX_VALIDATE(result);
          }
        }

        /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
         * bsdf sampling
         * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
        auto bsdf_sample = bsdf::sample({spect, medium_index, mat, PathSource::Camera, intersection, ray.d, {}}, scene, smp);
        if (bsdf_sample.valid() == false) {
          break;
        }

        if (bsdf_sample.properties & BSDFSample::MediumChanged) {
          medium_index = bsdf_sample.medium_index;
        }

        ETX_VALIDATE(throughput);
        throughput *= bsdf_sample.weight;
        ETX_VALIDATE(throughput);

        if (throughput.is_zero()) {
          break;
        }

        sampled_bsdf_pdf = bsdf_sample.pdf;
        sampled_delta_bsdf = bsdf_sample.is_delta();
        eta *= bsdf_sample.eta;

        ray.d = bsdf_sample.w_o;
        ray.o = shading_pos(scene.vertices, tri, intersection.barycentric, bsdf_sample.w_o);

      } else {
        for (uint32_t ie = 0; ie < scene.environment_emitters.count; ++ie) {
          const auto& emitter = scene.emitters[scene.environment_emitters.emitters[ie]];
          float pdf_emitter_area = 0.0f;
          float pdf_emitter_dir = 0.0f;
          float pdf_emitter_dir_out = 0.0f;
          auto e = emitter_evaluate_in_dist(emitter, spect, ray.d, pdf_emitter_area, pdf_emitter_dir, pdf_emitter_dir_out, scene);
          ETX_VALIDATE(e);
          if ((pdf_emitter_dir > 0) && (e.is_zero() == false)) {
            float pdf_emitter_discrete = emitter_discrete_pdf(emitter, scene.emitters_distribution);
            auto weight = ((path_length == 1) || sampled_delta_bsdf) ? 1.0f : power_heuristic(sampled_bsdf_pdf, pdf_emitter_discrete * pdf_emitter_dir);
            result += throughput * e * weight;
            ETX_VALIDATE(result);
          }
        }
        break;
      }

      if ((path_length >= rr_start) && (apply_rr(eta, smp.next(), throughput) == false)) {
        break;
      }

      ++path_length;
    }

    return (result / spectrum::sample_pdf()).to_xyz();
  }

  float3 preview_pixel(RNDSampler& smp, uint32_t x, uint32_t y) {
    auto ray = generate_ray(smp, rt.scene(), get_jittered_uv(smp, {x, y}, dimensions));

    float3 xyz = {0.1f, 0.1f, 0.1f};

    Intersection intersection;
    if (rt.trace(ray, intersection, smp)) {
      float d = fabsf(dot(intersection.nrm, ray.d));
      xyz = spectrum::rgb_to_xyz({d, d, d});
    }
    return xyz;
  }

  SpectralResponse transmittance(SpectralQuery spect, Sampler& smp, const float3& p0, const float3& p1, uint32_t medium_index) {
    const auto& scene = rt.scene();
    float3 w_o = p1 - p0;
    float max_t = length(w_o);
    w_o /= max_t;
    max_t -= kRayEpsilon;

    float3 origin = p0;

    SpectralResponse result = {spect.wavelength, 1.0f};

    for (;;) {
      Intersection intersection;
      if (rt.trace({origin, w_o, kRayEpsilon, max_t}, intersection, smp) == false) {
        if (medium_index != kInvalidIndex) {
          result *= scene.mediums[medium_index].transmittance(spect, smp, origin, w_o, max_t);
        }
        break;
      }

      const auto& tri = scene.triangles[intersection.triangle_index];
      const auto& mat = scene.materials[tri.material_index];
      if (mat.cls != Material::Class::Boundary) {
        result = {spect.wavelength, 0.0f};
        break;
      }

      if (medium_index != kInvalidIndex) {
        result *= scene.mediums[medium_index].transmittance(spect, smp, origin, w_o, intersection.t);
      }

      medium_index = (dot(intersection.nrm, w_o) < 0.0f) ? mat.int_medium : mat.ext_medium;
      origin = intersection.pos;
      max_t -= intersection.t;
    }

    return result;
  }
};

CPUPathTracing::CPUPathTracing(CPUPathTracing&& other) noexcept
  : Integrator(other.rt) {
  if (_private) {
    _private->~CPUPathTracingImpl();
  }
  memcpy(_private_storage, other._private_storage, sizeof(_private_storage));
  _private = reinterpret_cast<struct CPUPathTracingImpl*>(_private_storage);
  memset(other._private_storage, 0, sizeof(_private_storage));
  other._private = nullptr;
}

CPUPathTracing& CPUPathTracing::operator=(CPUPathTracing&& other) noexcept {
  if (_private) {
    _private->~CPUPathTracingImpl();
  }
  memcpy(_private_storage, other._private_storage, sizeof(_private_storage));
  _private = reinterpret_cast<struct CPUPathTracingImpl*>(_private_storage);
  memset(other._private_storage, 0, sizeof(_private_storage));
  other._private = nullptr;
  return *this;
}

CPUPathTracing::CPUPathTracing(Raytracing& rt)
  : Integrator(rt) {
  ETX_PIMPL_INIT(CPUPathTracing, rt, &current_state);
}

CPUPathTracing::~CPUPathTracing() {
  if (current_state != State::Stopped) {
    stop();
  }

  ETX_PIMPL_CLEANUP(CPUPathTracing);
}

void CPUPathTracing::set_output_size(const uint2& dim) {
  _private->dimensions = dim;
  _private->camera_image.resize(1llu * dim.x * dim.y);
}

float4* CPUPathTracing::get_updated_camera_image() {
  return _private->camera_image.data();
}

float4* CPUPathTracing::get_updated_light_image() {
  return nullptr;
}

const char* CPUPathTracing::status() const {
  return _private->status;
}

void CPUPathTracing::preview() {
  stop();

  if (rt.has_scene()) {
    current_state = State::Preview;
    _private->start();
  }
}

void CPUPathTracing::run(const Options& opt) {
  stop();

  _private->max_iterations = opt.get("spp", _private->max_iterations).to_integer();
  _private->max_depth = opt.get("pathlen", _private->max_depth).to_integer();
  _private->rr_start = opt.get("rrstart", _private->rr_start).to_integer();

  if (rt.has_scene()) {
    current_state = State::Running;
    _private->start();
  }
}

void CPUPathTracing::update() {
  if ((current_state != State::Stopped) && rt.scheduler().completed(_private->current_task)) {
    if (_private->iteration >= _private->max_iterations) {
      rt.scheduler().wait(_private->current_task);
      snprintf(_private->status, sizeof(_private->status), "[%u] Completed in %.2f seconds", _private->iteration, _private->total_time.measure());
      _private->current_task = {};
      current_state = Integrator::State::Stopped;
    } else {
      snprintf(_private->status, sizeof(_private->status), "[%u] %s... (%.3fms per iteration)", _private->iteration,
        (current_state == Integrator::State::Running ? "Running" : "Preview"), _private->iteration_time.measure_ms());
      _private->iteration_time = {};
      _private->iteration += 1;
      rt.scheduler().restart(_private->current_task);
    }
  }
}

void CPUPathTracing::stop() {
  if (current_state == State::Stopped) {
    return;
  }

  current_state = State::Stopped;
  rt.scheduler().wait(_private->current_task);
  _private->current_task = {};

  snprintf(_private->status, sizeof(_private->status), "[%u] Stopped", _private->iteration);
}

}  // namespace etx
