#include <etx/core/core.hxx>
#include <etx/log/log.hxx>

#include <etx/render/shared/base.hxx>
#include <etx/render/shared/bsdf.hxx>
#include <etx/render/host/rnd_sampler.hxx>
#include <etx/render/host/film.hxx>

#include <etx/rt/integrators/path_tracing.hxx>

namespace etx {

struct CPUPathTracingImpl : public Task {
  Raytracing& rt;
  Film camera_image;
  std::vector<RNDSampler> samplers;
  uint2 current_dimensions = {};
  char status[2048] = {};

  TimeMeasure total_time = {};
  TimeMeasure iteration_time = {};
  Task::Handle current_task = {};
  uint32_t iteration = 0u;
  uint32_t opt_max_iterations = ~0u;
  uint32_t opt_max_depth = ~0u;
  uint32_t opt_rr_start = ~0u;

  uint32_t preview_frames = 3;
  uint32_t current_max_depth = 1;
  uint32_t current_scale = 1u;
  std::atomic<Integrator::State>* state = nullptr;

  CPUPathTracingImpl(Raytracing& a_rt, std::atomic<Integrator::State>* st)
    : rt(a_rt)
    , samplers(rt.scheduler().max_thread_count())
    , state(st) {
  }

  void start(const Options& opt) {
    opt_max_iterations = opt.get("spp", opt_max_iterations).to_integer();
    opt_max_depth = opt.get("pathlen", opt_max_depth).to_integer();
    opt_rr_start = opt.get("rrstart", opt_rr_start).to_integer();

    iteration = 0;
    snprintf(status, sizeof(status), "[%u] %s ...", iteration, (state->load() == Integrator::State::Running ? "Running" : "Preview"));

    if (state->load() == Integrator::State::Running) {
      current_scale = 1;
      current_max_depth = opt_max_depth;
    } else {
      current_scale = max(1u, uint32_t(exp2(preview_frames)));
      current_max_depth = min(2u, opt_max_depth);
    }
    current_dimensions = camera_image.dimensions() / current_scale;

    total_time = {};
    iteration_time = {};
    current_task = rt.scheduler().schedule(this, current_dimensions.x * current_dimensions.y);
  }

  void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) override {
    auto& smp = samplers[thread_id];
    for (uint32_t i = begin; (state->load() != Integrator::State::Stopped) && (i < end); ++i) {
      uint32_t x = i % current_dimensions.x;
      uint32_t y = i / current_dimensions.x;
      float2 uv = get_jittered_uv(smp, {x, y}, current_dimensions);
      float4 xyz = {trace_pixel(smp, uv), 1.0f};

      if (state->load() == Integrator::State::Running) {
        camera_image.accumulate(xyz, uv, float(iteration) / float(iteration + 1));
      } else {
        float t = iteration < preview_frames ? 0.0f : float(iteration - preview_frames) / float(iteration - preview_frames + 1);
        for (uint32_t ay = 0; ay < current_scale; ++ay) {
          for (uint32_t ax = 0; ax < current_scale; ++ax) {
            uint32_t rx = x * current_scale + ax;
            uint32_t ry = y * current_scale + ay;
            camera_image.accumulate(xyz, rx, ry, t);
          }
        }
      }
    }
  }

  float3 trace_pixel(RNDSampler& smp, const float2& uv) {
    auto& scene = rt.scene();
    auto spect = spectrum::sample(smp.next());
    SpectralResponse result = {spect.wavelength, 0.0f};
    SpectralResponse throughput = {spect.wavelength, 1.0f};

    uint32_t path_length = 1;
    uint32_t medium_index = scene.camera_medium_index;

    float eta = 1.0f;
    float sampled_bsdf_pdf = 0.0f;
    bool sampled_delta_bsdf = false;

    Intersection intersection = {};
    Medium::Sample medium_sample = {};
    auto ray = generate_ray(smp, rt.scene(), uv);
    while ((state->load() != Integrator::State::Stopped) && (path_length <= current_max_depth)) {
      bool found_intersection = rt.trace(ray, intersection, smp);

      if (medium_index != kInvalidIndex) {
        medium_sample = scene.mediums[medium_index].sample(spect, smp, ray.o, ray.d, found_intersection ? intersection.t : std::numeric_limits<float>::max());
        throughput *= medium_sample.weight;
        ETX_VALIDATE(throughput);
      } else {
        medium_sample.sampled_medium = 0;
      }

      if (medium_sample.sampled_medium) {
        const auto& medium = scene.mediums[medium_index];
        /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
         * direct light sampling from medium
         * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
        if (path_length + 1 <= current_max_depth) {
          auto emitter_sample = sample_emitter(spect, smp, medium_sample.pos, scene);
          if (emitter_sample.pdf_dir > 0) {
            auto tr = transmittance(spect, smp, medium_sample.pos, emitter_sample.origin, medium_index, rt);
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
        if (path_length + 1 <= current_max_depth) {
          auto emitter_sample = sample_emitter(spect, smp, intersection.pos, scene);
          if (emitter_sample.pdf_dir > 0) {
            BSDFEval bsdf_eval = bsdf::evaluate({spect, medium_index, PathSource::Camera, intersection, ray.d, emitter_sample.direction}, mat, scene, smp);
            if (bsdf_eval.valid()) {
              auto pos = shading_pos(scene.vertices, tri, intersection.barycentric, emitter_sample.direction);
              auto tr = transmittance(spect, smp, pos, emitter_sample.origin, medium_index, rt);
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
            auto tr = transmittance(spect, smp, ray.o, intersection.pos, medium_index, rt);
            float pdf_emitter_discrete = emitter_discrete_pdf(emitter, scene.emitters_distribution);
            auto weight = ((path_length == 1) || sampled_delta_bsdf) ? 1.0f : power_heuristic(sampled_bsdf_pdf, pdf_emitter_discrete * pdf_emitter_dir);
            result += throughput * e * tr * weight;
            ETX_VALIDATE(result);
          }
        }

        /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
         * bsdf sampling
         * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
        auto bsdf_sample = bsdf::sample({spect, medium_index, PathSource::Camera, intersection, ray.d, {}}, mat, scene, smp);
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

        if ((path_length >= opt_rr_start) && (apply_rr(eta, smp.next(), throughput) == false)) {
          break;
        }

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

      ++path_length;
    }

    ETX_VALIDATE(result);
    result /= spectrum::sample_pdf();
    ETX_VALIDATE(result);

    return result.to_xyz();
  }
};

CPUPathTracing::CPUPathTracing(Raytracing& rt)
  : Integrator(rt) {
  ETX_PIMPL_INIT(CPUPathTracing, rt, &current_state);
}

CPUPathTracing::~CPUPathTracing() {
  if (current_state != State::Stopped) {
    stop(Stop::Immediate);
  }

  ETX_PIMPL_CLEANUP(CPUPathTracing);
}

void CPUPathTracing::set_output_size(const uint2& dim) {
  if (current_state != State::Stopped) {
    stop(Stop::Immediate);
  }
  _private->camera_image.resize(dim, 1);
}

const float4* CPUPathTracing::get_camera_image(bool force_update) {
  return _private->camera_image.data();
}

const float4* CPUPathTracing::get_light_image(bool force_update) {
  return nullptr;
}

const char* CPUPathTracing::status() const {
  return _private->status;
}

void CPUPathTracing::preview(const Options& opt) {
  stop(Stop::Immediate);

  if (rt.has_scene()) {
    current_state = State::Preview;
    _private->start(opt);
  }
}

void CPUPathTracing::run(const Options& opt) {
  stop(Stop::Immediate);

  if (rt.has_scene()) {
    current_state = State::Running;
    _private->start(opt);
  }
}

void CPUPathTracing::update() {
  bool should_stop = (current_state != State::Stopped) || (current_state == State::WaitingForCompletion);

  if (should_stop && rt.scheduler().completed(_private->current_task)) {
    if ((current_state == State::WaitingForCompletion) || (_private->iteration + 1 >= _private->opt_max_iterations)) {
      rt.scheduler().wait(_private->current_task);
      _private->current_task = {};
      if (current_state == State::Preview) {
        snprintf(_private->status, sizeof(_private->status), "[%u] Preview completed", _private->iteration);
        current_state = Integrator::State::Preview;
      } else {
        snprintf(_private->status, sizeof(_private->status), "[%u] Completed in %.2f seconds", _private->iteration, _private->total_time.measure());
        current_state = Integrator::State::Stopped;
      }
    } else {
      snprintf(_private->status, sizeof(_private->status), "[%u] %s... (%.3fms per iteration)", _private->iteration,
        (current_state == Integrator::State::Running ? "Running" : "Preview"), _private->iteration_time.measure_ms());
      _private->iteration_time = {};
      _private->iteration += 1;

      if (current_state == Integrator::State::Running) {
        _private->current_scale = 1;
        _private->current_max_depth = _private->opt_max_depth;
      } else {
        _private->current_scale = max(1u, uint32_t(exp2(_private->preview_frames - _private->iteration)));
        _private->current_max_depth = clamp(_private->iteration + 2, 2u, min(5u, _private->opt_max_depth));
      }
      _private->current_dimensions = _private->camera_image.dimensions() / _private->current_scale;

      rt.scheduler().restart(_private->current_task, _private->current_dimensions.x * _private->current_dimensions.y);
    }
  }
}

void CPUPathTracing::stop(Stop st) {
  if (current_state == State::Stopped) {
    return;
  }

  if (st == Stop::WaitForCompletion) {
    current_state = State::WaitingForCompletion;
    snprintf(_private->status, sizeof(_private->status), "[%u] Waiting for completion", _private->iteration);
  } else {
    current_state = State::Stopped;
    rt.scheduler().wait(_private->current_task);
    _private->current_task = {};
    snprintf(_private->status, sizeof(_private->status), "[%u] Stopped", _private->iteration);
  }
}

Options CPUPathTracing::options() const {
  Options result = {};
  result.add(1u, _private->opt_max_iterations, 0xffffu, "spp", "Samples per Pixel");
  result.add(1u, _private->opt_max_depth, 65536u, "pathlen", "Maximal Path Length");
  result.add(1u, _private->opt_rr_start, 65536u, "rrstart", "Start Russian Roulette at");
  return result;
}

void CPUPathTracing::update_options(const Options& opt) {
  if (current_state == State::Preview) {
    preview(opt);
  }
}

}  // namespace etx
