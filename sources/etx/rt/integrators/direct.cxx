#include <etx/core/core.hxx>
#include <etx/log/log.hxx>

#include <etx/render/shared/base.hxx>
#include <etx/render/shared/bsdf.hxx>
#include <etx/render/host/rnd_sampler.hxx>
#include <etx/render/host/film.hxx>

#include <etx/rt/integrators/direct.hxx>
#include <etx/rt/shared/path_tracing_shared.hxx>

namespace etx {

namespace restir {

struct Reservoir {
  EmitterSample sample = {};
  uint32_t sample_count = 0;
  float total_weight = 0.0f;
  float target_weight = 0.0f;
  float weight = 0.0f;

  void add_sample(float sample_weight, float sample_pdf, const EmitterSample& emitter_sample, Sampler& smp) {
    float weight = sample_weight / sample_pdf;
    total_weight += weight;

    if (smp.next() < weight / total_weight) {
      target_weight = sample_weight;
      sample = emitter_sample;
    }
  }

  void compute_weight() {
    weight = (total_weight == 0.0f) ? 0.0f : total_weight / (target_weight * float(sample_count));
    ETX_VALIDATE(weight);
  }
};

Reservoir generate_samples(const SpectralQuery& spect, Sampler& smp, const Intersection& intersection, const Scene& scene, bool use_bsdf) {
  const auto& tri = scene.triangles[intersection.triangle_index];
  const auto& mat = scene.materials[tri.material_index];
  const float emitter_sample_pdf = 1.0f / float(scene.emitters.count);

  Reservoir result = {};
  result.sample_count = 32u;
  for (uint32_t i = 0; i < result.sample_count; ++i) {
    uint32_t emitter_index = uint32_t(smp.next() / emitter_sample_pdf);
    auto emitter_sample = sample_emitter(spect, emitter_index, smp, intersection.pos, scene);
    if (emitter_sample.pdf_dir > 0.0f) {
      float bsdf_pdf = bsdf::pdf({spect, kInvalidIndex, PathSource::Camera, intersection, intersection.w_i, emitter_sample.direction}, mat, scene, smp);
      if ((use_bsdf == false) || (bsdf_pdf > 0.0f)) {
        float weight = (use_bsdf ? bsdf_pdf : 1.0f) * emitter_sample.value.to_xyz().y / emitter_sample.pdf_dir;
        result.add_sample(weight, emitter_sample_pdf, emitter_sample, smp);
      }
    }
  }
  result.compute_weight();
  return result;
}

}  // namespace restir

struct ETX_ALIGNED DirectOptions {
  uint32_t iterations = 1u;
  bool ris_sampling = false;
  bool ris_bsdf = false;
};

struct ETX_ALIGNED DirectPayload {
  Ray ray = {};
  SpectralResponse accumulated = {spectrum::kUndefinedWavelength, 0.0f};
  uint32_t medium = kInvalidIndex;
  uint32_t iteration = 0u;
  SpectralQuery spect = {};
  float2 uv = {};
  Sampler smp = {};

  static DirectPayload make(const Scene& scene, uint2 px, uint2 dim, uint32_t iteration) {
    uint32_t index = px.x + px.y * dim.x;
    DirectPayload payload = {};
    payload.iteration = iteration;
    payload.smp.init(index, payload.iteration);
    payload.spect = spectrum::sample(payload.smp.next());
    payload.uv = get_jittered_uv(payload.smp, px, dim);
    payload.ray = generate_ray(payload.smp, scene, payload.uv);
    payload.accumulated = {payload.spect.wavelength, 0.0f};
    payload.medium = scene.camera_medium_index;
    return payload;
  }
};

struct CPUDirectLightingImpl : public Task {
  Raytracing& rt;
  Film camera_image;
  uint2 current_dimensions = {};
  char status[2048] = {};

  TimeMeasure total_time = {};
  TimeMeasure iteration_time = {};
  Task::Handle current_task = {};
  uint32_t iteration = 0u;
  uint32_t preview_frames = 3;
  uint32_t current_scale = 1u;

  DirectOptions options = {};

  std::atomic<Integrator::State>* state = nullptr;

  CPUDirectLightingImpl(Raytracing& a_rt, std::atomic<Integrator::State>* st)
    : rt(a_rt)
    , state(st) {
  }

  void start(const Options& opt) {
    options.iterations = opt.get("spp", options.iterations).to_integer();
    options.ris_sampling = opt.get("ris", options.ris_sampling).to_bool();
    options.ris_bsdf = opt.get("ris-bsdf", options.ris_bsdf).to_bool();

    iteration = 0;
    snprintf(status, sizeof(status), "[%u] %s ...", iteration, (state->load() == Integrator::State::Running ? "Running" : "Preview"));

    if (true || state->load() == Integrator::State::Running) {
      current_scale = 1;
    } else {
      current_scale = max(1u, uint32_t(exp2(preview_frames)));
    }
    current_dimensions = camera_image.dimensions() / current_scale;

    total_time = {};
    iteration_time = {};
    current_task = rt.scheduler().schedule(this, current_dimensions.x * current_dimensions.y);
  }

  void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) override {
    for (uint32_t i = begin; (state->load() != Integrator::State::Stopped) && (i < end); ++i) {
      uint32_t x = i % current_dimensions.x;
      uint32_t y = i / current_dimensions.x;

      const auto& scene = rt.scene();
      auto payload = DirectPayload::make(scene, {x, y}, current_dimensions, iteration);

      Intersection intersection = {};
      if (rt.trace(scene, payload.ray, intersection, payload.smp)) {
        const auto& tri = scene.triangles[intersection.triangle_index];
        const auto& mat = scene.materials[tri.material_index];

        EmitterSample emitter_sample = {};

        if (options.ris_sampling) {
          auto reservoir = restir::generate_samples(payload.spect, payload.smp, intersection, scene, options.ris_bsdf);
          emitter_sample = reservoir.sample;
          emitter_sample.pdf_sample = 1.0f / reservoir.weight;
          if (emitter_sample.pdf_sample <= 0.0f) {
            printf("!");
          }
        } else {
          emitter_sample = sample_emitter(payload.spect, payload.smp, intersection.pos, scene);
        }

        if (emitter_sample.pdf_dir > 0) {
          BSDFEval bsdf_eval = bsdf::evaluate({payload.spect, payload.medium, PathSource::Camera, intersection, payload.ray.d, emitter_sample.direction}, mat, scene, payload.smp);
          if (bsdf_eval.valid()) {
            auto pos = shading_pos(scene.vertices, tri, intersection.barycentric, emitter_sample.direction);
            auto tr = transmittance(payload.spect, payload.smp, pos, emitter_sample.origin, payload.medium, scene, rt);
            payload.accumulated += bsdf_eval.bsdf * emitter_sample.value * tr / (emitter_sample.pdf_dir * emitter_sample.pdf_sample);
            ETX_VALIDATE(payload.accumulated);
          }
        }

        if (tri.emitter_index != kInvalidIndex) {
          const auto& emitter = scene.emitters[tri.emitter_index];
          float pdf_a = 0.0f;
          float pdf_do = 0.0f;
          float pdf_emitter_dir = 0.0f;
          auto e = emitter_get_radiance(emitter, payload.spect, intersection.tex, payload.ray.o, intersection.pos, pdf_a, pdf_emitter_dir, pdf_do, scene, true);
          if (pdf_emitter_dir > 0.0f) {
            auto tr = transmittance(payload.spect, payload.smp, payload.ray.o, intersection.pos, payload.medium, scene, rt);
            payload.accumulated += e * tr;
            ETX_VALIDATE(payload.accumulated);
          }
        }
      }

      auto xyz = (payload.accumulated / spectrum::sample_pdf()).to_xyz();
      ETX_VALIDATE(xyz);

      if (state->load() == Integrator::State::Running) {
        camera_image.accumulate({xyz.x, xyz.y, xyz.z, 1.0f}, payload.uv, float(iteration) / float(iteration + 1));
      } else {
        float t = iteration < preview_frames ? 0.0f : float(iteration - preview_frames) / float(iteration - preview_frames + 1);
        for (uint32_t ay = 0; ay < current_scale; ++ay) {
          for (uint32_t ax = 0; ax < current_scale; ++ax) {
            uint32_t rx = x * current_scale + ax;
            uint32_t ry = y * current_scale + ay;
            camera_image.accumulate({xyz.x, xyz.y, xyz.z, 1.0f}, rx, ry, t);
          }
        }
      }
    }
  }
};

CPUDirectLighting::CPUDirectLighting(Raytracing& rt)
  : Integrator(rt) {
  ETX_PIMPL_INIT(CPUDirectLighting, rt, &current_state);
}

CPUDirectLighting::~CPUDirectLighting() {
  if (current_state != State::Stopped) {
    stop(Stop::Immediate);
  }

  ETX_PIMPL_CLEANUP(CPUDirectLighting);
}

void CPUDirectLighting::set_output_size(const uint2& dim) {
  if (current_state != State::Stopped) {
    stop(Stop::Immediate);
  }
  _private->camera_image.resize(dim, 1);
}

const float4* CPUDirectLighting::get_camera_image(bool force_update) {
  return _private->camera_image.data();
}

const float4* CPUDirectLighting::get_light_image(bool force_update) {
  return nullptr;
}

const char* CPUDirectLighting::status() const {
  return _private->status;
}

void CPUDirectLighting::preview(const Options& opt) {
  stop(Stop::Immediate);

  if (rt.has_scene()) {
    current_state = State::Preview;
    _private->start(opt);
  }
}

void CPUDirectLighting::run(const Options& opt) {
  stop(Stop::Immediate);

  if (rt.has_scene()) {
    current_state = State::Running;
    _private->start(opt);
  }
}

void CPUDirectLighting::update() {
  bool should_stop = (current_state != State::Stopped) || (current_state == State::WaitingForCompletion);

  if (should_stop && rt.scheduler().completed(_private->current_task)) {
    if ((current_state == State::WaitingForCompletion) || (_private->iteration + 1 >= _private->options.iterations)) {
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
      } else {
        _private->current_scale = max(1u, uint32_t(exp2(_private->preview_frames - _private->iteration)));
      }
      _private->current_dimensions = _private->camera_image.dimensions() / _private->current_scale;

      rt.scheduler().restart(_private->current_task, _private->current_dimensions.x * _private->current_dimensions.y);
    }
  }
}

void CPUDirectLighting::stop(Stop st) {
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

Options CPUDirectLighting::options() const {
  Options result = {};
  result.add(1u, _private->options.iterations, 0xffffu, "spp", "Samples per Pixel");
  result.add(_private->options.ris_sampling, "ris", "RIS sampling");
  result.add(_private->options.ris_bsdf, "ris-bsdf", "BSDF in RIS sampling");
  return result;
}

void CPUDirectLighting::update_options(const Options& opt) {
  if (current_state == State::Preview) {
    preview(opt);
  }
}

}  // namespace etx
