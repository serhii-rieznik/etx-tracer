#include <etx/core/core.hxx>
#include <etx/core/core.hxx>

#include <etx/render/shared/base.hxx>
#include <etx/render/shared/bsdf.hxx>
#include <etx/render/host/rnd_sampler.hxx>
#include <etx/render/host/film.hxx>

#include <etx/rt/integrators/path_tracing.hxx>
#include <etx/rt/shared/path_tracing_shared.hxx>

namespace etx {

struct CPUPathTracingImpl : public Task {
  Raytracing& rt;
  Film& film;
  TaskScheduler& scheduler;

  TimeMeasure total_time = {};
  TimeMeasure iteration_time = {};
  Task::Handle current_task = {};
  Integrator::Status status = {};
  PTOptions options = {};

  std::atomic<uint32_t> pixels_processed = {};
  std::atomic<Integrator::State>* state = nullptr;

  CPUPathTracingImpl(Raytracing& a_rt, std::atomic<Integrator::State>* st)
    : rt(a_rt)
    , film(a_rt.film())
    , scheduler(a_rt.scheduler())
    , state(st) {
  }

  void start(const Options& opt) {
    const auto& scene = rt.scene();
    options.direct = scene.strategy_enabled(Scene::Strategy::DirectHit);
    options.nee = scene.strategy_enabled(Scene::Strategy::ConnectToLight);
    options.mis = scene.multiple_importance_sampling();
    options.blue_noise = scene.blue_noise();

    status = {};
    total_time = {};
    iteration_time = {};
    pixels_processed = 0;

    film.clear(Film::ClearCameraData);
    current_task = scheduler.schedule(film.pixel_count(), this);
  }

  void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) override {
    ETX_PROFILER_SCOPE();
    const auto& scene = rt.scene();
    const auto& camera = rt.camera();

    for (uint32_t i = begin; (state->load() != Integrator::State::Stopped) && (i < end); ++i) {
      uint2 pixel = {};
      if (film.active_pixel(i, pixel) == false) {
        continue;
      }

      pixels_processed++;
      PTRayPayload payload = make_ray_payload(scene, camera, film, pixel, i, status.current_iteration, scene.spectral(), options.blue_noise);

      while ((state->load() != Integrator::State::Stopped) && run_path_iteration(scene, options, rt, payload)) {
        ETX_VALIDATE(payload.accumulated);
      }

      auto normal = payload.view_normal;
      auto albedo = (payload.view_albedo / payload.spect.sampling_pdf()).to_rgb();
      ETX_CHECK_FINITE(albedo);
      auto color = (payload.accumulated / payload.spect.sampling_pdf()).to_rgb();
      ETX_CHECK_FINITE(color);

      if ((scene.radiance_clamp > 0.0f) && (payload.path_length > 1)) {
        float lum = luminance(color);
        if (lum > scene.radiance_clamp) {
          color *= scene.radiance_clamp / lum;
        }
      }

      film.accumulate_camera_image(pixel, color, normal, albedo);
    }
  }

  void update(std::atomic<Integrator::State>& current_state) {
    if ((current_task.data == kInvalidHandle) || (current_state == Integrator::State::Stopped) || (scheduler.completed(current_task) == false)) {
      return;
    }

    if (pixels_processed == 0) {
      current_state = Integrator::State::WaitingForCompletion;
    }

    status.last_iteration_time = iteration_time.measure();
    status.total_time += status.last_iteration_time;
    status.completed_iterations += 1u;

    scheduler.wait(current_task);
    film.estimate_noise_levels(status.current_iteration, rt.scene().samples, rt.scene().noise_threshold);

    if ((current_state == Integrator::State::WaitingForCompletion) || (status.current_iteration + 1 >= rt.scene().samples)) {
      current_task = {};
      current_state = Integrator::State::Stopped;
    } else {
      iteration_time = {};
      pixels_processed = 0;
      status.current_iteration += 1;
      current_task = scheduler.schedule(film.pixel_count(), this);
    }
  }

  void build_options(Options& integrator_options) {
    integrator_options.options.clear();
  }
};

CPUPathTracing::CPUPathTracing(Raytracing& rt)
  : Integrator(rt) {
  ETX_PIMPL_INIT(CPUPathTracing, rt, &current_state);
  _private->build_options(integrator_options);
}

CPUPathTracing::~CPUPathTracing() {
  if (current_state != State::Stopped) {
    stop(Stop::Immediate);
  }

  ETX_PIMPL_CLEANUP(CPUPathTracing);
}

const Integrator::Status& CPUPathTracing::status() const {
  return _private->status;
}

void CPUPathTracing::run() {
  stop(Stop::Immediate);

  if (can_run()) {
    current_state = State::Running;
    _private->start(integrator_options);
  }
}

void CPUPathTracing::update() {
  _private->update(current_state);
}

void CPUPathTracing::stop(Stop st) {
  if (current_state == State::Stopped) {
    return;
  }

  if (st == Stop::WaitForCompletion) {
    current_state = State::WaitingForCompletion;
  } else {
    current_state = State::Stopped;
    _private->scheduler.wait(_private->current_task);
    _private->current_task = {};
  }
}

void CPUPathTracing::update_options() {
  if (current_state == State::Running) {
    run();
  }
}

void CPUPathTracing::sync_from_options(const Options& options) {
  _private->build_options(integrator_options);
}

uint32_t CPUPathTracing::supported_strategies() const {
  return Scene::Strategy::DirectHit | Scene::Strategy::ConnectToLight;
}

float2 sample_blue_noise(const uint2& pixel, const uint32_t total_samples, const uint32_t current_sample, uint32_t dimension) {
  auto smp = BNSampler(pixel.x, pixel.y, total_samples, current_sample);
  float u = smp.get(dimension + 0u);
  float v = smp.get(dimension + 1u);
  return {u, v};
}

}  // namespace etx
