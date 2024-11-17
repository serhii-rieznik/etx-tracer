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

  TimeMeasure total_time = {};
  TimeMeasure iteration_time = {};
  Task::Handle current_task = {};
  Integrator::Status status = {};
  PTOptions options = {};

  std::atomic<uint32_t> pixels_processed = {};
  std::atomic<Integrator::State>* state = nullptr;

  CPUPathTracingImpl(Raytracing& a_rt, std::atomic<Integrator::State>* st)
    : rt(a_rt)
    , state(st) {
  }

  void start(const Options& opt) {
    ETX_PROFILER_RESET_COUNTERS();

    options.nee = opt.get("nee", options.nee).to_bool();
    options.mis = opt.get("mis", options.mis).to_bool();

    status = {};
    total_time = {};
    iteration_time = {};
    pixels_processed = 0;

    rt.film().clear();
    current_task = rt.scheduler().schedule(rt.film().active_pixel_count(), this);
  }

  void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) override {
    ETX_FUNCTION_SCOPE();
    auto& film = rt.film();
    const auto& scene = rt.scene();

    for (uint32_t i = begin; (state->load() != Integrator::State::Stopped) && (i < end); ++i) {
      uint2 pixel = {};
      if (film.active_pixel(i, pixel) == false) {
        continue;
      }

      pixels_processed++;
      PTRayPayload payload = make_ray_payload(scene, film, pixel, status.current_iteration, scene.spectral);

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

      film.accumulate(Film::CameraImage, {color.x, color.y, color.z, 1.0f}, pixel);
      film.accumulate(Film::Normals, {normal.x, normal.y, normal.z, 1.0f}, pixel);
      film.accumulate(Film::Albedo, {albedo.x, albedo.y, albedo.z, 1.0f}, pixel);
    }
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

const Integrator::Status& CPUPathTracing::status() const {
  return _private->status;
}

void CPUPathTracing::run(const Options& opt) {
  stop(Stop::Immediate);

  if (rt.has_scene()) {
    current_state = State::Running;
    _private->start(opt);
  }
}

void CPUPathTracing::update() {
  ETX_FUNCTION_SCOPE();
  if ((_private->current_task.data == kInvalidHandle) || (current_state == State::Stopped) || (rt.scheduler().completed(_private->current_task) == false)) {
    return;
  }

  if (_private->pixels_processed == 0) {
    current_state = State::WaitingForCompletion;
  }

  _private->status.last_iteration_time = _private->iteration_time.measure();
  _private->status.total_time += _private->status.last_iteration_time;
  _private->status.completed_iterations += 1u;

  rt.scheduler().wait(_private->current_task);
  rt.film().estimate_noise_levels(_private->status.current_iteration, rt.scene().samples, rt.scene().noise_threshold);

  if ((current_state == State::WaitingForCompletion) || (_private->status.current_iteration + 1 >= _private->rt.scene().samples)) {
    _private->current_task = {};
    current_state = Integrator::State::Stopped;
  } else {
    _private->iteration_time = {};
    _private->pixels_processed = 0;
    _private->status.current_iteration += 1;
    _private->current_task = rt.scheduler().schedule(rt.film().active_pixel_count(), _private);
  }
}

void CPUPathTracing::stop(Stop st) {
  if (current_state == State::Stopped) {
    return;
  }

  if (st == Stop::WaitForCompletion) {
    current_state = State::WaitingForCompletion;
  } else {
    current_state = State::Stopped;
    rt.scheduler().wait(_private->current_task);
    _private->current_task = {};
  }
}

Options CPUPathTracing::options() const {
  Options result = {};
  result.add(_private->options.nee, "nee", "Next Event Estimation");
  result.add(_private->options.mis, "mis", "Multiple Importance Sampling");
  return result;
}

void CPUPathTracing::update_options(const Options& opt) {
  if (current_state == State::Running) {
    run(opt);
  }
}

}  // namespace etx
