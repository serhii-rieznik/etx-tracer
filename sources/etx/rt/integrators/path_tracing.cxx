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
  uint2 current_dimensions = {};

  TimeMeasure total_time = {};
  TimeMeasure iteration_time = {};
  Task::Handle current_task = {};
  uint32_t current_scale = 1u;
  uint32_t max_samples = 1u;

  Integrator::Status status = {
    .preview_frames = 3,
  };

  PTOptions options = {};

  std::atomic<Integrator::State>* state = nullptr;

  CPUPathTracingImpl(Raytracing& a_rt, std::atomic<Integrator::State>* st)
    : rt(a_rt)
    , state(st) {
  }

  void start(const Options& opt) {
    ETX_PROFILER_RESET_COUNTERS();

    options.nee = opt.get("nee", options.nee).to_bool();
    options.mis = opt.get("mis", options.mis).to_bool();

    if (state->load() == Integrator::State::Running) {
      current_scale = 1;
    } else {
      current_scale = 1u << status.preview_frames;
    }
    current_dimensions = rt.film().dimensions() / current_scale;

    status = {
      .preview_frames = 3,
    };

    total_time = {};
    iteration_time = {};
    current_task = rt.scheduler().schedule(current_dimensions.x * current_dimensions.y, this);
  }

  void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) override {
    ETX_FUNCTION_SCOPE();
    auto& film = rt.film();
    for (uint32_t i = begin; (state->load() != Integrator::State::Stopped) && (i < end); ++i) {
      uint32_t x = i % current_dimensions.x;
      uint32_t y = i / current_dimensions.x;

      PTRayPayload payload = make_ray_payload(rt.scene(), {x, y}, current_dimensions, status.current_iteration, rt.scene().spectral);

      while ((state->load() != Integrator::State::Stopped) && run_path_iteration(rt.scene(), options, rt, payload)) {
        ETX_VALIDATE(payload.accumulated);
      }

      auto normal = payload.view_normal;

      auto albedo = (payload.view_albedo / payload.spect.sampling_pdf()).to_xyz();
      ETX_VALIDATE(albedo);

      auto xyz = (payload.accumulated / payload.spect.sampling_pdf()).to_xyz();
      ETX_VALIDATE(xyz);

      if (state->load() == Integrator::State::Running) {
        float t = float(status.current_iteration) / float(status.current_iteration + 1);
        film.accumulate(Film::Camera, {xyz.x, xyz.y, xyz.z, 1.0f}, payload.uv, t);
        film.accumulate(Film::Normals, {normal.x, normal.y, normal.z, 1.0f}, payload.uv, t);
        film.accumulate(Film::Albedo, {albedo.x, albedo.y, albedo.z, 1.0f}, payload.uv, t);
      } else {
        float t = status.current_iteration < status.preview_frames
                    ? 0.0f
                    : float(status.current_iteration - status.preview_frames) / float(status.current_iteration - status.preview_frames + 1);
        for (uint32_t ay = 0; ay < current_scale; ++ay) {
          for (uint32_t ax = 0; ax < current_scale; ++ax) {
            uint32_t rx = x * current_scale + ax;
            uint32_t ry = y * current_scale + ay;
            film.accumulate(Film::Camera, {xyz.x, xyz.y, xyz.z, 1.0f}, rx, ry, t);
            film.accumulate(Film::Normals, {normal.x, normal.y, normal.z, 1.0f}, rx, ry, t);
            film.accumulate(Film::Albedo, {albedo.x, albedo.y, albedo.z, 1.0f}, rx, ry, t);
          }
        }
      }
    }
  }

  void completed() override {
    status.last_iteration_time = iteration_time.measure();
    if (status.current_iteration >= status.preview_frames) {
      status.total_time += status.last_iteration_time;
      status.completed_iterations += 1u;
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

Integrator::Status CPUPathTracing::status() const {
  return _private->status;
}

void CPUPathTracing::preview(const Options& opt) {
  stop(Stop::Immediate);

  if (rt.has_scene()) {
    current_state = State::Preview;
    _private->max_samples = std::max(_private->status.preview_frames + 1u, rt.scene().samples);
    _private->start(opt);
  }
}

void CPUPathTracing::run(const Options& opt) {
  stop(Stop::Immediate);

  if (rt.has_scene()) {
    current_state = State::Running;
    _private->max_samples = rt.scene().samples;
    _private->start(opt);
  }
}

void CPUPathTracing::update() {
  ETX_FUNCTION_SCOPE();
  if ((current_state == State::Stopped) || (rt.scheduler().completed(_private->current_task) == false)) {
    return;
  }

  if ((current_state == State::WaitingForCompletion) || (_private->status.current_iteration + 1 >= _private->max_samples)) {
    rt.scheduler().wait(_private->current_task);
    _private->current_task = {};

    if (current_state == State::Preview) {
      current_state = Integrator::State::Preview;
    } else {
      current_state = Integrator::State::Stopped;
    }
  } else {
    _private->status.current_iteration += 1;

    if (current_state == Integrator::State::Running) {
      _private->current_scale = 1;
    } else {
      uint32_t d_frame = (_private->status.preview_frames >= _private->status.current_iteration) ? (_private->status.preview_frames - _private->status.current_iteration) : 0u;
      _private->current_scale = 1u << d_frame;
    }
    _private->current_dimensions = rt.film().dimensions() / _private->current_scale;

    _private->iteration_time = {};
    rt.scheduler().restart(_private->current_task, _private->current_dimensions.x * _private->current_dimensions.y);
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
  if (current_state == State::Preview) {
    preview(opt);
  }
}

}  // namespace etx
