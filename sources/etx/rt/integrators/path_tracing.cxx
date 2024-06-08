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
    current_task = rt.scheduler().schedule(rt.film().count(), this);
  }

  void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) override {
    ETX_FUNCTION_SCOPE();
    auto& film = rt.film();
    for (uint32_t i = begin; (state->load() != Integrator::State::Stopped) && (i < end); ++i) {
      uint2 pixel = {
        i % film.dimensions().x,
        i / film.dimensions().x,
      };

      PTRayPayload payload = make_ray_payload(rt.scene(), film, pixel, status.current_iteration, rt.scene().spectral);

      while ((state->load() != Integrator::State::Stopped) && run_path_iteration(rt.scene(), options, rt, payload)) {
        ETX_VALIDATE(payload.accumulated);
      }

      auto normal = payload.view_normal;

      auto albedo = (payload.view_albedo / payload.spect.sampling_pdf()).to_rgb();
      ETX_CHECK_FINITE(albedo);

      auto xyz = (payload.accumulated / payload.spect.sampling_pdf()).to_rgb();
      ETX_CHECK_FINITE(xyz);

      float t = float(status.current_iteration) / float(status.current_iteration + 1);
      film.accumulate(Film::CameraIteration, {xyz.x, xyz.y, xyz.z, 1.0f}, pixel, 0.0f);
      film.accumulate(Film::Normals, {normal.x, normal.y, normal.z, 1.0f}, pixel, t);
      film.accumulate(Film::Albedo, {albedo.x, albedo.y, albedo.z, 1.0f}, pixel, t);
    }
  }

  void completed() override {
    status.last_iteration_time = iteration_time.measure();
    status.total_time += status.last_iteration_time;
    status.completed_iterations += 1u;
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
  if ((current_state == State::Stopped) || (rt.scheduler().completed(_private->current_task) == false)) {
    return;
  }

  rt.film().commit_camera_iteration(_private->status.current_iteration);

  if ((current_state == State::WaitingForCompletion) || (_private->status.current_iteration + 1 >= _private->rt.scene().samples)) {
    rt.scheduler().wait(_private->current_task);
    _private->current_task = {};
    current_state = Integrator::State::Stopped;
  } else {
    _private->iteration_time = {};
    _private->status.current_iteration += 1;
    rt.scheduler().restart(_private->current_task, rt.film().count());
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
