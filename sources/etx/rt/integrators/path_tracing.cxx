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
  Film camera_image;
  uint2 current_dimensions = {};
  char status[2048] = {};

  TimeMeasure total_time = {};
  TimeMeasure iteration_time = {};
  Task::Handle current_task = {};
  uint32_t iteration = 0u;
  uint32_t preview_frames = 3;
  uint32_t current_scale = 1u;
  uint32_t max_samples = 1u;
  double last_iteration_time = 0.0;
  double total_iterations_time = 0.0;

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

    iteration = 0;

    if (state->load() == Integrator::State::Running) {
      current_scale = 1;
    } else {
      current_scale = 1u << preview_frames;
    }
    current_dimensions = camera_image.dimensions() / current_scale;

    total_time = {};
    iteration_time = {};
    last_iteration_time = 0.0;
    total_iterations_time = 0.0;
    current_task = rt.scheduler().schedule(current_dimensions.x * current_dimensions.y, this);
  }

  void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) override {
    ETX_FUNCTION_SCOPE();
    for (uint32_t i = begin; (state->load() != Integrator::State::Stopped) && (i < end); ++i) {
      uint32_t x = i % current_dimensions.x;
      uint32_t y = i / current_dimensions.x;

      PTRayPayload payload = make_ray_payload(rt.scene(), {x, y}, current_dimensions, iteration, rt.scene().spectral);

      while ((state->load() != Integrator::State::Stopped) && run_path_iteration(rt.scene(), options, rt, payload)) {
        ETX_VALIDATE(payload.accumulated);
      }

      auto xyz = (payload.accumulated / payload.spect.sampling_pdf()).to_xyz();
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

  void completed() override {
    last_iteration_time = iteration_time.measure();
    if (iteration >= preview_frames) {
      total_iterations_time += last_iteration_time;
    }
  }

  void update_status(Integrator::State st) {
    constexpr const char* status_str[] = {
      "Stopped",
      "Preview",
      "Running",
      "Completing",
    };

    uint32_t i_frame = (iteration >= preview_frames) ? (iteration - preview_frames) : 0u;
    double average_time = total_iterations_time / double(i_frame + 1u);

    snprintf(status, sizeof(status), "%-4d | %s | %.3fms last, %.3fms avg, %.3fs total",  //
      iteration, status_str[uint32_t(st)],                                                //
      last_iteration_time * 1000.0, average_time * 1000.0f, total_iterations_time);
  };
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
  _private->camera_image.allocate(dim, Film::Layer::CameraRays);
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
    _private->max_samples = std::max(_private->preview_frames + 1u, rt.scene().samples);
    _private->start(opt);
  }

  _private->update_status(current_state);
}

void CPUPathTracing::run(const Options& opt) {
  stop(Stop::Immediate);

  if (rt.has_scene()) {
    current_state = State::Running;
    _private->max_samples = rt.scene().samples;
    _private->start(opt);
  }

  _private->update_status(current_state);
}

void CPUPathTracing::update() {
  ETX_FUNCTION_SCOPE();
  if ((current_state == State::Stopped) || (rt.scheduler().completed(_private->current_task) == false)) {
    return;
  }

  if ((current_state == State::WaitingForCompletion) || (_private->iteration + 1 >= _private->max_samples)) {
    rt.scheduler().wait(_private->current_task);
    _private->current_task = {};

    if (current_state == State::Preview) {
      current_state = Integrator::State::Preview;
    } else {
      current_state = Integrator::State::Stopped;
    }

    _private->update_status(current_state);
  } else {
    _private->update_status(current_state);

    _private->iteration += 1;

    if (current_state == Integrator::State::Running) {
      _private->current_scale = 1;
    } else {
      uint32_t d_frame = (_private->preview_frames >= _private->iteration) ? (_private->preview_frames - _private->iteration) : 0u;
      _private->current_scale = 1u << d_frame;
    }
    _private->current_dimensions = _private->camera_image.dimensions() / _private->current_scale;

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

  _private->update_status(current_state);
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

uint32_t CPUPathTracing::sample_count() const {
  return _private->iteration;
}

}  // namespace etx
