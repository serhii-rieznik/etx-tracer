#include <etx/core/core.hxx>
#include <etx/log/log.hxx>

#include <etx/render/host/film.hxx>
#include <etx/rt/integrators/bidirectional.hxx>

#include <atomic>

namespace etx {

struct CPUBidirectionalImpl : public Task {
  char status[2048] = {};
  Raytracing& rt;
  std::vector<RNDSampler> samplers;
  std::atomic<Integrator::State>* state = {};
  Film camera_image;
  Film light_image;
  TimeMeasure total_time = {};
  TimeMeasure iteration_time = {};
  Handle current_task = {};
  uint2 film_dimensions = {};
  uint2 current_dimensions = {};
  uint32_t iteration = 0;
  uint32_t current_scale = 1;
  uint32_t current_max_depth = 1;
  uint32_t preview_frames = 3;
  uint32_t opt_max_iterations = 0x7fffffff;
  uint32_t opt_max_depth = 0x7fffffff;
  uint32_t opt_rr_start = 0x5;

  CPUBidirectionalImpl(Raytracing& r, std::atomic<Integrator::State>* st)
    : rt(r)
    , state(st)
    , samplers(rt.scheduler().max_thread_count()) {
  }

  void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) {
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
    return {smp.next(), smp.next(), smp.next()};
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
};

CPUBidirectional::CPUBidirectional(Raytracing& rt)
  : Integrator(rt) {
  ETX_PIMPL_INIT(CPUBidirectional, rt, &current_state);
}

CPUBidirectional::~CPUBidirectional() {
  if (current_state != State::Stopped) {
    stop(false);
  }
  ETX_PIMPL_CLEANUP(CPUBidirectional);
}

void CPUBidirectional::preview(const Options& opt) {
  stop(false);

  if (rt.has_scene()) {
    current_state = State::Preview;
    _private->start(opt);
  }
}

void CPUBidirectional::run(const Options& opt) {
  stop(false);

  if (rt.has_scene()) {
    current_state = State::Running;
    _private->start(opt);
  }
}

void CPUBidirectional::update() {
  bool should_stop = (current_state != State::Stopped) || (current_state == State::WaitingForCompletion);

  if (should_stop && rt.scheduler().completed(_private->current_task)) {
    if ((current_state == State::WaitingForCompletion) || (_private->iteration >= _private->opt_max_iterations)) {
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

void CPUBidirectional::stop(bool /* wait for completion */) {
  if (current_state == State::Stopped) {
    return;
  }

  current_state = State::Stopped;
  rt.scheduler().wait(_private->current_task);
  _private->current_task = {};
}

Options CPUBidirectional::options() const {
  Options result = {};
  return result;
}

void CPUBidirectional::update_options(const Options&) {
}

void CPUBidirectional::set_output_size(const uint2& dim) {
  if (current_state != State::Stopped) {
    stop(false);
  }
  _private->film_dimensions = dim;
  _private->camera_image.resize(dim);
  _private->light_image.resize(dim);
}

float4* CPUBidirectional::get_camera_image(bool) {
  return _private->camera_image.data();
}

float4* CPUBidirectional::get_light_image(bool) {
  return _private->light_image.data();
}

const char* CPUBidirectional::status() const {
  return _private->status;
}

}  // namespace etx
