#pragma once

#include <etx/core/profiler.hxx>
#include <etx/render/shared/scene.hxx>
#include <etx/util/options.hxx>
#include <etx/rt/rt.hxx>

#include <atomic>

namespace etx {

struct Integrator {
  enum class State : uint32_t {
    Stopped,
    Running,
    WaitingForCompletion,
  };

  enum class Stop : uint32_t {
    Immediate,
    WaitForCompletion,
  };

  struct Status {
    struct DebugInfo {
      const char* title = "";
      float value = 0.0f;
    };

    double last_iteration_time = 0.0;
    double total_time = 0.0;
    uint32_t completed_iterations = 0;
    uint32_t current_iteration = 0;

    DebugInfo* debug_info = nullptr;
    uint32_t debug_info_count = 0;
  };

  Integrator(Raytracing& r)
    : rt(r) {
  }

  virtual ~Integrator() = default;

  virtual const char* name() {
    return "Basic Integrator";
  }

  virtual bool enabled() const {
    return true;
  }

  virtual const char* status_str() const {
    return "Basic Integrator (not able to render anything)";
  }

  virtual Options options() const {
    Options result = {};
    result.set("desc", "No options available");
    return result;
  }

  virtual void run(const Options&) {
  }

  virtual void update() {
  }

  virtual void stop(Stop) {
  }

  virtual void update_options(const Options&) {
  }

  virtual bool have_updated_camera_image() const {
    return state() != State::Stopped;
  }

  virtual bool have_updated_light_image() const {
    return state() != State::Stopped;
  }

  virtual const Status& status() const = 0;

 public:
  bool can_run() const {
    return rt.has_scene();
  }
  State state() const {
    return current_state.load();
  }

 protected:
  Raytracing& rt;
  std::atomic<State> current_state = {State::Stopped};
};

struct TaskScheduler;
struct IntegratorThreadImpl;
struct IntegratorThread {
  enum Mode : uint32_t {
    ExternalControl,
    Async,
  };

  IntegratorThread(TaskScheduler&, Mode mode);
  ~IntegratorThread();

  void start();
  void terminate();

  void update();

  Integrator* integrator();
  void set_integrator(Integrator*);

  bool running();
  const Integrator::Status& status() const;

  void run(const Options&);
  void stop(Integrator::Stop);
  void restart(const Options&);
  void restart();

 private:
  ETX_DECLARE_PIMPL(IntegratorThread, 256);
};

}  // namespace etx
