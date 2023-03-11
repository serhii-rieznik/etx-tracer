#pragma once

#include <etx/core/options.hxx>
#include <etx/render/shared/scene.hxx>
#include <etx/rt/rt.hxx>

#include <atomic>

namespace etx {

struct Integrator {
  enum class State : uint32_t {
    Stopped,
    Preview,
    Running,
    WaitingForCompletion,
  };

  enum class Stop : uint32_t {
    Immediate,
    WaitForCompletion,
  };

  struct DebugInfo {
    const char* title = "";
    float value = 0.0f;
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

  virtual const char* status() const {
    return "Basic Integrator (not able to render anything)";
  }

  virtual Options options() const {
    Options result = {};
    result.set("desc", "No options available");
    return result;
  }

  virtual void set_output_size(const uint2&) {
  }

  virtual void preview(const Options&) {
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
    return true;
  }

  virtual const float4* get_camera_image(bool /* force update */) {
    return nullptr;
  }

  virtual bool have_updated_light_image() const {
    return true;
  }

  virtual const float4* get_light_image(bool /* force update */) {
    return nullptr;
  }

  virtual void reload() {
  }

  virtual uint64_t debug_info_count() const {
    return 0llu;
  }

  virtual DebugInfo* debug_info() const {
    return nullptr;
  }

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

}  // namespace etx
