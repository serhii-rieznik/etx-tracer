#pragma once

#include <etx/core/options.hxx>
#include <etx/render/shared/scene.hxx>
#include <etx/rt/rt.hxx>

#include <atomic>

namespace etx {

struct Integrator {
  enum class State {
    Stopped,
    Preview,
    Running,
    WaitingForCompletion,
  };

  Integrator(Raytracing& r)
    : rt(r) {
  }

  virtual ~Integrator() = default;

  virtual const char* name() {
    return "Basic Integrator";
  }

  virtual Options options() const {
    Options result = {};
    result.set("desc", "No options available");
    return result;
  }

  virtual void set_output_size(const uint2&) {
  }

  virtual State state() const {
    return current_state.load();
  }

  virtual void preview(const Options&) {
  }

  virtual void run(const Options&) {
  }

  virtual void update() {
  }

  virtual void stop(bool /* wait for completion */) {
  }

  virtual void update_options(const Options&) {
  }

  virtual float4* get_updated_camera_image() {
    return nullptr;
  }

  virtual float4* get_updated_light_image() {
    return nullptr;
  }

  virtual const char* status() const {
    return "Test Integrator";
  }

  virtual bool can_run() const {
    return rt.has_scene();
  }

 protected:
  Raytracing& rt;
  std::atomic<State> current_state = {State::Stopped};
};

}  // namespace etx
