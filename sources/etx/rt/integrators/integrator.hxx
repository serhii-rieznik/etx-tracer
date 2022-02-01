#pragma once

#include <etx/core/options.hxx>
#include <etx/render/shared/base.hxx>

namespace etx {

struct Integrator {
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

  virtual void stop() {
  }

  virtual void preview() {
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
};

}  // namespace etx
