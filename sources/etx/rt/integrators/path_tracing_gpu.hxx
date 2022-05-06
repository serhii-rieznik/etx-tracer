#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/rt/integrators/integrator.hxx>

namespace etx {

struct GPUPathTracing : public Integrator {
  const char* name() override {
    return "Path Tracing (GPU)";
  }

  const char* status() const {
    return "Path Tracing GPU";
  }

  GPUPathTracing(Raytracing& r);
  ~GPUPathTracing() override;

  Options options() const override;
  bool enabled() const override;
  void set_output_size(const uint2&) override;
  void preview(const Options&) override;
  void run(const Options&) override;
  void update() override;
  void stop(Stop) override;
  void update_options(const Options&) override;
  void reload() override;

  const float4* get_camera_image(bool force) override;

  bool have_updated_camera_image() const {
    return true;
  }

  bool have_updated_light_image() const {
    return false;
  }

  ETX_DECLARE_PIMPL(GPUPathTracing, 1024);
};

}  // namespace etx
