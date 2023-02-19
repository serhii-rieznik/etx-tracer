#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/rt/integrators/integrator.hxx>

namespace etx {

struct CPUWavefrontPT : public Integrator {
  CPUWavefrontPT(Raytracing&);
  ~CPUWavefrontPT() override;

  const char* name() override {
    return "Wavefront Path Tracing (CPU)";
  }

  Options options() const override;

  bool have_updated_light_image() const override {
    return false;
  }

  void set_output_size(const uint2&) override;
  const float4* get_camera_image(bool force_update) override;
  const float4* get_light_image(bool force_update) override;
  const char* status() const override;

  void preview(const Options&) override;
  void run(const Options&) override;
  void update() override;
  void stop(Stop) override;
  void update_options(const Options&) override;

  uint64_t debug_info_count() const override;
  DebugInfo* debug_info() const override;

  ETX_DECLARE_PIMPL(CPUWavefrontPT, 4096);
};

}  // namespace etx
