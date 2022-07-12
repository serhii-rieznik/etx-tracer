#pragma once

#include <etx/rt/integrators/integrator.hxx>

namespace etx {

struct GPUVCM : public Integrator {
  GPUVCM(Raytracing& r);
  ~GPUVCM() override;

  const char* name() override {
    return "VCM (GPU)";
  }

  bool enabled() const override;
  const char* status() const override;
  Options options() const override;
  void set_output_size(const uint2&) override;
  void preview(const Options&) override;
  void run(const Options&) override;
  void update() override;
  void stop(Stop) override;
  void update_options(const Options&) override;
  bool have_updated_camera_image() const override;
  const float4* get_camera_image(bool /* force update */) override;
  bool have_updated_light_image() const override;
  const float4* get_light_image(bool /* force update */) override;
  void reload() override;

  uint64_t debug_info_count() const override;
  DebugInfo* debug_info() const override;

  ETX_DECLARE_PIMPL(GPUVCM, 4096);
};

}  // namespace etx
