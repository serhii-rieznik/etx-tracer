#pragma once

#include <etx/rt/integrators/integrator.hxx>

namespace etx {

struct CPUVCM : public Integrator {
  CPUVCM(Raytracing&);
  ~CPUVCM();

  const char* name() {
    return "VCM (CPU)";
  }

  Options options() const override;
  void set_output_size(const uint2&) override;
  void preview(const Options&) override;
  void run(const Options&) override;
  void update() override;
  void stop(Stop) override;
  void update_options(const Options&) override;

  bool have_updated_camera_image() const override;
  bool have_updated_light_image() const override;
  const float4* get_camera_image(bool) override;
  const float4* get_light_image(bool) override;
  const char* status() const override;

 private:
  ETX_DECLARE_PIMPL(CPUVCM, 4096);
};

}  // namespace etx
