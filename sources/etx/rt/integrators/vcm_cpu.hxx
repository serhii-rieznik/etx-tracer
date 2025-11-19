#pragma once

#include <etx/rt/integrators/integrator.hxx>

namespace etx {

struct CPUVCM : public Integrator {
  CPUVCM(Raytracing&);
  ~CPUVCM();

  const char* name() override {
    return "VCM (CPU)";
  }

  Integrator::Type type() const override {
    return Integrator::Type::VCM;
  }

  void run() override;
  void update() override;
  void stop(Stop) override;
  void update_options() override;
  void sync_from_options(const Options& options) override;
  uint32_t supported_strategies() const override;

  bool have_updated_camera_image() const override;
  bool have_updated_light_image() const override;

  const Status& status() const override;

 private:
  ETX_DECLARE_PIMPL(CPUVCM, 768);
};

}  // namespace etx
