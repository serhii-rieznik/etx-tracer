#pragma once

#include <etx/rt/integrators/integrator.hxx>

namespace etx {

struct CPUUPBP : public Integrator {
  CPUUPBP(Raytracing&);
  ~CPUUPBP();

  const char* name() override {
    return "UPBP (CPU)";
  }

  Integrator::Type type() const override {
    return Integrator::Type::UPBP;
  }

  const char* status_str() const override;
  bool failed() const override;
  const char* failure_reason() const override;
  PathProgress path_progress() const override;
  void run() override;
  void update() override;
  void stop(Stop) override;
  void update_options() override;
  void sync_from_options(const Options& options) override;
  uint32_t supported_strategies() const override;
  const Status& status() const override;

 private:
  ETX_DECLARE_PIMPL(CPUUPBP, 4096);
};

}  // namespace etx
