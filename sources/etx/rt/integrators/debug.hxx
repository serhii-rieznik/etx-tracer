#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/rt/integrators/integrator.hxx>

namespace etx {

struct CPUDebugIntegrator : public Integrator {
  CPUDebugIntegrator(Raytracing&);
  ~CPUDebugIntegrator() override;

  const char* name() override {
    return "Debug (CPU)";
  }

  Integrator::Type type() const override {
    return Integrator::Type::Debug;
  }

  void update_options() override;
  void sync_from_options(const Options& options) override;
  uint32_t supported_strategies() const override;

  const Status& status() const override;

  void run() override;
  void update() override;
  void stop(Stop) override;

 private:
  ETX_DECLARE_PIMPL(CPUDebugIntegrator, 192);
};

}  // namespace etx
