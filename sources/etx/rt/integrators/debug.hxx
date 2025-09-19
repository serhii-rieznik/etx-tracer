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

  void update_options() override;

  const Status& status() const override;

  void run() override;
  void update() override;
  void stop(Stop) override;

 private:
  ETX_DECLARE_PIMPL(CPUDebugIntegrator, 192);
};

}  // namespace etx
