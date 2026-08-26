#pragma once

#include <etx/rt/integrators/integrator.hxx>
#include <etx/rt/shared/bdpt_mode.hxx>

namespace etx {

struct CPUBidirectional : public Integrator {
  CPUBidirectional(Raytracing&);
  CPUBidirectional(Raytracing&, BDPTMode, Integrator::Type);
  ~CPUBidirectional();

  const char* name() override {
    return (_type == Integrator::Type::PathTracing) ? "Path Tracing (CPU)" : "Bidirectional (CPU)";
  }

  Integrator::Type type() const override {
    return _type;
  }

  void run() override;
  void update() override;
  void stop(Stop) override;
  void update_options() override;
  void sync_from_options(const Options& options) override;
  uint32_t supported_strategies() const override;

  const Status& status() const override;

 private:
  ETX_DECLARE_PIMPL(CPUBidirectional, 256);
  Integrator::Type _type = Integrator::Type::Bidirectional;
};

}  // namespace etx
