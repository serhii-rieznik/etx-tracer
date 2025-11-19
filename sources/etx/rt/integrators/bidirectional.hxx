#pragma once

#include <etx/rt/integrators/integrator.hxx>

namespace etx {

struct CPUBidirectional : public Integrator {
  CPUBidirectional(Raytracing&);
  ~CPUBidirectional();

  const char* name() override {
    return "Bidirectional (CPU)";
  }

  Integrator::Type type() const override {
    return Integrator::Type::Bidirectional;
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
};

}  // namespace etx
