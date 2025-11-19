#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/rt/integrators/integrator.hxx>

namespace etx {

struct CPUPathTracing : public Integrator {
  CPUPathTracing(Raytracing&);
  ~CPUPathTracing() override;

  const char* name() override {
    return "Path Tracing (CPU)";
  }

  Integrator::Type type() const override {
    return Integrator::Type::PathTracing;
  }

  bool have_updated_light_image() const override {
    return false;
  }

  const Status& status() const override;

  void run() override;
  void update() override;
  void stop(Stop) override;
  void update_options() override;
  void sync_from_options(const Options& options) override;
  uint32_t supported_strategies() const override;

  ETX_DECLARE_PIMPL(CPUPathTracing, 160);
};

}  // namespace etx
