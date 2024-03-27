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

  Options options() const override;

  bool have_updated_light_image() const override {
    return false;
  }

  Status status() const override;

  void preview(const Options&) override;
  void run(const Options&) override;
  void update() override;
  void stop(Stop) override;
  void update_options(const Options&) override;

  ETX_DECLARE_PIMPL(CPUPathTracing, 4096);
};

}  // namespace etx
