#pragma once

#include <etx/rt/integrators/integrator.hxx>

namespace etx {

struct CPUBidirectional : public Integrator {
  CPUBidirectional(Raytracing&);
  ~CPUBidirectional();

  const char* name() override {
    return "Bidirectional (CPU)";
  }

  Options options() const override;

  void run(const Options&) override;
  void update() override;
  void stop(Stop) override;
  void update_options(const Options&) override;

  const Status& status() const override;

 private:
  ETX_DECLARE_PIMPL(CPUBidirectional, 128);
};

}  // namespace etx
