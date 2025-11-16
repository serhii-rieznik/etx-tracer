#pragma once

#include <etx/rt/integrators/integrator.hxx>

namespace etx {

struct CPUBidirectional : public Integrator {
  CPUBidirectional(Raytracing&);
  ~CPUBidirectional();

  const char* name() override {
    return "Bidirectional (CPU)";
  }

  void run() override;
  void update() override;
  void stop(Stop) override;
  void update_options() override;

  const Status& status() const override;

 private:
  ETX_DECLARE_PIMPL(CPUBidirectional, 256);
};

}  // namespace etx
