#pragma once

#include <etx/rt/integrators/integrator.hxx>

namespace etx {

struct BDPTDistilled : public Integrator {
  BDPTDistilled(Raytracing&);
  ~BDPTDistilled();

  const char* name() override {
    return "BDPT Distilled";
  }

  Integrator::Type type() const override {
    return Integrator::Type::BDPTDistilled;
  }

  void run() override;
  void update() override;
  void stop(Stop) override;
  void update_options() override;
  void sync_from_options(const Options& options) override;
  uint32_t supported_strategies() const override;

  const Status& status() const override;

 private:
  ETX_DECLARE_PIMPL(BDPTDistilled, 256);
};

}  // namespace etx