#pragma once

#include <etx/rt/integrators/integrator.hxx>

namespace etx {

struct CPUVCM : public Integrator {
  CPUVCM(Raytracing&);
  ~CPUVCM();

  const char* name() {
    return "VCM (CPU)";
  }

 private:
  ETX_DECLARE_PIMPL(CPUVCM, 256);
};

}  // namespace etx
