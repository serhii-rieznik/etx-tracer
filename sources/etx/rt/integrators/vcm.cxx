#include <etx/rt/integrators/vcm.hxx>

namespace etx {

struct CPUVCMImpl {
  Raytracing& rt;

  CPUVCMImpl(Raytracing& r)
    : rt(r) {
  }
};

CPUVCM::CPUVCM(Raytracing& rt)
  : Integrator(rt) {
  ETX_PIMPL_INIT(CPUVCM, rt);
}

CPUVCM::~CPUVCM() {
  if (current_state != State::Stopped) {
    stop(Stop::Immediate);
  }
  ETX_PIMPL_CLEANUP(CPUVCM);
}

}  // namespace etx
