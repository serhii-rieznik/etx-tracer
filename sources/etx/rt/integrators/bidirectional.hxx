#pragma once

#include <etx/rt/integrators/integrator.hxx>

namespace etx {

struct CPUBidirectional : public Integrator {
  CPUBidirectional(Raytracing&);
  ~CPUBidirectional();

  const char* name() {
    return "Bidirectional (CPU)";
  }

  Options options() const override;
  void set_output_size(const uint2&) override;
  void preview(const Options&) override;
  void run(const Options&) override;
  void update() override;
  void stop(bool /* wait for completion */) override;
  void update_options(const Options&) override;

  float4* get_updated_camera_image() override;
  float4* get_updated_light_image() override;
  const char* status() const override;

 private:
  ETX_DECLARE_PIMPL(CPUBidirectional, 4096);
};

}  // namespace etx
