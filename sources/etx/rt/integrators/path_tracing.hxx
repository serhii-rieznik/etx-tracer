#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/rt/integrators/integrator.hxx>

namespace etx {

struct CPUPathTracing : public Integrator {
  CPUPathTracing(const Raytracing&);
  ~CPUPathTracing() override;

  const char* name() override {
    return "Path Tracing (CPU)";
  }

  Options options() const override {
    Options result = {};
    result.set(1u, 256u, 65536u, "spp", "Samples per Pixel");
    result.set(1u, 4096u, 65536u, "pathlen", "Maximal Path Length");
    result.set(1u, 5u, 65536u, "rrstart", "Start Russian Roulette at");
    return result;
  }

  void set_output_size(const uint2&) override;
  float4* get_updated_camera_image() override;
  float4* get_updated_light_image() override;
  const char* status() const override;

  void preview() override;

  ETX_DECLARE_PIMPL(CPUPathTracing, 256);
};

}  // namespace etx
