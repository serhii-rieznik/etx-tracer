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

  Options options() const override {
    Options result = {};
    result.set(1u, 0x7fffu, 0xffffu, "spp", "Samples per Pixel");
    result.set(1u, 0x7fffu, 65536u, "pathlen", "Maximal Path Length");
    result.set(1u, 5u, 65536u, "rrstart", "Start Russian Roulette at");
    return result;
  }

  void set_output_size(const uint2&) override;
  float4* get_updated_camera_image() override;
  float4* get_updated_light_image() override;
  const char* status() const override;

  void preview() override;
  void run(const Options&) override;
  void update() override;
  void stop() override;

  ETX_DECLARE_PIMPL(CPUPathTracing, 4096);
};

}  // namespace etx
