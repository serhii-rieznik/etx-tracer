#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/rt/integrators/integrator.hxx>

namespace etx {

struct CPUDebugIntegrator : public Integrator {
  enum class Mode {
    Geometry,
    Barycentrics,
    Normals,
    Tangents,
    Bitangents,
    TexCoords,
    FaceOrientation,
    DiffuseColors,
    Fresnel,
    Count,
  };
  static std::string mode_to_string(uint32_t);

  CPUDebugIntegrator(Raytracing&);
  ~CPUDebugIntegrator() override;

  const char* name() override {
    return "Debug (CPU)";
  }

  Options options() const override;
  void update_options(const Options&) override;

  void set_output_size(const uint2&) override;
  float4* get_updated_camera_image() override;
  float4* get_updated_light_image() override;
  const char* status() const override;

  void preview(const Options&) override;
  void run(const Options&) override;
  void update() override;
  void stop(bool wait_for_completion) override;

 private:
  ETX_DECLARE_PIMPL(CPUDebugIntegrator, 4096);
};

}  // namespace etx
