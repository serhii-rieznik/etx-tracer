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
    Thickness,
    Spectrum,
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
  const float4* get_camera_image(bool) override;
  const float4* get_light_image(bool) override;
  const char* status() const override;

  void preview(const Options&) override;
  void run(const Options&) override;
  void update() override;
  void stop(Stop) override;

 private:
  ETX_DECLARE_PIMPL(CPUDebugIntegrator, 4096);
};

}  // namespace etx
