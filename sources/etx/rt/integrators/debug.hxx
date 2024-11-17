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
    TransmittanceColor,
    ReflectanceColor,
    Fresnel,
    Thickness,
    Thinfilm,
    Spectrums,
    IOR,

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

  const Status& status() const override;

  void run(const Options&) override;
  void update() override;
  void stop(Stop) override;

 private:
  ETX_DECLARE_PIMPL(CPUDebugIntegrator, 1024u * 32u);
};

}  // namespace etx
