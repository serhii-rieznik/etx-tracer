#pragma once

#include <etx/render/shared/base.hxx>
#include <etx/render/shared/spectrum.hxx>

namespace etx {

struct ETX_ALIGNED SpectralImage {
  SpectralDistribution spectrum = {};
  uint32_t image_index = kInvalidIndex;
};

struct ETX_ALIGNED Thinfilm {
  struct Eval {
    RefractiveIndex::Sample ior;
    float thickness = 0.0f;
    operator bool() const {
      return thickness > 0.0f;
    }
  };

  RefractiveIndex ior = {};
  uint32_t thinkness_image = kInvalidIndex;
  float min_thickness = 0.0f;
  float max_thickness = 0.0f;
  float pad;
};

struct ETX_ALIGNED Material {
  enum class Class : uint32_t {
    Diffuse,
    MultiscatteringDiffuse,
    Plastic,
    Conductor,
    MultiscatteringConductor,
    Dielectric,
    MultiscatteringDielectric,
    Thinfilm,
    Translucent,
    Mirror,
    Boundary,
    Generic,
    Coating,
    Mixture,

    Count,
    Undefined = kInvalidIndex,
  };

  Class cls = Class::Undefined;
  SpectralImage diffuse;
  SpectralImage specular;
  SpectralImage transmittance;

  uint32_t int_medium = kInvalidIndex;
  uint32_t ext_medium = kInvalidIndex;
  RefractiveIndex ext_ior = {};
  RefractiveIndex int_ior = {};
  Thinfilm thinfilm = {};

  float2 roughness = {};

  uint32_t normal_image_index = kInvalidIndex;
  uint32_t metal_roughness_image_index = kInvalidIndex;
  uint32_t mixture_0 = kInvalidIndex;
  uint32_t mixture_1 = kInvalidIndex;
  uint32_t mixture_image_index = kInvalidIndex;

  float mixture = {};
  float metalness = {};
  float normal_scale = 1.0f;
};

}  // namespace etx
