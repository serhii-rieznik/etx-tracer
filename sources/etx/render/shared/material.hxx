#pragma once

#include <etx/render/shared/base.hxx>
#include <etx/render/shared/spectrum.hxx>

namespace etx {

struct alignas(16) Thinfilm {
  uint32_t image_index = kInvalidIndex;
  float min_thickness = 0.0f;
  float max_thickness = 0.0f;
  float pad;
};

struct alignas(16) SpectralImage {
  SpectralDistribution spectrum = {};
  uint32_t image_index = kInvalidIndex;
};

struct alignas(16) Material {
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

  enum : uint32_t {
    DoubleSided = 1u << 0u,
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
