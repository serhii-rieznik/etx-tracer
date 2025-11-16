#pragma once

#include <etx/render/shared/base.hxx>
#include <etx/render/shared/spectrum.hxx>

namespace etx {

struct SpectralImage {
  uint32_t spectrum_index = kInvalidIndex;
  uint32_t image_index = kInvalidIndex;
};

struct SampledImage {
  float4 value = {};
  uint32_t image_index = kInvalidIndex;
  uint32_t channel = kInvalidIndex;
};

struct Thinfilm {
  constexpr static const float3 kRGBWavelengths = {610.0f, 537.0f, 450.0f};
  constexpr static const float3 kRGBWavelengthsSpan = {45.0f, 47.0f, 23.5f};

  struct Eval {
    RefractiveIndex::Sample ior;
    float3 rgb_wavelengths = kRGBWavelengths;
    float thickness = 0.0f;
  };

  RefractiveIndex ior = {};
  uint32_t thinkness_image = kInvalidIndex;
  float min_thickness = 0.0f;
  float max_thickness = 0.0f;
  float pad;
};

struct SubsurfaceMaterial : public SpectralImage {
  enum class Class : uint32_t {
    Disabled,
    RandomWalk,
    ChristensenBurley,
  };

  enum class Path : uint32_t {
    Diffuse,
    Refracted,
  };

  Class cls = Class::Disabled;
  Path path = Path::Diffuse;
};

struct Material {
  enum class Class : uint32_t {
    Diffuse,
    Translucent,
    Plastic,
    Conductor,
    Dielectric,
    Thinfilm,
    Mirror,
    Boundary,
    Velvet,
    Principled,
    Void,

    Count,
    Undefined = kInvalidIndex,
  };

  SpectralImage reflectance;
  SpectralImage scattering;
  SpectralImage emission = {};
  SampledImage roughness;
  SampledImage metalness;
  SampledImage transmission;
  SubsurfaceMaterial subsurface;
  Thinfilm thinfilm = {};

  RefractiveIndex ext_ior = {};
  RefractiveIndex int_ior = {};

  Class cls = Class::Undefined;
  uint32_t int_medium = kInvalidIndex;
  uint32_t ext_medium = kInvalidIndex;
  uint32_t normal_image_index = kInvalidIndex;
  uint32_t diffuse_variation = 0u;
  uint32_t two_sided = 0u;
  float normal_scale = 1.0f;
  float opacity = 1.0f;
  float emission_collimation = 0.0f;

  bool has_diffuse() const {
    return (cls == Class::Diffuse) || (cls == Class::Plastic);
  }
};

}  // namespace etx
