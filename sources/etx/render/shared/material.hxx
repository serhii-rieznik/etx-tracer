#pragma once

#include <etx/render/shared/base.hxx>
#include <etx/render/shared/spectrum.hxx>

namespace etx {

struct ETX_ALIGNED SpectralImage {
  SpectralDistribution spectrum = {};
  uint32_t image_index = kInvalidIndex;
};

struct ETX_ALIGNED Thinfilm {
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

struct SubsurfaceMaterial {
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
  SpectralDistribution scattering_distance;
  float scale = 1.0f;
};

struct ETX_ALIGNED Material {
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

    Count,
    Undefined = kInvalidIndex,
  };

  Class cls = Class::Undefined;
  SpectralImage reflectance;
  SpectralImage transmittance;
  SpectralImage emission;

  uint32_t int_medium = kInvalidIndex;
  uint32_t ext_medium = kInvalidIndex;
  RefractiveIndex ext_ior = {};
  RefractiveIndex int_ior = {};
  Thinfilm thinfilm = {};
  SubsurfaceMaterial subsurface = {};

  float2 roughness = {};

  uint32_t normal_image_index = kInvalidIndex;
  float normal_scale = 1.0f;

  uint32_t diffuse_variation = 0;

  bool has_diffuse() const {
    return (cls == Class::Diffuse) || (cls == Class::Plastic);
  }
};

}  // namespace etx
