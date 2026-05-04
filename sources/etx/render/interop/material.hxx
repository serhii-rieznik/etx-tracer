#pragma once

#include "interop.hxx"
#include "spectrum.hxx"

ETX_STATIC_CONST float3 kRGBWavelengths = float3(610.0f, 537.0f, 450.0f);
ETX_STATIC_CONST float3 kRGBWavelengthsSpan = float3(45.0f, 47.0f, 23.5f);

struct ETX_ALIGNED SpectralImage {
  uint32_t spectrum_index ETX_INIT(kInvalidIndex);
  uint32_t image_index ETX_INIT(kInvalidIndex);
};

struct ETX_ALIGNED SampledImage {
  float4 value ETX_INIT({});
  uint32_t image_index ETX_INIT(kInvalidIndex);
  uint32_t channel ETX_INIT(kInvalidIndex);
};

struct ETX_ALIGNED Thinfilm {
  RefractiveIndex ior ETX_INIT({});
  uint32_t thinkness_image ETX_INIT(kInvalidIndex);
  float min_thickness ETX_INIT(0.0f);
  float max_thickness ETX_INIT(0.0f);
  float pad ETX_INIT({});
};

struct ETX_ALIGNED ThinfilmEval {
  RefractiveIndexSample ior ETX_INIT({});
  float3 rgb_wavelengths ETX_INIT(kRGBWavelengths);
  float thickness ETX_INIT(0.0f);
};

struct SubsurfaceMaterial {
  using Class = uint32_t;
  enum : uint32_t {
    Disabled,
    RandomWalk,
    ChristensenBurley,
  };

  using Path = uint32_t;
  enum : uint32_t {
    DiffusePath,
    RefractedPath,
  };
};

struct MaterialClass {
  enum : uint32_t {
    Diffuse,
    Translucent,
    Plastic,
    Conductor,
    Dielectric,
    Thinfilm,
    Mirror,
    Boundary,
    Velvet,
    OpenPBR,
    Void,

    Count,
    Undefined = kInvalidIndex,
  };
};

struct ETX_ALIGNED Material {
  using Class = uint32_t;

  SpectralImage reflectance;
  SpectralImage scattering;
  SpectralImage emission;
  SpectralImage subsurface;
  SampledImage roughness;
  SampledImage metalness;
  SampledImage transmission;
  Thinfilm thinfilm;
  RefractiveIndex ext_ior;
  RefractiveIndex int_ior;
  uint32_t subsurface_cls ETX_INIT(SubsurfaceMaterial::Disabled);
  uint32_t subsurface_path ETX_INIT(SubsurfaceMaterial::DiffusePath);
  uint32_t cls ETX_INIT(MaterialClass::Undefined);
  uint32_t int_medium ETX_INIT(kInvalidIndex);
  uint32_t ext_medium ETX_INIT(kInvalidIndex);
  uint32_t normal_image_index ETX_INIT(kInvalidIndex);
  uint32_t two_sided ETX_INIT(0u);
  float normal_scale ETX_INIT(1.0f);
  float opacity ETX_INIT(1.0f);
  float emission_collimation ETX_INIT(0.0f);
  uint32_t energy_compensation_interface_index ETX_INIT(kInvalidIndex);
  uint32_t conductor_energy_compensation_interface_index ETX_INIT(kInvalidIndex);
};
