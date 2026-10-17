#pragma once

#include <etx/render/interop/material.hxx>

namespace etx {

struct MaterialLayerSettings {
  uint32_t explicit_layers = 0u;
  uint32_t coating_enabled = 0u;
  uint32_t thinfilm_enabled = 0u;
  uint32_t bsdf = MaterialClass::Undefined;
};

struct CoatingDescription {
  RefractiveIndex ior = {};
  SampledImage roughness = {};
  SampledImage thickness = {{0.01f, 0.01f, 0.01f, 0.01f}, kInvalidIndex, 0u};
  uint32_t thickness_nm_image = kInvalidIndex;
  float min_thickness_nm = 100.0f;
  float max_thickness_nm = 100.0f;
  SpectralImage absorption = {};
  SpectralImage scattering = {};
  float anisotropy = 0.0f;
};

// Authored resource references and parameters only. Commit owns all derived bindings.
struct MaterialDescription {
  using Class = uint32_t;

  uint32_t cls = MaterialClass::Undefined;
  SpectralImage reflectance = {};
  SpectralImage scattering = {};
  SampledImage roughness = {};
  SampledImage metalness = {};
  SampledImage transmission = {};
  RefractiveIndex int_ior = {};
  DiffractionGrating diffraction_grating = {};
  MaterialMix mix = {};

  MaterialLayerSettings layers = {};
  CoatingDescription coating = {};
  Thinfilm thinfilm = {};

  RefractiveIndex ext_ior = {};
  uint32_t int_medium = kInvalidIndex;
  uint32_t ext_medium = kInvalidIndex;
  uint32_t normal_image_index = kInvalidIndex;
  uint32_t two_sided = 0u;
  float normal_scale = 1.0f;
  float opacity = 1.0f;
  SampledImage alpha_mask = {{1.0f, 1.0f, 1.0f, 1.0f}, kInvalidIndex, kInvalidIndex};
  SampledImage bump = {{0.0f, 0.0f, 0.0f, 0.0f}, kInvalidIndex, 0u};

  uint32_t subsurface_cls = SubsurfaceMaterial::Disabled;
  uint32_t subsurface_path = SubsurfaceMaterial::DiffusePath;
  SpectralImage subsurface = {};
  float subsurface_packing = 0.0f;
  float subsurface_anisotropy = 0.0f;

  SpectralImage emission = {};
  float emission_collimation = 0.0f;
  float temperature_kelvin = 0.0f;
};

inline const char* material_class_display_name(uint32_t cls) {
  switch (cls) {
    case MaterialClass::Diffuse:
      return "Diffuse";
    case MaterialClass::Translucent:
      return "Translucent";
    case MaterialClass::Plastic:
      return "Coated diffuse";
    case MaterialClass::Conductor:
      return "Conductor";
    case MaterialClass::CoatedConductor:
      return "Coated conductor";
    case MaterialClass::Mix:
      return "Mix";
    case MaterialClass::Dielectric:
      return "Dielectric";
    case MaterialClass::Thinfilm:
      return "Thinfilm";
    case MaterialClass::Mirror:
      return "Mirror";
    case MaterialClass::Boundary:
      return "Boundary";
    case MaterialClass::Velvet:
      return "Velvet";
    case MaterialClass::OpenPBR:
      return "OpenPBR";
    case MaterialClass::Void:
      return "Void";
    case MaterialClass::DiffractionGrating:
      return "Diffraction Grating";
    case MaterialClass::Undefined:
      return "Custom layers";
    default:
      return "Unknown material model";
  }
}

inline void configure_material_base(MaterialDescription& material, uint32_t cls) {
  material.cls = cls;
  const uint32_t ior_class = cls == MaterialClass::Conductor                                                                               ? SpectralDistribution::Conductor
                             : ((cls == MaterialClass::Dielectric) || (cls == MaterialClass::Thinfilm) || (cls == MaterialClass::OpenPBR)) ? SpectralDistribution::Dielectric
                                                                                                                                           : SpectralDistribution::Invalid;
  if ((ior_class != SpectralDistribution::Invalid) && (material.int_ior.cls != ior_class)) {
    material.int_ior = {};
  }
  if ((ior_class != SpectralDistribution::Invalid) && (material.ext_ior.cls == SpectralDistribution::Conductor)) {
    material.ext_ior = {};
  }
}

inline void configure_material_bsdf(MaterialDescription& material, uint32_t bsdf) {
  const bool legacy = material.layers.explicit_layers == 0u;
  if (legacy) {
    material.layers.thinfilm_enabled = ((material.thinfilm.max_thickness > 0.0f) || (material.thinfilm.thinkness_image != kInvalidIndex)) ? 1u : 0u;
  }
  material.layers.explicit_layers = 1u;
  material.layers.bsdf = bsdf;
  const uint32_t selected = bsdf == MaterialClass::Undefined ? material.cls : bsdf;
  if ((selected == MaterialClass::Plastic) || (selected == MaterialClass::CoatedConductor)) {
    configure_material_base(material, selected == MaterialClass::Plastic ? MaterialClass::Diffuse : MaterialClass::Conductor);
    material.layers.coating_enabled = 1u;
    if (material.ext_ior.cls == SpectralDistribution::Conductor) {
      material.ext_ior = {};
    }
  } else if (bsdf != MaterialClass::Undefined) {
    configure_material_base(material, selected);
    material.layers.coating_enabled = 0u;
  }
  if (bsdf != MaterialClass::Undefined) {
    if (selected == MaterialClass::Thinfilm) {
      material.layers.thinfilm_enabled = 1u;
    } else if ((selected != MaterialClass::Plastic) && (selected != MaterialClass::CoatedConductor) && (selected != MaterialClass::Conductor) &&
               (selected != MaterialClass::Dielectric) && (selected != MaterialClass::OpenPBR)) {
      material.layers.thinfilm_enabled = 0u;
    }
  }
}

inline Material project_material_description(const MaterialDescription& source) {
  Material result = {};
  result.cls = source.cls;
  result.reflectance = source.reflectance;
  result.scattering = source.scattering;
  result.roughness = source.roughness;
  result.metalness = source.metalness;
  result.transmission = source.transmission;
  result.int_ior = source.int_ior;
  result.diffraction_grating = source.diffraction_grating;
  result.mix = source.mix;
  result.coating.ior = source.coating.ior;
  result.coating.roughness = source.coating.roughness;
  result.coating.thickness = source.coating.thickness;
  result.coating.absorption = source.coating.absorption;
  result.coating.scattering = source.coating.scattering;
  result.coating.anisotropy = source.coating.anisotropy;
  result.thinfilm = source.thinfilm;
  if ((source.layers.explicit_layers != 0u) && (source.layers.coating_enabled != 0u) && (source.coating.ior.cls == SpectralDistribution::Conductor)) {
    result.coherent_coating.ior = source.coating.ior;
    result.coherent_coating.thickness_image = source.coating.thickness_nm_image;
    result.coherent_coating.min_thickness_nm = source.coating.min_thickness_nm;
    result.coherent_coating.max_thickness_nm = source.coating.max_thickness_nm;
  }
  result.ext_ior = source.ext_ior;
  result.int_medium = source.int_medium;
  result.ext_medium = source.ext_medium;
  result.normal_image_index = source.normal_image_index;
  result.two_sided = source.two_sided;
  result.normal_scale = source.normal_scale;
  result.opacity = source.opacity;
  result.alpha_mask = source.alpha_mask;
  result.bump = source.bump;
  result.subsurface_cls = source.subsurface_cls;
  result.subsurface_path = source.subsurface_path;
  result.subsurface = source.subsurface;
  result.subsurface_packing = source.subsurface_packing;
  result.subsurface_anisotropy = source.subsurface_anisotropy;
  if ((source.layers.explicit_layers != 0u) && (source.cls != MaterialClass::Diffuse) && (source.cls != MaterialClass::Plastic) && (source.cls != MaterialClass::OpenPBR)) {
    result.subsurface_cls = SubsurfaceMaterial::Disabled;
  }
  result.emission = source.emission;
  result.emission_collimation = source.emission_collimation;
  result.temperature_kelvin = source.temperature_kelvin;
  return result;
}

}  // namespace etx
