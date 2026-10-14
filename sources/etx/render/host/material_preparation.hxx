#pragma once

#include <etx/render/host/scene_data.hxx>
#include <etx/core/log.hxx>
#include <etx/render/access/bsdf_resource_cpu.hxx>
#include <etx/render/interop/thermal_surface_shared.hxx>

#include <limits>

namespace etx {
inline bool coherent_dielectric_base_reflection_is_unit(const SceneData& data, const SpectralImage& reflectance) {
  if (reflectance.spectrum_index >= data.spectrum_values.size()) {
    return false;
  }
  const SpectralDistribution& spectrum = data.spectrum_values[reflectance.spectrum_index];
  if (reflectance.image_index == kInvalidIndex) {
    const auto source = data.spectrum_sources.find(reflectance.spectrum_index);
    if ((source != data.spectrum_sources.end()) && source->second.matches(spectrum)) {
      const SpectrumSource& filter = source->second;
      if ((filter.mode == SpectrumSource::Mode::Color) && (filter.kind == SpectrumSource::Kind::Reflectance) && (filter.color.x == 1.0f) && (filter.color.y == 1.0f) &&
          (filter.color.z == 1.0f) && (filter.strength == 1.0f)) {
        return true;
      }
    }
  }
  const float3 rgb = spectrum.integrated();
  const Image* image = nullptr;
  uint64_t pixel_count = 1u;
  float4 texture = {1.0f, 1.0f, 1.0f, 1.0f};
  if (reflectance.image_index != kInvalidIndex) {
    if (reflectance.image_index >= data.images_vector.size()) {
      return false;
    }
    image = &data.images_vector[reflectance.image_index];
    pixel_count = uint64_t(image->isize.x) * image->isize.y * image->isize.z;
    if ((image->format == Image::Format::Undefined) || (pixel_count == 0u) || (pixel_count > std::numeric_limits<uint32_t>::max())) {
      return false;
    }
    texture = image->pixel(0u);
  }
  for (uint32_t pixel = 0u; pixel < pixel_count; ++pixel) {
    const float4 texel = image != nullptr ? image->pixel(pixel) : float4{1.0f, 1.0f, 1.0f, 1.0f};
    if ((texel.x != texture.x) || (texel.y != texture.y) || (texel.z != texture.z) || ((rgb.x * texel.x) != 1.0f) || ((rgb.y * texel.y) != 1.0f) || ((rgb.z * texel.z) != 1.0f)) {
      return false;
    }
  }
  for (uint32_t wavelength = 0u; wavelength < WavelengthCount; ++wavelength) {
    const float lambda = kShortestWavelength + float(wavelength);
    const float factor = image != nullptr ? spectral_rgb_evaluate_wavelength(lambda, {texture.x, texture.y, texture.z}) : 1.0f;
    if ((spectrum({lambda, SpectralFlags::Spectral}).value * factor) != 1.0f) {
      return false;
    }
  }
  return true;
}

inline bool compile_material_layers(SceneData& data) {
  for (uint32_t index = 0u; index < data.material_descriptions.size(); ++index) {
    const MaterialLayerSettings& layers = data.material_descriptions[index].layers;
    if ((layers.bsdf != MaterialClass::Undefined) && (layers.bsdf >= MaterialClass::Count)) {
      data.fail_material(index, SceneDiagnosticKind::Validation, "Material parameters", "has an invalid BSDF selection");
      return false;
    }
    if (layers.explicit_layers == 0u) {
      continue;
    }
    if ((index >= data.materials.size()) || (layers.explicit_layers > 1u) || (layers.coating_enabled > 1u) || (layers.thinfilm_enabled > 1u)) {
      data.fail_material(index, SceneDiagnosticKind::Validation, "Material parameters", "has an invalid layer description");
      return false;
    }
    Material& material = data.materials[index];
    material.layer_base_class = material.cls + 1u;
    if ((material.cls == MaterialClass::Plastic) || (material.cls == MaterialClass::CoatedConductor)) {
      data.fail_material(index, SceneDiagnosticKind::Validation, "Material parameters", "must select a base rather than a legacy layered BSDF");
      return false;
    }
    if (layers.thinfilm_enabled == 0u) {
      material.thinfilm.weight = 0.0f;
      material.thinfilm.min_thickness = 0.0f;
      material.thinfilm.max_thickness = 0.0f;
      material.thinfilm.thinkness_image = kInvalidIndex;
    } else if ((layers.coating_enabled == 0u) && (material.cls != MaterialClass::Conductor) && (material.cls != MaterialClass::Dielectric) &&
               (material.cls != MaterialClass::Thinfilm) && (material.cls != MaterialClass::OpenPBR) && (material.cls != MaterialClass::Diffuse) &&
               (material.cls != MaterialClass::Velvet)) {
      data.fail_material(index, SceneDiagnosticKind::Unsupported, "Thin film", "requires a thin-film interface wrapper for base %u which is not implemented yet", material.cls);
      return false;
    }
    if (layers.coating_enabled == 0u) {
      const bool active_film = (material.thinfilm.weight > 0.0f) && (material.thinfilm.max_thickness > 0.0f);
      if ((active_film == false) || ((material.cls != MaterialClass::Diffuse) && (material.cls != MaterialClass::Velvet))) {
        continue;
      }
      if (material.int_ior.cls != SpectralDistribution::Dielectric) {
        data.fail_material(index, SceneDiagnosticKind::Validation, "IOR", "requires a dielectric substrate IOR beneath its bare thin film");
        return false;
      }
      // A bare film uses the base optical boundary and surface; disabled coating settings remain authored only.
      material.coating = {};
      material.coating.ior = material.int_ior;
      material.coating.roughness = material.roughness;
      material.coating.thickness.value = {};
      if (material.cls == MaterialClass::Diffuse) {
        if (data.thermal_unit_spectrum_index == kInvalidIndex) {
          data.thermal_unit_spectrum_index = data.add_spectrum(SpectralDistribution::constant(1.0f));
        }
        material.reflectance = {.spectrum_index = data.thermal_unit_spectrum_index};
      }
    }
    if (material.coating.ior.cls == SpectralDistribution::Conductor) {
      const bool opaque_support = (material.cls == MaterialClass::Diffuse) || (material.cls == MaterialClass::Velvet);
      const bool dielectric_support = (material.cls == MaterialClass::Dielectric) && (material.int_ior.cls == SpectralDistribution::Dielectric);
      const bool translucent_support = material.cls == MaterialClass::Translucent;
      if ((opaque_support == false) && (dielectric_support == false) && (translucent_support == false) &&
          ((material.cls != MaterialClass::Conductor) || (material.int_ior.cls != SpectralDistribution::Conductor))) {
        data.fail_material(index, SceneDiagnosticKind::Unsupported, "Coating", "requires finite-metal coating transport for its base which is not implemented yet");
        return false;
      }
      if (material.subsurface_cls != SubsurfaceMaterial::Disabled) {
        data.fail_material(index, SceneDiagnosticKind::Unsupported, "Coating", "requires subsurface transmission through the finite-metal coating which is not implemented yet");
        return false;
      }
      const SpectralImage coefficients[] = {material.coating.absorption, material.coating.scattering};
      for (const SpectralImage& coefficient : coefficients) {
        if ((coefficient.image_index != kInvalidIndex) ||
            ((coefficient.spectrum_index != kInvalidIndex) &&
              ((coefficient.spectrum_index >= data.spectrum_values.size()) || (data.spectrum_values[coefficient.spectrum_index].is_zero() == false)))) {
          data.fail_material(index, SceneDiagnosticKind::Unsupported, "Coating", "requires volume transport within a finite-metal coating which is not implemented yet");
          return false;
        }
      }
      if ((std::isfinite(material.coating.roughness.value.x) == false) || (std::isfinite(material.coating.roughness.value.y) == false) ||
          (material.coating.roughness.value.x < 0.0f) || (material.coating.roughness.value.y < 0.0f) ||
          ((material.coating.roughness.image_index != kInvalidIndex) && (material.coating.roughness.channel > 4u))) {
        data.fail_material(index, SceneDiagnosticKind::Validation, "Roughness", "has invalid finite-metal surface roughness");
        return false;
      }
      const RefractiveIndex interfaces[] = {material.ext_ior, material.coating.ior, material.int_ior};
      if (material.ext_ior.cls != SpectralDistribution::Dielectric) {
        data.fail_material(index, SceneDiagnosticKind::Validation, "IOR", "requires a dielectric exterior for its finite-metal coating");
        return false;
      }
      const uint32_t interface_count = (opaque_support || translucent_support) ? 2u : 3u;
      for (uint32_t interface_index = 0u; interface_index < interface_count; ++interface_index) {
        const RefractiveIndex& ior = interfaces[interface_index];
        if ((ior.eta_index >= data.spectrum_values.size()) || ((ior.k_index != kInvalidIndex) && (ior.k_index >= data.spectrum_values.size()))) {
          data.fail_material(index, SceneDiagnosticKind::Validation, "Material parameters", "has an undefined finite-metal interface spectrum");
          return false;
        }
        for (uint32_t wavelength = 0u; wavelength < WavelengthCount; ++wavelength) {
          const SpectralQuery query = {kShortestWavelength + float(wavelength), SpectralFlags::Spectral};
          const float eta = data.spectrum_values[ior.eta_index](query).value;
          const float k = ior.k_index != kInvalidIndex ? data.spectrum_values[ior.k_index](query).value : 0.0f;
          if ((std::isfinite(eta) == false) || (std::isfinite(k) == false) || (eta < 0.0f) || (k < 0.0f) || ((eta == 0.0f) && (k == 0.0f)) ||
              (((interface_index == 0u) || (dielectric_support && (interface_index == 2u))) && ((eta == 0.0f) || (k != 0.0f)))) {
            data.fail_material(index, SceneDiagnosticKind::Validation, "IOR", "requires passive finite-metal interfaces and a lossless dielectric exterior");
            return false;
          }
        }
      }
      if (dielectric_support && (coherent_dielectric_base_reflection_is_unit(data, material.reflectance) == false)) {
        data.fail_material(index, SceneDiagnosticKind::Unsupported, "Material parameters",
          "requires unit effective base reflection tint; non-unit tint and unresolved textures are unsupported by the coherent matched dielectric endpoint");
        return false;
      }
      if (dielectric_support || translucent_support) {
        material.coating.ior = dielectric_support ? material.int_ior : material.ext_ior;
        material.coating.thickness = {{0.0f, 0.0f, 0.0f, 0.0f}, kInvalidIndex, 0u};
        material.coating.anisotropy = 0.0f;
        if (dielectric_support) {
          material.roughness = material.coating.roughness;
          // Canonical neutral filters use exact unit transport; the description retains the authored RGB reconstruction.
          if (data.thermal_unit_spectrum_index == kInvalidIndex) {
            data.thermal_unit_spectrum_index = data.add_spectrum(SpectralDistribution::constant(1.0f));
          }
          material.reflectance = {.spectrum_index = data.thermal_unit_spectrum_index};
          const bool surface_source = (material.temperature_kelvin > 0.0f) || (material.emission.image_index != kInvalidIndex) ||
                                      ((material.emission.spectrum_index < data.spectrum_values.size()) &&
                                        ((data.spectrum_values[material.emission.spectrum_index].is_zero() == false) ||
                                          (data.spectrum_values[material.emission.spectrum_index].integrated().x != 0.0f) ||
                                          (data.spectrum_values[material.emission.spectrum_index].integrated().y != 0.0f) ||
                                          (data.spectrum_values[material.emission.spectrum_index].integrated().z != 0.0f)));
          const bool direct_interface = ((layers.bsdf == MaterialClass::Undefined) || (layers.bsdf == MaterialClass::Dielectric)) && (surface_source == false) &&
                                        coherent_dielectric_base_reflection_is_unit(data, material.scattering);
          if (direct_interface) {
            material.scattering = {.spectrum_index = data.thermal_unit_spectrum_index};
            continue;
          }
        }
        material.cls = MaterialClass::CoatedConductor;
      } else if (opaque_support) {
        // Index matching preserves the opaque base's zero-coating limit without an added dielectric boundary.
        material.coating.ior = material.ext_ior;
        material.coating.thickness = {{0.0f, 0.0f, 0.0f, 0.0f}, kInvalidIndex, 0u};
        material.coating.anisotropy = 0.0f;
        material.cls = MaterialClass::CoatedConductor;
      } else {
        // Coherent metal-on-metal interfaces share a surface. The description retains the independent base roughness.
        material.roughness = material.coating.roughness;
      }
      continue;
    }
    if (material.coating.ior.cls != SpectralDistribution::Dielectric) {
      data.fail_material(index, SceneDiagnosticKind::Validation, "IOR", "has an invalid coating IOR class");
      return false;
    }
    const Coating& coat = material.coating;
    if (material.ext_ior.cls != SpectralDistribution::Dielectric) {
      return data.fail_material(index, "Outside IOR", "A dielectric exterior is required for a coating; supplied IOR class is %u.", material.ext_ior.cls);
    }
    if ((std::isfinite(coat.thickness.value.x) == false) || (coat.thickness.value.x < 0.0f)) {
      return data.fail_material(index, "Coating / Physical thickness", "Thickness %.9g m must be finite and nonnegative.", coat.thickness.value.x);
    }
    if ((std::isfinite(coat.roughness.value.x) == false) || (std::isfinite(coat.roughness.value.y) == false) || (coat.roughness.value.x < 0.0f) ||
        (coat.roughness.value.y < 0.0f)) {
      return data.fail_material(index, "Coating / Roughness", "Roughness (%.9g, %.9g) must have finite, nonnegative components.", coat.roughness.value.x, coat.roughness.value.y);
    }
    if ((std::isfinite(coat.anisotropy) == false) || (std::abs(coat.anisotropy) >= 1.0f)) {
      return data.fail_material(index, "Coating / Volume anisotropy", "Anisotropy %.9g must be finite and strictly between -1 and 1.", coat.anisotropy);
    }
    if ((coat.roughness.image_index != kInvalidIndex) && (coat.roughness.channel > 4u)) {
      return data.fail_material(index, "Coating / Roughness texture channel", "Texture channel %u must be in [0, 4].", coat.roughness.channel);
    }
    if ((coat.thickness.image_index != kInvalidIndex) && (coat.thickness.channel > 4u)) {
      return data.fail_material(index, "Coating / Thickness texture channel", "Texture channel %u must be in [0, 4].", coat.thickness.channel);
    }
    const RefractiveIndex interfaces[] = {material.ext_ior, coat.ior};
    for (const RefractiveIndex& ior : interfaces) {
      if ((ior.eta_index >= data.spectrum_values.size()) || ((ior.k_index != kInvalidIndex) && (ior.k_index >= data.spectrum_values.size()))) {
        data.fail_material(index, SceneDiagnosticKind::Validation, "Coating", "has an undefined coating interface spectrum");
        return false;
      }
      for (uint32_t wavelength = 0u; wavelength < WavelengthCount; ++wavelength) {
        const SpectralQuery query = {kShortestWavelength + float(wavelength), SpectralFlags::Spectral};
        const float eta = data.spectrum_values[ior.eta_index](query).value;
        const float k = ior.k_index != kInvalidIndex ? data.spectrum_values[ior.k_index](query).value : 0.0f;
        if ((std::isfinite(eta) == false) || (eta <= 0.0f) || (k != 0.0f)) {
          data.fail_material(index, "IOR", "%s at %.1f nm has eta=%.9g, k=%.9g. A lossless dielectric requires finite eta > 0 and k = 0.",
            &ior == &interfaces[0] ? "Outside IOR" : "Coating IOR", query.wavelength, eta, k);
          return false;
        }
      }
    }
    const bool transmitting_base = (material.cls == MaterialClass::Dielectric) || (material.cls == MaterialClass::Translucent);
    if (transmitting_base) {
      if (data.material_descriptions[index].subsurface_cls != SubsurfaceMaterial::Disabled) {
        data.fail_material(index, SceneDiagnosticKind::Unsupported, "Coating", "requires subsurface transport through its transmitting-base coating which is not implemented yet");
        return false;
      }
    }
    if (material.cls == MaterialClass::Diffuse) {
      bool needs_transport = (material.temperature_kelvin > 0.0f) || (material.emission.image_index != kInvalidIndex) ||
                             ((material.emission.spectrum_index < data.spectrum_values.size()) && (data.spectrum_values[material.emission.spectrum_index].is_zero() == false));
      const SpectralImage coefficients[] = {material.coating.absorption, material.coating.scattering};
      for (const SpectralImage& coefficient : coefficients) {
        if ((coefficient.spectrum_index != kInvalidIndex) && (coefficient.spectrum_index >= data.spectrum_values.size())) {
          data.fail_material(index, SceneDiagnosticKind::Validation, "Coating", "has an undefined coating medium spectrum");
          return false;
        }
        needs_transport = needs_transport || (coefficient.image_index != kInvalidIndex) ||
                          ((coefficient.spectrum_index < data.spectrum_values.size()) && (data.spectrum_values[coefficient.spectrum_index].is_zero() == false));
      }
      if (needs_transport) {
        if (material.subsurface_cls != SubsurfaceMaterial::Disabled) {
          data.fail_material(index, SceneDiagnosticKind::Unsupported, "Coating", "requires subsurface transmission through the volume coating which is not implemented yet");
          return false;
        }
        material.cls = MaterialClass::CoatedConductor;
        continue;
      }
      material.cls = MaterialClass::Plastic;
      material.int_ior = material.coating.ior;
      if ((material.coating.roughness.value.x > 0.0f) || (material.coating.roughness.value.y > 0.0f)) {
        material.coating.roughness.value.x = max(kEpsilon, material.coating.roughness.value.x);
        material.coating.roughness.value.y = max(kEpsilon, material.coating.roughness.value.y);
      }
    } else if ((material.cls == MaterialClass::Conductor) || (material.cls == MaterialClass::Mirror) || (material.cls == MaterialClass::Velvet) || transmitting_base) {
      material.cls = MaterialClass::CoatedConductor;
    } else {
      data.fail_material(index, SceneDiagnosticKind::Unsupported, "Coating", "requires coating transport for base %u which is not implemented yet", material.cls);
      return false;
    }
  }
  for (uint32_t index = 0u; index < data.material_descriptions.size(); ++index) {
    const MaterialLayerSettings& layers = data.material_descriptions[index].layers;
    if ((layers.explicit_layers != 0u) && (layers.bsdf != MaterialClass::Undefined) && (data.materials[index].cls != layers.bsdf)) {
      data.fail_material(index, SceneDiagnosticKind::Validation, "Material parameters",
        "parameters require BSDF %u, but explicit BSDF %u was selected. Select Auto or adjust the layers.", data.materials[index].cls, layers.bsdf);
      return false;
    }
  }
  return true;
}

template <typename MaterialType>
inline bool material_graph_valid(const SceneData& data, const std::vector<MaterialType>& materials) {
  std::vector<uint8_t> states(materials.size(), 0u);
  std::vector<uint2> stack;
  stack.reserve(materials.size());
  for (uint32_t root = 0u; root < materials.size(); ++root) {
    if (states[root] != 0u) {
      continue;
    }
    stack.push_back({root, 0u});
    while (stack.empty() == false) {
      const uint32_t index = stack.back().x;
      const auto& material = materials[index];
      states[index] = 1u;
      if ((material.cls != MaterialClass::Mix) || (stack.back().y == 2u)) {
        states[index] = 2u;
        stack.pop_back();
        continue;
      }
      if ((std::isfinite(material.mix.amount.value.x) == false) || ((material.mix.amount.image_index != kInvalidIndex) && (material.mix.amount.channel > 4u))) {
        data.fail_material(index, SceneDiagnosticKind::Validation, "Material parameters", "has an invalid amount");
        return false;
      }
      const uint32_t child = stack.back().y++ == 0u ? material.mix.first : material.mix.second;
      if ((child >= materials.size()) || (states[child] == 1u)) {
        data.fail_material(index, SceneDiagnosticKind::Validation, "Material parameters", "has an undefined child or cyclic reference");
        return false;
      }
      if (states[child] == 0u) {
        stack.push_back({child, 0u});
      }
    }
  }
  for (uint32_t index = 0u; index < materials.size(); ++index) {
    const auto& material = materials[index];
    if constexpr (std::is_same_v<MaterialType, MaterialDescription>) {
      if ((std::isfinite(material.coating.min_thickness_nm) == false) || (std::isfinite(material.coating.max_thickness_nm) == false) ||
          (material.coating.min_thickness_nm < 0.0f) || (material.coating.max_thickness_nm < material.coating.min_thickness_nm)) {
        data.fail_material(index, SceneDiagnosticKind::Validation, "Coating / Optical thickness",
          "Minimum %.9g nm and maximum %.9g nm must be finite and satisfy 0 <= minimum <= maximum.", material.coating.min_thickness_nm, material.coating.max_thickness_nm);
        return false;
      }
    }
    if ((material.cls >= MaterialClass::Count) || (std::isfinite(material.roughness.value.x) == false) || (std::isfinite(material.roughness.value.y) == false) ||
        (material.roughness.value.x < 0.0f) || (material.roughness.value.y < 0.0f) ||
        ((material.subsurface_cls != SubsurfaceMaterial::Disabled) && (material.subsurface_cls != SubsurfaceMaterial::RandomWalk))) {
      data.fail_material(index, SceneDiagnosticKind::Validation, "Roughness", "has an invalid class, roughness or subsurface model");
      return false;
    }
    if (material.cls == MaterialClass::DiffractionGrating) {
      const DiffractionGrating& grating = material.diffraction_grating;
      if ((std::isfinite(grating.period_nm) == false) || (grating.period_nm < kDiffractionGratingMinimumPeriodNm) || (grating.period_nm > kDiffractionGratingMaximumPeriodNm) ||
          (std::isfinite(grating.optical_path_difference_nm) == false) || (grating.optical_path_difference_nm < kDiffractionGratingMinimumOpticalPathDifferenceNm) ||
          (grating.optical_path_difference_nm > kDiffractionGratingMaximumOpticalPathDifferenceNm) || (std::isfinite(grating.duty_cycle) == false) || (grating.duty_cycle < 0.0f) ||
          (grating.duty_cycle > 1.0f) || (std::isfinite(grating.rotation) == false)) {
        data.fail_material(index, SceneDiagnosticKind::Validation, "Material parameters",
          "Diffraction material %u has invalid period, optical path difference, duty cycle or rotation", index);
        return false;
      }
    }
    if ((std::isfinite(material.thinfilm.weight) == false) || (std::isfinite(material.thinfilm.min_thickness) == false) ||
        (std::isfinite(material.thinfilm.max_thickness) == false) || (material.thinfilm.weight < 0.0f) || (material.thinfilm.weight > 1.0f) ||
        (material.thinfilm.min_thickness < 0.0f) || (material.thinfilm.max_thickness < material.thinfilm.min_thickness)) {
      data.fail_material(index, SceneDiagnosticKind::Validation, "Thin film",
        "Coverage %.9g must be finite and in [0, 1]. Thickness range %.9g..%.9g nm must be finite and satisfy 0 <= minimum <= maximum.", material.thinfilm.weight,
        material.thinfilm.min_thickness, material.thinfilm.max_thickness);
      return false;
    }
    if (material.cls == MaterialClass::Thinfilm) {
      const RefractiveIndex media[] = {material.ext_ior, material.int_ior};
      for (const RefractiveIndex& medium : media) {
        if (((medium.cls != SpectralDistribution::Invalid) && (medium.cls != SpectralDistribution::Dielectric)) ||
            ((medium.eta_index != kInvalidIndex) && (medium.eta_index >= data.spectrum_values.size())) ||
            ((medium.k_index != kInvalidIndex) && (medium.k_index >= data.spectrum_values.size()))) {
          data.fail_material(index, SceneDiagnosticKind::Validation, "Material parameters", "requires dielectric surrounding media");
          return false;
        }
        for (uint32_t wavelength = 0u; wavelength < WavelengthCount; ++wavelength) {
          const SpectralQuery query = {kShortestWavelength + float(wavelength), SpectralFlags::Spectral};
          const float eta = medium.eta_index != kInvalidIndex ? data.spectrum_values[medium.eta_index](query).value : 1.0f;
          const float k = medium.k_index != kInvalidIndex ? data.spectrum_values[medium.k_index](query).value : 0.0f;
          if ((std::isfinite(eta) == false) || (eta <= 0.0f) || (k != 0.0f)) {
            data.fail_material(index, SceneDiagnosticKind::Validation, "IOR", "requires real positive surrounding IORs");
            return false;
          }
        }
      }
    }
    const bool dielectric_base = (material.cls == MaterialClass::Dielectric) || (material.cls == MaterialClass::Plastic);
    if (dielectric_base || (material.cls == MaterialClass::Conductor)) {
      const uint32_t base_ior_class = dielectric_base ? SpectralDistribution::Dielectric : SpectralDistribution::Conductor;
      if (((material.ext_ior.cls != SpectralDistribution::Invalid) && (material.ext_ior.cls != SpectralDistribution::Dielectric)) ||
          ((material.int_ior.cls != SpectralDistribution::Invalid) && (material.int_ior.cls != base_ior_class))) {
        data.fail_material(index, SceneDiagnosticKind::Validation, "IOR", "has an IOR class incompatible with its base or exterior");
        return false;
      }
      const RefractiveIndex interfaces[] = {material.ext_ior, material.int_ior};
      for (uint32_t interface_index = 0u; interface_index < 2u; ++interface_index) {
        const RefractiveIndex& ior = interfaces[interface_index];
        if (((ior.eta_index != kInvalidIndex) && (ior.eta_index >= data.spectrum_values.size())) ||
            ((ior.k_index != kInvalidIndex) && (ior.k_index >= data.spectrum_values.size()))) {
          data.fail_material(index, SceneDiagnosticKind::Validation, "IOR", "has an undefined base or exterior IOR spectrum");
          return false;
        }
        for (uint32_t wavelength = 0u; wavelength < WavelengthCount; ++wavelength) {
          const SpectralQuery query = {kShortestWavelength + float(wavelength), SpectralFlags::Spectral};
          const float eta = ior.eta_index != kInvalidIndex ? data.spectrum_values[ior.eta_index](query).value : 1.0f;
          const float k = ior.k_index != kInvalidIndex ? data.spectrum_values[ior.k_index](query).value : 0.0f;
          const bool dielectric_interface = (interface_index == 0u) || dielectric_base;
          if ((std::isfinite(eta) == false) || (std::isfinite(k) == false) || (eta < 0.0f) || (k < 0.0f) || ((eta == 0.0f) && (k == 0.0f)) ||
              (dielectric_interface && ((eta == 0.0f) || (k != 0.0f)))) {
            data.fail_material(index, "IOR", "%s at %.1f nm has eta=%.9g, k=%.9g. %s", interface_index == 0u ? "Outside IOR" : "Base IOR", query.wavelength, eta, k,
              dielectric_interface ? "A lossless dielectric requires finite eta > 0 and k = 0."
                                   : "A passive conductor requires finite eta >= 0 and k >= 0, with at least one nonzero.");
            return false;
          }
        }
      }
    }
    bool active_film = (material.thinfilm.weight > 0.0f) && (max(material.thinfilm.min_thickness, material.thinfilm.max_thickness) > 0.0f);
    bool bare_film_support = false;
    if constexpr (std::is_same_v<MaterialType, MaterialDescription>) {
      active_film = active_film && ((material.layers.explicit_layers == 0u) || (material.layers.thinfilm_enabled != 0u));
      bare_film_support =
        (material.layers.explicit_layers != 0u) && (material.layers.coating_enabled == 0u) && ((material.cls == MaterialClass::Diffuse) || (material.cls == MaterialClass::Velvet));
    }
    if (active_film) {
      if (bare_film_support && ((material.ext_ior.cls != SpectralDistribution::Dielectric) || (material.int_ior.cls != SpectralDistribution::Dielectric) ||
                                 (material.int_ior.eta_index >= data.spectrum_values.size()))) {
        data.fail_material(index, SceneDiagnosticKind::Validation, "IOR", "requires a dielectric substrate IOR beneath its bare thin film");
        return false;
      }
      const RefractiveIndex& film = material.thinfilm.ior;
      const RefractiveIndex dielectric_media[] = {material.ext_ior, material.int_ior};
      const uint32_t dielectric_count = ((material.cls == MaterialClass::Dielectric) || (material.cls == MaterialClass::Plastic) || bare_film_support) ? 2u : 1u;
      for (uint32_t medium_index = 0u; medium_index < dielectric_count; ++medium_index) {
        const RefractiveIndex& medium = dielectric_media[medium_index];
        if (((medium.eta_index != kInvalidIndex) && (medium.eta_index >= data.spectrum_values.size())) ||
            ((medium.k_index != kInvalidIndex) && (medium.k_index >= data.spectrum_values.size()))) {
          data.fail_material(index, SceneDiagnosticKind::Validation, "Thin film", "has an undefined thin-film dielectric IOR");
          return false;
        }
      }
      if (((film.cls != SpectralDistribution::Dielectric) && (film.cls != SpectralDistribution::Conductor)) || (film.eta_index >= data.spectrum_values.size()) ||
          ((film.k_index != kInvalidIndex) && (film.k_index >= data.spectrum_values.size())) || (std::isfinite(material.thinfilm.min_thickness) == false) ||
          (std::isfinite(material.thinfilm.max_thickness) == false) || (std::isfinite(material.thinfilm.weight) == false)) {
        data.fail_material(index, SceneDiagnosticKind::Validation, "Thin film", "has invalid thin-film optical parameters");
        return false;
      }
      for (uint32_t wavelength = 0u; wavelength < WavelengthCount; ++wavelength) {
        const SpectralQuery query = {kShortestWavelength + float(wavelength), SpectralFlags::Spectral};
        for (uint32_t medium_index = 0u; medium_index < dielectric_count; ++medium_index) {
          const RefractiveIndex& medium = dielectric_media[medium_index];
          const float medium_eta = medium.eta_index != kInvalidIndex ? data.spectrum_values[medium.eta_index](query).value : 1.0f;
          const float medium_k = medium.k_index != kInvalidIndex ? data.spectrum_values[medium.k_index](query).value : 0.0f;
          if ((std::isfinite(medium_eta) == false) || (medium_eta <= 0.0f) || (medium_k != 0.0f)) {
            data.fail_material(index, SceneDiagnosticKind::Validation, "Thin film", "requires real positive thin-film dielectric media");
            return false;
          }
        }
        const float eta = data.spectrum_values[film.eta_index](query).value;
        const float k = film.k_index != kInvalidIndex ? data.spectrum_values[film.k_index](query).value : 0.0f;
        if ((std::isfinite(eta) == false) || (std::isfinite(k) == false) || (eta < 0.0f) || (k < 0.0f) || ((eta == 0.0f) && (k == 0.0f))) {
          data.fail_material(index, SceneDiagnosticKind::Validation, "Thin film", "requires a finite passive thin-film IOR");
          return false;
        }
      }
    }
    if (material.cls != MaterialClass::CoatedConductor) {
      continue;
    }
    uint32_t substrate_class = MaterialClass::Conductor;
    if constexpr (requires { material.layer_base_class; }) {
      substrate_class = material_coating_base_class(material);
    }
    if (((substrate_class != MaterialClass::Conductor) && (substrate_class != MaterialClass::Diffuse) && (substrate_class != MaterialClass::Mirror) &&
          (substrate_class != MaterialClass::Velvet) && (substrate_class != MaterialClass::Dielectric) && (substrate_class != MaterialClass::Translucent)) ||
        (material.subsurface_cls != SubsurfaceMaterial::Disabled)) {
      data.fail_material(index, SceneDiagnosticKind::Validation, "Material parameters", "has no supported substrate transport");
      return false;
    }
    const auto& coat = material.coating;
    if ((material.ext_ior.cls != SpectralDistribution::Dielectric) || (coat.ior.cls != SpectralDistribution::Dielectric) ||
        ((substrate_class == MaterialClass::Conductor) && (material.int_ior.cls != SpectralDistribution::Conductor)) ||
        ((substrate_class == MaterialClass::Dielectric) && (material.int_ior.cls != SpectralDistribution::Dielectric)) || (std::isfinite(coat.thickness.value.x) == false) ||
        (coat.thickness.value.x < 0.0f) || ((coat.thickness.image_index != kInvalidIndex) && (coat.thickness.channel > 4u)) ||
        ((coat.roughness.image_index != kInvalidIndex) && (coat.roughness.channel > 4u)) || (std::isfinite(coat.roughness.value.x) == false) ||
        (std::isfinite(coat.roughness.value.y) == false) || (coat.roughness.value.x < 0.0f) || (coat.roughness.value.y < 0.0f) || (std::isfinite(coat.anisotropy) == false) ||
        (std::abs(coat.anisotropy) >= 1.0f)) {
      data.fail_material(index, SceneDiagnosticKind::Validation, "Coating", "requires valid dielectric coating and substrate parameters");
      return false;
    }
    for (uint32_t wavelength = 0u; wavelength < WavelengthCount; ++wavelength) {
      const SpectralQuery query = {kShortestWavelength + float(wavelength), SpectralFlags::Spectral};
      const uint32_t coefficients[] = {coat.absorption.spectrum_index, coat.scattering.spectrum_index};
      for (uint32_t coefficient : coefficients) {
        if (coefficient == kInvalidIndex) {
          continue;
        }
        if (coefficient >= data.spectrum_values.size()) {
          data.fail_material(index, SceneDiagnosticKind::Validation, "Material parameters", "has an undefined medium spectrum");
          return false;
        }
        const float value = data.spectrum_values[coefficient](query).value;
        if ((std::isfinite(value) == false) || (value < 0.0f)) {
          data.fail_material(index, SceneDiagnosticKind::Validation, "Volume coefficients", "requires finite nonnegative medium coefficients");
          return false;
        }
      }
      const RefractiveIndex interfaces[] = {material.ext_ior, coat.ior, material.int_ior};
      const uint32_t interface_count = ((substrate_class == MaterialClass::Conductor) || (substrate_class == MaterialClass::Dielectric)) ? 3u : 2u;
      for (uint32_t interface_index = 0u; interface_index < interface_count; ++interface_index) {
        const RefractiveIndex& ior = interfaces[interface_index];
        if (((ior.eta_index != kInvalidIndex) && (ior.eta_index >= data.spectrum_values.size())) ||
            ((ior.k_index != kInvalidIndex) && (ior.k_index >= data.spectrum_values.size()))) {
          data.fail_material(index, SceneDiagnosticKind::Validation, "IOR", "has an undefined IOR spectrum");
          return false;
        }
        const float eta = ior.eta_index != kInvalidIndex ? data.spectrum_values[ior.eta_index](query).value : 1.0f;
        const float k = ior.k_index != kInvalidIndex ? data.spectrum_values[ior.k_index](query).value : 0.0f;
        const bool dielectric_interface = (interface_index < 2u) || (substrate_class == MaterialClass::Dielectric);
        if ((std::isfinite(eta) == false) || (std::isfinite(k) == false) || (eta < 0.0f) || (k < 0.0f) || (dielectric_interface && ((eta == 0.0f) || (k != 0.0f))) ||
            ((eta == 0.0f) && (k == 0.0f))) {
          data.fail_material(index, SceneDiagnosticKind::Validation, "Material parameters", "requires lossless dielectric interfaces and a passive substrate");
          return false;
        }
      }
    }
  }
  return true;
}

inline bool material_graph_valid(const SceneData& data) {
  return material_graph_valid(data, data.material_descriptions);
}

inline void prepare_mix_emitters(SceneData& data) {
  std::vector<uint8_t> ready(data.materials.size(), 0u);
  std::vector<uint32_t> stack;
  stack.reserve(data.materials.size());
  const auto emission_weight = [&](uint32_t index) {
    const Material& material = data.materials[index];
    const float authored = material.emission.spectrum_index < data.spectrum_values.size() ? data.spectrum_values[material.emission.spectrum_index].luminance() : 0.0f;
    return material.cls == MaterialClass::Mix ? material.thermal_emission_weight : authored + material.thermal_emission_weight;
  };
  for (uint32_t root = 0u; root < data.materials.size(); ++root) {
    if (ready[root] != 0u)
      continue;
    stack.push_back(root);
    while (stack.empty() == false) {
      const uint32_t index = stack.back();
      Material& material = data.materials[index];
      if (material.cls == MaterialClass::Mix) {
        if (ready[material.mix.first] == 0u) {
          stack.push_back(material.mix.first);
          continue;
        }
        if (ready[material.mix.second] == 0u) {
          stack.push_back(material.mix.second);
          continue;
        }
        material.thermal_emission_weight = max(emission_weight(material.mix.first), emission_weight(material.mix.second));
        const Material& first = data.materials[material.mix.first];
        const Material& second = data.materials[material.mix.second];
        material.two_sided = first.two_sided | second.two_sided | uint32_t(thermal_surface_emits_both_sides(first)) | uint32_t(thermal_surface_emits_both_sides(second));
      }
      ready[index] = 1u;
      stack.pop_back();
    }
  }
}
}  // namespace etx
