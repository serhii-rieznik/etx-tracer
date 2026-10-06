#pragma once

#include <etx/render/host/scene_data.hxx>
#include <etx/core/log.hxx>

namespace etx {
inline bool subsurface_materials_valid(const SceneData& data) {
  for (const Material& material : data.materials) {
    if (material.subsurface_cls == SubsurfaceMaterial::Disabled) {
      continue;
    }
    if ((material.cls != MaterialClass::Diffuse) && (material.cls != MaterialClass::Plastic)) {
      log::error("Random-walk SSS requires a diffuse or plastic material");
      return false;
    }
    if ((std::isfinite(material.subsurface_packing) == false) || (material.subsurface_packing < 0.0f) || (material.subsurface_packing >= 1.0f) ||
        (std::isfinite(material.subsurface_anisotropy) == false) || (material.subsurface_anisotropy <= -1.0f) || (material.subsurface_anisotropy >= 1.0f)) {
      log::error("SSS requires finite packing in [0, 1) and anisotropy in (-1, 1)");
      return false;
    }
    if ((material.cls == MaterialClass::Diffuse) && (material.subsurface_path != SubsurfaceMaterial::DiffusePath) &&
        (material.subsurface_path != SubsurfaceMaterial::RefractedPath)) {
      log::error("Random-walk SSS requires Diffuse Transmittance or Incident Direction");
      return false;
    }
    if ((material.int_medium == kInvalidIndex) && (material.subsurface.image_index != kInvalidIndex)) {
      log::warning("SSS distance textures do not affect uniform bulk coefficients");
    }
    if (material.int_medium != kInvalidIndex) {
      if (material.int_medium >= data.mediums_vector.size()) {
        log::error("SSS has an invalid internal medium");
        return false;
      }
      const Medium& medium = data.mediums_vector[material.int_medium];
      const bool mapped_coated_medium = (material.cls == MaterialClass::Plastic) && (material.subsurface_packing == 0.0f);
      if ((medium.cls != Medium::Homogeneous) && ((mapped_coated_medium == false) || (medium.cls != Medium::Heterogeneous))) {
        log::error("Diffuse and Exclusion SSS require a homogeneous internal medium");
        return false;
      }
      if ((std::isfinite(medium.phase_function_g) == false) || (std::abs(medium.phase_function_g) >= 1.0f) || (medium.absorption_index >= data.spectrum_values.size()) ||
          (medium.scattering_index >= data.spectrum_values.size())) {
        log::error("SSS requires valid medium spectra and finite anisotropy in (-1, 1)");
        return false;
      }
      const float3 absorption_rgb = data.spectrum_values[medium.absorption_index].integrated();
      const float3 scattering_rgb = data.spectrum_values[medium.scattering_index].integrated();
      if ((valid_value(absorption_rgb) == false) || (valid_value(scattering_rgb) == false) || (absorption_rgb.x < 0.0f) || (absorption_rgb.y < 0.0f) || (absorption_rgb.z < 0.0f) ||
          (scattering_rgb.x < 0.0f) || (scattering_rgb.y < 0.0f) || (scattering_rgb.z < 0.0f)) {
        log::error("SSS requires finite, nonnegative medium RGB optical coefficients");
        return false;
      }
      for (uint32_t i = 0u; i < WavelengthCount; ++i) {
        const SpectralQuery spect = {kShortestWavelength + float(i), SpectralFlags::Spectral};
        const float absorption = data.spectrum_values[medium.absorption_index](spect).value;
        const float scattering = data.spectrum_values[medium.scattering_index](spect).value;
        if ((std::isfinite(absorption + scattering) == false) || (absorption < 0.0f) || (scattering < 0.0f)) {
          log::error("SSS requires finite, nonnegative medium optical coefficients");
          return false;
        }
      }
    }
  }
  return true;
}
}  // namespace etx
