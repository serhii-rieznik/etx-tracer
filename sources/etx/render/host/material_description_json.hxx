#pragma once

#include <etx/render/host/material_description.hxx>
#include <etx/core/json.hxx>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace etx {

inline constexpr const char* kMaterialDescriptionTypes[] = {"diffuse", "translucent", "plastic", "conductor", "dielectric", "thinfilm", "mirror", "boundary", "velvet", "openpbr",
  "void", "diffraction_grating", "coated_conductor", "mix"};
static_assert(std::size(kMaterialDescriptionTypes) == MaterialClass::Count);

inline bool material_description_serializable(const MaterialDescription& material) {
  if ((material.cls >= MaterialClass::Count) || (material.layers.explicit_layers > 1u) || (material.layers.coating_enabled > 1u) || (material.layers.thinfilm_enabled > 1u) ||
      ((material.layers.bsdf != MaterialClass::Undefined) && (material.layers.bsdf >= MaterialClass::Count)) || (material.two_sided > 1u) ||
      ((material.subsurface_cls != SubsurfaceMaterial::Disabled) && (material.subsurface_cls != SubsurfaceMaterial::RandomWalk)) ||
      ((material.subsurface_path != SubsurfaceMaterial::DiffusePath) && (material.subsurface_path != SubsurfaceMaterial::RefractedPath))) {
    return false;
  }
  const RefractiveIndex interfaces[] = {material.int_ior, material.ext_ior, material.coating.ior, material.thinfilm.ior};
  for (const auto& optical_interface : interfaces) {
    if ((optical_interface.cls != SpectralDistribution::Invalid) && (optical_interface.cls != SpectralDistribution::Dielectric) &&
        (optical_interface.cls != SpectralDistribution::Conductor)) {
      return false;
    }
  }
  const SampledImage sampled[] = {material.roughness, material.metalness, material.transmission, material.coating.roughness, material.coating.thickness, material.alpha_mask,
    material.bump, material.mix.amount};
  for (const auto& value : sampled) {
    if ((std::isfinite(value.value.x) == false) || (std::isfinite(value.value.y) == false) || (std::isfinite(value.value.z) == false) || (std::isfinite(value.value.w) == false) ||
        ((value.channel != kInvalidIndex) && (value.channel > 4u))) {
      return false;
    }
  }
  const float parameters[] = {material.diffraction_grating.period_nm, material.diffraction_grating.optical_path_difference_nm, material.diffraction_grating.duty_cycle,
    material.diffraction_grating.rotation, material.coating.anisotropy, material.thinfilm.min_thickness, material.thinfilm.max_thickness, material.thinfilm.weight,
    material.normal_scale, material.opacity, material.subsurface_packing, material.subsurface_anisotropy, material.emission_collimation, material.temperature_kelvin,
    material.coating.min_thickness_nm, material.coating.max_thickness_nm};
  for (const float value : parameters) {
    if (std::isfinite(value) == false) {
      return false;
    }
  }
  return true;
}

template <typename SpectrumReference, typename TextureReference, typename MaterialReference, typename MediumReference>
inline nlohmann::json serialize_material_description(const MaterialDescription& material, SpectrumReference spectrum, TextureReference texture,
  MaterialReference material_reference, MediumReference medium_reference) {
  if (material_description_serializable(material) == false) {
    throw std::runtime_error("Material contains invalid authored parameters.");
  }
  const auto spectral = [&](const SpectralImage& value) {
    return nlohmann::json{{"spectrum", spectrum(value.spectrum_index)}, {"texture", texture(value.image_index)}};
  };
  const auto sampled = [&](const SampledImage& value) {
    return nlohmann::json{{"value", {value.value.x, value.value.y, value.value.z, value.value.w}}, {"texture", texture(value.image_index)},
      {"channel", value.channel == kInvalidIndex ? nlohmann::json(nullptr) : nlohmann::json(value.channel)}};
  };
  const auto ior = [&](const RefractiveIndex& value) {
    const char* type = value.cls == SpectralDistribution::Dielectric ? "dielectric" : value.cls == SpectralDistribution::Conductor ? "conductor" : "unspecified";
    return nlohmann::json{{"type", type}, {"eta", spectrum(value.eta_index)}, {"k", spectrum(value.k_index)}};
  };
  nlohmann::json result = {
    {"model", material.layers.explicit_layers != 0u ? "layered" : "legacy_bsdf"},
    {"base", {{"type", material.cls < MaterialClass::Count ? kMaterialDescriptionTypes[material.cls] : "undefined"}, {"reflectance", spectral(material.reflectance)},
               {"scattering", spectral(material.scattering)}, {"roughness", sampled(material.roughness)}, {"metalness", sampled(material.metalness)},
               {"transmission", sampled(material.transmission)}, {"ior", ior(material.int_ior)},
               {"diffraction", {{"period_nm", material.diffraction_grating.period_nm}, {"optical_path_difference_nm", material.diffraction_grating.optical_path_difference_nm},
                                 {"duty_cycle", material.diffraction_grating.duty_cycle}, {"rotation", material.diffraction_grating.rotation}}}}},
    {"coating", {{"enabled", material.layers.coating_enabled != 0u}, {"ior", ior(material.coating.ior)}, {"roughness", sampled(material.coating.roughness)},
                  {"thickness", sampled(material.coating.thickness)}, {"absorption", spectral(material.coating.absorption)}, {"scattering", spectral(material.coating.scattering)},
                  {"anisotropy", material.coating.anisotropy},
                  {"optical_thickness",
                    {{"min_nm", material.coating.min_thickness_nm}, {"max_nm", material.coating.max_thickness_nm}, {"texture", texture(material.coating.thickness_nm_image)}}}}},
    {"thinfilm", {{"enabled", material.layers.thinfilm_enabled != 0u}, {"ior", ior(material.thinfilm.ior)}, {"thickness_texture", texture(material.thinfilm.thinkness_image)},
                   {"min_thickness_nm", material.thinfilm.min_thickness}, {"max_thickness_nm", material.thinfilm.max_thickness}, {"weight", material.thinfilm.weight}}},
    {"surface", {{"two_sided", material.two_sided != 0u}, {"opacity", material.opacity}, {"normal_texture", texture(material.normal_image_index)},
                  {"normal_scale", material.normal_scale}, {"alpha_mask", sampled(material.alpha_mask)}, {"bump", sampled(material.bump)}}},
    {"media", {{"exterior_ior", ior(material.ext_ior)}, {"interior", medium_reference(material.int_medium)}, {"exterior", medium_reference(material.ext_medium)}}},
    {"subsurface", {{"type", material.subsurface_cls == SubsurfaceMaterial::Disabled ? "disabled" : "random_walk"},
                     {"path", material.subsurface_path == SubsurfaceMaterial::DiffusePath ? "diffuse" : "refracted"}, {"distance", spectral(material.subsurface)},
                     {"packing", material.subsurface_packing}, {"anisotropy", material.subsurface_anisotropy}}},
    {"emission", {{"radiance", spectral(material.emission)}, {"collimation", material.emission_collimation}, {"temperature_kelvin", material.temperature_kelvin}}},
    {"mix", {{"first", material_reference(material.mix.first)}, {"second", material_reference(material.mix.second)}, {"amount", sampled(material.mix.amount)}}},
  };
  if (material.layers.explicit_layers != 0u) {
    result["bsdf"] = material.layers.bsdf == MaterialClass::Undefined ? "auto" : kMaterialDescriptionTypes[material.layers.bsdf];
  }
  return result;
}

inline float material_description_json_float(const nlohmann::json& value) {
  if (value.is_number() == false) {
    throw std::runtime_error("Material parameter must be numeric.");
  }
  const double parsed = value.get<double>();
  if ((std::isfinite(parsed) == false) || (std::abs(parsed) > std::numeric_limits<float>::max())) {
    throw std::runtime_error("Material parameter must be finite and representable.");
  }
  return static_cast<float>(parsed);
}

inline uint32_t material_description_json_channel(const nlohmann::json& value) {
  if (value.is_null()) {
    return kInvalidIndex;
  }
  if ((value.is_number_unsigned() == false) || (value.get<uint64_t>() > 4u)) {
    throw std::runtime_error("Material texture channel must be null or an integer in [0, 4].");
  }
  return value.get<uint32_t>();
}

template <typename SpectrumReference, typename TextureReference, typename MaterialReference, typename MediumReference>
inline bool deserialize_material_description(const nlohmann::json& source, MaterialDescription& target, SpectrumReference spectrum, TextureReference texture,
  MaterialReference material_reference, MediumReference medium_reference, std::string& error) {
  try {
    MaterialDescription material;
    const auto resolve = [](auto reference, const nlohmann::json& value, uint32_t& index) {
      if (reference(value, index) == false) {
        throw std::runtime_error("Material contains an invalid resource reference.");
      }
    };
    const auto spectral = [&](const nlohmann::json& value, SpectralImage& output) {
      resolve(spectrum, value.at("spectrum"), output.spectrum_index);
      resolve(texture, value.at("texture"), output.image_index);
    };
    const auto sampled = [&](const nlohmann::json& value, SampledImage& output) {
      const auto& values = value.at("value");
      if ((values.is_array() == false) || (values.size() != 4u)) {
        throw std::runtime_error("Material sampled value must have four components.");
      }
      output.value = {material_description_json_float(values[0]), material_description_json_float(values[1]), material_description_json_float(values[2]),
        material_description_json_float(values[3])};
      output.channel = material_description_json_channel(value.at("channel"));
      resolve(texture, value.at("texture"), output.image_index);
    };
    const auto ior = [&](const nlohmann::json& value, RefractiveIndex& output) {
      const std::string type = value.at("type").template get<std::string>();
      if ((type != "dielectric") && (type != "conductor") && (type != "unspecified")) {
        throw std::runtime_error("Material has an unknown optical interface type.");
      }
      output.cls = type == "dielectric" ? SpectralDistribution::Dielectric : type == "conductor" ? SpectralDistribution::Conductor : SpectralDistribution::Invalid;
      resolve(spectrum, value.at("eta"), output.eta_index);
      resolve(spectrum, value.at("k"), output.k_index);
    };
    const std::string model = source.at("model").template get<std::string>();
    if ((model != "layered") && (model != "legacy_bsdf")) {
      throw std::runtime_error("Material has an unknown authoring model.");
    }
    material.layers.explicit_layers = model == "layered" ? 1u : 0u;
    if ((material.layers.explicit_layers != 0u) && source.contains("bsdf")) {
      const std::string bsdf = source.at("bsdf").template get<std::string>();
      if (bsdf != "auto") {
        material.layers.bsdf = MaterialClass::Count;
        for (uint32_t index = 0u; index < MaterialClass::Count; ++index) {
          if (bsdf == kMaterialDescriptionTypes[index]) {
            material.layers.bsdf = index;
            break;
          }
        }
        if (material.layers.bsdf == MaterialClass::Count) {
          throw std::runtime_error("Material has an unknown BSDF selection.");
        }
      }
    }
    const auto& base = source.at("base");
    const std::string type = base.at("type").template get<std::string>();
    for (uint32_t index = 0u; index < MaterialClass::Count; ++index) {
      if (type == kMaterialDescriptionTypes[index]) {
        material.cls = index;
        break;
      }
    }
    if (material.cls == MaterialClass::Undefined) {
      throw std::runtime_error("Material has an unknown base type.");
    }
    spectral(base.at("reflectance"), material.reflectance);
    spectral(base.at("scattering"), material.scattering);
    sampled(base.at("roughness"), material.roughness);
    sampled(base.at("metalness"), material.metalness);
    sampled(base.at("transmission"), material.transmission);
    ior(base.at("ior"), material.int_ior);
    const auto& diffraction = base.at("diffraction");
    material.diffraction_grating.period_nm = material_description_json_float(diffraction.at("period_nm"));
    material.diffraction_grating.optical_path_difference_nm = material_description_json_float(diffraction.at("optical_path_difference_nm"));
    material.diffraction_grating.duty_cycle = material_description_json_float(diffraction.at("duty_cycle"));
    material.diffraction_grating.rotation = material_description_json_float(diffraction.at("rotation"));
    const auto& coating = source.at("coating");
    material.layers.coating_enabled = coating.at("enabled").template get<bool>() ? 1u : 0u;
    ior(coating.at("ior"), material.coating.ior);
    sampled(coating.at("roughness"), material.coating.roughness);
    sampled(coating.at("thickness"), material.coating.thickness);
    spectral(coating.at("absorption"), material.coating.absorption);
    spectral(coating.at("scattering"), material.coating.scattering);
    material.coating.anisotropy = material_description_json_float(coating.at("anisotropy"));
    if (coating.contains("optical_thickness")) {
      const auto& optical_thickness = coating.at("optical_thickness");
      material.coating.min_thickness_nm = material_description_json_float(optical_thickness.at("min_nm"));
      material.coating.max_thickness_nm = material_description_json_float(optical_thickness.at("max_nm"));
      resolve(texture, optical_thickness.at("texture"), material.coating.thickness_nm_image);
    }
    const auto& film = source.at("thinfilm");
    material.layers.thinfilm_enabled = film.at("enabled").template get<bool>() ? 1u : 0u;
    ior(film.at("ior"), material.thinfilm.ior);
    resolve(texture, film.at("thickness_texture"), material.thinfilm.thinkness_image);
    material.thinfilm.min_thickness = material_description_json_float(film.at("min_thickness_nm"));
    material.thinfilm.max_thickness = material_description_json_float(film.at("max_thickness_nm"));
    material.thinfilm.weight = material_description_json_float(film.at("weight"));
    const auto& surface = source.at("surface");
    material.two_sided = surface.at("two_sided").template get<bool>() ? 1u : 0u;
    material.opacity = material_description_json_float(surface.at("opacity"));
    material.normal_scale = material_description_json_float(surface.at("normal_scale"));
    resolve(texture, surface.at("normal_texture"), material.normal_image_index);
    sampled(surface.at("alpha_mask"), material.alpha_mask);
    sampled(surface.at("bump"), material.bump);
    const auto& media = source.at("media");
    ior(media.at("exterior_ior"), material.ext_ior);
    resolve(medium_reference, media.at("interior"), material.int_medium);
    resolve(medium_reference, media.at("exterior"), material.ext_medium);
    const auto& subsurface = source.at("subsurface");
    const std::string subsurface_type = subsurface.at("type").template get<std::string>();
    const std::string subsurface_path = subsurface.at("path").template get<std::string>();
    if (((subsurface_type != "disabled") && (subsurface_type != "random_walk")) || ((subsurface_path != "diffuse") && (subsurface_path != "refracted"))) {
      throw std::runtime_error("Material has an unknown subsurface model.");
    }
    material.subsurface_cls = subsurface_type == "disabled" ? SubsurfaceMaterial::Disabled : SubsurfaceMaterial::RandomWalk;
    material.subsurface_path = subsurface_path == "diffuse" ? SubsurfaceMaterial::DiffusePath : SubsurfaceMaterial::RefractedPath;
    spectral(subsurface.at("distance"), material.subsurface);
    material.subsurface_packing = material_description_json_float(subsurface.at("packing"));
    material.subsurface_anisotropy = material_description_json_float(subsurface.at("anisotropy"));
    const auto& emission = source.at("emission");
    spectral(emission.at("radiance"), material.emission);
    material.emission_collimation = material_description_json_float(emission.at("collimation"));
    material.temperature_kelvin = material_description_json_float(emission.at("temperature_kelvin"));
    const auto& mix = source.at("mix");
    resolve(material_reference, mix.at("first"), material.mix.first);
    resolve(material_reference, mix.at("second"), material.mix.second);
    sampled(mix.at("amount"), material.mix.amount);
    target = material;
    error.clear();
    return true;
  } catch (const std::exception& exception) {
    error = exception.what();
    return false;
  }
}

}  // namespace etx
