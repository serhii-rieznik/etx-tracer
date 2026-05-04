#include <etx/render/host/openpbr_material_loader.hxx>
#include <etx/core/log.hxx>
#include <etx/render/shared/scene.hxx>
#include <etx/render/shared/scattering.hxx>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>

namespace etx {

namespace {

bool read_text_file(const std::filesystem::path& path, std::string& out_text) {
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (file.is_open() == false) {
    return false;
  }

  file.seekg(0, std::ios::end);
  const std::streamoff size = file.tellg();
  file.seekg(0, std::ios::beg);
  if (size <= 0) {
    out_text.clear();
    return true;
  }

  out_text.resize(static_cast<size_t>(size));
  file.read(out_text.data(), size);
  return file.good();
}

bool find_xml_attribute(const std::string& tag, const char* attribute_name, std::string& out_value) {
  const std::string prefix = std::string(attribute_name) + "=";
  size_t attribute_pos = tag.find(prefix);
  if (attribute_pos == std::string::npos) {
    return false;
  }

  size_t value_begin = attribute_pos + prefix.size();
  if (value_begin >= tag.size()) {
    return false;
  }

  const char quote = tag[value_begin];
  if ((quote != '"') && (quote != '\'')) {
    return false;
  }

  ++value_begin;
  const size_t value_end = tag.find(quote, value_begin);
  if (value_end == std::string::npos) {
    return false;
  }

  out_value = tag.substr(value_begin, value_end - value_begin);
  return true;
}

bool find_openpbr_surface_block(const std::string& text, std::string& out_block) {
  const size_t begin = text.find("<open_pbr_surface");
  if (begin == std::string::npos) {
    return false;
  }

  const size_t begin_close = text.find('>', begin);
  if (begin_close == std::string::npos) {
    return false;
  }

  const size_t end = text.find("</open_pbr_surface>", begin_close);
  if (end == std::string::npos) {
    return false;
  }

  out_block = text.substr(begin_close + 1u, end - begin_close - 1u);
  return true;
}

bool find_openpbr_input_tag(const std::string& block, const char* input_name, std::string& out_tag) {
  size_t pos = 0u;
  while (pos < block.size()) {
    const size_t input_begin = block.find("<input", pos);
    if (input_begin == std::string::npos) {
      return false;
    }

    const size_t input_end = block.find('>', input_begin);
    if (input_end == std::string::npos) {
      return false;
    }

    const std::string tag = block.substr(input_begin, input_end - input_begin + 1u);
    std::string name;
    if ((find_xml_attribute(tag, "name", name)) && (name == input_name)) {
      out_tag = tag;
      return true;
    }

    pos = input_end + 1u;
  }

  return false;
}

bool find_openpbr_input_value(const std::string& block, const char* input_name, std::string& out_value) {
  std::string tag;
  if (find_openpbr_input_tag(block, input_name, tag) == false) {
    return false;
  }

  return find_xml_attribute(tag, "value", out_value);
}

bool parse_openpbr_float(const std::string& value, float& out_value) {
  return sscanf(value.c_str(), "%f", &out_value) == 1;
}

bool parse_openpbr_bool(const std::string& value, bool& out_value) {
  std::string normalized = value;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
  if ((normalized == "true") || (normalized == "1")) {
    out_value = true;
    return true;
  }
  if ((normalized == "false") || (normalized == "0")) {
    out_value = false;
    return true;
  }
  return false;
}

bool parse_openpbr_color3(const std::string& value, float3& out_value) {
  std::string normalized = value;
  std::replace(normalized.begin(), normalized.end(), ',', ' ');
  float values[3] = {};
  if (sscanf(normalized.c_str(), "%f %f %f", values + 0, values + 1, values + 2) != 3) {
    return false;
  }

  out_value = {values[0], values[1], values[2]};
  return true;
}

bool materialx_uses_acescg(const std::string& text) {
  const size_t materialx_begin = text.find("<materialx");
  if (materialx_begin == std::string::npos) {
    return false;
  }

  const size_t materialx_end = text.find('>', materialx_begin);
  if (materialx_end == std::string::npos) {
    return false;
  }

  const std::string tag = text.substr(materialx_begin, materialx_end - materialx_begin + 1u);
  std::string colorspace;
  if (find_xml_attribute(tag, "colorspace", colorspace) == false) {
    return false;
  }

  std::transform(colorspace.begin(), colorspace.end(), colorspace.begin(), ::tolower);
  return colorspace == "acescg";
}

float3 acescg_to_linear_srgb(const float3& color) {
  return {
    1.7048873f * color.x - 0.6217906f * color.y - 0.0832584f * color.z,
    -0.1305105f * color.x + 1.1408028f * color.y - 0.0105485f * color.z,
    -0.0240046f * color.x - 0.1289690f * color.y + 1.1535549f * color.z,
  };
}

float3 openpbr_color_to_render_rgb(const float3& color, bool acescg) {
  const float3 converted = acescg ? acescg_to_linear_srgb(color) : color;
  return max(converted, float3{});
}

float openpbr_input_float(const std::string& block, const char* input_name, float fallback) {
  std::string value;
  float result = fallback;
  if ((find_openpbr_input_value(block, input_name, value)) && (parse_openpbr_float(value, result))) {
    return result;
  }
  return fallback;
}

bool openpbr_input_bool(const std::string& block, const char* input_name, bool fallback) {
  std::string value;
  bool result = fallback;
  if ((find_openpbr_input_value(block, input_name, value)) && (parse_openpbr_bool(value, result))) {
    return result;
  }
  return fallback;
}

float3 openpbr_input_color3(const std::string& block, const char* input_name, const float3& fallback, bool acescg) {
  std::string value;
  float3 result = fallback;
  if ((find_openpbr_input_value(block, input_name, value)) && (parse_openpbr_color3(value, result))) {
    return openpbr_color_to_render_rgb(result, acescg);
  }
  return openpbr_color_to_render_rgb(fallback, acescg);
}

float2 openpbr_roughness_to_alpha(float roughness, float anisotropy) {
  const float r = clamp(roughness, 0.0f, 1.0f);
  const float a = clamp(anisotropy, 0.0f, 1.0f);
  const float aniso_invert = 1.0f - a;
  const float aniso_invert_sq = aniso_invert * aniso_invert;
  const float denom = aniso_invert_sq + 1.0f;
  const float alpha_x = max(kEpsilon, r * r * sqrtf(2.0f / denom));
  const float alpha_y = max(kEpsilon, aniso_invert * alpha_x);
  return {alpha_x, alpha_y};
}

float openpbr_modulated_specular_ior(const float specular_ior, const float specular_weight) {
  const float eta = max(kEpsilon, specular_ior);
  const float f0_sqrt = (eta - 1.0f) / (eta + 1.0f);
  const float f0 = f0_sqrt * f0_sqrt;
  const float scaled_f0 = clamp(max(0.0f, specular_weight) * f0, 0.0f, 0.99999f);
  const float eta_sign = (eta >= 1.0f) ? 1.0f : -1.0f;
  const float epsilon = eta_sign * sqrtf(scaled_f0);
  return max(kEpsilon, (1.0f + epsilon) / max(kEpsilon, 1.0f - epsilon));
}

}  // namespace

bool load_openpbr_material_file(const std::filesystem::path& path, SceneData& data, Material& material) {
  std::string text;
  if (read_text_file(path, text) == false) {
    log::error("Failed to open OpenPBR material file: %s", path.string().c_str());
    return false;
  }

  std::string block;
  if (find_openpbr_surface_block(text, block) == false) {
    log::error("OpenPBR material file does not contain an open_pbr_surface node: %s", path.string().c_str());
    return false;
  }

  const bool acescg = materialx_uses_acescg(text);
  const float base_weight = clamp(openpbr_input_float(block, "base_weight", 1.0f), 0.0f, 1.0f);
  const float3 base_color = openpbr_input_color3(block, "base_color", {0.8f, 0.8f, 0.8f}, acescg);
  const float metalness = clamp(openpbr_input_float(block, "base_metalness", 0.0f), 0.0f, 1.0f);

  const float specular_weight = max(0.0f, openpbr_input_float(block, "specular_weight", 1.0f));
  const float3 specular_color = openpbr_input_color3(block, "specular_color", {1.0f, 1.0f, 1.0f}, acescg);
  const float specular_roughness = openpbr_input_float(block, "specular_roughness", 0.3f);
  const float specular_anisotropy = openpbr_input_float(block, "specular_roughness_anisotropy", 0.0f);
  const float specular_ior = openpbr_modulated_specular_ior(openpbr_input_float(block, "specular_ior", 1.5f), specular_weight);

  const float transmission = clamp(openpbr_input_float(block, "transmission_weight", 0.0f), 0.0f, 1.0f);
  const float opacity = clamp(openpbr_input_float(block, "geometry_opacity", 1.0f), 0.0f, 1.0f);
  const bool thin_walled = openpbr_input_bool(block, "geometry_thin_walled", false);

  material.cls = MaterialClass::OpenPBR;
  material.scattering.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance(base_color * base_weight));
  material.reflectance.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance(specular_color));
  const float2 alpha = openpbr_roughness_to_alpha(specular_roughness, specular_anisotropy);
  material.roughness.value = {alpha.x, alpha.y, alpha.x, alpha.y};
  material.metalness.value = {metalness, metalness, metalness, metalness};
  material.transmission.value = {transmission, transmission, transmission, transmission};
  material.opacity = opacity;
  material.two_sided = thin_walled ? 1u : material.two_sided;

  material.ext_ior.cls = SpectralDistribution::Dielectric;
  material.ext_ior.eta_index = data.add_spectrum(SpectralDistribution::constant(1.0f));
  material.ext_ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
  material.int_ior.cls = SpectralDistribution::Dielectric;
  material.int_ior.eta_index = data.add_spectrum(SpectralDistribution::constant(specular_ior));
  material.int_ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));

  const float thin_film_weight = clamp(openpbr_input_float(block, "thin_film_weight", 0.0f), 0.0f, 1.0f);
  const float thin_film_thickness = openpbr_input_float(block, "thin_film_thickness", 0.5f);
  const float thin_film_ior = max(kEpsilon, openpbr_input_float(block, "thin_film_ior", 1.4f));
  if ((thin_film_weight > 0.0f) && (thin_film_thickness > 0.0f)) {
    const float thickness_nm = thin_film_thickness * 1000.0f;
    material.thinfilm.min_thickness = thickness_nm;
    material.thinfilm.max_thickness = thickness_nm;
    material.thinfilm.ior.cls = SpectralDistribution::Dielectric;
    material.thinfilm.ior.eta_index = data.add_spectrum(SpectralDistribution::constant(thin_film_ior));
    material.thinfilm.ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
  }

  const float emission_luminance = max(0.0f, openpbr_input_float(block, "emission_luminance", 0.0f));
  const float3 emission_color = openpbr_input_color3(block, "emission_color", {1.0f, 1.0f, 1.0f}, acescg);
  if (emission_luminance > 0.0f) {
    material.emission.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_luminance(emission_color * emission_luminance));
  }

  const float subsurface_weight = clamp(openpbr_input_float(block, "subsurface_weight", 0.0f), 0.0f, 1.0f);
  if (subsurface_weight > 0.0f) {
    const float subsurface_radius = max(0.0f, openpbr_input_float(block, "subsurface_radius", 1.0f));
    const float3 subsurface_radius_scale = openpbr_input_color3(block, "subsurface_radius_scale", {1.0f, 0.5f, 0.25f}, false);
    material.subsurface_cls = SubsurfaceMaterial::RandomWalk;
    material.subsurface.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance(max(float3{}, subsurface_radius_scale * subsurface_radius)));
  }

  return true;
}

}  // namespace etx
