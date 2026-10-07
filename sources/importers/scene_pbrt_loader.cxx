#include <etx/std.hxx>
#include <etx/core/core.hxx>
#include "scene_pbrt_loader.hxx"
#include "scene_pbrt_input.hxx"
#include "scene_pbrt_mesh.hxx"
#include "scene_pbrt_spectrum.hxx"
#include <etx/render/shared/ior_database.hxx>
#include <etx/render/shared/vertex_utils.hxx>
#include <etx/render/host/scene_loader_utils.hxx>
#include <etx/render/host/scene_procedural_geometry.hxx>
#include <etx/core/log.hxx>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace etx {
namespace {

[[noreturn]] void fail(const PbrtStatement& statement, const std::string& text) {
  throw std::runtime_error(statement.location.describe() + ": " + text);
}

float scalar(const PbrtStatement& statement, const char* name, float fallback) {
  const auto* value = statement.find(name);
  if (value == nullptr)
    return fallback;
  if ((value->numbers.size() != 1u) || (std::abs(value->numbers[0]) > std::numeric_limits<float>::max()))
    fail(statement, std::string("Expected a scalar parameter: ") + name);
  return static_cast<float>(value->numbers[0]);
}

uint32_t integer(const PbrtStatement& statement, const char* name, uint32_t fallback, uint32_t minimum, uint32_t maximum) {
  const auto* parameter = statement.find(name);
  if (parameter == nullptr)
    return fallback;
  if ((parameter->numbers.size() != 1u) || (parameter->numbers[0] < minimum) || (parameter->numbers[0] > maximum) || (std::floor(parameter->numbers[0]) != parameter->numbers[0]))
    fail(statement, std::string("Invalid integer parameter: ") + name);
  return static_cast<uint32_t>(parameter->numbers[0]);
}

std::string texture_key(const std::string& name, const std::string& type) {
  return (type == "float" ? "float:" : "spectrum:") + name;
}

std::string string(const PbrtStatement& statement, const char* name, const std::string& fallback) {
  const auto* value = statement.find(name);
  if (value == nullptr)
    return fallback;
  if (value->strings.size() != 1u)
    fail(statement, std::string("Expected a string parameter: ") + name);
  return value->strings[0];
}

bool boolean(const PbrtStatement& statement, const char* name, bool fallback) {
  const std::string value = string(statement, name, fallback ? "true" : "false");
  if ((value != "true") && (value != "false"))
    fail(statement, std::string("Invalid Boolean parameter: ") + name);
  return value == "true";
}

float3 vector(const PbrtStatement& statement, const char* name, const float3& fallback) {
  const auto* value = statement.find(name);
  if (value == nullptr)
    return fallback;
  if (value->numbers.size() != 3u)
    fail(statement, std::string("Expected a three-component parameter: ") + name);
  for (const double component : value->numbers) {
    if (std::abs(component) > std::numeric_limits<float>::max())
      fail(statement, std::string("Parameter exceeds native floating-point range: ") + name);
  }
  return {static_cast<float>(value->numbers[0]), static_cast<float>(value->numbers[1]), static_cast<float>(value->numbers[2])};
}

float4x4 identity() {
  return {{{1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 1.0f}}};
}

float3 pbrt_coordinate_tangent(const float3& normal) {
  const float sign = std::copysign(1.0f, normal.z);
  const float a = -1.0f / (sign + normal.z);
  const float b = normal.x * normal.y * a;
  return {1.0f + sign * normal.x * normal.x * a, sign * b, -sign * normal.x};
}

struct TextureValue {
  SpectralDistribution spectrum;
  uint32_t image = kInvalidIndex;
};

struct GraphicsState {
  float4x4 transform = identity();
  uint32_t material = kInvalidIndex;
  PbrtStatement area_light;
  bool reverse = false;
};

struct ObjectShape {
  uint32_t mesh;
  float4x4 transform;
};

struct Loader {
  Loader(const std::filesystem::path& root, SceneData& scene, TaskScheduler& scheduler, Camera& camera, PbrtVersion version)
    : _root(root)
    , _data(scene)
    , _scheduler(scheduler)
    , _camera(camera)
    , _version(version) {
    PbrtStatement material;
    material.location = {root / "scene.pbrt", 1u};
    _state.material = make_material(material, "diffuse", "PBRT default material");
  }

  const TextureValue& texture_value(const PbrtStatement& statement, const char* name, const std::string& value_type) {
    const std::string texture_name = texture_key(string(statement, name, ""), value_type);
    if ((_version == PbrtVersion::V4) && (_textures.contains(texture_name) == false)) {
      const auto definition = _texture_definitions.find(texture_name);
      if (definition != _texture_definitions.end()) {
        if (_resolving_textures.insert(texture_name).second == false)
          fail(statement, "Cyclic texture definition: " + texture_name);
        texture(definition->second);
        _resolving_textures.erase(texture_name);
      }
    }
    const auto found = _textures.find(texture_name);
    if (found == _textures.end())
      fail(statement, std::string("Unknown texture: ") + name);
    return found->second;
  }

  TextureValue spectrum(const PbrtStatement& statement, const char* name, float fallback, bool emission) {
    const auto* parameter = statement.find(name);
    if (parameter == nullptr)
      return {SpectralDistribution::constant(fallback), kInvalidIndex};
    if (parameter->type == "texture")
      return texture_value(statement, name, statement.directive == "Texture" ? statement.arguments[1] : "spectrum");
    if ((parameter->type == "rgb") || (parameter->type == "color")) {
      const float3 rgb = vector(statement, name, {});
      if ((rgb.x < 0.0f) || (rgb.y < 0.0f) || (rgb.z < 0.0f))
        fail(statement, "RGB spectrum components cannot be negative.");
      return {emission ? (_version == PbrtVersion::V4 ? pbrt_rgb_illuminant(rgb) : SpectralDistribution::rgb_luminance(rgb)) : SpectralDistribution::rgb_reflectance(rgb),
        kInvalidIndex};
    }
    if (parameter->type == "blackbody") {
      if (parameter->numbers.empty() || (parameter->numbers.size() > 2u) || (parameter->numbers[0] <= 0.0))
        fail(statement, "Invalid blackbody spectrum.");
      const float scale = parameter->numbers.size() == 2u ? static_cast<float>(parameter->numbers[1]) : 1.0f;
      const float temperature = static_cast<float>(parameter->numbers[0]);
      const float peak = black_body_radiation(black_body_radiation_maximum_wavelength(temperature), temperature);
      if ((std::isfinite(scale) == false) || (scale < 0.0f) || (std::isfinite(peak) == false) || (peak <= 0.0f))
        fail(statement, "Blackbody temperature or scale exceeds the supported range.");
      return {SpectralDistribution::from_black_body(temperature, scale / peak), kInvalidIndex};
    }
    if (parameter->strings.empty() == false) {
      const std::string identifier = string(statement, name, "");
      SpectralDistribution named;
      if ((_version == PbrtVersion::V4) && load_pbrt_named_spectrum(identifier, named))
        return {named, kInvalidIndex};
      const auto path = _root / std::filesystem::u8path(identifier);
      std::ifstream file(path);
      if (file.is_open() == false)
        fail(statement, "Cannot load spectrum: " + path_to_utf8(path));
      PbrtStatement loaded = statement;
      PbrtParameter samples;
      samples.name = name;
      samples.type = "spectrum";
      std::string line;
      while (std::getline(file, line)) {
        line = line.substr(0u, line.find('#'));
        std::istringstream tokens(line);
        double value;
        while (tokens >> value) {
          if ((std::isfinite(value) == false) || (std::abs(value) > std::numeric_limits<float>::max()))
            fail(statement, "Spectrum value exceeds the native floating-point range: " + identifier);
          samples.numbers.push_back(value);
        }
        if (tokens.eof() == false)
          fail(statement, "Invalid spectrum data: " + identifier);
      }
      if (file.bad() || samples.numbers.empty() || ((samples.numbers.size() % 2u) != 0u))
        fail(statement, "Spectrum file requires wavelength and power pairs: " + identifier);
      loaded.parameters.clear();
      loaded.parameters.emplace_back(std::move(samples));
      return spectrum(loaded, name, fallback, emission);
    }
    if (parameter->numbers.size() == 1u) {
      const float value = scalar(statement, name, fallback);
      if (value < 0.0f)
        fail(statement, "Spectrum power cannot be negative.");
      return {SpectralDistribution::constant(value), kInvalidIndex};
    }
    if ((parameter->numbers.empty() == false) && ((parameter->numbers.size() % 2u) == 0u)) {
      std::vector<float2> samples;
      samples.reserve(parameter->numbers.size() / 2u);
      for (size_t index = 0u; index < parameter->numbers.size(); index += 2u) {
        const float2 sample = {static_cast<float>(parameter->numbers[index]), static_cast<float>(parameter->numbers[index + 1u])};
        if ((std::isfinite(sample.x) == false) || (std::isfinite(sample.y) == false) || (sample.x <= 0.0f) || (sample.y < 0.0f))
          fail(statement, "Spectrum requires positive wavelengths and non-negative finite powers.");
        if ((samples.empty() == false) && (sample.x <= samples.back().x))
          fail(statement, "Spectrum wavelengths must increase.");
        samples.push_back(sample);
      }
      return {pbrt_sampled_spectrum(samples), kInvalidIndex};
    }
    fail(statement, std::string("Invalid spectrum parameter: ") + name);
  }

  TextureValue float_texture(const PbrtStatement& statement, const char* name, float fallback) {
    const auto* parameter = statement.find(name);
    if ((parameter == nullptr) || (parameter->type == "float"))
      return {SpectralDistribution::constant(scalar(statement, name, fallback)), kInvalidIndex};
    if (parameter->type != "texture")
      fail(statement, std::string("Expected a float or float texture: ") + name);
    return texture_value(statement, name, "float");
  }

  SpectralImage spectral_image(const TextureValue& value) {
    return {_data.add_spectrum(value.spectrum), value.image};
  }

  TextureValue radiance(const PbrtStatement& statement, const char* name) {
    if ((_version == PbrtVersion::V4) && ((scalar(statement, "power", -1.0f) > 0.0f) || (scalar(statement, "illuminance", -1.0f) > 0.0f)))
      fail(statement, "PBRT light power and illuminance controls are not implemented.");
    TextureValue result = spectrum(statement, name, 1.0f, true);
    float scale = _exposure;
    if (_version == PbrtVersion::V3) {
      const auto multiplier = spectrum(statement, "scale", 1.0f, false);
      if (multiplier.image != kInvalidIndex)
        fail(statement, "Light scale cannot reference an image texture.");
      result.spectrum = pbrt_multiply_spectra(result.spectrum, multiplier.spectrum);
    } else
      scale *= scalar(statement, "scale", 1.0f);
    if (scale < 0.0f)
      fail(statement, "Light scale cannot be negative.");
    if (_version == PbrtVersion::V4) {
      const auto* parameter = statement.find(name);
      const bool rgb_illuminant = (parameter == nullptr) || (parameter->type == "rgb") || (parameter->type == "color");
      if (parameter == nullptr)
        load_pbrt_named_spectrum("stdillum-D65", result.spectrum);
      const float photometric = rgb_illuminant ? (1.0f / kInvCIEYIntegral) : pbrt_photometric_response(result.spectrum);
      if (photometric > 0.0f)
        scale /= kInvCIEYIntegral * photometric;
    }
    if (std::isfinite(scale) == false)
      fail(statement, "Light scale exceeds the native floating-point range.");
    result.spectrum = pbrt_multiply_spectra(result.spectrum, SpectralDistribution::constant(scale));
    return result;
  }

  float alpha(const PbrtStatement& statement, const char* name, float fallback) const {
    float roughness = scalar(statement, name, fallback);
    if (roughness < 0.0f)
      fail(statement, "Roughness cannot be negative.");
    if (boolean(statement, "remaproughness", true) == false)
      return roughness;
    if (_version == PbrtVersion::V4)
      return std::sqrt(roughness);
    const float x = std::log(std::max(roughness, 0.001f));
    return 1.62142f + x * (0.819955f + x * (0.1734f + x * (0.0171201f + x * 0.000640711f)));
  }

  uint32_t make_material(const PbrtStatement& statement, const std::string& type, const std::string& name) {
    Material material;
    material.ext_ior = {SpectralDistribution::Dielectric, _data.add_spectrum(SpectralDistribution::constant(1.0f)), _data.add_spectrum(SpectralDistribution::constant(0.0f))};
    const bool conductor = (type == "metal") || (type == "conductor");
    const TextureValue eta =
      conductor ? TextureValue{SpectralDistribution::constant(1.5f), kInvalidIndex} : spectrum(statement, statement.find("eta") != nullptr ? "eta" : "index", 1.5f, false);
    if (eta.image != kInvalidIndex)
      fail(statement, "Spatially varying refractive indices are not implemented.");
    material.int_ior = {SpectralDistribution::Dielectric, _data.add_spectrum(eta.spectrum), _data.add_spectrum(SpectralDistribution::constant(0.0f))};
    material.reflectance = spectral_image({SpectralDistribution::constant(1.0f), kInvalidIndex});
    material.scattering = spectral_image(spectrum(statement, _version == PbrtVersion::V3 ? "Kd" : "reflectance", 0.5f, false));
    const float default_roughness = (_version == PbrtVersion::V3) ? (conductor ? 0.01f : (type == "glass" ? 0.0f : 0.1f)) : 0.0f;
    const float roughness = scalar(statement, "roughness", default_roughness);
    material.roughness.value = {alpha(statement, "uroughness", roughness), alpha(statement, "vroughness", roughness), 0.0f, 0.0f};
    if ((type == "diffuse") || (type == "matte")) {
      material.cls = MaterialClass::Diffuse;
      const float sigma = std::clamp(scalar(statement, "sigma", 0.0f), 0.0f, 90.0f) * kPi / 180.0f;
      material.roughness.value = {sigma, sigma, 0.0f, 0.0f};
    } else if ((_version == PbrtVersion::V4) && (type == "diffusetransmission")) {
      material.cls = MaterialClass::Translucent;
      const auto scale = SpectralDistribution::constant(std::max(0.0f, scalar(statement, "scale", 1.0f)));
      TextureValue reflection = spectrum(statement, "reflectance", 0.25f, false);
      TextureValue transmission = spectrum(statement, "transmittance", 0.25f, false);
      reflection.spectrum = pbrt_multiply_spectra(reflection.spectrum, scale);
      transmission.spectrum = pbrt_multiply_spectra(transmission.spectrum, scale);
      material.reflectance = spectral_image(reflection);
      material.scattering = spectral_image(transmission);
      material.roughness.value = {};
    } else if ((type == "plastic") || (type == "coateddiffuse") || (type == "substrate")) {
      material.cls = MaterialClass::Plastic;
      if (_version == PbrtVersion::V3)
        material.reflectance = spectral_image(spectrum(statement, "Ks", 0.25f, false));
    } else if ((type == "dielectric") || (type == "glass")) {
      material.cls = MaterialClass::Dielectric;
      material.scattering = spectral_image(spectrum(statement, "Kt", 1.0f, false));
      material.reflectance = spectral_image(spectrum(statement, "Kr", 1.0f, false));
      if (((statement.find("roughness") == nullptr) && (statement.find("uroughness") == nullptr) && (statement.find("vroughness") == nullptr)) ||
          ((scalar(statement, "uroughness", roughness) == 0.0f) && (scalar(statement, "vroughness", roughness) == 0.0f)))
        material.roughness.value = {};
    } else if (type == "mirror") {
      material.cls = MaterialClass::Mirror;
      material.reflectance = spectral_image(spectrum(statement, "Kr", 0.9f, false));
      material.roughness.value = {};
    } else if ((type == "metal") || (type == "conductor")) {
      material.cls = MaterialClass::Conductor;
      material.int_ior.cls = SpectralDistribution::Conductor;
      SpectralDistribution conductor_eta, conductor_k;
      if ((_version == PbrtVersion::V4) && (statement.find("reflectance") != nullptr)) {
        if ((statement.find("eta") != nullptr) || (statement.find("k") != nullptr))
          fail(statement, "Conductor reflectance cannot be combined with eta or k.");
        const auto reflectance = spectrum(statement, "reflectance", 1.0f, false);
        if (reflectance.image != kInvalidIndex)
          fail(statement, "Spatially varying conductor reflectance is not implemented.");
        const auto extinction = [](float value) {
          const float reflectance = std::clamp(value, 0.0f, 0.9999f);
          return 2.0f * std::sqrt(reflectance / (1.0f - reflectance));
        };
        float2 samples[WavelengthCount];
        for (uint32_t index = 0u; index < WavelengthCount; ++index) {
          const float wavelength = static_cast<float>(ShortestWavelength + index);
          samples[index] = {wavelength, extinction(reflectance.spectrum.query(SpectralQuery(wavelength, SpectralFlags::Spectral)).value)};
        }
        conductor_eta = SpectralDistribution::constant(1.0f);
        conductor_k = SpectralDistribution::from_samples(samples, WavelengthCount);
        const auto rgb = reflectance.spectrum.integrated();
        conductor_k.integrated_value = {extinction(rgb.x), extinction(rgb.y), extinction(rgb.z)};
      } else {
        if (statement.find("eta") != nullptr) {
          const auto value = spectrum(statement, "eta", 1.0f, false);
          if (value.image != kInvalidIndex)
            fail(statement, "Spatially varying refractive indices are not implemented.");
          conductor_eta = value.spectrum;
        } else
          load_pbrt_named_spectrum("metal-Cu-eta", conductor_eta);
        if (statement.find("k") != nullptr) {
          const auto value = spectrum(statement, "k", 0.0f, false);
          if (value.image != kInvalidIndex)
            fail(statement, "Spatially varying extinction coefficients are not implemented.");
          conductor_k = value.spectrum;
        } else
          load_pbrt_named_spectrum("metal-Cu-k", conductor_k);
      }
      material.int_ior.eta_index = _data.add_spectrum(conductor_eta);
      material.int_ior.k_index = _data.add_spectrum(conductor_k);
      if (_version == PbrtVersion::V3)
        material.reflectance = spectral_image(spectrum(statement, "Kr", 1.0f, false));
    } else if ((type.empty()) || (type == "interface") || (type == "none")) {
      material.cls = MaterialClass::Boundary;
    } else
      fail(statement, "Unsupported PBRT material: " + type);
    if (const auto* parameter = statement.find("normalmap")) {
      if (parameter->type != "string")
        fail(statement, "PBRT normalmap requires an image filename.");
      const std::string filename = string(statement, "normalmap", "");
      if (filename.empty() == false) {
        const auto path = _root / std::filesystem::u8path(filename);
        const uint32_t options = Image::RepeatU | Image::RepeatV | Image::SkipSRGBConversion | Image::TextureUVTransform | Image::TexelCenteredUV;
        material.normal_image_index = _data.add_image(path_to_utf8(path).c_str(), options, {}, {1.0f, 1.0f});
      }
    }
    const char* bump_parameter = _version == PbrtVersion::V3 ? "bumpmap" : "displacement";
    if (statement.find(bump_parameter) != nullptr) {
      const TextureValue value = float_texture(statement, bump_parameter, 0.0f);
      material.bump = {{value.spectrum.integrated().x, 0.0f, 0.0f, 0.0f}, value.image, 4u};
      _has_bump_maps = _has_bump_maps || (value.image != kInvalidIndex);
    }
    std::string unique_name = name;
    while (_data.has_material(unique_name.c_str()))
      unique_name += "#" + std::to_string(_data.materials.size());
    const uint32_t index = _data.add_material(unique_name.c_str());
    _data.materials[index] = material;
    return index;
  }

  void texture(const PbrtStatement& statement) {
    if ((statement.arguments[1] != "float") && (statement.arguments[1] != "spectrum") && ((_version != PbrtVersion::V3) || (statement.arguments[1] != "color")))
      fail(statement, "Unsupported texture value type: " + statement.arguments[1]);
    const std::string& type = statement.arguments[2];
    TextureValue value;
    if (type == "constant")
      value = statement.arguments[1] == "float" ? TextureValue{SpectralDistribution::constant(scalar(statement, "value", 1.0f)), kInvalidIndex}
                                                : spectrum(statement, "value", 1.0f, false);
    else if (type == "imagemap") {
      if (string(statement, "mapping", "uv") != "uv")
        fail(statement, "Only UV image texture mapping is currently implemented.");
      const auto path = _root / std::filesystem::u8path(string(statement, "filename", ""));
      uint32_t options = Image::RepeatU | Image::RepeatV;
      const std::string wrap = string(statement, "wrap", "repeat");
      if (wrap == "clamp")
        options = Image::Regular;
      else if (wrap != "repeat")
        fail(statement, "Unsupported PBRT texture wrap mode: " + wrap);
      const std::string encoding = string(statement, "encoding", "");
      const bool float_texture = statement.arguments[1] == "float";
      if (float_texture && boolean(statement, "invert", false))
        fail(statement, "Inverted PBRT float image textures are not implemented.");
      if ((encoding == "linear") || (boolean(statement, "gamma", true) == false) || (float_texture && encoding.empty() && (path.extension() != ".png")))
        options |= Image::SkipSRGBConversion;
      if ((encoding.empty() == false) && (encoding != "sRGB") && (encoding != "linear"))
        fail(statement, "Unsupported image encoding: " + encoding);
      options |= Image::TextureUVTransform | Image::TexelCenteredUV;
      const float2 scale = {scalar(statement, "uscale", 1.0f), scalar(statement, "vscale", 1.0f)};
      const float2 offset = {scalar(statement, "udelta", 0.0f), 1.0f - scale.y - scalar(statement, "vdelta", 0.0f)};
      const float strength = scalar(statement, "scale", 1.0f);
      if ((strength < 0.0f) && (float_texture == false))
        fail(statement, "Image texture scale cannot be negative.");
      value = {SpectralDistribution::constant(strength), _data.add_image(path_to_utf8(path).c_str(), options, offset, scale)};
    } else if (type == "scale") {
      const bool scalar_texture = statement.arguments[1] == "float";
      const char* source = _version == PbrtVersion::V4 ? "tex" : "tex1";
      const char* multiplier = _version == PbrtVersion::V4 ? "scale" : "tex2";
      value = scalar_texture ? float_texture(statement, source, 1.0f) : spectrum(statement, source, 1.0f, false);
      const TextureValue factor = scalar_texture ? float_texture(statement, multiplier, 1.0f) : spectrum(statement, multiplier, 1.0f, false);
      if ((value.image != kInvalidIndex) && (factor.image != kInvalidIndex))
        fail(statement, "Multiplication of two image textures is not implemented.");
      if (value.image == kInvalidIndex)
        value.image = factor.image;
      value.spectrum = pbrt_multiply_spectra(value.spectrum, factor.spectrum);
    } else
      fail(statement, "Unsupported PBRT texture: " + type);
    _textures[texture_key(statement.arguments[0], statement.arguments[1])] = value;
  }

  uint32_t mesh(const PbrtStatement& statement) {
    float alpha_scale = 1.0f;
    uint32_t alpha_image = kInvalidIndex;
    if (const auto* alpha = statement.find("alpha")) {
      if (alpha->type == "texture") {
        const auto& value = texture_value(statement, "alpha", "float");
        alpha_scale = value.spectrum.integrated().x;
        alpha_image = value.image;
      } else if (alpha->type == "float")
        alpha_scale = scalar(statement, "alpha", 1.0f);
      else
        fail(statement, "PBRT shape alpha requires a float or float texture.");
    }
    if (statement.find("shadowalpha") != nullptr)
      fail(statement, "PBRT shadow alpha mapping is not implemented.");
    if ((_state.area_light.directive.empty() == false) && ((alpha_image != kInvalidIndex) || (alpha_scale != 1.0f)))
      fail(statement, "PBRT alpha-masked area lights are not implemented.");
    PbrtMesh source;
    const auto& type = statement.arguments[0];
    if (type == "plymesh")
      source = load_pbrt_ply(_root / std::filesystem::u8path(string(statement, "filename", "")));
    else if ((type == "trianglemesh") || (type == "loopsubdiv") || (type == "bilinearmesh")) {
      const bool bilinear = type == "bilinearmesh";
      const uint32_t index_count = bilinear ? 4u : 3u;
      const auto* positions = statement.find("P");
      const auto* indices = statement.find("indices");
      if ((positions == nullptr) || ((bilinear == false) && (indices == nullptr)) || ((positions->numbers.size() % 3u) != 0u) ||
          ((indices != nullptr) && ((indices->numbers.size() % index_count) != 0u)))
        fail(statement, "Mesh requires position triples and correctly sized index groups.");
      if ((bilinear) && (indices == nullptr) && (positions->numbers.size() != 12u) && (positions->numbers.empty() == false))
        fail(statement, "Bilinear meshes require index quadruples unless there is a single patch.");
      for (size_t index = 0u; index < positions->numbers.size(); index += 3u)
        source.positions.push_back(
          {static_cast<float>(positions->numbers[index]), static_cast<float>(positions->numbers[index + 1u]), static_cast<float>(positions->numbers[index + 2u])});
      if (bilinear == false) {
        for (size_t index = 0u; index < indices->numbers.size(); index += 3u) {
          uint3 triangle = {};
          uint32_t* entries[] = {&triangle.x, &triangle.y, &triangle.z};
          for (size_t corner = 0u; corner < 3u; ++corner) {
            const double value = indices->numbers[index + corner];
            if ((value < 0.0) || (value >= source.positions.size()) || (std::floor(value) != value))
              fail(statement, "Triangle index is outside the vertex array.");
            *entries[corner] = static_cast<uint32_t>(value);
          }
          source.indices.push_back(triangle);
        }
      }
      if (const auto* normals = statement.find("N")) {
        if (normals->numbers.size() != positions->numbers.size())
          fail(statement, "Normal count differs from vertex count.");
        for (size_t index = 0u; index < normals->numbers.size(); index += 3u)
          source.normals.push_back(
            {static_cast<float>(normals->numbers[index]), static_cast<float>(normals->numbers[index + 1u]), static_cast<float>(normals->numbers[index + 2u])});
      }
      const auto* uv = statement.find("uv");
      if (uv == nullptr)
        uv = statement.find("st");
      if (uv != nullptr) {
        if (uv->numbers.size() != (source.positions.size() * 2u))
          fail(statement, "UV count differs from vertex count.");
        for (size_t index = 0u; index < uv->numbers.size(); index += 2u)
          source.texcoords.push_back({static_cast<float>(uv->numbers[index]), static_cast<float>(uv->numbers[index + 1u])});
      }
      if (bilinear) {
        discard_degenerate_pbrt_triangles(source);
        const size_t patch_count = indices != nullptr ? indices->numbers.size() / 4u : (source.positions.empty() ? 0u : 1u);
        std::vector<Vertex> vertices;
        std::vector<uint3> triangles;
        PbrtMesh tessellated;
        const size_t vertex_count = patch_count * (BilinearPatch::DefaultSubdivisions + 1u) * (BilinearPatch::DefaultSubdivisions + 1u);
        const size_t triangle_count = patch_count * 2u * BilinearPatch::DefaultSubdivisions * BilinearPatch::DefaultSubdivisions;
        if ((vertex_count >= kInvalidIndex) || (triangle_count >= kInvalidIndex))
          fail(statement, "Bilinear mesh exceeds native geometry index capacity.");
        tessellated.positions.reserve(vertex_count);
        tessellated.normals.reserve(vertex_count);
        tessellated.texcoords.reserve(vertex_count);
        tessellated.indices.reserve(triangle_count);
        for (size_t patch_index = 0u; patch_index < patch_count; ++patch_index) {
          BilinearPatch patch;
          for (uint32_t corner = 0u; corner < 4u; ++corner) {
            const double value = indices != nullptr ? indices->numbers[patch_index * 4u + corner] : corner;
            if ((value < 0.0) || (value >= source.positions.size()) || (std::floor(value) != value))
              fail(statement, "Bilinear index is outside the vertex array.");
            const size_t index = static_cast<size_t>(value);
            patch.positions[corner] = source.positions[index];
            if (source.normals.empty() == false)
              patch.normals[corner] = source.normals[index];
            if (source.texcoords.empty() == false)
              patch.texcoords[corner] = source.texcoords[index];
          }
          vertices.clear();
          triangles.clear();
          if (tessellate_bilinear_patch(patch, BilinearPatch::DefaultSubdivisions, vertices, triangles) == false)
            fail(statement, "Invalid bilinear patch coordinates.");
          const uint32_t offset = static_cast<uint32_t>(tessellated.positions.size());
          for (const Vertex& vertex : vertices) {
            tessellated.positions.push_back(vertex.pos);
            tessellated.normals.push_back(vertex.nrm);
            tessellated.texcoords.push_back(vertex.tex);
          }
          for (const uint3& triangle : triangles)
            tessellated.indices.push_back({offset + triangle.x, offset + triangle.y, offset + triangle.z});
        }
        source = std::move(tessellated);
      }
      if (type == "loopsubdiv") {
        const float levels = scalar(statement, statement.find("levels") != nullptr ? "levels" : "nlevels", 3.0f);
        if ((levels < 0.0f) || (static_cast<double>(levels) > std::numeric_limits<uint32_t>::max()) || (std::floor(levels) != levels))
          fail(statement, "Invalid Loop subdivision level count.");
        source = subdivide_pbrt_loop(std::move(source), static_cast<uint32_t>(levels));
      }
    } else if ((type == "sphere") || (type == "disk")) {
      const float radius = scalar(statement, "radius", 1.0f);
      const float phi_max = scalar(statement, "phimax", 360.0f) * kPi / 180.0f;
      if ((radius <= 0.0f) || (phi_max <= 0.0f) || (phi_max > kDoublePi))
        fail(statement, "Invalid analytic shape radius or azimuth.");
      constexpr uint32_t segments = 128u;
      const uint32_t rings = type == "sphere" ? 64u : 1u;
      const float z_min = std::clamp(scalar(statement, "zmin", -radius), -radius, radius);
      const float z_max = std::clamp(scalar(statement, "zmax", radius), -radius, radius);
      const float inner = scalar(statement, "innerradius", 0.0f), height = scalar(statement, "height", 0.0f);
      if (((type == "sphere") && (z_min >= z_max)) || ((type == "disk") && ((inner < 0.0f) || (inner >= radius))))
        fail(statement, "Invalid analytic shape bounds.");
      const float theta_min = std::acos(z_max / radius), theta_max = std::acos(z_min / radius);
      for (uint32_t y = 0u; y <= rings; ++y)
        for (uint32_t x = 0u; x <= segments; ++x) {
          const float u = static_cast<float>(x) / segments, v = static_cast<float>(y) / rings;
          const float phi = u * phi_max;
          float3 position, normal;
          if (type == "sphere") {
            const float theta = theta_min + (theta_max - theta_min) * v;
            normal = {std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)};
            position = radius * normal;
          } else {
            const float r = inner + (radius - inner) * v;
            position = {r * std::cos(phi), r * std::sin(phi), height};
            normal = {0.0f, 0.0f, 1.0f};
          }
          source.positions.push_back(position);
          source.normals.push_back(normal);
          source.texcoords.push_back({u, type == "disk" ? 1.0f - v : v});
        }
      for (uint32_t y = 0u; y < rings; ++y)
        for (uint32_t x = 0u; x < segments; ++x) {
          const uint32_t a = y * (segments + 1u) + x, b = a + 1u, c = a + segments + 1u, d = c + 1u;
          source.indices.push_back({a, c, b});
          source.indices.push_back({b, c, d});
        }
    } else
      fail(statement, "Unsupported PBRT shape: " + type);
    discard_degenerate_pbrt_triangles(source);
    if (source.indices.empty())
      return kInvalidIndex;
    const bool normals = source.normals.empty() == false;
    const bool normal_mapped = _data.materials[_state.material].normal_image_index != kInvalidIndex;
    const bool shared_vertices = normals && (source.texcoords.empty() == false) && (normal_mapped == false);
    const size_t vertex_count = shared_vertices ? source.positions.size() : source.indices.size() * 3u;
    if ((vertex_count > (std::numeric_limits<uint32_t>::max() - _data.vertices.pos.size())) ||
        (source.indices.size() > (std::numeric_limits<uint32_t>::max() - _data.triangles.size())))
      fail(statement, "Shape exceeds native geometry index capacity.");
    uint32_t material = _state.material;
    if (_state.area_light.directive.empty() == false) {
      const auto& light = _state.area_light;
      if (light.arguments[0] != "diffuse")
        fail(light, "Unsupported area light type.");
      if (string(light, "filename", "").empty() == false)
        fail(light, "PBRT image area lights are not implemented.");
      material = _data.clone_material(_data.materials[material], "");
      TextureValue emission = radiance(light, "L");
      _data.materials[material].emission = spectral_image(emission);
      _data.materials[material].two_sided = boolean(light, "twosided", false) ? 1u : 0u;
    }
    if (std::isfinite(alpha_scale) == false)
      fail(statement, "PBRT alpha texture scale exceeds the native floating-point range.");
    if ((alpha_image != kInvalidIndex) || (alpha_scale != 1.0f)) {
      if (material == _state.material)
        material = _data.clone_material(_data.materials[material], "");
      _data.materials[material].alpha_mask = {{max(0.0f, alpha_scale), 1.0f, 1.0f, 1.0f}, alpha_image, 4u};
      _has_alpha_masks = _has_alpha_masks || (alpha_image != kInvalidIndex);
    }
    const uint32_t vertex_start = static_cast<uint32_t>(_data.vertices.pos.size());
    const uint32_t triangle_start = static_cast<uint32_t>(_data.triangles.size());
    const bool reverse = _state.reverse != (_world_conversion.col[0].x < 0.0f);
    const auto reserve_vertices = [vertex_count](auto& values) {
      const size_t required = values.size() + vertex_count;
      if (required > values.capacity())
        values.reserve(std::max(required, values.capacity() * 2u));
    };
    reserve_vertices(_data.vertices.pos);
    reserve_vertices(_data.vertices.nrm);
    reserve_vertices(_data.vertices.tan);
    reserve_vertices(_data.vertices.btn);
    reserve_vertices(_data.vertices.tex);
    float3 lower = {kMaxFloat, kMaxFloat, kMaxFloat}, upper = {-kMaxFloat, -kMaxFloat, -kMaxFloat};
    const auto append_vertex = [&](size_t index, const float3& normal, const float2& uv) {
      const auto position = source.positions[index];
      if ((std::isfinite(position.x) == false) || (std::isfinite(position.y) == false) || (std::isfinite(position.z) == false))
        fail(statement, "Non-finite vertex position.");
      lower = min(lower, position);
      upper = max(upper, position);
      _data.vertices.pos.push_back(position);
      _data.vertices.nrm.push_back(normal);
      _data.vertices.tan.push_back({});
      _data.vertices.btn.push_back({});
      _data.vertices.tex.push_back({uv.x, 1.0f - uv.y});
    };
    if (shared_vertices) {
      for (size_t index = 0u; index < source.positions.size(); ++index)
        append_vertex(index, reverse ? -source.normals[index] : source.normals[index], source.texcoords[index]);
    }
    for (auto index : source.indices) {
      if (reverse)
        std::swap(index.y, index.z);
      Triangle triangle;
      triangle.i[0] = vertex_start + index.x;
      triangle.i[1] = vertex_start + index.y;
      triangle.i[2] = vertex_start + index.z;
      triangle.material_index = material;
      if (shared_vertices == false) {
        const float3 geometric = cross(source.positions[index.y] - source.positions[index.x], source.positions[index.z] - source.positions[index.x]);
        if (dot(geometric, geometric) == 0.0f)
          continue;
        const float3 normal = normalize(geometric);
        const uint32_t start = static_cast<uint32_t>(_data.vertices.pos.size());
        const uint32_t corners[] = {index.x, index.y, index.z};
        float2 default_uv[] = {{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};
        if (reverse)
          std::swap(default_uv[1], default_uv[2]);
        for (uint32_t corner = 0u; corner < 3u; ++corner) {
          const uint32_t source_index = corners[corner];
          const float3 vertex_normal =
            (normals && is_valid_vector(source.normals[source_index])) ? (reverse ? -source.normals[source_index] : source.normals[source_index]) : normal;
          const float2 uv = source.texcoords.empty() ? default_uv[corner] : source.texcoords[source_index];
          append_vertex(source_index, vertex_normal, uv);
        }
        triangle.i[0] = start;
        triangle.i[1] = start + 1u;
        triangle.i[2] = start + 2u;
      }
      if (validate_triangle(triangle, _data.vertices.pos) == false)
        continue;
      if (normal_mapped) {
        const float2 uv0 = _data.vertices.tex[triangle.i[0]], uv1 = _data.vertices.tex[triangle.i[1]], uv2 = _data.vertices.tex[triangle.i[2]];
        const float2 duv02 = uv0 - uv2, duv12 = uv1 - uv2;
        const float3 dp02 = _data.vertices.pos[triangle.i[0]] - _data.vertices.pos[triangle.i[2]];
        const float3 dp12 = _data.vertices.pos[triangle.i[1]] - _data.vertices.pos[triangle.i[2]];
        const float determinant = duv02.x * duv12.y - duv02.y * duv12.x;
        float3 tangent = {};
        if (std::abs(determinant) >= 1.0e-9f)
          tangent = (duv12.y * dp02 - duv02.y * dp12) / determinant;
        if ((is_valid_vector(tangent) == false) || (std::abs(determinant) < 1.0e-9f))
          tangent = pbrt_coordinate_tangent((reverse ? 1.0f : -1.0f) * triangle.geo_n);
        // PBRT derives its frame from dp/du and the shading normal, independent of UV handedness.
        const float orientation = _world_conversion.col[0].x < 0.0f ? -1.0f : 1.0f;
        for (const uint32_t vertex : triangle.i) {
          const float3 normal = is_valid_vector(_data.vertices.nrm[vertex]) ? normalize(_data.vertices.nrm[vertex]) : triangle.geo_n;
          _data.vertices.nrm[vertex] = normal;
          _data.vertices.tan[vertex] = tangent;
          _data.vertices.btn[vertex] = cross(normal, tangent) * orientation;
          if (is_valid_vector(_data.vertices.btn[vertex]) == false) {
            _data.vertices.tan[vertex] = pbrt_coordinate_tangent(normal);
            _data.vertices.btn[vertex] = cross(normal, _data.vertices.tan[vertex]) * orientation;
          }
        }
      }
      _data.triangles.push_back(triangle);
    }
    if (_data.triangles.size() == triangle_start) {
      _data.vertices.pos.resize(vertex_start);
      _data.vertices.nrm.resize(vertex_start);
      _data.vertices.tan.resize(vertex_start);
      _data.vertices.btn.resize(vertex_start);
      _data.vertices.tex.resize(vertex_start);
      return kInvalidIndex;
    }
    const std::string name = "PBRT shape " + std::to_string(_data.meshes.size());
    const uint32_t mesh = _data.add_mesh_asset(name.c_str(), triangle_start, static_cast<uint32_t>(_data.triangles.size() - triangle_start), lower, upper);
    _mesh_vertices.emplace(mesh, uint2{vertex_start, static_cast<uint32_t>(_data.vertices.pos.size() - vertex_start)});
    return mesh;
  }

  uint32_t opposite_mesh(uint32_t mesh_index) {
    const auto found = _opposite_meshes.find(mesh_index);
    if (found != _opposite_meshes.end())
      return found->second;
    const Mesh mesh = _data.meshes[mesh_index];
    const auto vertices = _mesh_vertices.at(mesh_index);
    if ((vertices.y > (std::numeric_limits<uint32_t>::max() - _data.vertices.pos.size())) ||
        (mesh.triangle_count > (std::numeric_limits<uint32_t>::max() - _data.triangles.size())))
      throw std::runtime_error("Mirrored instance exceeds native geometry index capacity.");
    const uint32_t vertex_start = static_cast<uint32_t>(_data.vertices.pos.size());
    const uint32_t triangle_start = static_cast<uint32_t>(_data.triangles.size());
    const auto copy = [&](auto& values) {
      values.reserve(values.size() + vertices.y);
      for (uint32_t index = 0u; index < vertices.y; ++index) {
        const auto value = values[vertices.x + index];
        values.push_back(value);
      }
    };
    copy(_data.vertices.pos);
    copy(_data.vertices.nrm);
    copy(_data.vertices.tan);
    copy(_data.vertices.btn);
    copy(_data.vertices.tex);
    for (uint32_t index = 0u; index < vertices.y; ++index) {
      _data.vertices.nrm[vertex_start + index] = -_data.vertices.nrm[vertex_start + index];
      _data.vertices.btn[vertex_start + index] = -_data.vertices.btn[vertex_start + index];
    }
    _data.triangles.reserve(_data.triangles.size() + mesh.triangle_count);
    for (uint32_t index = 0u; index < mesh.triangle_count; ++index) {
      Triangle triangle = _data.triangles[mesh.triangle_offset + index];
      for (uint32_t& vertex : triangle.i)
        vertex = vertex_start + (vertex - vertices.x);
      std::swap(triangle.i[1], triangle.i[2]);
      triangle.geo_n = -triangle.geo_n;
      _data.triangles.push_back(triangle);
    }
    const std::string name = "PBRT orientation " + std::to_string(mesh_index);
    const uint32_t result = _data.add_mesh_asset(name.c_str(), triangle_start, mesh.triangle_count, mesh.bbox_min, mesh.bbox_max);
    _opposite_meshes.emplace(mesh_index, result);
    return result;
  }

  void attach(const ObjectShape& shape) {
    const auto transform = affine_from_matrix(_world_conversion * shape.transform);
    AffineTransform inverse_transform;
    double determinant;
    if (invert_affine(transform, inverse_transform, determinant) == false)
      throw std::runtime_error("Invalid or singular PBRT instance transform.");
    // PBRT transforms normals without the winding flip applied by native mirrored instances.
    const uint32_t mesh = ((determinant < 0.0) != (_world_conversion.col[0].x < 0.0f)) ? opposite_mesh(shape.mesh) : shape.mesh;
    const std::string name = "PBRT instance " + std::to_string(_data.hierarchy.nodes.size());
    const uint32_t node = _data.hierarchy.add_node(name.c_str(), kInvalidIndex, transform);
    if (_data.hierarchy.add_attachment(node, {SceneAttachment::Type::Mesh, mesh, 0u, 0u}) == false)
      throw std::runtime_error("Cannot create native mesh instance.");
  }

  void camera(const PbrtStatement& statement) {
    if (statement.arguments[0] != "perspective")
      fail(statement, "Unsupported PBRT camera: " + statement.arguments[0]);
    _camera_to_world = inverse(_state.transform);
    const auto transform = affine_from_matrix(_camera_to_world);
    AffineTransform inverted;
    double determinant;
    if (invert_affine(transform, inverted, determinant) == false)
      fail(statement, "Singular camera transform.");
    // Convert PBRT's camera handedness into ETX's right-handed camera frame.
    _world_conversion = identity();
    if (determinant > 0.0)
      _world_conversion.col[0].x = -1.0f;
    const auto converted = affine_from_matrix(_world_conversion * _camera_to_world);
    const float3 position = transform_point(converted, {});
    const float3 direction = normalize(transform_vector(converted, {0.0f, 0.0f, 1.0f}));
    const float3 up = normalize(transform_vector(converted, {0.0f, 1.0f, 0.0f}));
    float fov = scalar(statement, "fov", 90.0f);
    if ((fov <= 0.0f) || (fov >= 180.0f))
      fail(statement, "Perspective field of view must be between zero and 180 degrees.");
    const float aspect = static_cast<float>(_resolution.x) / static_cast<float>(_resolution.y);
    if (aspect > 1.0f)
      fov = 2.0f * std::atan(std::tan(fov * kPi / 360.0f) * aspect) * 180.0f / kPi;
    build_camera(_camera, position, direction, up, _resolution, fov);
    _camera.lens_radius = scalar(statement, "lensradius", 0.0f);
    _camera.focal_distance = scalar(statement, "focaldistance", 1e6f);
    if ((_camera.lens_radius < 0.0f) || (_camera.focal_distance <= 0.0f))
      fail(statement, "Camera lens radius must be non-negative and focal distance positive.");
    _data.cameras.push_back({_camera, "PBRT camera", true});
    _coordinates["camera"] = _camera_to_world;
    _has_camera = true;
  }

  void light(const PbrtStatement& statement) {
    EmitterProfile emitter;
    TextureValue emission = radiance(statement, "L");
    emitter.emission = spectral_image(emission);
    const std::string& type = statement.arguments[0];
    if (type == "distant") {
      emitter.cls = EmitterProfile::Class::Directional;
      const auto transform = affine_from_matrix(_world_conversion * _state.transform);
      const float3 from = vector(statement, "from", {}), to = vector(statement, "to", {0.0f, 0.0f, 1.0f});
      const float3 direction = transform_vector(transform, from - to);
      const float length_squared = dot(direction, direction);
      if ((length_squared <= 0.0f) || (std::isfinite(length_squared) == false))
        fail(statement, "Distant light requires a finite non-zero direction.");
      emitter.directional.direction = normalize(direction);
    } else if (type == "infinite") {
      emitter.cls = EmitterProfile::Class::Environment;
      const std::string filename = string(statement, "filename", string(statement, "mapname", ""));
      if (filename.empty()) {
        const float4 white = {1.0f, 1.0f, 1.0f, 1.0f};
        emitter.emission.image_index = _data.add_image(&white, {1u, 1u}, Image::BuildSamplingTable | Image::SkipSRGBConversion, {}, {1.0f, 1.0f});
      } else {
        const auto path = _root / std::filesystem::u8path(filename);
        const uint32_t source = _data.add_image(path_to_utf8(path).c_str(), _version == PbrtVersion::V3 ? Image::RepeatU : Image::Regular);
        _data.images.load_images(_scheduler);
        if (_data.images.loading_succeeded() == false)
          fail(statement, "Cannot load environment image: " + filename);
        const Image& image = _data.images.get(source);
        const uint2 dimensions = {_version == PbrtVersion::V4 ? image.isize.x * 2u : image.isize.x, image.isize.y};
        const auto light_transform = affine_from_matrix(_world_conversion * _state.transform);
        AffineTransform world_to_light;
        double determinant;
        if (invert_affine(light_transform, world_to_light, determinant) == false)
          fail(statement, "Singular environment transform.");
        std::vector<float4> pixels(static_cast<size_t>(dimensions.x) * dimensions.y);
        _scheduler.execute(dimensions.y, [&](uint32_t begin, uint32_t end, uint32_t) {
          for (uint32_t y = begin; y < end; ++y)
            for (uint32_t x = 0u; x < dimensions.x; ++x) {
              const float2 uv = {static_cast<float>(x) / dimensions.x, static_cast<float>(y) / dimensions.y};
              const float3 direction = normalize(transform_vector(world_to_light, uv_to_direction(uv, {}, 1.0f, Projection::Equirectangular)));
              float2 source_uv;
              if (_version == PbrtVersion::V3) {
                float phi = std::atan2(direction.y, direction.x);
                if (phi < 0.0f)
                  phi += kDoublePi;
                source_uv = {phi / kDoublePi, std::acos(std::clamp(direction.z, -1.0f, 1.0f)) / kPi};
              } else {
                const float radius = std::sqrt(std::max(0.0f, 1.0f - std::abs(direction.z)));
                const float phi = std::atan2(std::abs(direction.y), std::abs(direction.x)) / kHalfPi;
                float v = radius * phi, u = radius - v;
                if (direction.z < 0.0f) {
                  std::swap(u, v);
                  u = 1.0f - u;
                  v = 1.0f - v;
                }
                source_uv = {0.5f * (std::copysign(u, direction.x) + 1.0f), 0.5f * (std::copysign(v, direction.y) + 1.0f)};
              }
              source_uv -= float2{0.5f / image.isize.x, 0.5f / image.isize.y};
              pixels[static_cast<size_t>(y) * dimensions.x + x] = image.evaluate(source_uv, nullptr);
            }
        });
        emitter.emission.image_index = _data.add_image(pixels.data(), dimensions, Image::BuildSamplingTable | Image::SkipSRGBConversion | Image::RepeatU, {}, {1.0f, 1.0f});
      }
    } else
      fail(statement, "Unsupported PBRT light: " + type);
    _data.emitter_profiles.push_back(emitter);
    _data.emitter_names.push_back("PBRT light " + std::to_string(_data.emitter_profiles.size()));
  }

  void statement(const PbrtStatement& statement) {
    for (const double value : statement.numbers) {
      if (std::abs(value) > std::numeric_limits<float>::max())
        fail(statement, "Directive exceeds the native floating-point range.");
    }
    for (const auto& parameter : statement.parameters) {
      for (const double value : parameter.numbers) {
        if (std::abs(value) > std::numeric_limits<float>::max())
          fail(statement, "Parameter exceeds the native floating-point range: " + parameter.name);
      }
    }
    const auto& directive = statement.directive;
    if (directive == "Film") {
      _resolution = {integer(statement, "xresolution", 1280u, 1u, 65535u), integer(statement, "yresolution", 720u, 1u, 65535u)};
      _exposure = _version == PbrtVersion::V4 ? scalar(statement, "iso", 100.0f) / 100.0f : scalar(statement, "scale", 1.0f);
      if (_exposure < 0.0f)
        fail(statement, "Film sensitivity cannot be negative.");
    } else if (directive == "Camera") {
      _camera_statement = statement;
      _camera_transform = _state.transform;
      _coordinates["camera"] = inverse(_state.transform);
    } else if (directive == "WorldBegin") {
      if (_world)
        fail(statement, "Duplicate WorldBegin.");
      if (_camera_statement.directive.empty()) {
        _camera_statement.location = statement.location;
        _camera_statement.directive = "Camera";
        _camera_statement.arguments = {"perspective"};
        _camera_transform = _state.transform;
      }
      _state.transform = _camera_transform;
      const float shutter = scalar(_camera_statement, "shutterclose", 1.0f) - scalar(_camera_statement, "shutteropen", 0.0f);
      if ((_version == PbrtVersion::V4) && (shutter < 0.0f))
        fail(_camera_statement, "Camera shutter interval is reversed.");
      if (_version == PbrtVersion::V4)
        _exposure *= shutter;
      camera(_camera_statement);
      _world = true;
      _state.transform = identity();
      _coordinates["world"] = identity();
    } else if (directive == "WorldEnd")
      _world_end = true;
    else if ((directive == "Include") || (directive == "Accelerator") || (directive == "Integrator") || (directive == "Sampler") || (directive == "PixelFilter") ||
             (directive == "Option")) {
    } else if (directive == "Identity")
      _state.transform = identity();
    else if ((directive == "Transform") || (directive == "ConcatTransform")) {
      float4x4 transform = {};
      for (uint32_t index = 0u; index < 16u; ++index)
        transform.val[index] = static_cast<float>(statement.numbers[index]);
      if ((transform.col[0].w != 0.0f) || (transform.col[1].w != 0.0f) || (transform.col[2].w != 0.0f) || (transform.col[3].w != 1.0f))
        fail(statement, "Projective object transforms are not supported.");
      _state.transform = directive == "Transform" ? transform : _state.transform * transform;
    } else if ((directive == "Translate") || (directive == "Scale") || (directive == "Rotate")) {
      float4x4 transform = identity();
      if (directive == "Translate")
        transform.col[3] = {static_cast<float>(statement.numbers[0]), static_cast<float>(statement.numbers[1]), static_cast<float>(statement.numbers[2]), 1.0f};
      else if (directive == "Scale") {
        transform.col[0].x = static_cast<float>(statement.numbers[0]);
        transform.col[1].y = static_cast<float>(statement.numbers[1]);
        transform.col[2].z = static_cast<float>(statement.numbers[2]);
      } else {
        const float3 axis = normalize(float3{static_cast<float>(statement.numbers[1]), static_cast<float>(statement.numbers[2]), static_cast<float>(statement.numbers[3])});
        const float half = static_cast<float>(statement.numbers[0]) * kPi / 360.0f;
        transform = transform_matrix({}, {axis.x * std::sin(half), axis.y * std::sin(half), axis.z * std::sin(half), std::cos(half)}, {1.0f, 1.0f, 1.0f});
      }
      _state.transform = _state.transform * transform;
    } else if (directive == "LookAt") {
      const auto point = [&](size_t begin) {
        return float3{static_cast<float>(statement.numbers[begin]), static_cast<float>(statement.numbers[begin + 1u]), static_cast<float>(statement.numbers[begin + 2u])};
      };
      const float3 position = point(0u), direction = normalize(point(3u) - position), right = normalize(cross(normalize(point(6u)), direction)), up = cross(direction, right);
      const float4x4 camera_world = {
        {{right.x, right.y, right.z, 0.0f}, {up.x, up.y, up.z, 0.0f}, {direction.x, direction.y, direction.z, 0.0f}, {position.x, position.y, position.z, 1.0f}}};
      _state.transform = _state.transform * inverse(camera_world);
    } else if (directive == "AttributeBegin")
      _attributes.push_back(_state);
    else if (directive == "AttributeEnd") {
      if (_attributes.empty())
        fail(statement, "Unmatched AttributeEnd.");
      _state = _attributes.back();
      _attributes.pop_back();
    } else if (directive == "TransformBegin")
      _transforms.push_back(_state.transform);
    else if (directive == "TransformEnd") {
      if (_transforms.empty())
        fail(statement, "Unmatched TransformEnd.");
      _state.transform = _transforms.back();
      _transforms.pop_back();
    } else if (directive == "CoordinateSystem")
      _coordinates[statement.arguments[0]] = _state.transform;
    else if (directive == "CoordSysTransform") {
      const auto found = _coordinates.find(statement.arguments[0]);
      if (found == _coordinates.end())
        fail(statement, "Unknown coordinate system.");
      _state.transform = found->second;
    } else if (directive == "ReverseOrientation")
      _state.reverse = _state.reverse == false;
    else if (directive == "Texture") {
      if (_version == PbrtVersion::V3)
        texture(statement);
    } else if (directive == "MakeNamedMaterial") {
      if (_version == PbrtVersion::V3)
        _materials[statement.arguments[0]] = make_material(statement, string(statement, "type", ""), statement.arguments[0]);
    } else if (directive == "Material")
      _state.material = make_material(statement, statement.arguments[0], "PBRT material " + std::to_string(_data.materials.size()));
    else if (directive == "NamedMaterial") {
      if ((_version == PbrtVersion::V4) && (_materials.contains(statement.arguments[0]) == false)) {
        const auto definition = _material_definitions.find(statement.arguments[0]);
        if (definition != _material_definitions.end())
          _materials[statement.arguments[0]] = make_material(definition->second, string(definition->second, "type", ""), statement.arguments[0]);
      }
      const auto found = _materials.find(statement.arguments[0]);
      if (found == _materials.end())
        fail(statement, "Unknown named material: " + statement.arguments[0]);
      _state.material = found->second;
    } else if (directive == "AreaLightSource") {
      _state.area_light = statement;
      if (statement.arguments[0].empty())
        _state.area_light = {};
    } else if (directive == "LightSource")
      light(statement);
    else if (directive == "Shape") {
      if (_world == false)
        fail(statement, "Shape appears before WorldBegin.");
      const ObjectShape shape = {mesh(statement), _state.transform};
      if (shape.mesh != kInvalidIndex) {
        if (_object.empty())
          attach(shape);
        else
          _objects[_object].push_back(shape);
      }
    } else if (directive == "ObjectBegin") {
      if (_object.empty() == false)
        fail(statement, "Nested ObjectBegin is not allowed.");
      _attributes.push_back(_state);
      _object = statement.arguments[0];
      if (_object.empty())
        fail(statement, "Object name must not be empty.");
      if (_objects.contains(_object))
        fail(statement, "Duplicate object definition: " + _object);
      _objects.emplace(_object, std::vector<ObjectShape>{});
    } else if (directive == "ObjectEnd") {
      if (_object.empty() || _attributes.empty())
        fail(statement, "Unmatched ObjectEnd.");
      _object.clear();
      _state = _attributes.back();
      _attributes.pop_back();
    } else if (directive == "ObjectInstance") {
      if (statement.arguments[0] == _object)
        fail(statement, "An object cannot instantiate itself.");
      const auto found = _objects.find(statement.arguments[0]);
      if (found == _objects.end())
        fail(statement, "Unknown object instance: " + statement.arguments[0]);
      for (const auto& definition : found->second) {
        const ObjectShape instance = {definition.mesh, _state.transform * definition.transform};
        if (_object.empty())
          attach(instance);
        else
          _objects[_object].push_back(instance);
      }
    } else if (directive == "ColorSpace") {
      if (statement.arguments[0] != "srgb")
        fail(statement, "Unsupported RGB color space: " + statement.arguments[0]);
    } else if (directive == "ActiveTransform") {
      if (statement.arguments[0] != "All")
        fail(statement, "Animated transforms are not implemented.");
    } else if (directive == "TransformTimes") {
      if (statement.numbers[0] > statement.numbers[1])
        fail(statement, "Transform time interval is reversed.");
    } else
      fail(statement, "Unsupported PBRT directive: " + directive);
    for (const float value : _state.transform.val) {
      if (std::isfinite(value) == false)
        fail(statement, "Transform exceeds the native floating-point range.");
    }
  }

  uint32_t finish() {
    if ((_world == false) || ((_version == PbrtVersion::V3) && (_world_end == false)) || (_attributes.empty() == false) || (_transforms.empty() == false) ||
        (_object.empty() == false))
      throw std::runtime_error("Incomplete PBRT world or unmatched graphics-state scope.");
    if (_has_alpha_masks || _has_bump_maps) {
      _data.images.load_images(_scheduler);
      if (_data.images.loading_succeeded() == false)
        throw std::runtime_error("Cannot decode a PBRT scalar texture image.");
      for (auto& material : _data.materials) {
        if (material.alpha_mask.image_index != kInvalidIndex) {
          const auto& image = _data.images.get(material.alpha_mask.image_index);
          material.alpha_mask.channel = (image.options & Image::HasAlphaChannel) != 0u ? 3u : 4u;
        }
        if (material.bump.image_index != kInvalidIndex) {
          const auto& image = _data.images.get(material.bump.image_index);
          material.bump.channel = (image.options & Image::HasAlphaChannel) != 0u ? 3u : 4u;
        }
      }
    }
    return SceneLoadSucceeded | (_has_camera ? SceneLoadCameraInfo : 0u);
  }

  const std::filesystem::path _root;
  SceneData& _data;
  TaskScheduler& _scheduler;
  Camera& _camera;
  const PbrtVersion _version;
  float _exposure = 1.0f;
  GraphicsState _state;
  PbrtStatement _camera_statement;
  float4x4 _camera_transform = identity();
  float4x4 _world_conversion = identity(), _camera_to_world = identity();
  uint2 _resolution = {1280u, 720u};
  bool _world = false, _world_end = false, _has_camera = false, _has_alpha_masks = false, _has_bump_maps = false;
  std::vector<GraphicsState> _attributes;
  std::vector<float4x4> _transforms;
  std::unordered_map<std::string, float4x4> _coordinates;
  std::unordered_map<std::string, uint32_t> _materials;
  std::unordered_map<std::string, TextureValue> _textures;
  std::unordered_map<std::string, PbrtStatement> _texture_definitions;
  std::unordered_map<std::string, PbrtStatement> _material_definitions;
  std::unordered_set<std::string> _resolving_textures;
  std::unordered_map<std::string, std::vector<ObjectShape>> _objects;
  std::unordered_map<uint32_t, uint2> _mesh_vertices;
  std::unordered_map<uint32_t, uint32_t> _opposite_meshes;
  std::string _object;
};

}  // namespace

int32_t probe_pbrt_file(const char* source) {
  if (source == nullptr)
    return 0;
  try {
    if (is_pbrt_scene_file(std::filesystem::u8path(source)) == false)
      return 0;
    bool world = false;
    visit_pbrt_file(std::filesystem::u8path(source), false, [&](const auto& statement) {
      world = world || (statement.directive == "WorldBegin");
    });
    return world ? 1 : 0;
  } catch (...) {
    return 0;
  }
}

SceneDependencyInspection inspect_pbrt_dependencies(const std::filesystem::path& file, std::string_view) {
  SceneDependencyInspection result;
  if (is_pbrt_scene_file(file) == false)
    return result;
  const auto root = std::filesystem::absolute(file).parent_path();
  std::unordered_set<std::string> references;
  std::vector<std::filesystem::path> stack;
  const auto add = [&](const std::string& name) {
    const auto path = (root / std::filesystem::u8path(name)).lexically_normal();
    const std::string relative = path_to_utf8(path.lexically_relative(root));
    if (references.insert(relative).second)
      result.references.push_back(relative);
  };
  const auto visit = [&](const auto& self, const std::filesystem::path& path) -> void {
    const auto canonical = std::filesystem::weakly_canonical(path);
    if (std::find(stack.begin(), stack.end(), canonical) != stack.end())
      throw std::runtime_error("Cyclic PBRT include: " + path_to_utf8(path));
    stack.push_back(canonical);
    visit_pbrt_file(path, false, [&](const PbrtStatement& statement) {
      if ((statement.directive == "Include") || (statement.directive == "Import")) {
        const std::string& name = statement.arguments[0];
        add(name);
        const auto included = root / std::filesystem::u8path(name);
        if (std::filesystem::is_regular_file(included))
          self(self, included);
      }
      for (const auto& parameter : statement.parameters) {
        const bool file_parameter =
          ((parameter.name == "filename") && (statement.directive != "Film")) || (parameter.name == "mapname") || (parameter.name == "bsdffile") || (parameter.name == "normalmap");
        for (const auto& value : parameter.strings) {
          if ((parameter.name == "normalmap") && value.empty())
            continue;
          if (file_parameter || ((parameter.type == "spectrum") && (is_pbrt_named_spectrum(value) == false)))
            add(value);
        }
      }
    });
    stack.pop_back();
  };
  try {
    visit(visit, file);
  } catch (const std::exception& error) {
    result.error = error.what();
  }
  return result;
}

uint32_t load_pbrt_file(const char* source, SceneData& data, const IORDatabase&, TaskScheduler& scheduler, Camera& camera, PbrtVersion version) {
  const auto path = std::filesystem::absolute(std::filesystem::u8path(source));
  Loader loader(path.parent_path(), data, scheduler, camera, version);
  if (version == PbrtVersion::V4) {
    visit_pbrt_file(path, true, [&](const PbrtStatement& statement) {
      if (statement.directive == "Texture") {
        if (loader._texture_definitions.emplace(texture_key(statement.arguments[0], statement.arguments[1]), statement).second == false)
          fail(statement, "Texture is redefined: " + statement.arguments[0]);
      } else if (statement.directive == "MakeNamedMaterial") {
        if (loader._material_definitions.emplace(statement.arguments[0], statement).second == false)
          fail(statement, "Named material is redefined: " + statement.arguments[0]);
      }
    });
  }
  visit_pbrt_file(path, true, [&](const auto& statement) {
    loader.statement(statement);
  });
  return loader.finish();
}

void configure_pbrt_scene(const char* source, SceneRepresentation& scene, PbrtVersion version) {
  auto integrator = scene.integrator_data();
  integrator.selected = Integrator::Type::PathTracing;
  scene.data().options.max_path_length = 6u;
  scene.data().options.samples = 16u;
  visit_pbrt_file(std::filesystem::u8path(source), true, [&](const PbrtStatement& statement) {
    if (statement.directive == "Integrator") {
      const std::string& type = statement.arguments[0];
      integrator.selected = (type == "bdpt") ? Integrator::Type::Bidirectional : (type == "sppm") ? Integrator::Type::VCM : Integrator::Type::PathTracing;
      scene.data().options.max_path_length = integer(statement, "maxdepth", 5u, 0u, kMaximumPathLength - 1u) + 1u;
    } else if (statement.directive == "Sampler") {
      scene.data().options.samples = integer(statement, "pixelsamples", 16u, 1u, std::numeric_limits<uint32_t>::max());
    }
  });
  scene.data().options.properties[Scene::Properties::Spectral] = version == PbrtVersion::V4;
  scene.set_integrator_data(integrator);
}

}  // namespace etx
