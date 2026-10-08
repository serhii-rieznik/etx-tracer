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
#include <map>
#include <tuple>
#include <set>

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

float pbrt_roughness_to_alpha(float value, PbrtVersion version) {
  const float roughness = std::max(0.0f, value);
  if (version == PbrtVersion::V4)
    return std::sqrt(roughness);
  const float x = std::log(std::max(roughness, 0.001f));
  return 1.62142f + x * (0.819955f + x * (0.1734f + x * (0.0171201f + x * 0.000640711f)));
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
  uint32_t projection = kInvalidIndex;
  bool constant_float = false;
};

struct GraphicsState {
  float4x4 transform = identity();
  uint32_t material = kInvalidIndex;
  std::string material_name;
  PbrtStatement area_light;
  bool reverse = false;
  std::string inside_medium, outside_medium;
};

bool apply_transform(const PbrtStatement& statement, GraphicsState& state, std::vector<GraphicsState>& attributes, std::vector<float4x4>& transforms,
  std::unordered_map<std::string, float4x4>& coordinates) {
  const auto& directive = statement.directive;
  if (directive == "Identity")
    state.transform = identity();
  else if ((directive == "Transform") || (directive == "ConcatTransform")) {
    float4x4 transform = {};
    for (uint32_t index = 0u; index < 16u; ++index)
      transform.val[index] = static_cast<float>(statement.numbers[index]);
    if ((transform.col[0].w != 0.0f) || (transform.col[1].w != 0.0f) || (transform.col[2].w != 0.0f) || (transform.col[3].w != 1.0f))
      fail(statement, "Projective object transforms are not supported.");
    state.transform = directive == "Transform" ? transform : state.transform * transform;
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
    state.transform = state.transform * transform;
  } else if (directive == "LookAt") {
    const auto point = [&](size_t begin) {
      return float3{static_cast<float>(statement.numbers[begin]), static_cast<float>(statement.numbers[begin + 1u]), static_cast<float>(statement.numbers[begin + 2u])};
    };
    const float3 position = point(0u), direction = normalize(point(3u) - position), right = normalize(cross(normalize(point(6u)), direction)), up = cross(direction, right);
    const float4x4 camera_world = {
      {{right.x, right.y, right.z, 0.0f}, {up.x, up.y, up.z, 0.0f}, {direction.x, direction.y, direction.z, 0.0f}, {position.x, position.y, position.z, 1.0f}}};
    state.transform = state.transform * inverse(camera_world);
  } else if (directive == "AttributeBegin")
    attributes.push_back(state);
  else if (directive == "AttributeEnd") {
    if (attributes.empty())
      fail(statement, "Unmatched AttributeEnd.");
    state = attributes.back();
    attributes.pop_back();
  } else if (directive == "TransformBegin")
    transforms.push_back(state.transform);
  else if (directive == "TransformEnd") {
    if (transforms.empty())
      fail(statement, "Unmatched TransformEnd.");
    state.transform = transforms.back();
    transforms.pop_back();
  } else if (directive == "CoordinateSystem")
    coordinates[statement.arguments[0]] = state.transform;
  else if (directive == "CoordSysTransform") {
    const auto found = coordinates.find(statement.arguments[0]);
    if (found == coordinates.end())
      fail(statement, "Unknown coordinate system.");
    state.transform = found->second;
  } else if (directive == "ReverseOrientation")
    state.reverse = state.reverse == false;
  else
    return false;
  for (const float value : state.transform.val) {
    if (std::isfinite(value) == false)
      fail(statement, "Transform exceeds the native floating-point range.");
  }
  return true;
}

struct MaterialBinding {
  PbrtStatement statement;
  uint32_t material = kInvalidIndex;
};

struct ObjectShape {
  uint32_t mesh;
  float4x4 transform;
  uint32_t flags;
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
    if (_version == PbrtVersion::V4) {
      _inline_materials.push_back({material, _state.material});
      _state.material = 0u;
    }
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
      return {SpectralDistribution::constant(scalar(statement, name, fallback)), kInvalidIndex, kInvalidIndex, true};
    if (parameter->type != "texture")
      fail(statement, std::string("Expected a float or float texture: ") + name);
    return texture_value(statement, name, "float");
  }

  SpectralImage spectral_image(const TextureValue& value) {
    const uint32_t spectrum = _data.add_spectrum(value.spectrum);
    _spectral_projections.emplace(spectrum, value.projection);
    return {spectrum, value.image};
  }

  uint32_t common_projection(const PbrtStatement& statement, std::initializer_list<TextureValue> textures) const {
    uint32_t projection = kInvalidIndex;
    bool assigned = false;
    for (const auto& texture : textures) {
      if (texture.image == kInvalidIndex)
        continue;
      if (assigned && (projection != texture.projection))
        fail(statement, "Material textures require incompatible projections; native meshes store one UV set.");
      projection = texture.projection;
      assigned = true;
    }
    return projection;
  }

  TextureValue projected_spectrum(const SpectralImage& image) const {
    if (image.image_index == kInvalidIndex)
      return {};
    return {{}, image.image_index, _spectral_projections.at(image.spectrum_index)};
  }

  uint32_t planar_projection(const PbrtStatement& statement) {
    // PBRT v3 planar mapping uses world positions directly; v4 applies the texture transform.
    const auto transform = _version == PbrtVersion::V4 ? _texture_transforms.at(texture_key(statement.arguments[0], statement.arguments[1])) : identity();
    AffineTransform inverse_transform;
    double determinant;
    if (invert_affine(affine_from_matrix(transform), inverse_transform, determinant) == false)
      fail(statement, "Invalid or singular planar texture transform.");
    const auto texture_from_world = inverse(transform);
    const float3 u = vector(statement, "v1", {1.0f, 0.0f, 0.0f});
    const float3 v = vector(statement, "v2", {0.0f, 1.0f, 0.0f});
    std::array<float, 8> rows;
    for (uint32_t column = 0u; column < 4u; ++column) {
      const auto& c = texture_from_world.col[column];
      rows[column] = dot(u, float3{c.x, c.y, c.z});
      rows[column + 4u] = dot(v, float3{c.x, c.y, c.z});
    }
    for (const float value : rows) {
      if (std::isfinite(value) == false)
        fail(statement, "Planar projection exceeds the native floating-point range.");
    }
    const auto found = _projection_indices.find(rows);
    if (found != _projection_indices.end())
      return found->second;
    const uint32_t index = static_cast<uint32_t>(_planar_projections.size());
    _planar_projections.push_back(rows);
    _projection_indices.emplace(rows, index);
    return index;
  }

  void load_texture_images(const PbrtStatement& statement) {
    _data.images.load_images(_scheduler);
    if (_data.images.loading_succeeded() == false)
      fail(statement, "Cannot decode a PBRT texture image for baking.");
  }

  float texture_scalar(const TextureValue& value, const float2& uv) const {
    float result = value.spectrum.integrated().x;
    if (value.image != kInvalidIndex) {
      const auto& image = _data.images.get(value.image);
      const float4 pixel = image.evaluate(uv, nullptr);
      result *= (image.options & Image::HasAlphaChannel) != 0u ? pixel.w : (pixel.x + pixel.y + pixel.z) / 3.0f;
    }
    return result;
  }

  float3 texture_rgb(const TextureValue& value, const float2& uv) const {
    float3 result = value.spectrum.integrated();
    if (value.image != kInvalidIndex) {
      const float4 pixel = _data.images.get(value.image).evaluate(uv, nullptr);
      result *= float3{pixel.x, pixel.y, pixel.z};
    }
    return result;
  }

  template <typename Evaluate>
  uint32_t bake_texture(const PbrtStatement& statement, const std::array<uint32_t, 3>& images, const Evaluate& evaluate) {
    load_texture_images(statement);
    const uint32_t anchor = *std::find_if(images.begin(), images.end(), [](uint32_t index) {
      return index != kInvalidIndex;
    });
    const Image& source = _data.images.get(anchor);
    const uint32_t sampling = Image::RepeatU | Image::RepeatV | Image::ReflectU | Image::ReflectV | Image::TextureUVTransform | Image::TexelCenteredUV;
    uint2 dimensions = {source.isize.x, source.isize.y};
    for (uint32_t index : images) {
      if (index == kInvalidIndex)
        continue;
      const Image& other = _data.images.get(index);
      const bool offset_u_matches = (source.offset.x == other.offset.x) || (((source.options & Image::RepeatU) != 0u) && ((source.options & Image::ReflectU) == 0u));
      const bool offset_v_matches = (source.offset.y == other.offset.y) || (((source.options & Image::RepeatV) != 0u) && ((source.options & Image::ReflectV) == 0u));
      if (((source.options & sampling) != (other.options & sampling)) || (offset_u_matches == false) || (offset_v_matches == false) || (source.scale.x != other.scale.x) ||
          (source.scale.y != other.scale.y))
        fail(statement, "Texture baking requires matching UV periods and wrap modes.");
      dimensions.x = std::max(dimensions.x, other.isize.x);
      dimensions.y = std::max(dimensions.y, other.isize.y);
    }
    std::vector<float4> pixels(static_cast<size_t>(dimensions.x) * dimensions.y);
    _scheduler.execute(dimensions.y, [&](uint32_t begin, uint32_t end, uint32_t) {
      for (uint32_t y = begin; y < end; ++y)
        for (uint32_t x = 0u; x < dimensions.x; ++x) {
          const float center = (source.options & Image::TexelCenteredUV) != 0u ? 0.5f : 0.0f;
          float2 uv = {(x + center) / dimensions.x, (y + center) / dimensions.y};
          if ((source.options & Image::TextureUVTransform) != 0u) {
            uv.x = source.scale.x != 0.0f ? (uv.x - source.offset.x) / source.scale.x : 0.0f;
            uv.y = source.scale.y != 0.0f ? (uv.y - source.offset.y) / source.scale.y : 0.0f;
          }
          const float3 value = evaluate(uv);
          pixels[static_cast<size_t>(y) * dimensions.x + x] = {value.x, value.y, value.z, 1.0f};
        }
    });
    for (const float4& pixel : pixels) {
      if ((std::isfinite(pixel.x) == false) || (std::isfinite(pixel.y) == false) || (std::isfinite(pixel.z) == false))
        fail(statement, "Baked texture exceeds the native floating-point range.");
    }
    return _data.add_image(pixels.data(), dimensions, (source.options & sampling) | Image::SkipSRGBConversion, {source.offset.x, source.offset.y},
      {source.scale.x, source.scale.y});
  }

  TextureValue mix_texture(const PbrtStatement& statement, const TextureValue& a, const TextureValue& b, const TextureValue& amount, bool scalar_texture) {
    if ((a.image == kInvalidIndex) && (b.image == kInvalidIndex) && (amount.image == kInvalidIndex)) {
      const float weight = amount.spectrum.integrated().x;
      float2 samples[WavelengthCount];
      for (uint32_t index = 0u; index < WavelengthCount; ++index) {
        const float wavelength = static_cast<float>(ShortestWavelength + index);
        const SpectralQuery query(wavelength, SpectralFlags::Spectral);
        const float value = a.spectrum.query(query).value * (1.0f - weight) + b.spectrum.query(query).value * weight;
        if (std::isfinite(value) == false)
          fail(statement, "Texture mix exceeds the native floating-point range.");
        samples[index] = {wavelength, value};
      }
      auto result = SpectralDistribution::from_samples(samples, WavelengthCount);
      result.integrated_value = a.spectrum.integrated() * (1.0f - weight) + b.spectrum.integrated() * weight;
      return {result, kInvalidIndex};
    }
    const uint32_t projection = common_projection(statement, {a, b, amount});
    const uint32_t image = bake_texture(statement, {a.image, b.image, amount.image}, [&](const float2& uv) {
      const float weight = texture_scalar(amount, uv);
      if (scalar_texture) {
        const float value = texture_scalar(a, uv) * (1.0f - weight) + texture_scalar(b, uv) * weight;
        return float3{value, value, value};
      }
      return texture_rgb(a, uv) * (1.0f - weight) + texture_rgb(b, uv) * weight;
    });
    return {SpectralDistribution::constant(1.0f), image, projection};
  }

  uint32_t named_material(const PbrtStatement& statement, const std::string& name) {
    if ((_version == PbrtVersion::V4) && (_materials.contains(name) == false)) {
      const auto definition = _material_definitions.find(name);
      if (definition != _material_definitions.end()) {
        if (_resolving_materials.insert(name).second == false)
          fail(statement, "Cyclic material definition: " + name);
        const uint32_t material = make_material(definition->second, string(definition->second, "type", ""), name);
        _materials[name] = material;
        _resolving_materials.erase(name);
      }
    }
    const auto found = _materials.find(name);
    if (found == _materials.end())
      fail(statement, "Unknown named material: " + name);
    return found->second;
  }

  uint32_t bound_material(const PbrtStatement& statement) {
    if (_version == PbrtVersion::V3)
      return _state.material;
    if (_state.material_name.empty() == false)
      return named_material(statement, _state.material_name);
    auto& binding = _inline_materials[_state.material];
    if (binding.material == kInvalidIndex)
      binding.material = make_material(binding.statement, binding.statement.arguments[0], "PBRT material " + std::to_string(_data.materials.size()));
    return binding.material;
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

  uint32_t roughness_image(const PbrtStatement& statement, uint32_t source_index, float strength, bool smooth_zero) {
    const auto key = std::make_tuple(source_index, strength, smooth_zero);
    const auto found = _roughness_images.find(key);
    if (found != _roughness_images.end())
      return found->second;
    if (_data.images.get(source_index).format == Image::Format::Undefined) {
      _data.images.load_images(_scheduler);
      if (_data.images.loading_succeeded() == false)
        fail(statement, "Cannot decode a PBRT roughness texture image.");
    }
    const Image& source = _data.images.get(source_index);
    const uint2 dimensions = {source.isize.x, source.isize.y};
    std::vector<float4> pixels(static_cast<size_t>(dimensions.x) * dimensions.y);
    _scheduler.execute(dimensions.y, [&](uint32_t begin, uint32_t end, uint32_t) {
      for (uint32_t y = begin; y < end; ++y)
        for (uint32_t x = 0u; x < dimensions.x; ++x) {
          const float4 pixel = source.pixel(x, y);
          const float value = strength * ((source.options & Image::HasAlphaChannel) != 0u ? pixel.w : (pixel.x + pixel.y + pixel.z) / 3.0f);
          const float alpha = (smooth_zero && (value == 0.0f)) ? 0.0f : pbrt_roughness_to_alpha(value, _version);
          pixels[static_cast<size_t>(y) * dimensions.x + x] = {alpha, alpha, alpha, 1.0f};
        }
    });
    const uint32_t options =
      (source.options & (Image::RepeatU | Image::RepeatV | Image::ReflectU | Image::ReflectV | Image::TextureUVTransform | Image::TexelCenteredUV)) | Image::SkipSRGBConversion;
    const uint32_t image_index = _data.add_image(pixels.data(), dimensions, options, {source.offset.x, source.offset.y}, {source.scale.x, source.scale.y});
    _roughness_images.emplace(key, image_index);
    return image_index;
  }

  uint32_t roughness(const PbrtStatement& statement, const std::string& type, Material& material) {
    const bool conductor = (type == "metal") || (type == "conductor");
    const bool smooth_default = (type == "glass") || (type == "subsurface");
    const float fallback = _version == PbrtVersion::V3 ? (conductor ? 0.01f : (smooth_default ? 0.0f : 0.1f)) : 0.0f;
    const TextureValue u = float_texture(statement, statement.find("uroughness") != nullptr ? "uroughness" : "roughness", fallback);
    const TextureValue v = float_texture(statement, statement.find("vroughness") != nullptr ? "vroughness" : "roughness", fallback);
    const float strength_u = u.spectrum.integrated().x, strength_v = v.spectrum.integrated().x;
    if ((strength_u < 0.0f) || (strength_v < 0.0f))
      fail(statement, "Roughness cannot be negative.");
    if ((u.image != v.image) || (u.projection != v.projection))
      fail(statement, "Independent U/V roughness textures are not supported; use a shared roughness texture or constant U/V values.");
    material.roughness = {{strength_u, strength_v, 0.0f, 0.0f}, u.image, 4u};
    const bool remap = boolean(statement, "remaproughness", true);
    const bool smooth_zero = (type == "glass") || (type == "dielectric") || (type == "subsurface");
    if ((u.image == kInvalidIndex) && (v.image == kInvalidIndex)) {
      float2 alpha = remap ? float2{pbrt_roughness_to_alpha(strength_u, _version), pbrt_roughness_to_alpha(strength_v, _version)} : float2{strength_u, strength_v};
      if (smooth_zero && (strength_u == 0.0f) && (strength_v == 0.0f))
        alpha = {};
      material.roughness.value = {alpha.x, alpha.y, 0.0f, 0.0f};
    } else {
      if (remap) {
        if ((_version == PbrtVersion::V3) && (strength_u != strength_v))
          fail(statement, "PBRT v3 remapped roughness requires matching U/V texture scales.");
        material.roughness.image_index = roughness_image(statement, u.image, _version == PbrtVersion::V3 ? strength_u : 1.0f, smooth_zero && (_version == PbrtVersion::V3));
        const float2 alpha = _version == PbrtVersion::V4 ? float2{std::sqrt(strength_u), std::sqrt(strength_v)} : float2{1.0f, 1.0f};
        material.roughness.value = {alpha.x, alpha.y, 0.0f, 0.0f};
      }
      _has_roughness_maps = true;
    }
    return u.projection;
  }

  SpectralDistribution coefficient(const PbrtStatement& statement, const char* name, float fallback) {
    const TextureValue value = spectrum(statement, name, fallback, false);
    if (value.image != kInvalidIndex)
      fail(statement, std::string("Spatially varying scattering coefficients are not implemented: ") + name);
    return value.spectrum;
  }

  uint32_t make_medium(const PbrtStatement& statement, const std::string& name) {
    const std::string type = string(statement, "type", "");
    if (type != "homogeneous")
      fail(statement, "Unsupported PBRT medium: " + type);
    SpectralDistribution absorption, scattering;
    const std::string preset = string(statement, "preset", "");
    const bool has_preset = (preset.empty() == false) && load_pbrt_medium_preset(preset, absorption, scattering);
    if ((preset.empty() == false) && (has_preset == false))
      log::warning("%s: medium preset %s was not found; using explicit coefficients or defaults", statement.location.describe().c_str(), preset.c_str());
    if (_version == PbrtVersion::V3) {
      if (has_preset == false)
        load_pbrt_medium_preset("Wholemilk", absorption, scattering);
      if (statement.find("sigma_a") != nullptr)
        absorption = coefficient(statement, "sigma_a", 0.0f);
      if (statement.find("sigma_s") != nullptr)
        scattering = coefficient(statement, "sigma_s", 0.0f);
    } else if (has_preset == false) {
      absorption = coefficient(statement, "sigma_a", 1.0f);
      scattering = coefficient(statement, "sigma_s", 1.0f);
    }
    const float scale = scalar(statement, "scale", 1.0f), g = scalar(statement, "g", 0.0f);
    if ((scale < 0.0f) || (std::abs(g) >= 1.0f))
      fail(statement, "Medium scale must be non-negative and anisotropy must lie in (-1, 1).");
    absorption = pbrt_multiply_spectra(absorption, SpectralDistribution::constant(scale));
    scattering = pbrt_multiply_spectra(scattering, SpectralDistribution::constant(scale));
    std::string unique_name = name;
    for (uint32_t suffix = _data.mediums.array_size(); _data.mediums.find(unique_name.c_str()) != kInvalidIndex; ++suffix)
      unique_name = name + "#" + std::to_string(suffix);
    const uint32_t index = _data.add_medium(Medium::Homogeneous, unique_name.c_str(), nullptr, absorption, scattering, g, true);
    if (statement.find("Le") != nullptr) {
      TextureValue emission = spectrum(statement, "Le", 0.0f, true);
      if (emission.image != kInvalidIndex)
        fail(statement, "Homogeneous emission cannot reference a texture.");
      const float scale_le = scalar(statement, "Lescale", 1.0f);
      if (scale_le < 0.0f)
        fail(statement, "Medium emission scale cannot be negative.");
      const auto* parameter = statement.find("Le");
      const bool rgb_illuminant = (parameter->type == "rgb") || (parameter->type == "color");
      const float photometric = rgb_illuminant ? (1.0f / kInvCIEYIntegral) : pbrt_photometric_response(emission.spectrum);
      if (photometric > 0.0f)
        emission.spectrum = pbrt_multiply_spectra(emission.spectrum, SpectralDistribution::constant(_exposure * scale_le / (kInvCIEYIntegral * photometric)));
      _data.spectrum_values[_data.mediums.get(index).emission_index] = pbrt_multiply_spectra(absorption, emission.spectrum);
    }
    return index;
  }

  uint32_t medium(const PbrtStatement& statement, const std::string& name) {
    if (name.empty())
      return kInvalidIndex;
    if ((_version == PbrtVersion::V4) && (_mediums.contains(name) == false)) {
      const auto definition = _medium_definitions.find(name);
      if (definition != _medium_definitions.end())
        _mediums[name] = make_medium(definition->second, name);
    }
    const auto found = _mediums.find(name);
    if (found == _mediums.end())
      fail(statement, "Unknown named medium: " + name);
    return found->second;
  }

  void subsurface(const PbrtStatement& statement, Material& material, const std::string& name) {
    SpectralDistribution absorption, scattering;
    float g = scalar(statement, "g", 0.0f);
    const float scale = scalar(statement, "scale", 1.0f);
    if (scale < 0.0f)
      fail(statement, "Subsurface scale must be non-negative.");
    const std::string preset = string(statement, "name", "");
    if (_version == PbrtVersion::V3) {
      const bool has_preset = (preset.empty() == false) && load_pbrt_medium_preset(preset, absorption, scattering);
      if (has_preset)
        g = 0.0f;
      else {
        load_pbrt_medium_preset("Wholemilk", absorption, scattering);
        if (preset.empty() == false)
          log::warning("%s: subsurface preset %s was not found; using defaults", statement.location.describe().c_str(), preset.c_str());
      }
      if (statement.find("sigma_a") != nullptr)
        absorption = coefficient(statement, "sigma_a", 0.0f);
      if (statement.find("sigma_s") != nullptr)
        scattering = coefficient(statement, "sigma_s", 0.0f);
    } else if (preset.empty() == false) {
      if (load_pbrt_medium_preset(preset, absorption, scattering) == false)
        fail(statement, "Unknown subsurface preset: " + preset);
      g = 0.0f;
    } else if ((statement.find("sigma_a") != nullptr) || (statement.find("sigma_s") != nullptr)) {
      if ((statement.find("sigma_a") == nullptr) || (statement.find("sigma_s") == nullptr))
        fail(statement, "Subsurface sigma_a and sigma_s must be specified together.");
      absorption = coefficient(statement, "sigma_a", 0.0f);
      scattering = coefficient(statement, "sigma_s", 0.0f);
    } else if (statement.find("reflectance") != nullptr) {
      if (std::abs(g) >= 1.0f)
        fail(statement, "Subsurface anisotropy must lie in (-1, 1).");
      const auto reflectance = spectrum(statement, "reflectance", 0.5f, false);
      const auto mfp = spectrum(statement, "mfp", 1.0f, false);
      if ((reflectance.image != kInvalidIndex) || (mfp.image != kInvalidIndex))
        fail(statement, "Spatially varying subsurface reflectance/mfp requires bulk coefficient textures, which ETX does not support.");
      material.cls = MaterialClass::Plastic;
      material.subsurface_cls = SubsurfaceMaterial::RandomWalk;
      material.subsurface_anisotropy = g;
      material.scattering = spectral_image(reflectance);
      material.subsurface = spectral_image({pbrt_multiply_spectra(mfp.spectrum, SpectralDistribution::constant(scale)), kInvalidIndex});
      return;
    } else {
      absorption = SpectralDistribution::rgb_reflectance({0.0011f, 0.0024f, 0.014f});
      scattering = SpectralDistribution::rgb_reflectance({2.55f, 3.21f, 3.77f});
    }
    if (std::abs(g) >= 1.0f)
      fail(statement, "Subsurface anisotropy must lie in (-1, 1).");
    absorption = pbrt_multiply_spectra(absorption, SpectralDistribution::constant(scale));
    scattering = pbrt_multiply_spectra(scattering, SpectralDistribution::constant(scale));
    const std::string medium_name = "PBRT SSS " + name + " " + std::to_string(_data.materials.size());
    material.cls = MaterialClass::Plastic;
    material.subsurface_cls = SubsurfaceMaterial::RandomWalk;
    material.subsurface_anisotropy = g;
    material.int_medium = _data.add_medium(Medium::Homogeneous, medium_name.c_str(), nullptr, absorption, scattering, g, true);
    material.scattering = spectral_image(_version == PbrtVersion::V3 ? spectrum(statement, "Kt", 1.0f, false) : TextureValue{SpectralDistribution::constant(1.0f), kInvalidIndex});
    if (_version == PbrtVersion::V3)
      material.reflectance = spectral_image(spectrum(statement, "Kr", 1.0f, false));
  }

  uint32_t make_material(const PbrtStatement& statement, const std::string& type, const std::string& name) {
    if (type == "coatedconductor") {
      PbrtStatement native = statement;
      for (auto& parameter : native.parameters) {
        if (parameter.name.starts_with("conductor."))
          parameter.name.erase(0u, std::string_view("conductor.").size());
      }
      if (native.find("roughness") == nullptr) {
        if (const auto* roughness = statement.find("interface.roughness")) {
          native.parameters.push_back(*roughness);
          native.parameters.back().name = "roughness";
        }
      }
      log::warning("%s: coatedconductor is imported as a native conductor without its coating", statement.location.describe().c_str());
      return make_material(native, "conductor", name);
    }
    Material material;
    material.ext_ior = {SpectralDistribution::Dielectric, _data.add_spectrum(SpectralDistribution::constant(1.0f)), _data.add_spectrum(SpectralDistribution::constant(0.0f))};
    uint32_t roughness_projection = kInvalidIndex, bump_projection = kInvalidIndex;
    const bool conductor = (type == "metal") || (type == "conductor");
    const TextureValue eta = conductor ? TextureValue{SpectralDistribution::constant(1.5f), kInvalidIndex}
                                       : spectrum(statement, statement.find("eta") != nullptr ? "eta" : "index", type == "subsurface" ? 1.33f : 1.5f, false);
    if (eta.image != kInvalidIndex)
      fail(statement, "Spatially varying refractive indices are not implemented.");
    material.int_ior = {SpectralDistribution::Dielectric, _data.add_spectrum(eta.spectrum), _data.add_spectrum(SpectralDistribution::constant(0.0f))};
    material.reflectance = spectral_image({SpectralDistribution::constant(1.0f), kInvalidIndex});
    material.scattering = spectral_image(spectrum(statement, _version == PbrtVersion::V3 ? "Kd" : "reflectance", 0.5f, false));
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
    } else if ((type == "plastic") || (type == "coateddiffuse") || (type == "substrate") || (type == "mix")) {
      material.cls = MaterialClass::Plastic;
      if (_version == PbrtVersion::V3)
        material.reflectance = spectral_image(spectrum(statement, "Ks", 0.25f, false));
      if (type == "mix") {
        const auto* children = statement.find("materials");
        if ((children == nullptr) || (children->strings.size() != 2u))
          fail(statement, "Mix material requires two named materials.");
        const uint32_t first = named_material(statement, children->strings[0]);
        material.scattering = _data.materials[first].scattering;
        material.roughness = _data.materials[first].roughness;
        roughness_projection = _roughness_projections.at(first);
        log::warning("%s: mix material uses native plastic with the first child's base color and roughness", statement.location.describe().c_str());
      }
    } else if ((type == "dielectric") || (type == "glass")) {
      material.cls = MaterialClass::Dielectric;
      material.scattering = spectral_image(spectrum(statement, "Kt", 1.0f, false));
      material.reflectance = spectral_image(spectrum(statement, "Kr", 1.0f, false));
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
        const auto extinction = [](float value) {
          const float reflectance = std::clamp(value, 0.0f, 0.9999f);
          return 2.0f * std::sqrt(reflectance / (1.0f - reflectance));
        };
        if (reflectance.image != kInvalidIndex) {
          conductor_eta = SpectralDistribution::constant(0.0f);
          conductor_k = SpectralDistribution::constant(1.0f);
          material.reflectance = spectral_image(reflectance);
          log::warning("%s: textured conductor reflectance uses native Fresnel modulation", statement.location.describe().c_str());
        } else {
          float2 samples[WavelengthCount];
          for (uint32_t index = 0u; index < WavelengthCount; ++index) {
            const float wavelength = static_cast<float>(ShortestWavelength + index);
            samples[index] = {wavelength, extinction(reflectance.spectrum.query(SpectralQuery(wavelength, SpectralFlags::Spectral)).value)};
          }
          conductor_eta = SpectralDistribution::constant(1.0f);
          conductor_k = SpectralDistribution::from_samples(samples, WavelengthCount);
          const auto rgb = reflectance.spectrum.integrated();
          conductor_k.integrated_value = {extinction(rgb.x), extinction(rgb.y), extinction(rgb.z)};
        }
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
    } else if (type == "subsurface")
      subsurface(statement, material, name);
    else if ((type.empty()) || (type == "interface") || (type == "none")) {
      material.cls = MaterialClass::Boundary;
    } else
      fail(statement, "Unsupported PBRT material: " + type);
    if ((type != "mix") && ((material.cls == MaterialClass::Plastic) || (material.cls == MaterialClass::Conductor) || (material.cls == MaterialClass::Dielectric)))
      roughness_projection = roughness(statement, type, material);
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
      bump_projection = value.projection;
      material.bump = {{value.spectrum.integrated().x, 0.0f, 0.0f, 0.0f}, value.image, 4u};
      _has_bump_maps = _has_bump_maps || (value.image != kInvalidIndex);
    }
    std::string unique_name = name;
    while (_data.has_material(unique_name.c_str()))
      unique_name += "#" + std::to_string(_data.materials.size());
    const uint32_t index = _data.add_material(unique_name.c_str());
    _data.materials[index] = material;
    _roughness_projections.emplace(index, roughness_projection);
    const uint32_t projection = common_projection(statement,
      {projected_spectrum(material.reflectance), projected_spectrum(material.scattering),
        material.subsurface.image_index != kInvalidIndex ? projected_spectrum(material.subsurface) : TextureValue{}, {{}, material.roughness.image_index, roughness_projection},
        {{}, material.bump.image_index, bump_projection}, {{}, material.normal_image_index, kInvalidIndex}});
    _material_projections.emplace(index, projection);
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
      const std::string mapping = string(statement, "mapping", "uv");
      if ((mapping == "cylindrical") || (mapping == "spherical"))
        log::warning("%s: %s texture mapping uses the mesh UVs", statement.location.describe().c_str(), mapping.c_str());
      else if ((mapping != "uv") && (mapping != "planar"))
        fail(statement, "Unsupported PBRT texture mapping: " + mapping);
      const auto path = _root / std::filesystem::u8path(string(statement, "filename", ""));
      uint32_t options = Image::RepeatU | Image::RepeatV;
      const std::string wrap = string(statement, "wrap", "repeat");
      if (wrap == "clamp")
        options = Image::Regular;
      else if ((wrap == "reflect") || (wrap == "mirror"))
        options = Image::ReflectU | Image::ReflectV;
      else if (wrap != "repeat")
        fail(statement, "Unsupported PBRT texture wrap mode: " + wrap);
      const std::string encoding = string(statement, "encoding", "");
      const bool float_texture = statement.arguments[1] == "float";
      if ((encoding == "linear") || (boolean(statement, "gamma", true) == false) || (float_texture && encoding.empty() && (path.extension() != ".png")))
        options |= Image::SkipSRGBConversion;
      if ((encoding.empty() == false) && (encoding != "sRGB") && (encoding != "linear"))
        fail(statement, "Unsupported image encoding: " + encoding);
      options |= Image::TextureUVTransform | Image::TexelCenteredUV;
      const float2 scale = mapping == "planar" ? float2{1.0f, 1.0f} : float2{scalar(statement, "uscale", 1.0f), scalar(statement, "vscale", 1.0f)};
      const float2 offset = {scalar(statement, "udelta", 0.0f), 1.0f - scale.y - scalar(statement, "vdelta", 0.0f)};
      const float strength = scalar(statement, "scale", 1.0f);
      if ((strength < 0.0f) && (float_texture == false))
        fail(statement, "Image texture scale cannot be negative.");
      value = {SpectralDistribution::constant(strength), _data.add_image(path_to_utf8(path).c_str(), options, offset, scale),
        mapping == "planar" ? planar_projection(statement) : kInvalidIndex};
      const bool invert = boolean(statement, "invert", false);
      if (float_texture || invert) {
        const auto key = std::make_tuple(value.image, strength, invert, float_texture);
        const auto found = _image_texture_bakes.find(key);
        if (found != _image_texture_bakes.end())
          value.image = found->second;
        else {
          load_texture_images(statement);
          const Image& source = _data.images.get(value.image);
          bool use_alpha = float_texture && ((source.options & Image::HasAlphaChannel) != 0u);
          if (float_texture && (use_alpha == false)) {
            for (uint32_t index = 0u, count = source.isize.x * source.isize.y; index < count; ++index) {
              if (source.pixel(index).w != 1.0f) {
                use_alpha = true;
                break;
              }
            }
          }
          const uint32_t image = bake_texture(statement, {value.image, kInvalidIndex, kInvalidIndex}, [&](const float2& uv) {
            const float4 pixel = _data.images.get(value.image).evaluate(uv, nullptr);
            if (float_texture == false) {
              const float3 sampled = strength * float3{pixel.x, pixel.y, pixel.z};
              return max(float3{1.0f, 1.0f, 1.0f} - sampled, float3{});
            }
            const float sampled = strength * (use_alpha ? pixel.w : (pixel.x + pixel.y + pixel.z) / 3.0f);
            const float result = invert ? std::max(0.0f, 1.0f - sampled) : sampled;
            return float3{result, result, result};
          });
          _image_texture_bakes.emplace(key, image);
          value.image = image;
        }
        value.spectrum = SpectralDistribution::constant(1.0f);
      }
    } else if (type == "scale") {
      const bool scalar_texture = statement.arguments[1] == "float";
      const char* source = _version == PbrtVersion::V4 ? "tex" : "tex1";
      const char* multiplier = _version == PbrtVersion::V4 ? "scale" : "tex2";
      value = scalar_texture ? float_texture(statement, source, 1.0f) : spectrum(statement, source, 1.0f, false);
      const TextureValue factor = (scalar_texture || (_version == PbrtVersion::V4)) ? float_texture(statement, multiplier, 1.0f) : spectrum(statement, multiplier, 1.0f, false);
      // PBRT preserves the constant texture type only when unwrapping an identity scale.
      const bool constant_float = (_version == PbrtVersion::V4) && scalar_texture && value.constant_float && factor.constant_float &&
                                  ((value.spectrum.integrated().x == 1.0f) || (factor.spectrum.integrated().x == 1.0f));
      if ((value.image != kInvalidIndex) && (factor.image != kInvalidIndex)) {
        common_projection(statement, {value, factor});
        const TextureValue image_a = {SpectralDistribution::constant(1.0f), value.image};
        const TextureValue image_b = {SpectralDistribution::constant(1.0f), factor.image};
        value.image = bake_texture(statement, {value.image, factor.image, kInvalidIndex}, [&](const float2& uv) {
          if (scalar_texture) {
            const float result = texture_scalar(image_a, uv) * texture_scalar(image_b, uv);
            return float3{result, result, result};
          }
          return texture_rgb(image_a, uv) * texture_rgb(image_b, uv);
        });
      } else if (value.image == kInvalidIndex) {
        value.image = factor.image;
        value.projection = factor.projection;
      }
      value.spectrum = pbrt_multiply_spectra(value.spectrum, factor.spectrum);
      value.constant_float = constant_float;
    } else if (type == "mix") {
      const bool scalar_texture = statement.arguments[1] == "float";
      const TextureValue a = scalar_texture ? float_texture(statement, "tex1", 0.0f) : spectrum(statement, "tex1", 0.0f, false);
      const TextureValue b = scalar_texture ? float_texture(statement, "tex2", 1.0f) : spectrum(statement, "tex2", 1.0f, false);
      value = mix_texture(statement, a, b, float_texture(statement, "amount", 0.5f), scalar_texture);
    } else
      fail(statement, "Unsupported PBRT texture: " + type);
    if (type == "constant")
      value.constant_float = statement.arguments[1] == "float";
    _textures[texture_key(statement.arguments[0], statement.arguments[1])] = value;
  }

  uint32_t mesh(const PbrtStatement& statement, uint32_t& attachment_flags) {
    float alpha_scale = 1.0f;
    uint32_t alpha_image = kInvalidIndex, alpha_projection = kInvalidIndex;
    bool constant_alpha = true;
    if (const auto* alpha = statement.find("alpha")) {
      if (alpha->type == "texture") {
        const auto& value = texture_value(statement, "alpha", "float");
        alpha_scale = value.spectrum.integrated().x;
        alpha_image = value.image;
        alpha_projection = value.projection;
        constant_alpha = value.constant_float;
      } else if (alpha->type == "float")
        alpha_scale = scalar(statement, "alpha", 1.0f);
      else
        fail(statement, "PBRT shape alpha requires a float or float texture.");
    }
    if (statement.find("shadowalpha") != nullptr)
      fail(statement, "PBRT shadow alpha mapping is not implemented.");
    const bool sample_only = (_version == PbrtVersion::V4) && (_state.area_light.directive.empty() == false) && constant_alpha && (alpha_scale == 0.0f);
    attachment_flags = sample_only ? SceneAttachment::SampleOnlyEmitter : 0u;
    if ((_state.area_light.directive.empty() == false) && (sample_only == false) && ((alpha_image != kInvalidIndex) || (alpha_scale != 1.0f)))
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
    uint32_t material = bound_material(statement);
    uint32_t projection = _material_projections.at(material);
    const auto& bound = _data.materials[material];
    const bool has_image = (bound.reflectance.image_index != kInvalidIndex) || (bound.scattering.image_index != kInvalidIndex) || (bound.subsurface.image_index != kInvalidIndex) ||
                           (bound.roughness.image_index != kInvalidIndex) || (bound.bump.image_index != kInvalidIndex) || (bound.normal_image_index != kInvalidIndex);
    uint32_t projection_image = has_image ? 0u : kInvalidIndex;
    projection = common_projection(statement, {{{}, projection_image, projection}, {{}, alpha_image, alpha_projection}});
    if (alpha_image != kInvalidIndex)
      projection_image = 0u;
    const bool normal_mapped = _data.materials[material].normal_image_index != kInvalidIndex;
    const bool shared_vertices = normals && (source.texcoords.empty() == false) && (normal_mapped == false);
    const size_t vertex_count = shared_vertices ? source.positions.size() : source.indices.size() * 3u;
    if ((vertex_count > (std::numeric_limits<uint32_t>::max() - _data.vertices.pos.size())) ||
        (source.indices.size() > (std::numeric_limits<uint32_t>::max() - _data.triangles.size())))
      fail(statement, "Shape exceeds native geometry index capacity.");
    const uint32_t inside = medium(statement, _state.inside_medium), outside = medium(statement, _state.outside_medium);
    const auto& base = _data.materials[material];
    if ((base.subsurface_cls != SubsurfaceMaterial::Disabled) && (inside != kInvalidIndex))
      fail(statement, "A subsurface material cannot also bind an explicit interior medium.");
    const uint32_t internal = base.subsurface_cls != SubsurfaceMaterial::Disabled ? base.int_medium : inside;
    if ((internal != base.int_medium) || (outside != base.ext_medium)) {
      const auto key = std::make_tuple(material, internal, outside);
      const auto found = _medium_materials.find(key);
      if (found != _medium_materials.end())
        material = found->second;
      else {
        material = _data.clone_material(_data.materials[material], "");
        _data.materials[material].int_medium = internal;
        _data.materials[material].ext_medium = outside;
        _medium_materials.emplace(key, material);
      }
    }
    if (_state.area_light.directive.empty() == false) {
      const auto& light = _state.area_light;
      if (light.arguments[0] != "diffuse")
        fail(light, "Unsupported area light type.");
      if (string(light, "filename", "").empty() == false)
        fail(light, "PBRT image area lights are not implemented.");
      material = _data.clone_material(_data.materials[material], "");
      TextureValue emission = radiance(light, "L");
      projection = common_projection(light, {{{}, projection_image, projection}, emission});
      _data.materials[material].emission = spectral_image(emission);
      _data.materials[material].two_sided = boolean(light, "twosided", false) ? 1u : 0u;
    }
    if (std::isfinite(alpha_scale) == false)
      fail(statement, "PBRT alpha texture scale exceeds the native floating-point range.");
    if ((sample_only == false) && ((alpha_image != kInvalidIndex) || (alpha_scale != 1.0f))) {
      if (_state.area_light.directive.empty())
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
    _mesh_projections.emplace(mesh, projection);
    return mesh;
  }

  uint32_t copy_mesh(uint32_t mesh_index, bool reverse) {
    const Mesh mesh = _data.meshes[mesh_index];
    const auto vertices = _mesh_vertices.at(mesh_index);
    if ((vertices.y > (std::numeric_limits<uint32_t>::max() - _data.vertices.pos.size())) ||
        (mesh.triangle_count > (std::numeric_limits<uint32_t>::max() - _data.triangles.size())))
      throw std::runtime_error("PBRT instance exceeds native geometry index capacity.");
    const uint32_t vertex_start = static_cast<uint32_t>(_data.vertices.pos.size());
    const uint32_t triangle_start = static_cast<uint32_t>(_data.triangles.size());
    const auto copy = [&](auto& values) {
      const size_t required = values.size() + vertices.y;
      if (required > values.capacity())
        values.reserve(std::max(required, values.capacity() * 2u));
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
    if (reverse)
      for (uint32_t index = 0u; index < vertices.y; ++index) {
        _data.vertices.nrm[vertex_start + index] = -_data.vertices.nrm[vertex_start + index];
        _data.vertices.btn[vertex_start + index] = -_data.vertices.btn[vertex_start + index];
      }
    const size_t required_triangles = _data.triangles.size() + mesh.triangle_count;
    if (required_triangles > _data.triangles.capacity())
      _data.triangles.reserve(std::max(required_triangles, _data.triangles.capacity() * 2u));
    for (uint32_t index = 0u; index < mesh.triangle_count; ++index) {
      Triangle triangle = _data.triangles[mesh.triangle_offset + index];
      for (uint32_t& vertex : triangle.i)
        vertex = vertex_start + (vertex - vertices.x);
      if (reverse) {
        std::swap(triangle.i[1], triangle.i[2]);
        triangle.geo_n = -triangle.geo_n;
      }
      _data.triangles.push_back(triangle);
    }
    const std::string name = (reverse ? "PBRT orientation " : "PBRT projection ") + std::to_string(mesh_index);
    const uint32_t result = _data.add_mesh_asset(name.c_str(), triangle_start, mesh.triangle_count, mesh.bbox_min, mesh.bbox_max);
    _mesh_vertices.emplace(result, uint2{vertex_start, vertices.y});
    return result;
  }

  uint32_t opposite_mesh(uint32_t mesh_index) {
    const auto found = _opposite_meshes.find(mesh_index);
    if (found != _opposite_meshes.end())
      return found->second;
    const uint32_t result = copy_mesh(mesh_index, true);
    _opposite_meshes.emplace(mesh_index, result);
    return result;
  }

  uint32_t projected_mesh(const ObjectShape& shape) {
    const uint32_t projection = _mesh_projections.at(shape.mesh);
    if (projection == kInvalidIndex)
      return shape.mesh;
    const auto& rows = _planar_projections[projection];
    std::array<float, 8> local_rows;
    for (uint32_t row = 0u; row < 2u; ++row)
      for (uint32_t column = 0u; column < 4u; ++column) {
        const auto& c = shape.transform.col[column];
        const auto offset = row * 4u;
        local_rows[offset + column] = rows[offset] * c.x + rows[offset + 1u] * c.y + rows[offset + 2u] * c.z + rows[offset + 3u] * c.w;
      }
    for (const float value : local_rows) {
      if (std::isfinite(value) == false)
        throw std::runtime_error("Planar instance projection exceeds the native floating-point range.");
    }
    const auto key = std::make_pair(shape.mesh, local_rows);
    const auto found = _projected_meshes.find(key);
    if (found != _projected_meshes.end())
      return found->second;
    const uint32_t mesh = _projected_sources.insert(shape.mesh).second ? shape.mesh : copy_mesh(shape.mesh, false);
    const auto vertices = _mesh_vertices.at(mesh);
    for (uint32_t index = vertices.x; index < (vertices.x + vertices.y); ++index) {
      const auto& p = _data.vertices.pos[index];
      const float u = local_rows[0] * p.x + local_rows[1] * p.y + local_rows[2] * p.z + local_rows[3];
      const float v = local_rows[4] * p.x + local_rows[5] * p.y + local_rows[6] * p.z + local_rows[7];
      if ((std::isfinite(u) == false) || (std::isfinite(v) == false))
        throw std::runtime_error("Planar vertex UV exceeds the native floating-point range.");
      _data.vertices.tex[index] = {u, 1.0f - v};
    }
    _projected_meshes.emplace(key, mesh);
    return mesh;
  }

  void attach(const ObjectShape& shape) {
    const auto transform = affine_from_matrix(_world_conversion * shape.transform);
    AffineTransform inverse_transform;
    double determinant;
    if (invert_affine(transform, inverse_transform, determinant) == false)
      throw std::runtime_error("Invalid or singular PBRT instance transform.");
    // PBRT transforms normals without the winding flip applied by native mirrored instances.
    const uint32_t projected = projected_mesh(shape);
    const uint32_t mesh = ((determinant < 0.0) != (_world_conversion.col[0].x < 0.0f)) ? opposite_mesh(projected) : projected;
    const std::string name = "PBRT instance " + std::to_string(_data.hierarchy.nodes.size());
    const uint32_t node = _data.hierarchy.add_node(name.c_str(), kInvalidIndex, transform);
    if (_data.hierarchy.add_attachment(node, {SceneAttachment::Type::Mesh, mesh, shape.flags, 0u}) == false)
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
    _camera.medium_index = medium(statement, _camera_medium);
    if ((_camera.lens_radius < 0.0f) || (_camera.focal_distance <= 0.0f))
      fail(statement, "Camera lens radius must be non-negative and focal distance positive.");
    _data.cameras.push_back({_camera, "PBRT camera", true});
    _coordinates["camera"] = _camera_to_world;
    _has_camera = true;
  }

  void light(const PbrtStatement& statement) {
    EmitterProfile emitter;
    emitter.medium_index = medium(statement, _state.outside_medium);
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
      if (statement.find("portal") != nullptr) {
        std::array<float, 16> transform;
        std::copy(std::begin(_state.transform.val), std::end(_state.transform.val), transform.begin());
        std::array<float, WavelengthCount + 3u> spectrum;
        for (uint32_t index = 0u; index < WavelengthCount; ++index)
          spectrum[index] = emission.spectrum.query(SpectralQuery(static_cast<float>(ShortestWavelength + index), SpectralFlags::Spectral)).value;
        const float3 rgb = emission.spectrum.integrated();
        spectrum[WavelengthCount] = rgb.x;
        spectrum[WavelengthCount + 1u] = rgb.y;
        spectrum[WavelengthCount + 2u] = rgb.z;
        const auto key = std::make_tuple(filename, emitter.medium_index, transform, spectrum);
        if (_portal_environments.insert(key).second == false) {
          log::warning("%s: duplicate portal environment is omitted", statement.location.describe().c_str());
          return;
        }
        log::warning("%s: portal environment is imported as an ordinary environment light", statement.location.describe().c_str());
      }
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
      _camera_medium = _state.outside_medium;
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
    } else if (apply_transform(statement, _state, _attributes, _transforms, _coordinates)) {
    } else if (directive == "Texture") {
      if (_version == PbrtVersion::V3)
        texture(statement);
    } else if (directive == "MakeNamedMaterial") {
      if (_version == PbrtVersion::V3)
        _materials[statement.arguments[0]] = make_material(statement, string(statement, "type", ""), statement.arguments[0]);
    } else if (directive == "MakeNamedMedium") {
      if (_version == PbrtVersion::V3)
        _mediums[statement.arguments[0]] = make_medium(statement, statement.arguments[0]);
    } else if (directive == "MediumInterface") {
      _state.inside_medium = statement.arguments[0];
      _state.outside_medium = statement.arguments[1];
    } else if (directive == "Material") {
      _state.material_name.clear();
      if (_version == PbrtVersion::V3)
        _state.material = make_material(statement, statement.arguments[0], "PBRT material " + std::to_string(_data.materials.size()));
      else {
        _state.material = static_cast<uint32_t>(_inline_materials.size());
        _inline_materials.push_back({statement, kInvalidIndex});
      }
    } else if (directive == "NamedMaterial") {
      if (statement.arguments[0].empty())
        fail(statement, "Named material name must not be empty.");
      if (_version == PbrtVersion::V3) {
        _state.material = named_material(statement, statement.arguments[0]);
        _state.material_name.clear();
      } else
        _state.material_name = statement.arguments[0];
    } else if (directive == "AreaLightSource") {
      _state.area_light = statement;
      if (statement.arguments[0].empty())
        _state.area_light = {};
    } else if (directive == "LightSource")
      light(statement);
    else if (directive == "Shape") {
      if (_world == false)
        fail(statement, "Shape appears before WorldBegin.");
      ObjectShape shape = {kInvalidIndex, _state.transform, 0u};
      shape.mesh = mesh(statement, shape.flags);
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
        const ObjectShape instance = {definition.mesh, _state.transform * definition.transform, definition.flags};
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
    if (_has_alpha_masks || _has_bump_maps || _has_roughness_maps) {
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
        if (material.roughness.image_index != kInvalidIndex) {
          const auto& image = _data.images.get(material.roughness.image_index);
          material.roughness.channel = (image.options & Image::HasAlphaChannel) != 0u ? 3u : 4u;
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
  bool _world = false, _world_end = false, _has_camera = false, _has_alpha_masks = false, _has_bump_maps = false, _has_roughness_maps = false;
  std::vector<GraphicsState> _attributes;
  std::vector<float4x4> _transforms;
  std::unordered_map<std::string, float4x4> _coordinates;
  std::unordered_map<std::string, uint32_t> _materials;
  std::vector<MaterialBinding> _inline_materials;
  std::unordered_set<std::string> _resolving_materials;
  std::unordered_map<std::string, uint32_t> _mediums;
  std::map<std::tuple<uint32_t, uint32_t, uint32_t>, uint32_t> _medium_materials;
  std::map<std::tuple<uint32_t, float, bool>, uint32_t> _roughness_images;
  std::map<std::tuple<uint32_t, float, bool, bool>, uint32_t> _image_texture_bakes;
  std::set<std::tuple<std::string, uint32_t, std::array<float, 16>, std::array<float, WavelengthCount + 3u>>> _portal_environments;
  std::unordered_map<std::string, TextureValue> _textures;
  std::unordered_map<std::string, PbrtStatement> _texture_definitions;
  std::unordered_map<std::string, float4x4> _texture_transforms;
  std::map<std::array<float, 8>, uint32_t> _projection_indices;
  std::vector<std::array<float, 8>> _planar_projections;
  std::unordered_map<uint32_t, uint32_t> _spectral_projections, _material_projections, _roughness_projections, _mesh_projections;
  std::map<std::pair<uint32_t, std::array<float, 8>>, uint32_t> _projected_meshes;
  std::unordered_set<uint32_t> _projected_sources;
  std::unordered_map<std::string, PbrtStatement> _material_definitions;
  std::unordered_map<std::string, PbrtStatement> _medium_definitions;
  std::unordered_set<std::string> _resolving_textures;
  std::unordered_map<std::string, std::vector<ObjectShape>> _objects;
  std::unordered_map<uint32_t, uint2> _mesh_vertices;
  std::unordered_map<uint32_t, uint32_t> _opposite_meshes;
  std::string _object;
  std::string _camera_medium;
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
    GraphicsState state;
    std::vector<GraphicsState> attributes;
    std::vector<float4x4> transforms;
    std::unordered_map<std::string, float4x4> coordinates;
    visit_pbrt_file(path, true, [&](const PbrtStatement& statement) {
      if (apply_transform(statement, state, attributes, transforms, coordinates))
        return;
      if (statement.directive == "Camera")
        coordinates["camera"] = inverse(state.transform);
      else if (statement.directive == "WorldBegin") {
        state.transform = identity();
        coordinates["world"] = identity();
      } else if (statement.directive == "ObjectBegin")
        attributes.push_back(state);
      else if (statement.directive == "ObjectEnd") {
        if (attributes.empty())
          fail(statement, "Unmatched ObjectEnd.");
        state = attributes.back();
        attributes.pop_back();
      } else if (statement.directive == "Texture") {
        loader._texture_transforms.emplace(texture_key(statement.arguments[0], statement.arguments[1]), state.transform);
        if (loader._texture_definitions.emplace(texture_key(statement.arguments[0], statement.arguments[1]), statement).second == false)
          fail(statement, "Texture is redefined: " + statement.arguments[0]);
      } else if (statement.directive == "MakeNamedMaterial") {
        if (loader._material_definitions.emplace(statement.arguments[0], statement).second == false)
          fail(statement, "Named material is redefined: " + statement.arguments[0]);
      } else if (statement.directive == "MakeNamedMedium") {
        if (loader._medium_definitions.emplace(statement.arguments[0], statement).second == false)
          fail(statement, "Named medium is redefined: " + statement.arguments[0]);
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
