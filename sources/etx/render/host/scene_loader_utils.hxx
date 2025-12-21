#pragma once

#include <etx/core/environment.hxx>
#include <etx/render/shared/math.hxx>
#include <etx/render/shared/scene.hxx>
#include <etx/render/shared/spectrum.hxx>
#include <etx/render/shared/ior_database.hxx>

#include <vector>
#include <map>
#include <string>
#include <filesystem>

namespace etx {

struct MaterialDefinition {
  std::string name;
  std::map<std::string, std::string> properties;
};

enum : uint32_t {
  SceneLoadFailed = 0u,
  SceneLoadSucceeded = 1u << 0u,
  SceneLoadCameraInfo = 1u << 1u,
};

inline float2 make_float2(const float values[]) {
  return {values[0], values[1]};
}

inline float3 make_float3(const float values[]) {
  return {values[0], values[1], values[2]};
}

inline bool validate_triangle(Triangle& tri, const std::vector<float3>& vertices) {
  if (tri.i[0] >= vertices.size() || tri.i[1] >= vertices.size() || tri.i[2] >= vertices.size()) {
    return false;
  }
  tri.geo_n = cross(vertices[tri.i[1]] - vertices[tri.i[0]], vertices[tri.i[2]] - vertices[tri.i[0]]);
  float l = length(tri.geo_n);
  if (l == 0.0f) {
    return false;
  }
  tri.geo_n /= l;
  return true;
}

inline std::filesystem::path locate_spectrum_file(const char* identifier, std::initializer_list<const char*> fallback_folders) {
  if ((identifier == nullptr) || (identifier[0] == 0))
    return {};

  std::filesystem::path requested(identifier);
  if (requested.has_extension() == false)
    requested.replace_extension(".spd");

  std::error_code ec;
  if (requested.is_absolute()) {
    if (std::filesystem::exists(requested, ec))
      return requested;
    return {};
  }

  std::filesystem::path data_root = std::filesystem::path(env().data_folder()) / "spectrum";

  std::filesystem::path combined = data_root / requested;
  if (std::filesystem::exists(combined, ec))
    return combined;

  for (const char* folder : fallback_folders) {
    std::filesystem::path candidate = data_root / folder / requested.filename();
    if (std::filesystem::exists(candidate, ec))
      return candidate;
  }

  return {};
}

inline void trim_whitespace(std::string& str) {
  if (str.empty()) {
    return;
  }

  const size_t len = str.size();
  size_t start = 0;
  while ((start < len) && std::isspace(str[start])) {
    ++start;
  }

  size_t end = len;
  while ((end > start) && std::isspace(str[end - 1])) {
    --end;
  }

  if (start == 0 && end == len) {
    return;  // No trimming needed
  }

  str = str.substr(start, end - start);
}

inline bool load_ior_from_identifier(const char* identifier, const IORDatabase& ior_database, SpectralDistribution& eta, SpectralDistribution& k,
  SpectralDistribution::Class& cls) {
  if ((identifier == nullptr) || (identifier[0] == 0))
    return false;

  if (const IORDefinition* def = ior_database.find_by_name(identifier)) {
    cls = def->cls;
    eta = def->eta;
    k = def->k;
    return true;
  }

  std::filesystem::path candidate = locate_spectrum_file(identifier, {"conductor", "dielectric"});
  if (candidate.empty())
    return false;

  cls = RefractiveIndex::load_from_file(candidate.string().c_str(), eta, k);
  return cls != SpectralDistribution::Class::Invalid;
}

}  // namespace etx
