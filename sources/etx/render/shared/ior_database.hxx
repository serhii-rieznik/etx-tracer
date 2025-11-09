#pragma once

#include <etx/core/environment.hxx>
#include <etx/render/shared/spectrum.hxx>

#include <filesystem>
#include <vector>
#include <array>
#include <string>
#include <unordered_map>
#include <cctype>
#include <algorithm>
#include <system_error>
#include <cmath>

namespace etx {

struct IORDefinition {
  std::string filename;
  std::string name;
  std::string title;
  SpectralDistribution::Class cls = SpectralDistribution::Class::Invalid;
  SpectralDistribution eta;
  SpectralDistribution k;
  float3 eta_integrated = {};
  float3 k_integrated = {};
};

inline std::string make_ior_title(const std::string& base_name) {
  if (base_name.empty())
    return {};

  std::string result = base_name;
  result[0] = char(std::toupper(static_cast<unsigned char>(result[0])));
  for (size_t i = 1; i < result.size(); ++i) {
    if ((result[i] == '_') || (result[i] == '-')) {
      result[i] = ' ';
      if (i + 1 < result.size()) {
        result[i + 1] = char(std::toupper(static_cast<unsigned char>(result[i + 1])));
      }
    }
  }

  return result;
}

struct IORDatabase {
  std::vector<IORDefinition> definitions;

  const std::vector<size_t>& class_entries(SpectralDistribution::Class cls) const {
    static const std::vector<size_t> kEmpty;
    size_t index = class_to_index(cls);
    if (index >= _class_indices.size())
      return kEmpty;
    return _class_indices[index];
  }

  const IORDefinition* find_by_name(const char* name, SpectralDistribution::Class cls = SpectralDistribution::Class::Invalid) const {
    if ((name == nullptr) || (name[0] == 0))
      return nullptr;

    std::string key = normalize_key(std::filesystem::path(name).stem().string());
    if (key.empty())
      key = normalize_key(name);

    auto range = _definitions_by_name.equal_range(key);
    for (auto it = range.first; it != range.second; ++it) {
      const IORDefinition& def = definitions[it->second];
      if ((cls == SpectralDistribution::Class::Invalid) || (def.cls == cls))
        return &def;
    }
    return nullptr;
  }

  void load(const char* folder) {
    definitions.clear();
    _definitions_by_name.clear();
    for (auto& v : _class_indices) {
      v.clear();
    }
    if ((folder == nullptr) || (folder[0] == 0))
      return;

    std::filesystem::path root(folder);
    std::error_code ec;
    if (std::filesystem::exists(root, ec) == false)
      return;

    std::filesystem::recursive_directory_iterator iterator(root, std::filesystem::directory_options::skip_permission_denied, ec);
    if (ec)
      return;

    const std::filesystem::recursive_directory_iterator end;
    while (iterator != end) {
      const std::filesystem::path current_path = iterator->path();
      bool is_file = std::filesystem::is_regular_file(current_path, ec);
      if (ec) {
        ec.clear();
      } else if (is_file && current_path.extension() == ".spd") {
        SpectralDistribution eta = {};
        SpectralDistribution k = {};
        std::string file_title;
        auto cls = RefractiveIndex::load_from_file(current_path.string().c_str(), eta, k, &file_title);
        if (cls != SpectralDistribution::Class::Invalid) {
          IORDefinition def = {};
          def.filename = current_path.string();
          def.name = current_path.stem().string();
          def.title = file_title.empty() ? make_ior_title(def.name) : file_title;
          def.cls = cls;
          def.eta = eta;
          def.k = k;
          def.eta_integrated = eta.integrated();
          def.k_integrated = k.integrated();
          definitions.emplace_back(def);
        }
      }

      iterator.increment(ec);
      if (ec) {
        ec.clear();
      }
    }

    std::sort(definitions.begin(), definitions.end(), [](const IORDefinition& a, const IORDefinition& b) {
      if (a.cls != b.cls)
        return a.cls < b.cls;
      return a.title < b.title;
    });

    rebuild_lookup();
  }

  int find_matching_index(const SpectralDistribution& eta, const SpectralDistribution& k, SpectralDistribution::Class cls) const {
    constexpr float kMatchTolerance = 1.0e-4f;
    if (cls == SpectralDistribution::Class::Invalid)
      return -1;

    const float3 eta_target = eta.integrated();
    const float3 k_target = k.integrated();
    for (int index = 0; index < static_cast<int>(definitions.size()); ++index) {
      const IORDefinition& def = definitions[static_cast<size_t>(index)];
      if (def.cls != cls)
        continue;
      float diff = fabsf(eta_target.x - def.eta_integrated.x) + fabsf(eta_target.y - def.eta_integrated.y) + fabsf(eta_target.z - def.eta_integrated.z);
      if (cls == SpectralDistribution::Class::Conductor) {
        diff += fabsf(k_target.x - def.k_integrated.x) + fabsf(k_target.y - def.k_integrated.y) + fabsf(k_target.z - def.k_integrated.z);
      }
      if (diff < kMatchTolerance)
        return index;
    }

    return -1;
  }

 private:
  static constexpr size_t kClassCount = static_cast<size_t>(SpectralDistribution::Class::Illuminant) + 1u;

  static size_t class_to_index(SpectralDistribution::Class cls) {
    return static_cast<size_t>(cls);
  }

  static std::string normalize_key(std::string name) {
    if (name.empty())
      return name;
    for (char& c : name) {
      c = char(std::tolower(static_cast<unsigned char>(c)));
    }
    return name;
  }

  void rebuild_lookup() {
    _definitions_by_name.clear();
    for (auto& v : _class_indices) {
      v.clear();
    }

    for (size_t i = 0; i < definitions.size(); ++i) {
      const IORDefinition& def = definitions[i];
      size_t idx = class_to_index(def.cls);
      if (idx < _class_indices.size()) {
        _class_indices[idx].push_back(i);
      }
      _definitions_by_name.emplace(normalize_key(def.name), i);
    }
  }

  std::array<std::vector<size_t>, kClassCount> _class_indices = {};
  std::unordered_multimap<std::string, size_t> _definitions_by_name;
};

}  // namespace etx
