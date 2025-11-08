#pragma once

#include <etx/core/environment.hxx>
#include <etx/render/shared/spectrum.hxx>

#include <filesystem>
#include <vector>
#include <string>
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

  void load(const char* folder) {
    definitions.clear();
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
        auto cls = RefractiveIndex::load_from_file(current_path.string().c_str(), eta, k);
        if (cls != SpectralDistribution::Class::Invalid) {
          IORDefinition def = {};
          def.filename = current_path.string();
          def.name = current_path.stem().string();
          def.title = make_ior_title(def.name);
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
};

}  // namespace etx
