#include <etx/render/host/temperature_optics.hxx>
#include <etx/render/host/scene_data.hxx>
#include <etx/core/log.hxx>
#include <fstream>
#include <sstream>

namespace etx {
namespace {
SpectralDistribution interpolate_optics(const SpectralDistribution& a, const SpectralDistribution& b, float weight) {
  SpectralDistribution result = a;
  result.integrated_value = a.integrated_value * (1.0f - weight) + b.integrated_value * weight;
  for (uint32_t i = 0u; i < result.spectral_entry_count; ++i) {
    result.spectral_entries[i].power = a.spectral_entries[i].power * (1.0f - weight) + b.spectral_entries[i].power * weight;
  }
  return result;
}
}  // namespace

bool TemperatureOpticalProfile::evaluate(float temperature_kelvin, bool hold_endpoints, SpectralDistribution& eta, SpectralDistribution& k) const {
  if ((std::isfinite(temperature_kelvin) == false) || (temperature_kelvin < 0.0f) || samples.empty()) {
    return false;
  }
  // Zero is the application's thermal-off state; use the reference optical table.
  if (temperature_kelvin == 0.0f) {
    temperature_kelvin = samples.front().temperature_kelvin;
  }
  if ((hold_endpoints == false) && ((temperature_kelvin < samples.front().temperature_kelvin) || (temperature_kelvin > samples.back().temperature_kelvin))) {
    return false;
  }
  const auto upper = std::lower_bound(samples.begin(), samples.end(), temperature_kelvin, [](const Sample& sample, float temperature) {
    return sample.temperature_kelvin < temperature;
  });
  if ((upper == samples.begin()) || (upper == samples.end())) {
    const Sample& sample = upper == samples.end() ? samples.back() : *upper;
    eta = sample.eta;
    k = sample.k;
    return true;
  }
  const Sample& lower = *(upper - 1);
  const float weight = (temperature_kelvin - lower.temperature_kelvin) / (upper->temperature_kelvin - lower.temperature_kelvin);
  eta = interpolate_optics(lower.eta, upper->eta, weight);
  k = interpolate_optics(lower.k, upper->k, weight);
  return true;
}

std::shared_ptr<const TemperatureOpticalProfile> parse_temperature_optics(const std::string& text) {
  auto profile = std::make_shared<TemperatureOpticalProfile>();
  profile->text = text;
  std::istringstream input(text);
  std::string line;
  std::vector<float2> eta_points, k_points;
  float temperature = 0.0f;
  bool nanometers = false;
  bool absorption_per_meter = false;
  bool has_extinction = false;
  const auto finish = [&]() {
    if ((temperature <= 0.0f) || (eta_points.size() < 2u) || (eta_points.front().x > kShortestWavelength) || (eta_points.back().x < kLongestWavelength)) {
      return false;
    }
    TemperatureOpticalProfile::Sample sample{temperature, {}, {}};
    // Resample before from_samples so knots outside the renderer's range are not clamped into it.
    std::array<float2, WavelengthCount> eta_grid, k_grid;
    size_t segment = 0u;
    for (uint32_t i = 0u; i < WavelengthCount; ++i) {
      const float wavelength = kShortestWavelength + float(i);
      while (((segment + 2u) < eta_points.size()) && (eta_points[segment + 1u].x < wavelength)) {
        ++segment;
      }
      const float weight = (wavelength - eta_points[segment].x) / (eta_points[segment + 1u].x - eta_points[segment].x);
      eta_grid[i] = {wavelength, eta_points[segment].y * (1.0f - weight) + eta_points[segment + 1u].y * weight};
      k_grid[i] = {wavelength, k_points[segment].y * (1.0f - weight) + k_points[segment + 1u].y * weight};
    }
    sample.eta = SpectralDistribution::from_samples(eta_grid.data(), eta_grid.size());
    sample.k = SpectralDistribution::from_samples(k_grid.data(), k_grid.size());
    sample.eta.update_integrated_value(SpectralDistribution::Integration::Coefficient);
    sample.k.update_integrated_value(SpectralDistribution::Integration::Coefficient);
    if ((valid_value(sample.eta.integrated()) == false) || (valid_value(sample.k.integrated()) == false)) {
      return false;
    }
    profile->samples.push_back(std::move(sample));
    eta_points.clear();
    k_points.clear();
    return true;
  };
  while (std::getline(input, line)) {
    if (line.starts_with("\xef\xbb\xbf")) {
      line.erase(0u, 3u);
    }
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos) {
      continue;
    }
    line.erase(0u, first);
    if (line.starts_with("#temperature:")) {
      if ((temperature > 0.0f) && (finish() == false)) {
        return {};
      }
      std::istringstream value(line.substr(13u));
      std::string trailing;
      if ((bool(value >> temperature) == false) || bool(value >> trailing) || (std::isfinite(temperature) == false) || (temperature <= 0.0f) ||
          ((profile->samples.empty() == false) && (temperature <= profile->samples.back().temperature_kelvin))) {
        return {};
      }
    } else if (line.starts_with("#class:")) {
      std::istringstream value(line.substr(7u));
      std::string name;
      value >> name;
      const auto cls = name == "conductor" ? SpectralDistribution::Conductor : name == "dielectric" ? SpectralDistribution::Dielectric : SpectralDistribution::Invalid;
      if ((cls == SpectralDistribution::Invalid) || ((profile->cls != SpectralDistribution::Invalid) && (profile->cls != cls))) {
        return {};
      }
      profile->cls = cls;
    } else if (line.starts_with("#title:")) {
      profile->title = line.substr(7u);
    } else if (line.starts_with("#wavelength-unit:")) {
      std::istringstream value(line.substr(17u));
      std::string unit;
      value >> unit;
      if (unit != "nm") {
        return {};
      }
      nanometers = true;
    } else if (line.starts_with("#extinction-unit:")) {
      std::istringstream value(line.substr(17u));
      std::string unit;
      value >> unit;
      if (unit != "per-m") {
        return {};
      }
      absorption_per_meter = true;
    } else if (line[0] != '#') {
      float wavelength = 0.0f, eta = 0.0f, k = 0.0f;
      std::string trailing;
      std::istringstream row(line);
      if ((bool(row >> wavelength >> eta >> k) == false) || bool(row >> trailing) || (temperature <= 0.0f) || (std::isfinite(wavelength) == false) ||
          (std::isfinite(eta) == false) || (std::isfinite(k) == false) || (wavelength <= 0.0f) || (eta < 0.0f) || (k < 0.0f) ||
          ((eta_points.empty() == false) && (wavelength <= eta_points.back().x))) {
        return {};
      }
      eta_points.push_back({wavelength, eta});
      k_points.push_back({wavelength, k});
      has_extinction |= k > 0.0f;
    }
  }
  if ((nanometers == false) || ((profile->cls != SpectralDistribution::Conductor) && (profile->cls != SpectralDistribution::Dielectric)) ||
      ((profile->cls == SpectralDistribution::Conductor) && absorption_per_meter) ||
      ((profile->cls == SpectralDistribution::Dielectric) && has_extinction && (absorption_per_meter == false)) || (finish() == false)) {
    return {};
  }
  for (const auto& sample : profile->samples) {
    for (uint32_t i = 0u; i < sample.eta.spectral_entry_count; ++i) {
      if (((profile->cls == SpectralDistribution::Dielectric) && (sample.eta.spectral_entries[i].power <= 0.0f)) ||
          ((sample.eta.spectral_entries[i].power == 0.0f) && (sample.k.spectral_entries[i].power == 0.0f))) {
        return {};
      }
    }
  }
  return profile;
}

std::shared_ptr<const TemperatureOpticalProfile> load_temperature_optics(const char* path, bool& recognized) {
  recognized = false;
  std::ifstream input(std::filesystem::u8path(path), std::ios::binary);
  if (input.is_open() == false) {
    return {};
  }
  const std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  std::istringstream lines(text);
  std::string line;
  while (std::getline(lines, line)) {
    if (line.starts_with("\xef\xbb\xbf")) {
      line.erase(0u, 3u);
    }
    const auto first = line.find_first_not_of(" \t\r");
    if ((first != std::string::npos) && (line.compare(first, 13u, "#temperature:") == 0)) {
      recognized = true;
      break;
    }
  }
  return recognized ? parse_temperature_optics(text) : nullptr;
}

bool prepare_temperature_optics(SceneData& data) {
  std::unordered_map<uint32_t, float> temperatures;
  for (uint32_t material_index = 0u; material_index < data.materials.size(); ++material_index) {
    Material& material = data.materials[material_index];
    const auto eta_author = data.spectrum_sources.find(material.int_ior.eta_index);
    const auto k_author = data.spectrum_sources.find(material.int_ior.k_index);
    const bool eta_profile = (eta_author != data.spectrum_sources.end()) && (eta_author->second.temperature_profile != nullptr);
    const bool k_profile = (k_author != data.spectrum_sources.end()) && (k_author->second.temperature_profile != nullptr);
    if (eta_profile != k_profile) {
      log::error("Material %u requires paired temperature optical profiles for internal n and extinction", material_index);
      return false;
    }
    if (eta_profile) {
      const SpectrumSource& eta_source = eta_author->second;
      const SpectrumSource& k_source = k_author->second;
      if ((eta_source.temperature_profile_component != 0u) || (k_source.temperature_profile_component != 1u) ||
          (eta_source.temperature_profile_enabled != k_source.temperature_profile_enabled) ||
          (eta_source.temperature_profile_hold_endpoints != k_source.temperature_profile_hold_endpoints) ||
          ((eta_source.temperature_profile != k_source.temperature_profile) && (eta_source.temperature_profile->text != k_source.temperature_profile->text))) {
        log::error("Material %u has inconsistent internal temperature optical profiles", material_index);
        return false;
      }
    }
    for (uint32_t* slot : {&material.int_ior.eta_index, &material.int_ior.k_index}) {
      auto found = data.spectrum_sources.find(*slot);
      if ((found == data.spectrum_sources.end()) || (found->second.temperature_profile == nullptr)) {
        continue;
      }
      const float temperature = found->second.temperature_profile_enabled ? material.temperature_kelvin : 0.0f;
      if (const auto previous = temperatures.find(*slot); (previous != temperatures.end()) && (previous->second != temperature)) {
        *slot = data.copy_spectrum(*slot);
        found = data.spectrum_sources.find(*slot);
      }
      temperatures[*slot] = temperature;
      SpectrumSource& source = found->second;
      SpectralDistribution eta, k;
      if (source.temperature_profile->evaluate(temperature, source.temperature_profile_hold_endpoints, eta, k) == false) {
        log::error("Material %u temperature %.1f K is outside optical profile range %.1f..%.1f K; explicitly enable endpoint optics to render outside it", material_index,
          temperature, source.temperature_profile->samples.front().temperature_kelvin, source.temperature_profile->samples.back().temperature_kelvin);
        return false;
      }
      source.base = source.temperature_profile_component == 0u                            ? eta
                    : source.temperature_profile->cls == SpectralDistribution::Dielectric ? SpectralDistribution::constant(0.0f)
                                                                                          : k;
      const SpectralDistribution prepared = source.output();
      if (prepared.empty() || (prepared.valid() == false) || (valid_value(prepared.integrated()) == false)) {
        log::error("Material %u temperature optics exceed the supported numeric range", material_index);
        return false;
      }
      data.spectrum_values[*slot] = prepared;
      material.int_ior.cls = source.temperature_profile->cls;
    }
    const auto extinction = data.spectrum_sources.find(material.int_ior.k_index);
    if ((extinction != data.spectrum_sources.end()) && (extinction->second.temperature_profile != nullptr)) {
      const auto& profile = *extinction->second.temperature_profile;
      const SpectralDistribution& eta = data.spectrum_values[material.int_ior.eta_index];
      const SpectralDistribution& k = data.spectrum_values[material.int_ior.k_index];
      for (uint32_t i = 0u; i < eta.spectral_entry_count; ++i) {
        if (((profile.cls == SpectralDistribution::Dielectric) && (eta.spectral_entries[i].power <= 0.0f)) ||
            ((eta.spectral_entries[i].power == 0.0f) && (k.spectral_entries[i].power == 0.0f))) {
          log::error("Material %u has a singular prepared refractive index", material_index);
          return false;
        }
      }
      if ((profile.cls == SpectralDistribution::Conductor) && (material.cls != MaterialClass::Conductor)) {
        log::error("Material %u requires a conductor BSDF for a conductor optical profile", material_index);
        return false;
      }
      if ((profile.cls == SpectralDistribution::Dielectric) && (material.cls != MaterialClass::Dielectric) &&
          std::any_of(profile.samples.begin(), profile.samples.end(), [](const auto& sample) {
            return sample.k.is_zero() == false;
          })) {
        log::error("Material %u requires a dielectric BSDF for an optical profile with bulk absorption", material_index);
        return false;
      }
    }
    if ((material.cls == MaterialClass::Dielectric) && (extinction != data.spectrum_sources.end()) && (extinction->second.temperature_profile != nullptr) &&
        (extinction->second.temperature_profile->cls == SpectralDistribution::Dielectric) && (material.int_medium == kInvalidIndex)) {
      const uint32_t zero = data.add_spectrum(SpectralDistribution::constant(0.0f));
      material.int_medium = data.mediums.add(Medium::Homogeneous, "Optical profile medium " + std::to_string(material_index), "", zero, zero, 0.0f, true);
    }
  }
  return true;
}
}  // namespace etx
