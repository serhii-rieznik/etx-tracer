#pragma once

#include <etx/render/shared/spectrum.hxx>
#include <memory>

namespace etx {
struct SceneData;

struct TemperatureOpticalProfile {
  struct Sample {
    float temperature_kelvin;
    SpectralDistribution eta;
    SpectralDistribution k;
  };
  std::string text;
  std::string title;
  SpectralDistribution::Class cls = SpectralDistribution::Invalid;
  // Conductors store dimensionless k; dielectrics store bulk absorption in m^-1.
  std::vector<Sample> samples;

  bool evaluate(float temperature_kelvin, bool hold_endpoints, SpectralDistribution& eta, SpectralDistribution& k) const;
};

std::shared_ptr<const TemperatureOpticalProfile> parse_temperature_optics(const std::string& text);
std::shared_ptr<const TemperatureOpticalProfile> load_temperature_optics(const char* path, bool& recognized);
bool prepare_temperature_optics(SceneData& data);
}  // namespace etx
