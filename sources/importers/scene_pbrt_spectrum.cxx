#include <etx/std.hxx>
#include "scene_pbrt_spectrum.hxx"
#include <span>

namespace etx {
namespace {

struct NamedSpectrum {
  std::string_view name;
  std::span<const float2> samples;
  bool normalize;
};

#include "scene_pbrt_spectrum_tables.inl.hxx"

const NamedSpectrum* find(std::string_view name) {
  const auto found = std::find_if(std::begin(kNamedSpectra), std::end(kNamedSpectra), [&](const auto& spectrum) {
    return spectrum.name == name;
  });
  return found == std::end(kNamedSpectra) ? nullptr : found;
}

}  // namespace

bool is_pbrt_named_spectrum(std::string_view name) {
  return find(name) != nullptr;
}

SpectralDistribution pbrt_sampled_spectrum(std::span<const float2> samples) {
  float2 values[WavelengthCount];
  size_t segment = 0u;
  for (uint32_t index = 0u; index < WavelengthCount; ++index) {
    const float wavelength = static_cast<float>(ShortestWavelength + index);
    while (((segment + 1u) < samples.size()) && (samples[segment + 1u].x < wavelength))
      ++segment;
    float power = samples.front().y;
    if (wavelength >= samples.back().x)
      power = samples.back().y;
    else if (wavelength > samples.front().x) {
      const auto& a = samples[segment];
      const auto& b = samples[segment + 1u];
      power = std::lerp(a.y, b.y, (wavelength - a.x) / (b.x - a.x));
    }
    values[index] = {wavelength, power};
  }
  return SpectralDistribution::from_samples(values, WavelengthCount);
}

SpectralDistribution pbrt_multiply_spectra(const SpectralDistribution& a, const SpectralDistribution& b) {
  float2 samples[WavelengthCount];
  for (uint32_t index = 0u; index < WavelengthCount; ++index) {
    const float wavelength = static_cast<float>(ShortestWavelength + index);
    const SpectralQuery query(wavelength, SpectralFlags::Spectral);
    const float power = a.query(query).value * b.query(query).value;
    if (std::isfinite(power) == false)
      throw std::runtime_error("Spectrum multiplication exceeds the native floating-point range.");
    samples[index] = {wavelength, power};
  }
  const float3 rgb = a.integrated() * b.integrated();
  if ((std::isfinite(rgb.x) == false) || (std::isfinite(rgb.y) == false) || (std::isfinite(rgb.z) == false))
    throw std::runtime_error("RGB multiplication exceeds the native floating-point range.");
  auto result = SpectralDistribution::from_samples(samples, WavelengthCount);
  result.integrated_value = rgb;
  return result;
}

float pbrt_photometric_response(const SpectralDistribution& spectrum) {
  float result = 0.0f;
  for (uint32_t wavelength = ShortestWavelength; wavelength <= LongestWavelength; ++wavelength)
    result += kPbrtCIEY[wavelength - 360u] * spectrum.query(SpectralQuery(static_cast<float>(wavelength), SpectralFlags::Spectral)).value;
  return result;
}

bool load_pbrt_named_spectrum(std::string_view name, SpectralDistribution& result) {
  const auto* definition = find(name);
  if (definition == nullptr)
    return false;
  result = pbrt_sampled_spectrum(definition->samples);
  if (definition->normalize)
    result.scale((1.0f / kInvCIEYIntegral) / pbrt_photometric_response(result));
  return true;
}

SpectralDistribution pbrt_rgb_illuminant(const float3& rgb) {
  SpectralDistribution daylight;
  load_pbrt_named_spectrum("stdillum-D65", daylight);
  const auto color = SpectralDistribution::rgb_reflectance(rgb);
  float2 samples[WavelengthCount];
  for (uint32_t index = 0u; index < WavelengthCount; ++index) {
    const float wavelength = static_cast<float>(ShortestWavelength + index);
    const SpectralQuery query(wavelength, SpectralFlags::Spectral);
    samples[index] = {wavelength, color.query(query).value * daylight.query(query).value};
  }
  return SpectralDistribution::from_samples(samples, WavelengthCount);
}

}  // namespace etx
