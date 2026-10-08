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

bool load_pbrt_medium_preset(std::string_view name, SpectralDistribution& absorption, SpectralDistribution& scattering) {
  struct Preset {
    std::string_view name;
    float3 scattering, absorption;
  };
  // PBRT's measured subsurface coefficients (mm^-1), media.cpp, Apache-2.0.
  static constexpr Preset presets[] = {
    {"Apple", {2.29f, 2.39f, 1.97f}, {0.0030f, 0.0034f, 0.046f}},
    {"Chicken1", {0.15f, 0.21f, 0.38f}, {0.015f, 0.077f, 0.19f}},
    {"Chicken2", {0.19f, 0.25f, 0.32f}, {0.018f, 0.088f, 0.20f}},
    {"Cream", {7.38f, 5.47f, 3.15f}, {0.0002f, 0.0028f, 0.0163f}},
    {"Ketchup", {0.18f, 0.07f, 0.03f}, {0.061f, 0.97f, 1.45f}},
    {"Marble", {2.19f, 2.62f, 3.00f}, {0.0021f, 0.0041f, 0.0071f}},
    {"Potato", {0.68f, 0.70f, 0.55f}, {0.0024f, 0.0090f, 0.12f}},
    {"Skimmilk", {0.70f, 1.22f, 1.90f}, {0.0014f, 0.0025f, 0.0142f}},
    {"Skin1", {0.74f, 0.88f, 1.01f}, {0.032f, 0.17f, 0.48f}},
    {"Skin2", {1.09f, 1.59f, 1.79f}, {0.013f, 0.070f, 0.145f}},
    {"Spectralon", {11.6f, 20.4f, 14.9f}, {0.00f, 0.00f, 0.00f}},
    {"Wholemilk", {2.55f, 3.21f, 3.77f}, {0.0011f, 0.0024f, 0.014f}},
    {"Lowfat Milk", {0.89187f, 1.5136f, 2.532f}, {0.002875f, 0.00575f, 0.0115f}},
    {"Reduced Milk", {2.4858f, 3.1669f, 4.5214f}, {0.0025556f, 0.0051111f, 0.012778f}},
    {"Regular Milk", {4.5513f, 5.8294f, 7.136f}, {0.0015333f, 0.0046f, 0.019933f}},
    {"Espresso", {0.72378f, 0.84557f, 1.0247f}, {4.7984f, 6.5751f, 8.8493f}},
    {"Mint Mocha Coffee", {0.31602f, 0.38538f, 0.48131f}, {3.772f, 5.8228f, 7.82f}},
    {"Lowfat Soy Milk", {0.30576f, 0.34233f, 0.61664f}, {0.0014375f, 0.0071875f, 0.035937f}},
    {"Regular Soy Milk", {0.59223f, 0.73866f, 1.4693f}, {0.0019167f, 0.0095833f, 0.065167f}},
    {"Lowfat Chocolate Milk", {0.64925f, 0.83916f, 1.1057f}, {0.0115f, 0.0368f, 0.1564f}},
    {"Regular Chocolate Milk", {1.4585f, 2.1289f, 2.9527f}, {0.010063f, 0.043125f, 0.14375f}},
    {"Coke", {8.9053e-05f, 8.372e-05f, 0.0f}, {0.10014f, 0.16503f, 0.2468f}},
    {"Pepsi", {6.1697e-05f, 4.2564e-05f, 0.0f}, {0.091641f, 0.14158f, 0.20729f}},
    {"Sprite", {6.0306e-06f, 6.4139e-06f, 6.5504e-06f}, {0.001886f, 0.0018308f, 0.0020025f}},
    {"Gatorade", {0.0024574f, 0.003007f, 0.0037325f}, {0.024794f, 0.019289f, 0.008878f}},
    {"Chardonnay", {1.7982e-05f, 1.3758e-05f, 1.2023e-05f}, {0.010782f, 0.011855f, 0.023997f}},
    {"White Zinfandel", {1.7501e-05f, 1.9069e-05f, 1.288e-05f}, {0.012072f, 0.016184f, 0.019843f}},
    {"Merlot", {2.1129e-05f, 0.0f, 0.0f}, {0.11632f, 0.25191f, 0.29434f}},
    {"Budweiser Beer", {2.4356e-05f, 2.4079e-05f, 1.0564e-05f}, {0.011492f, 0.024911f, 0.057786f}},
    {"Coors Light Beer", {5.0922e-05f, 4.301e-05f, 0.0f}, {0.006164f, 0.013984f, 0.034983f}},
    {"Clorox", {0.0024035f, 0.0031373f, 0.003991f}, {0.0033542f, 0.014892f, 0.026297f}},
    {"Apple Juice", {0.00013612f, 0.00015836f, 0.000227f}, {0.012957f, 0.023741f, 0.052184f}},
    {"Cranberry Juice", {0.00010402f, 0.00011646f, 7.8139e-05f}, {0.039437f, 0.094223f, 0.12426f}},
    {"Grape Juice", {5.382e-05f, 0.0f, 0.0f}, {0.10404f, 0.23958f, 0.29325f}},
    {"Ruby Grapefruit Juice", {0.011002f, 0.010927f, 0.011036f}, {0.085867f, 0.18314f, 0.25262f}},
    {"White Grapefruit Juice", {0.22826f, 0.23998f, 0.32748f}, {0.0138f, 0.018831f, 0.056781f}},
    {"Shampoo", {0.0007176f, 0.0008303f, 0.0009016f}, {0.014107f, 0.045693f, 0.061717f}},
    {"Strawberry Shampoo", {0.00015671f, 0.00015947f, 1.518e-05f}, {0.01449f, 0.05796f, 0.075823f}},
    {"Head & Shoulders Shampoo", {0.023805f, 0.028804f, 0.034306f}, {0.084621f, 0.15688f, 0.20365f}},
    {"Lemon Tea Powder", {0.040224f, 0.045264f, 0.051081f}, {2.4288f, 4.5757f, 7.2127f}},
    {"Orange Powder", {0.00015617f, 0.00017482f, 0.0001762f}, {0.001449f, 0.003441f, 0.007863f}},
    {"Pink Lemonade Powder", {0.00012103f, 0.00013073f, 0.00012528f}, {0.001165f, 0.002366f, 0.003195f}},
    {"Cappuccino Powder", {1.8436f, 2.5851f, 2.1662f}, {35.844f, 49.547f, 61.084f}},
    {"Salt Powder", {0.027333f, 0.032451f, 0.031979f}, {0.28415f, 0.3257f, 0.34148f}},
    {"Sugar Powder", {0.00022272f, 0.00025513f, 0.000271f}, {0.012638f, 0.031051f, 0.050124f}},
    {"Suisse Mocha Powder", {2.7979f, 3.5452f, 4.3365f}, {17.502f, 27.004f, 35.433f}},
    {"Pacific Ocean Surface Water", {0.0001764f, 0.00032095f, 0.00019617f}, {0.031845f, 0.031324f, 0.030147f}},
  };
  for (const auto& preset : presets) {
    if (preset.name == name) {
      absorption = SpectralDistribution::rgb_reflectance(preset.absorption);
      scattering = SpectralDistribution::rgb_reflectance(preset.scattering);
      return true;
    }
  }
  return false;
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
