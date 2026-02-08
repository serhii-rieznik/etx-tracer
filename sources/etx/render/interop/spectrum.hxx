#pragma once

#include "interop.hxx"

ETX_STATIC_CONST uint32_t RGBResponseShortestWavelength = 390u;
ETX_STATIC_CONST uint32_t RGBResponseLongestWavelength = 780u;
ETX_STATIC_CONST uint32_t RGBResponseWavelengthCount = RGBResponseLongestWavelength - RGBResponseShortestWavelength + 1u;
ETX_STATIC_CONST uint32_t ShortestWavelength = 390u;
ETX_STATIC_CONST uint32_t WavelengthCount = 441u;
ETX_STATIC_CONST uint32_t LongestWavelength = ShortestWavelength + WavelengthCount - 1u;
ETX_STATIC_CONST float kShortestWavelength = float(ShortestWavelength);
ETX_STATIC_CONST float kLongestWavelength = float(LongestWavelength);
ETX_STATIC_CONST float kWavelengthCount = float(WavelengthCount);
ETX_STATIC_CONST float kRGBResponseShortestWavelength = float(RGBResponseShortestWavelength);
ETX_STATIC_CONST float kRGBResponseLongestWavelength = float(RGBResponseLongestWavelength);
ETX_STATIC_CONST float kRGBResponseWavelengthCount = float(RGBResponseWavelengthCount);
ETX_STATIC_CONST float kUndefinedWavelength = -1.0f;

struct SpectralFlags {
  enum : uint32_t {
    Spectral = 1u << 0u,
  };
};

struct ETX_ALIGNED SpectralDistribution {
  using Class = uint32_t;
  enum : uint32_t {
    Invalid,
    Reflectance,
    Conductor,
    Dielectric,
    Illuminant,
  };

  struct Entry {
    float wavelength ETX_INIT(0.0f);
    float power ETX_INIT(0.0f);
  };

  float3 integrated_value ETX_INIT({});
  uint32_t spectral_entry_count ETX_INIT(0u);
  Entry spectral_entries[WavelengthCount] ETX_INIT({});
  Entry pad ETX_INIT({});
};

struct ETX_ALIGNED RefractiveIndex {
  uint32_t cls ETX_INIT(SpectralDistribution::Invalid);
  uint32_t eta_index ETX_INIT(kInvalidIndex);
  uint32_t k_index ETX_INIT(kInvalidIndex);
  uint32_t pad ETX_INIT(0);
};

struct SpectralQuery {
  float wavelength ETX_INIT(kUndefinedWavelength);
  uint32_t flags ETX_INIT(0u);
};

struct SpectralResponse {
  float3 integrated ETX_INIT({});
  float value ETX_INIT(0.0f);
  float wavelength ETX_INIT(kUndefinedWavelength);
  uint32_t flags ETX_INIT(0u);
};

struct RefractiveIndexSample {
  SpectralResponse eta ETX_INIT({});
  SpectralResponse k ETX_INIT({});
  SpectralDistribution::Class cls ETX_INIT(SpectralDistribution::Invalid);
};

ETX_SHARED_INLINE float luminance(ETX_IN(float3, value)) {
  return value.x * 0.212671f + value.y * 0.715160f + value.z * 0.072169f;
}

ETX_SHARED_INLINE bool spectral_response_is_spectral(ETX_IN(SpectralResponse, value)) {
  return value.flags & SpectralFlags::Spectral;
}

ETX_SHARED_INLINE float spectral_response_monochromatic(ETX_IN(SpectralResponse, value)) {
  return spectral_response_is_spectral(value) ? value.value : luminance(value.integrated);
}

ETX_SHARED_INLINE bool spectral_response_is_zero(ETX_IN(SpectralResponse, value)) {
  return spectral_response_is_spectral(value) ? (value.value <= kEpsilon)
                                              : (value.integrated.x <= kEpsilon) && (value.integrated.y <= kEpsilon) && (value.integrated.z <= kEpsilon);
}

ETX_SHARED_INLINE SpectralResponse spectral_response_make(float wavelength, float spectral_val) {
  SpectralResponse result;
  result.flags = SpectralFlags::Spectral;
  result.wavelength = wavelength;
  result.value = spectral_val;
  result.integrated = make_float3(spectral_val, spectral_val, spectral_val);
  return result;
}

ETX_SHARED_INLINE SpectralResponse spectral_response_make(ETX_IN(float3, integrated_val)) {
  SpectralResponse result;
  result.flags = 0;
  result.wavelength = kUndefinedWavelength;
  result.value = 0.0f;
  result.integrated = integrated_val;
  return result;
}

ETX_SHARED_INLINE SpectralResponse spectral_response_div(ETX_IN(SpectralResponse, a), ETX_IN(SpectralResponse, b)) {
  if (spectral_response_is_spectral(a) && spectral_response_is_spectral(b))
    return spectral_response_make(a.wavelength, a.value / b.value);

  return spectral_response_make(a.integrated / b.integrated);
}
