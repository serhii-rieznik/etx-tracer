#pragma once

#include "interop.hxx"

#define ETX_SPECTRAL_MODE_RUNTIME 0
#define ETX_SPECTRAL_MODE_RGB     1

#if defined(ETX_SPECTRAL_MODE) == false
# define ETX_SPECTRAL_MODE ETX_SPECTRAL_MODE_RUNTIME
#endif

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
ETX_STATIC_CONST float kInvCIEYIntegral = 1.0f / 106.856895f;
ETX_STATIC_CONST float3 kSpectralDistributionRGBLuminanceScale = float3(0.817660332f, 1.05418909f, 1.09945524f);

ETX_STATIC_CONST float3 kCIE2006[WavelengthCount] = {
#include "spectrum_cie2006_table.inl"
};

ETX_STATIC_CONST float3 kRGBResponse[RGBResponseWavelengthCount] = {
#include "spectrum_rgb_response_table.inl"
};

struct SpectralFlags {
  enum : uint32_t {
    Spectral = 1u << 0u,
    Packet = 1u << 1u,
    HeroOnly = 1u << 2u,
  };
};

ETX_STATIC_CONST uint32_t kSpectralPacketSize = 4u;

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

struct ETX_ALIGNED SpectralResponse {
  float3 integrated ETX_INIT({});
  float value ETX_INIT(0.0f);
  float wavelength ETX_INIT(kUndefinedWavelength);
  uint32_t flags ETX_INIT(0u);
  uint32_t pad0 ETX_INIT(0u);
  uint32_t pad1 ETX_INIT(0u);
};

struct RefractiveIndexSample {
  SpectralResponse eta ETX_INIT({});
  SpectralResponse k ETX_INIT({});
  SpectralDistribution::Class cls ETX_INIT(SpectralDistribution::Invalid);
};

ETX_SHARED_INLINE bool spectral_response_is_spectral(ETX_IN(SpectralResponse, value));
ETX_SHARED_INLINE SpectralResponse spectral_response_make(float wavelength, float spectral_val);
ETX_SHARED_INLINE SpectralResponse spectral_response_make(ETX_IN(SpectralQuery, query), float spectral_val);
ETX_SHARED_INLINE SpectralResponse spectral_response_make(ETX_IN(SpectralQuery, query), ETX_IN(float3, integrated_val));
ETX_SHARED_INLINE SpectralQuery spectral_response_as_query(ETX_IN(SpectralResponse, value));

ETX_SHARED_INLINE float3 spectral_xyz(uint32_t index) {
#if defined(__cplusplus)
  ETX_ASSERT(index < WavelengthCount);
#endif
  return kCIE2006[index];
}

ETX_SHARED_INLINE float3 spectral_xyz_to_rgb(ETX_IN(float3, xyz)) {
  return make_float3(3.24045420f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z, -0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z,
    0.05564340f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z);
}

ETX_SHARED_INLINE float3 spectral_rgb_to_xyz(ETX_IN(float3, rgb)) {
  return make_float3(0.4124564f * rgb.x + 0.3575760f * rgb.y + 0.1804375f * rgb.z, 0.2126729f * rgb.x + 0.7151521f * rgb.y + 0.0721750f * rgb.z,
    0.0193339f * rgb.x + 0.1191920f * rgb.y + 0.9503041f * rgb.z);
}

ETX_SHARED_INLINE bool spectral_query_is_spectral(ETX_IN(SpectralQuery, query)) {
#if ETX_SPECTRAL_MODE == ETX_SPECTRAL_MODE_RGB
  (void)query;
  return false;
#else
  return (query.flags & SpectralFlags::Spectral) != 0u;
#endif
}

ETX_SHARED_INLINE bool spectral_query_is_packet(ETX_IN(SpectralQuery, query)) {
#if ETX_SPECTRAL_MODE == ETX_SPECTRAL_MODE_RGB
  (void)query;
  return false;
#else
  return (query.flags & SpectralFlags::Packet) != 0u;
#endif
}

ETX_SHARED_INLINE bool spectral_query_is_hero_only(ETX_IN(SpectralQuery, query)) {
#if ETX_SPECTRAL_MODE == ETX_SPECTRAL_MODE_RGB
  (void)query;
  return false;
#else
  return (query.flags & SpectralFlags::HeroOnly) != 0u;
#endif
}

ETX_SHARED_INLINE float spectral_query_wavelength_pdf(float wavelength) {
  float x = 0.0072f * (wavelength - 538.0f);
  float x_exp = exp(x);
  float x_inv_exp = 1.0f / x_exp;
  float x_cosh = 0.5f * (x_exp + x_inv_exp);
  return 0.0039398042f / (x_cosh * x_cosh);
}

ETX_SHARED_INLINE float spectral_query_sampling_pdf(ETX_IN(SpectralQuery, query)) {
  return spectral_query_is_spectral(query) ? spectral_query_wavelength_pdf(query.wavelength) : 1.0f;
}

ETX_SHARED_INLINE SpectralQuery spectral_query_sample() {
  SpectralQuery result;
  result.wavelength = kUndefinedWavelength;
  result.flags = 0u;
  return result;
}

ETX_SHARED_INLINE SpectralQuery spectral_query_spectral_sample(float rnd) {
  const float kSamplingOffset = 0.03781818226f;
  const float kSamplingScale = 1.0f - kSamplingOffset;
  float clamped_rnd = clamp(rnd, 0.0f, 1.0f - kEpsilon);
  float x = 0.85691062f - 1.82750197f * (clamped_rnd * kSamplingScale + kSamplingOffset);
  x = clamp(x, -0.999999f, 0.999999f);
  float inverse_hyperbolic_tangent = 0.5f * log((1.0f + x) / (1.0f - x));

  SpectralQuery result;
  result.wavelength = clamp(538.0f - 138.888889f * inverse_hyperbolic_tangent, kShortestWavelength, kLongestWavelength);
  result.flags = SpectralFlags::Spectral;
  return result;
}

ETX_SHARED_INLINE SpectralQuery spectral_query_packet_sample(float rnd) {
  SpectralQuery result = spectral_query_spectral_sample(rnd);
  result.flags |= SpectralFlags::Packet;
  return result;
}

ETX_SHARED_INLINE float spectral_query_sample_coordinate(float wavelength) {
  const float kSamplingOffset = 0.03781818226f;
  const float kSamplingScale = 1.0f - kSamplingOffset;
  const float inverse_hyperbolic_tangent = (538.0f - wavelength) / 138.888889f;
  const float x = tanh(inverse_hyperbolic_tangent);
  return clamp((((0.85691062f - x) / 1.82750197f) - kSamplingOffset) / kSamplingScale, 0.0f, 1.0f - kEpsilon);
}

ETX_SHARED_INLINE SpectralQuery spectral_query_packet_lane(ETX_IN(SpectralQuery, query), uint32_t lane) {
  SpectralQuery result = query;
  if ((spectral_query_is_packet(query) == false) || (lane == 0u)) {
    result.flags = SpectralFlags::Spectral;
    return result;
  }

  float sample = spectral_query_sample_coordinate(query.wavelength) + float(lane) / float(kSpectralPacketSize);
  sample -= floor(sample);
  result = spectral_query_spectral_sample(sample);
  return result;
}

ETX_SHARED_INLINE float spectral_query_packet_lane_pdf(ETX_IN(SpectralQuery, query), uint32_t lane) {
  return spectral_query_wavelength_pdf(spectral_query_packet_lane(query, lane).wavelength);
}

ETX_SHARED_INLINE void spectral_query_terminate_secondary(ETX_INOUT(SpectralQuery, query)) {
  if (spectral_query_is_packet(query)) {
    query.flags |= SpectralFlags::HeroOnly;
  }
}

ETX_SHARED_INLINE bool spectral_query_compatible(ETX_IN(SpectralQuery, a), ETX_IN(SpectralQuery, b)) {
  const uint32_t relevant_flags = SpectralFlags::Spectral | SpectralFlags::Packet;
  return ((a.flags & relevant_flags) == (b.flags & relevant_flags)) && (((a.flags & SpectralFlags::Spectral) == 0u) || (a.wavelength == b.wavelength));
}

ETX_SHARED_INLINE float spectral_distribution_evaluate_wavelength(ETX_IN(SpectralDistribution, distribution), float wavelength) {
  if (distribution.spectral_entry_count == 0u) {
    return 0.0f;
  }

  uint32_t begin = 0u;
  uint32_t end = distribution.spectral_entry_count;
  while ((end - begin) > 1u) {
    uint32_t middle = begin + (end - begin) / 2u;
    if (distribution.spectral_entries[middle].wavelength > wavelength) {
      end = middle;
    } else {
      begin = middle;
    }
  }

  uint32_t i = begin;
  if (i >= distribution.spectral_entry_count) {
    return 0.0f;
  }

  if ((i == 0u) && (wavelength < distribution.spectral_entries[i].wavelength)) {
    return 0.0f;
  }

  if (((i + 1u) == distribution.spectral_entry_count) && (wavelength > distribution.spectral_entries[i].wavelength)) {
    return 0.0f;
  }

  uint32_t j = min(i + 1u, distribution.spectral_entry_count - 1u);
  float t =
    (i == j) ? 0.0f : (wavelength - distribution.spectral_entries[i].wavelength) / (distribution.spectral_entries[j].wavelength - distribution.spectral_entries[i].wavelength);
  return distribution.spectral_entries[i].power + (distribution.spectral_entries[j].power - distribution.spectral_entries[i].power) * t;
}

ETX_SHARED_INLINE SpectralResponse spectral_distribution_query(ETX_IN(SpectralDistribution, distribution), ETX_IN(SpectralQuery, query)) {
  if (spectral_query_is_spectral(query) == false) {
    return spectral_response_make(query, distribution.integrated_value);
  }

  SpectralResponse result = spectral_response_make(query, spectral_distribution_evaluate_wavelength(distribution, query.wavelength));
  if (spectral_query_is_packet(query) && (spectral_query_is_hero_only(query) == false)) {
    result.integrated.x = spectral_distribution_evaluate_wavelength(distribution, spectral_query_packet_lane(query, 1u).wavelength);
    result.integrated.y = spectral_distribution_evaluate_wavelength(distribution, spectral_query_packet_lane(query, 2u).wavelength);
    result.integrated.z = spectral_distribution_evaluate_wavelength(distribution, spectral_query_packet_lane(query, 3u).wavelength);
  }
  return result;
}

ETX_SHARED_INLINE float3 spectral_response_to_xyz(ETX_IN(SpectralResponse, response)) {
  if (spectral_response_is_spectral(response) == false) {
    return spectral_rgb_to_xyz(response.integrated);
  }

  if ((response.value == 0.0f) || (response.wavelength < kShortestWavelength) || (response.wavelength > kLongestWavelength)) {
    return make_float3(0.0f, 0.0f, 0.0f);
  }

  float wavelength_floor = floor(response.wavelength);
  float wavelength_fraction = response.wavelength - wavelength_floor;
  uint32_t i = uint32_t(wavelength_floor - kShortestWavelength);
  uint32_t j = min(i + 1u, WavelengthCount - 1u);
  float3 xyz0 = spectral_xyz(i);
  float3 xyz1 = spectral_xyz(j);
  float3 xyz = lerp(xyz0, xyz1, wavelength_fraction);
  return xyz * (response.value * kInvCIEYIntegral);
}

ETX_SHARED_INLINE float spectral_response_packet_lane(ETX_IN(SpectralResponse, response), uint32_t lane) {
  if (lane == 0u) {
    return response.value;
  }
  if (lane == 1u) {
    return response.integrated.x;
  }
  if (lane == 2u) {
    return response.integrated.y;
  }
  return response.integrated.z;
}

ETX_SHARED_INLINE void spectral_response_set_packet_lane(ETX_INOUT(SpectralResponse, response), uint32_t lane, float value) {
  if (lane == 0u) {
    response.value = value;
  } else if (lane == 1u) {
    response.integrated.x = value;
  } else if (lane == 2u) {
    response.integrated.y = value;
  } else {
    response.integrated.z = value;
  }
}

ETX_SHARED_INLINE float3 spectral_response_to_xyz_estimate(ETX_IN(SpectralResponse, response)) {
  if (spectral_response_is_spectral(response) == false) {
    return spectral_rgb_to_xyz(response.integrated);
  }

  const SpectralQuery query = spectral_response_as_query(response);
  if ((spectral_query_is_packet(query) == false) || spectral_query_is_hero_only(query)) {
    const float pdf = spectral_query_sampling_pdf(query);
    return (pdf > 0.0f) ? (spectral_response_to_xyz(response) / pdf) : make_float3(0.0f, 0.0f, 0.0f);
  }

  float3 result = make_float3(0.0f, 0.0f, 0.0f);
  for (uint32_t lane = 0u; lane < kSpectralPacketSize; ++lane) {
    const SpectralQuery lane_query = spectral_query_packet_lane(query, lane);
    const float pdf = spectral_query_wavelength_pdf(lane_query.wavelength);
    if (pdf > 0.0f) {
      const SpectralResponse lane_response = spectral_response_make(lane_query, spectral_response_packet_lane(response, lane));
      result += spectral_response_to_xyz(lane_response) / pdf;
    }
  }
  return result / float(kSpectralPacketSize);
}

ETX_SHARED_INLINE float3 spectral_response_to_rgb_estimate(ETX_IN(SpectralResponse, response)) {
  return spectral_response_is_spectral(response) ? spectral_xyz_to_rgb(spectral_response_to_xyz_estimate(response)) : response.integrated;
}

ETX_SHARED_INLINE float3 spectral_response_to_rgb(ETX_IN(SpectralResponse, response)) {
  return spectral_response_is_spectral(response) ? spectral_xyz_to_rgb(spectral_response_to_xyz(response)) : response.integrated;
}

ETX_SHARED_INLINE float spectral_rgb_evaluate_wavelength(float wavelength, ETX_IN(float3, rgb)) {
  float rgb_luminance = rgb.x * 0.212671f + rgb.y * 0.715160f + rgb.z * 0.072169f;
  if (rgb_luminance <= 0.0f) {
    return 0.0f;
  }

  if ((wavelength < kRGBResponseShortestWavelength) || (wavelength > kRGBResponseLongestWavelength)) {
    return 0.0f;
  }

  uint32_t wi = uint32_t(wavelength - kRGBResponseShortestWavelength);
  uint32_t wj = min(wi + 1u, RGBResponseWavelengthCount - 1u);
  float dw = wavelength - floor(wavelength);
  float3 weight = lerp(kRGBResponse[wi], kRGBResponse[wj], dw);
  return rgb.x * weight.x + rgb.y * weight.y + rgb.z * weight.z;
}

ETX_SHARED_INLINE SpectralResponse spectral_rgb_response(ETX_IN(SpectralQuery, query), ETX_IN(float3, rgb)) {
  if (spectral_query_is_spectral(query) == false) {
    return spectral_response_make(query, rgb);
  }

  SpectralResponse result = spectral_response_make(query, spectral_rgb_evaluate_wavelength(query.wavelength, rgb));
  if (spectral_query_is_packet(query) && (spectral_query_is_hero_only(query) == false)) {
    result.integrated.x = spectral_rgb_evaluate_wavelength(spectral_query_packet_lane(query, 1u).wavelength, rgb);
    result.integrated.y = spectral_rgb_evaluate_wavelength(spectral_query_packet_lane(query, 2u).wavelength, rgb);
    result.integrated.z = spectral_rgb_evaluate_wavelength(spectral_query_packet_lane(query, 3u).wavelength, rgb);
  }
  return result;
}

ETX_SHARED_INLINE float luminance(ETX_IN(float3, value)) {
  return value.x * 0.212671f + value.y * 0.715160f + value.z * 0.072169f;
}

ETX_SHARED_INLINE bool spectral_response_is_spectral(ETX_IN(SpectralResponse, value)) {
#if ETX_SPECTRAL_MODE == ETX_SPECTRAL_MODE_RGB
  (void)value;
  return false;
#else
  return value.flags & SpectralFlags::Spectral;
#endif
}

ETX_SHARED_INLINE float spectral_response_monochromatic(ETX_IN(SpectralResponse, value)) {
  return spectral_response_is_spectral(value) ? value.value : luminance(value.integrated);
}

ETX_SHARED_INLINE SpectralQuery spectral_response_as_query(ETX_IN(SpectralResponse, value)) {
  SpectralQuery result;
  result.wavelength = value.wavelength;
  result.flags = value.flags;
  return result;
}

ETX_SHARED_INLINE bool spectral_response_is_zero(ETX_IN(SpectralResponse, value)) {
  if (spectral_response_is_spectral(value) == false) {
    return (value.integrated.x <= kEpsilon) && (value.integrated.y <= kEpsilon) && (value.integrated.z <= kEpsilon);
  }
  const SpectralQuery query = spectral_response_as_query(value);
  return (value.value <= kEpsilon) && ((spectral_query_is_packet(query) == false) || spectral_query_is_hero_only(query) ||
                                        ((value.integrated.x <= kEpsilon) && (value.integrated.y <= kEpsilon) && (value.integrated.z <= kEpsilon)));
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

ETX_SHARED_INLINE SpectralResponse spectral_response_make(ETX_IN(SpectralQuery, query), float spectral_val) {
  SpectralResponse result;
  result.flags = query.flags;
  result.wavelength = query.wavelength;
  result.value = spectral_val;
  result.integrated = spectral_query_is_hero_only(query) ? make_float3(0.0f, 0.0f, 0.0f) : make_float3(spectral_val, spectral_val, spectral_val);
  return result;
}

ETX_SHARED_INLINE SpectralResponse spectral_response_make(ETX_IN(SpectralQuery, query), ETX_IN(float3, integrated_val)) {
  SpectralResponse result;
  result.flags = query.flags;
  result.wavelength = query.wavelength;
  result.value = 0.0f;
  result.integrated = integrated_val;
  return result;
}

ETX_SHARED_INLINE SpectralResponse spectral_response_make_packet(ETX_IN(SpectralQuery, query), ETX_IN(float3, secondary), float hero) {
  SpectralResponse result;
  result.flags = query.flags;
  result.wavelength = query.wavelength;
  result.value = hero;
  result.integrated = spectral_query_is_hero_only(query) ? make_float3(0.0f, 0.0f, 0.0f) : secondary;
  return result;
}

ETX_SHARED_INLINE void spectral_response_terminate_secondary(ETX_INOUT(SpectralResponse, response)) {
  if (spectral_query_is_packet(spectral_response_as_query(response))) {
    response.flags |= SpectralFlags::HeroOnly;
    response.integrated = make_float3(0.0f, 0.0f, 0.0f);
  }
}

ETX_SHARED_INLINE SpectralResponse spectral_response_add(ETX_IN(SpectralResponse, a), ETX_IN(SpectralResponse, b)) {
  if (spectral_response_is_spectral(a) && spectral_response_is_spectral(b)) {
    SpectralQuery query = spectral_response_as_query(a);
    if (spectral_query_is_hero_only(spectral_response_as_query(b))) {
      spectral_query_terminate_secondary(query);
    }
    return spectral_response_make_packet(query, a.integrated + b.integrated, a.value + b.value);
  }

  return spectral_response_make(a.integrated + b.integrated);
}

ETX_SHARED_INLINE SpectralResponse spectral_response_sub(ETX_IN(SpectralResponse, a), ETX_IN(SpectralResponse, b)) {
  if (spectral_response_is_spectral(a) && spectral_response_is_spectral(b)) {
    SpectralQuery query = spectral_response_as_query(a);
    if (spectral_query_is_hero_only(spectral_response_as_query(b))) {
      spectral_query_terminate_secondary(query);
    }
    return spectral_response_make_packet(query, a.integrated - b.integrated, a.value - b.value);
  }

  return spectral_response_make(a.integrated - b.integrated);
}

ETX_SHARED_INLINE SpectralResponse spectral_response_mul(ETX_IN(SpectralResponse, a), ETX_IN(SpectralResponse, b)) {
  if (spectral_response_is_spectral(a) && spectral_response_is_spectral(b)) {
    SpectralQuery query = spectral_response_as_query(a);
    if (spectral_query_is_hero_only(spectral_response_as_query(b))) {
      spectral_query_terminate_secondary(query);
    }
    return spectral_response_make_packet(query, a.integrated * b.integrated, a.value * b.value);
  }

  return spectral_response_make(a.integrated * b.integrated);
}

ETX_SHARED_INLINE SpectralResponse spectral_response_div(ETX_IN(SpectralResponse, a), ETX_IN(SpectralResponse, b)) {
  if (spectral_response_is_spectral(a) && spectral_response_is_spectral(b)) {
    SpectralQuery query = spectral_response_as_query(a);
    if (spectral_query_is_hero_only(spectral_response_as_query(b))) {
      spectral_query_terminate_secondary(query);
    }
    return spectral_response_make_packet(query, a.integrated / b.integrated, a.value / b.value);
  }

  return spectral_response_make(a.integrated / b.integrated);
}

ETX_SHARED_INLINE SpectralResponse spectral_response_add(ETX_IN(SpectralResponse, a), float b) {
  if (spectral_response_is_spectral(a))
    return spectral_response_make_packet(spectral_response_as_query(a), a.integrated + b, a.value + b);

  return spectral_response_make(a.integrated + b);
}

ETX_SHARED_INLINE SpectralResponse spectral_response_sub(ETX_IN(SpectralResponse, a), float b) {
  if (spectral_response_is_spectral(a))
    return spectral_response_make_packet(spectral_response_as_query(a), a.integrated - b, a.value - b);

  return spectral_response_make(a.integrated - b);
}

ETX_SHARED_INLINE SpectralResponse spectral_response_mul(ETX_IN(SpectralResponse, a), float b) {
  if (spectral_response_is_spectral(a))
    return spectral_response_make_packet(spectral_response_as_query(a), a.integrated * b, a.value * b);

  return spectral_response_make(a.integrated * b);
}

ETX_SHARED_INLINE SpectralResponse spectral_response_div(ETX_IN(SpectralResponse, a), float b) {
  if (spectral_response_is_spectral(a))
    return spectral_response_make_packet(spectral_response_as_query(a), a.integrated / b, a.value / b);

  return spectral_response_make(a.integrated / b);
}

ETX_SHARED_INLINE SpectralResponse spectral_response_exp(ETX_IN(SpectralResponse, a)) {
  if (spectral_response_is_spectral(a))
    return spectral_response_make_packet(spectral_response_as_query(a), exp(a.integrated), exp(a.value));

  return spectral_response_make(exp(a.integrated));
}

ETX_SHARED_INLINE SpectralResponse spectral_response_sqrt(ETX_IN(SpectralResponse, a)) {
  if (spectral_response_is_spectral(a))
    return spectral_response_make_packet(spectral_response_as_query(a), sqrt(a.integrated), sqrt(a.value));

  return spectral_response_make(sqrt(a.integrated));
}

ETX_SHARED_INLINE SpectralResponse spectral_response_cos(ETX_IN(SpectralResponse, a)) {
  if (spectral_response_is_spectral(a))
    return spectral_response_make_packet(spectral_response_as_query(a), cos(a.integrated), cos(a.value));

  return spectral_response_make(cos(a.integrated));
}

ETX_SHARED_INLINE SpectralResponse spectral_response_abs(ETX_IN(SpectralResponse, a)) {
  if (spectral_response_is_spectral(a))
    return spectral_response_make_packet(spectral_response_as_query(a), abs(a.integrated), abs(a.value));

  return spectral_response_make(abs(a.integrated));
}

ETX_SHARED_INLINE SpectralResponse spectral_response_saturate(ETX_IN(SpectralResponse, a)) {
  if (spectral_response_is_spectral(a))
    return spectral_response_make_packet(spectral_response_as_query(a), saturate(a.integrated), saturate(a.value));

  return spectral_response_make(saturate(a.integrated));
}

ETX_SHARED_INLINE SpectralResponse spectral_response_sign(ETX_IN(SpectralResponse, a)) {
  if (spectral_response_is_spectral(a))
    return spectral_response_make_packet(spectral_response_as_query(a), sign(a.integrated), sign(a.value));

  return spectral_response_make(sign(a.integrated));
}

ETX_SHARED_INLINE SpectralResponse spectral_response_atan(ETX_IN(SpectralResponse, a)) {
  if (spectral_response_is_spectral(a))
    return spectral_response_make_packet(spectral_response_as_query(a), atan(a.integrated), atan(a.value));

  return spectral_response_make(atan(a.integrated));
}

ETX_SHARED_INLINE SpectralResponse spectral_response_pow(ETX_IN(SpectralResponse, a), float b) {
  if (spectral_response_is_spectral(a))
    return spectral_response_make_packet(spectral_response_as_query(a), pow(a.integrated, b), pow(a.value, b));

  return spectral_response_make(pow(a.integrated, b));
}

ETX_SHARED_INLINE SpectralResponse spectral_response_pow(ETX_IN(SpectralResponse, a), ETX_IN(SpectralResponse, b)) {
  if (spectral_response_is_spectral(a) && spectral_response_is_spectral(b)) {
    SpectralQuery query = spectral_response_as_query(a);
    if (spectral_query_is_hero_only(spectral_response_as_query(b))) {
      spectral_query_terminate_secondary(query);
    }
    return spectral_response_make_packet(query, pow(a.integrated, b.integrated), pow(a.value, b.value));
  }

  return spectral_response_make(pow(a.integrated, b.integrated));
}

ETX_SHARED_INLINE SpectralResponse spectral_response_max(ETX_IN(SpectralResponse, a), float b) {
  if (spectral_response_is_spectral(a))
    return spectral_response_make_packet(spectral_response_as_query(a), max(a.integrated, b), max(a.value, b));

  return spectral_response_make(max(a.integrated, b));
}

ETX_SHARED_INLINE SpectralResponse spectral_response_max(float a, ETX_IN(SpectralResponse, b)) {
  if (spectral_response_is_spectral(b))
    return spectral_response_make_packet(spectral_response_as_query(b), max(b.integrated, a), max(a, b.value));

  return spectral_response_make(max(b.integrated, a));
}

ETX_SHARED_INLINE SpectralResponse spectral_response_min(ETX_IN(SpectralResponse, a), float b) {
  if (spectral_response_is_spectral(a))
    return spectral_response_make_packet(spectral_response_as_query(a), min(a.integrated, b), min(a.value, b));

  return spectral_response_make(min(a.integrated, b));
}

ETX_SHARED_INLINE SpectralResponse spectral_response_min(float a, ETX_IN(SpectralResponse, b)) {
  if (spectral_response_is_spectral(b))
    return spectral_response_make_packet(spectral_response_as_query(b), min(b.integrated, a), min(a, b.value));

  return spectral_response_make(min(b.integrated, a));
}

ETX_SHARED_INLINE void spectral_response_add_assign(ETX_INOUT(SpectralResponse, a), ETX_IN(SpectralResponse, b)) {
  a = spectral_response_add(a, b);
}

ETX_SHARED_INLINE void spectral_response_sub_assign(ETX_INOUT(SpectralResponse, a), ETX_IN(SpectralResponse, b)) {
  a = spectral_response_sub(a, b);
}

ETX_SHARED_INLINE void spectral_response_mul_assign(ETX_INOUT(SpectralResponse, a), ETX_IN(SpectralResponse, b)) {
  a = spectral_response_mul(a, b);
}

ETX_SHARED_INLINE void spectral_response_div_assign(ETX_INOUT(SpectralResponse, a), ETX_IN(SpectralResponse, b)) {
  a = spectral_response_div(a, b);
}

ETX_SHARED_INLINE void spectral_response_add_assign(ETX_INOUT(SpectralResponse, a), float b) {
  a.integrated += b;
  a.value += b;
}

ETX_SHARED_INLINE void spectral_response_sub_assign(ETX_INOUT(SpectralResponse, a), float b) {
  a.integrated -= b;
  a.value -= b;
}

ETX_SHARED_INLINE void spectral_response_mul_assign(ETX_INOUT(SpectralResponse, a), float b) {
  a.integrated *= b;
  a.value *= b;
}

ETX_SHARED_INLINE void spectral_response_div_assign(ETX_INOUT(SpectralResponse, a), float b) {
  a.integrated /= b;
  a.value /= b;
}

ETX_SHARED_INLINE void spectral_response_exp_assign(ETX_INOUT(SpectralResponse, a)) {
  a.integrated = exp(a.integrated);
  a.value = exp(a.value);
}

ETX_SHARED_INLINE SpectralResponse spectral_response_zero(ETX_IN(SpectralQuery, query)) {
  return spectral_response_make(query, 0.0f);
}

ETX_SHARED_INLINE float3 spectral_rgb_clamp_non_negative(ETX_IN(float3, value)) {
  return max(value, make_float3(0.0f, 0.0f, 0.0f));
}

ETX_SHARED_INLINE SpectralResponse spectral_response_apply_rgb_scale(ETX_IN(SpectralQuery, query), ETX_IN(SpectralResponse, response), ETX_IN(float3, rgb)) {
  SpectralResponse result = response;
  if (spectral_query_is_spectral(query)) {
    result = spectral_response_mul(result, spectral_rgb_response(query, rgb));
  } else {
    result.integrated *= rgb;
  }

  return result;
}

ETX_SHARED_INLINE SpectralResponse spectral_response_clamp_non_negative(ETX_IN(SpectralResponse, value)) {
  SpectralResponse result = value;
  if (spectral_response_is_spectral(result)) {
    result.value = max(result.value, 0.0f);
    result.integrated = max(result.integrated, make_float3(0.0f, 0.0f, 0.0f));
  } else {
    result.integrated = spectral_rgb_clamp_non_negative(result.integrated);
  }

  return result;
}

ETX_SHARED_INLINE float spectral_response_maximum(ETX_IN(SpectralResponse, val)) {
  if (spectral_response_is_spectral(val)) {
    const SpectralQuery query = spectral_response_as_query(val);
    return ((spectral_query_is_packet(query) == false) || spectral_query_is_hero_only(query)) ? val.value
                                                                                              : max(val.value, max(val.integrated.x, max(val.integrated.y, val.integrated.z)));
  }

  return max(val.integrated.x, max(val.integrated.y, val.integrated.z));
}

ETX_SHARED_INLINE complex refractive_index_sample_as_complex_spectral(ETX_IN(RefractiveIndexSample, value)) {
  return make_complex(value.eta.value, value.k.value);
}

ETX_SHARED_INLINE complex refractive_index_sample_as_complex_x(ETX_IN(RefractiveIndexSample, value)) {
  return make_complex(value.eta.integrated.x, value.k.integrated.x);
}

ETX_SHARED_INLINE complex refractive_index_sample_as_complex_y(ETX_IN(RefractiveIndexSample, value)) {
  return make_complex(value.eta.integrated.y, value.k.integrated.y);
}

ETX_SHARED_INLINE complex refractive_index_sample_as_complex_z(ETX_IN(RefractiveIndexSample, value)) {
  return make_complex(value.eta.integrated.z, value.k.integrated.z);
}
