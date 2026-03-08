#pragma once

#include <cstddef>

#include <etx/render/interop/spectrum.hxx>
#include <etx/render/interop/gpu_abi_constants.hxx>
#include <etx/render/access/spectrum_access_cpu.hxx>

ETX_STATIC_ASSERT((sizeof(::SpectralDistribution) == kSpectralDistributionStride), "SpectralDistribution ABI size mismatch");
ETX_STATIC_ASSERT((offsetof(::SpectralDistribution, integrated_value) == kSpectralDistributionIntegratedOffset), "SpectralDistribution.integrated_value ABI mismatch");
ETX_STATIC_ASSERT((offsetof(::SpectralDistribution, spectral_entry_count) == kSpectralDistributionEntryCountOffset), "SpectralDistribution.spectral_entry_count ABI mismatch");
ETX_STATIC_ASSERT((offsetof(::SpectralDistribution, spectral_entries) == kSpectralDistributionEntriesOffset), "SpectralDistribution.spectral_entries ABI mismatch");
ETX_STATIC_ASSERT((sizeof(::SpectralDistribution::Entry) == kSpectralDistributionEntryStride), "SpectralDistribution::Entry ABI size mismatch");

namespace etx {

ETX_SHARED_INLINE float3 spectral_xyz(uint32_t i) {
  ETX_ASSERT(i < WavelengthCount);
  return ::spectral_xyz(i);
}

ETX_SHARED_INLINE float3 xyz_to_rgb(const float3& xyz) {
  return ::spectral_xyz_to_rgb(xyz);
}

ETX_SHARED_INLINE float3 rgb_to_xyz(const float3& rgb) {
  return ::spectral_rgb_to_xyz(rgb);
}

ETX_SHARED_INLINE float4 rgb_to_xyz4(const float3& rgb) {
  const float3 xyz = rgb_to_xyz(rgb);
  return {xyz.x, xyz.y, xyz.z, 1.0f};
}

ETX_SHARED_INLINE float black_body_radiation_maximum_wavelength(float t_kelvins) {
  return 2.8977729e+6f / t_kelvins;
}

ETX_SHARED_INLINE float black_body_radiation(float wavelength_nm, float t_kelvins) {
  ETX_ASSERT(t_kelvins > 0);

  // wavelength (in nm) is scaled to reduce floating point errors, constants are scaled correspondingly
  constexpr float wavelengt_scale = 1.0f / 1000.0f;
  constexpr float Lc1 = 3.7417712e+5f;  // 2 * pi * h * c * c * (10^21 - from wavelength scale)
  constexpr float Lc2 = 1.4387752e+4f;  // h * c / k * (10^-6 - from wavelength scale)

  wavelength_nm *= wavelengt_scale;
  float wl5 = wavelength_nm * (wavelength_nm * wavelength_nm) * (wavelength_nm * wavelength_nm);

  float e0 = expf(Lc2 / (wavelength_nm * t_kelvins));
  ETX_ASSERT(isnan(e0) == false);

  float d = wl5 * (e0 - 1.0f);
  ETX_ASSERT(isnan(d) == false);

  return isinf(d) ? 0.0f : (Lc1 / d);
}

ETX_SHARED_INLINE float kYIntegral() {
  float result = 0.0f;
  for (uint32_t i = 0; i < WavelengthCount; ++i) {
    result += spectral_xyz(i).y;
  }
  return result;
}

namespace scattering {

struct Parameters {
  float altitude = 1000.0f;
  float anisotropy = 0.825f;
  float rayleigh_scale = 1.0f;
  float mie_scale = 1.0f;
  float ozone_scale = 1.0f;
  uint32_t primary_scattering = 1u;
  uint32_t secondary_scattering = 1u;
};

}  // namespace scattering

struct SpectralQuery : public ::SpectralQuery {
  SpectralQuery() = default;

  SpectralQuery(float w, const uint32_t& f)
    : ::SpectralQuery{w, f} {
  }

  bool spectral() const {
    return (flags & SpectralFlags::Spectral) != 0;
  };

  float sampling_pdf() const {
    return spectral() ? ::spectral_query_wavelength_pdf(wavelength) : 1.0f;
  }

  bool valid() const {
    return (wavelength >= kShortestWavelength) && (wavelength <= kLongestWavelength);
  }

  static SpectralQuery sample() {
    const ::SpectralQuery query = ::spectral_query_sample();
    return SpectralQuery{query.wavelength, query.flags};
  }

  static SpectralQuery spectral_sample(float rnd) {
    const ::SpectralQuery query = ::spectral_query_spectral_sample(rnd);
    return SpectralQuery{query.wavelength, query.flags};
  }

  float static const spectral_sample_pdf(float wavelength) {
    return ::spectral_query_wavelength_pdf(wavelength);
  }
};

struct SpectralResponse : public ::SpectralResponse {
  SpectralResponse() = default;

  SpectralResponse(const SpectralQuery q)
    : ::SpectralResponse({}, 0.0f, q.wavelength, q.flags) {
  }

  SpectralResponse(const SpectralQuery q, float a)
    : ::SpectralResponse({a, a, a}, a, q.wavelength, q.flags) {
  }

  SpectralResponse(const SpectralResponse q, float a)
    : ::SpectralResponse({a, a, a}, a, q.wavelength, q.flags) {
  }

  SpectralResponse(const SpectralQuery q, const float3& c)
    : ::SpectralResponse(c, 0.0f, q.wavelength, q.flags) {
  }

  SpectralResponse(const SpectralResponse q, const float3& c)
    : ::SpectralResponse(c, 0.0f, q.wavelength, q.flags) {
  }

  bool spectral() const {
    return (flags & SpectralFlags::Spectral) != 0;
  }

  float component_count() const {
    return spectral() ? 1.0f : 3.0f;
  }

  float sampling_pdf() const {
    return spectral() ? SpectralQuery::spectral_sample_pdf(wavelength) : 1.0f;
  }

  SpectralQuery as_query() const {
    return {wavelength, flags};
  }

  ETX_SHARED_INLINE float3 to_xyz() const {
    return ::spectral_response_to_xyz(static_cast<const ::SpectralResponse&>(*this));
  }

  ETX_SHARED_INLINE float3 to_rgb() const {
    return ::spectral_response_to_rgb(static_cast<const ::SpectralResponse&>(*this));
  }

  ETX_SHARED_INLINE float minimum() const {
    return spectral() ? value : min(integrated.x, min(integrated.y, integrated.z));
  }

  ETX_SHARED_INLINE float maximum() const {
    return spectral() ? value : max(integrated.x, max(integrated.y, integrated.z));
  }

  ETX_SHARED_INLINE float monochromatic() const {
    return spectral() ? value : ::luminance(integrated);
  }

  ETX_SHARED_INLINE float sum() const {
    return spectral() ? value : integrated.x + integrated.y + integrated.z;
  }

  ETX_SHARED_INLINE float average() const {
    return spectral() ? value : (integrated.x + integrated.y + integrated.z) / 3.0f;
  }

  ETX_SHARED_INLINE float luminance() const {
    return to_xyz().y;
  }

  ETX_SHARED_INLINE float component(uint32_t i) const {
    ETX_ASSERT(i < 3);
    return spectral() ? value : *(&integrated.x + i);
  }

  ETX_SHARED_INLINE bool valid() const {
    return spectral() ? valid_value(value) : valid_value(integrated);
  }

  ETX_SHARED_INLINE bool is_zero() const {
    return spectral_response_is_zero(*this);
  }

#define SPECTRAL_OP(OP)                                                             \
  ETX_SHARED_INLINE SpectralResponse& operator OP(const SpectralResponse & other) { \
    ETX_ASSERT_EQUAL(wavelength, other.wavelength);                                 \
    integrated OP other.integrated;                                                 \
    value OP other.value;                                                           \
    return *this;                                                                   \
  }
  SPECTRAL_OP(+=)
  SPECTRAL_OP(-=)
  SPECTRAL_OP(*=)
  SPECTRAL_OP(/=)
#undef SPECTRAL_OP

#define SPECTRAL_OP(OP)                                                                                                                    \
  ETX_SHARED_INLINE SpectralResponse operator OP(const SpectralResponse& other) const {                                                    \
    ETX_ASSERT_EQUAL(wavelength, other.wavelength);                                                                                        \
    ETX_ASSERT((spectral() && other.spectral()) || ((spectral() == false) && (other.spectral() == false)));                                \
    return spectral() ? SpectralResponse{as_query(), value OP other.value} : SpectralResponse{as_query(), integrated OP other.integrated}; \
  }
  SPECTRAL_OP(+)
  SPECTRAL_OP(-)
  SPECTRAL_OP(*)
  SPECTRAL_OP(/)
#undef SPECTRAL_OP

#define SPECTRAL_OP(OP)                                          \
  ETX_SHARED_INLINE SpectralResponse& operator OP(float other) { \
    integrated OP other;                                         \
    value OP other;                                              \
    return *this;                                                \
  }
  SPECTRAL_OP(+=)
  SPECTRAL_OP(-=)
  SPECTRAL_OP(*=)
  SPECTRAL_OP(/=)
#undef SPECTRAL_OP

#define SPECTRAL_OP(OP)                                                                                                   \
  ETX_SHARED_INLINE SpectralResponse operator OP(float other) const {                                                     \
    return spectral() ? SpectralResponse{as_query(), value OP other} : SpectralResponse{as_query(), integrated OP other}; \
  }
  SPECTRAL_OP(+)
  SPECTRAL_OP(-)
  SPECTRAL_OP(*)
  SPECTRAL_OP(/)
#undef SPECTRAL_OP
};

ETX_SHARED_INLINE SpectralResponse operator*(float other, const SpectralResponse& s) {
  return s * other;
}
ETX_SHARED_INLINE SpectralResponse operator/(float other, const SpectralResponse& s) {
  return s.spectral() ? SpectralResponse{s.as_query(), other / s.value} : SpectralResponse{s.as_query(), other / s.integrated};
}
ETX_SHARED_INLINE SpectralResponse operator+(float other, const SpectralResponse& s) {
  return s + other;
}
ETX_SHARED_INLINE SpectralResponse operator-(const SpectralResponse& s) {
  return s.spectral() ? SpectralResponse{s.as_query(), -s.value} : SpectralResponse{s.as_query(), -s.integrated};
}
ETX_SHARED_INLINE SpectralResponse operator-(float other, const SpectralResponse& s) {
  return s.spectral() ? SpectralResponse{s.as_query(), other - s.value} : SpectralResponse{s.as_query(), other - s.integrated};
}
ETX_SHARED_INLINE SpectralResponse spectrum_exp(const SpectralResponse& s) {
  return s.spectral() ? SpectralResponse{s.as_query(), expf(s.value)} : SpectralResponse{s.as_query(), exp(s.integrated)};
}
ETX_SHARED_INLINE SpectralResponse spectrum_sqrt(const SpectralResponse& s) {
  return s.spectral() ? SpectralResponse{s.as_query(), sqrtf(s.value)} : SpectralResponse{s.as_query(), sqrt(s.integrated)};
}
ETX_SHARED_INLINE SpectralResponse spectrum_cos(const SpectralResponse& s) {
  return s.spectral() ? SpectralResponse{s.as_query(), cosf(s.value)} : SpectralResponse{s.as_query(), cos(s.integrated)};
}
ETX_SHARED_INLINE SpectralResponse spectrum_abs(const SpectralResponse& s) {
  return s.spectral() ? SpectralResponse{s.as_query(), fabsf(s.value)} : SpectralResponse{s.as_query(), abs(s.integrated)};
}
ETX_SHARED_INLINE SpectralResponse spectrum_saturate(const SpectralResponse& s) {
  return s.spectral() ? SpectralResponse{s.as_query(), saturate(s.value)} : SpectralResponse{s.as_query(), saturate(s.integrated)};
}
ETX_SHARED_INLINE SpectralResponse spectrum_sign(const SpectralResponse& b) {
  return b.spectral() ? SpectralResponse{b.as_query(), sign(b.value)} : SpectralResponse(b.as_query(), sign(b.integrated));
}
ETX_SHARED_INLINE SpectralResponse spectrum_atan(const SpectralResponse& b) {
  return b.spectral() ? SpectralResponse{b.as_query(), atanf(b.value)} : SpectralResponse(b.as_query(), atan(b.integrated));
}
ETX_SHARED_INLINE SpectralResponse spectrum_pow(const SpectralResponse& a, float b) {
  return a.spectral() ? SpectralResponse{a.as_query(), powf(a.value, b)} : SpectralResponse(a.as_query(), pow(a.integrated, b));
}
ETX_SHARED_INLINE SpectralResponse spectrum_pow(const SpectralResponse& a, const SpectralResponse& b) {
  return a.spectral() ? SpectralResponse{b.as_query(), powf(a.value, b.value)} : SpectralResponse(b.as_query(), pow(a.integrated, b.integrated));
}
ETX_SHARED_INLINE SpectralResponse spectrum_max(const SpectralResponse& a, float b) {
  return a.spectral() ? SpectralResponse{a.as_query(), fmaxf(a.value, b)} : SpectralResponse(a.as_query(), max(a.integrated, b));
}
ETX_SHARED_INLINE SpectralResponse spectrum_max(float a, const SpectralResponse& b) {
  return b.spectral() ? SpectralResponse{b.as_query(), fmaxf(b.value, a)} : SpectralResponse(b.as_query(), max(b.integrated, a));
}
ETX_SHARED_INLINE SpectralResponse spectrum_min(const SpectralResponse& a, float b) {
  return a.spectral() ? SpectralResponse{a.as_query(), fminf(a.value, b)} : SpectralResponse(a.as_query(), min(a.integrated, b));
}
ETX_SHARED_INLINE SpectralResponse spectrum_min(float a, const SpectralResponse& b) {
  return b.spectral() ? SpectralResponse{b.as_query(), fminf(b.value, a)} : SpectralResponse(b.as_query(), min(b.integrated, a));
}

ETX_SHARED_INLINE SpectralResponse spectral_response_apply_rgb_scale(const SpectralQuery spect, const SpectralResponse& response, const float3& rgb) {
  const ::SpectralResponse result =
    ::spectral_response_apply_rgb_scale(static_cast<const ::SpectralQuery&>(spect), static_cast<const ::SpectralResponse&>(response), rgb);
  SpectralQuery result_query{result.wavelength, result.flags};
  return ::spectral_response_is_spectral(result) ? SpectralResponse{result_query, result.value} : SpectralResponse{result_query, result.integrated};
}

ETX_SHARED_INLINE SpectralResponse spectral_response_clamp_non_negative(const SpectralResponse& response) {
  const ::SpectralResponse result = ::spectral_response_clamp_non_negative(static_cast<const ::SpectralResponse&>(response));
  SpectralQuery result_query{result.wavelength, result.flags};
  return ::spectral_response_is_spectral(result) ? SpectralResponse{result_query, result.value} : SpectralResponse{result_query, result.integrated};
}

ETX_SHARED_INLINE bool valid_value(const SpectralResponse& v) {
  return v.valid();
}

ETX_SHARED_INLINE bool valid_value(const ::SpectralResponse& v) {
  return valid_value(v.integrated) && valid_value(v.value) && valid_value(v.wavelength);
}

#if (ETX_DEBUG || ETX_FORCE_VALIDATION)
template <>
ETX_SHARED_INLINE void print_value<complex>(const char* name, const complex& z, const char* filename, uint32_t line) {
  printf("Validation failed: %s (%f + i * %f) at %s [%u]\n", name, z.real(), z.imag(), filename, line);
}

template <>
ETX_SHARED_INLINE void print_value<SpectralResponse>(const char* name, const SpectralResponse& v, const char* filename, uint32_t line) {
  printf("Validation failed: %s (%f : %f %f %f / %f) at %s [%u]\n", name, v.wavelength, v.integrated.x, v.integrated.y, v.integrated.z, v.value, filename, line);
}

template <>
ETX_SHARED_INLINE void print_value<::SpectralResponse>(const char* name, const ::SpectralResponse& v, const char* filename, uint32_t line) {
  printf("Validation failed: %s (%f : %f %f %f / %f) at %s [%u]\n", name, v.wavelength, v.integrated.x, v.integrated.y, v.integrated.z, v.value, filename, line);
}
#endif

struct Spectrums;

struct SpectralDistribution : public ::SpectralDistribution {
  constexpr static const float3 kRGBLuminanceScale = {0.817660332f, 1.05418909f, 1.09945524f};

 public:  // device
  ETX_SHARED_INLINE SpectralResponse query(const SpectralQuery q) const {
    if (q.spectral()) {
      ETX_ASSERT(q.valid());
    }

    SpectrumAccessCPUContext access_context = make_spectrum_access_cpu_context(static_cast<const ::SpectralDistribution*>(this), 1u);
    const ::SpectralResponse shared_response = spectrum_access_evaluate(access_context, 0u, static_cast<const ::SpectralQuery&>(q));
    SpectralQuery response_query{shared_response.wavelength, shared_response.flags};
    SpectralResponse result =
      ::spectral_response_is_spectral(shared_response) ? SpectralResponse{response_query, shared_response.value} : SpectralResponse{response_query, shared_response.integrated};
    ETX_VALIDATE(result);
    return result;
  }

  ETX_SHARED_INLINE SpectralResponse operator()(const SpectralQuery q) const {
    return query(q);
  }

  ETX_SHARED_INLINE bool empty() const {
    return spectral_entry_count == 0;
  }

  ETX_SHARED_INLINE bool is_zero() const {
    for (uint32_t i = 0; i < spectral_entry_count; ++i) {
      if (spectral_entries[i].power != 0.0f) {
        return false;
      }
    }
    return true;
  }

 public:
  SpectralDistribution() = default;

  void scale(float factor);

  float3 integrate_to_xyz() const;

  const float3& integrated() const;

  float luminance() const;
  float maximum_spectral_power() const;

  bool valid() const;

  static SpectralDistribution from_samples(const float2 wavelengths_power[], uint64_t count);

  static SpectralDistribution constant(float value);
  static SpectralDistribution from_black_body(float temperature, float scale);
  static SpectralDistribution from_normalized_black_body(float temperature, float scale);
  static SpectralDistribution rgb_reflectance(const float3& rgb);
  static SpectralDistribution rgb_luminance(const float3& rgb);

  static SpectralDistribution::Class load_from_file(const char* file_name, SpectralDistribution& values0, SpectralDistribution* values1, bool extend_range, std::string& out_title);
  static SpectralDistribution::Class load_refractive_index(const char* file_name, SpectralDistribution& eta, SpectralDistribution& k, std::string& out_title);

 private:
  friend struct RefractiveIndex;
};

using RefractiveIndex = ::RefractiveIndex;

struct RefractiveIndexSample : public ::RefractiveIndexSample {
  ETX_SHARED_INLINE complex as_complex_x() const {
    ETX_ASSERT((spectral_response_is_spectral(eta) == false) && (spectral_response_is_spectral(k) == false));
    return complex{eta.integrated.x, k.integrated.x};
  }

  ETX_SHARED_INLINE complex as_complex_y() const {
    ETX_ASSERT((spectral_response_is_spectral(eta) == false) && (spectral_response_is_spectral(k) == false));
    return {eta.integrated.y, k.integrated.y};
  }

  ETX_SHARED_INLINE complex as_complex_z() const {
    ETX_ASSERT((spectral_response_is_spectral(eta) == false) && (spectral_response_is_spectral(k) == false));
    return {eta.integrated.z, k.integrated.z};
  }

  ETX_SHARED_INLINE complex as_complex() const {
    ETX_ASSERT((spectral_response_is_spectral(eta) == false) && (spectral_response_is_spectral(k) == false));
    return {eta.value, k.value};
  }

  ETX_SHARED_INLINE complex as_monochromatic_complex() const {
    return {spectral_response_monochromatic(eta), spectral_response_monochromatic(k)};
  }
};

SpectralResponse rgb_response(const SpectralQuery spect, const float3& rgb);

void init(Spectrums&);

}  // namespace etx

