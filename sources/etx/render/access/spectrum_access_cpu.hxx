#pragma once

#include <etx/render/interop/spectrum.hxx>

namespace etx {

struct SpectrumAccessCPUContext {
  const ::SpectralDistribution* spectrums = nullptr;
  uint32_t spectrum_count = 0u;
};

ETX_SHARED_INLINE SpectrumAccessCPUContext make_spectrum_access_cpu_context(const ::SpectralDistribution* spectrums, uint32_t spectrum_count) {
  SpectrumAccessCPUContext result = {};
  result.spectrums = spectrums;
  result.spectrum_count = spectrum_count;
  return result;
}

ETX_SHARED_INLINE bool spectrum_access_can_evaluate(ETX_IN(SpectrumAccessCPUContext, context), uint32_t spectrum_index) {
  return spectrum_index < context.spectrum_count;
}

ETX_SHARED_INLINE const ::SpectralDistribution& spectrum_access_cpu_distribution(ETX_IN(SpectrumAccessCPUContext, context), uint32_t spectrum_index) {
  ETX_ASSERT(spectrum_index < context.spectrum_count);
  return context.spectrums[spectrum_index];
}

ETX_SHARED_INLINE float3 spectrum_access_cpu_integrated(ETX_IN(SpectrumAccessCPUContext, context), uint32_t spectrum_index) {
  return spectrum_access_cpu_distribution(context, spectrum_index).integrated_value;
}

ETX_SHARED_INLINE uint32_t spectrum_access_cpu_entry_count(ETX_IN(SpectrumAccessCPUContext, context), uint32_t spectrum_index) {
  return spectrum_access_cpu_distribution(context, spectrum_index).spectral_entry_count;
}

ETX_SHARED_INLINE float spectrum_access_cpu_entry_wavelength(ETX_IN(SpectrumAccessCPUContext, context), uint32_t spectrum_index, uint32_t entry_index) {
  const auto& distribution = spectrum_access_cpu_distribution(context, spectrum_index);
  ETX_ASSERT(entry_index < distribution.spectral_entry_count);
  return distribution.spectral_entries[entry_index].wavelength;
}

ETX_SHARED_INLINE float spectrum_access_cpu_entry_power(ETX_IN(SpectrumAccessCPUContext, context), uint32_t spectrum_index, uint32_t entry_index) {
  const auto& distribution = spectrum_access_cpu_distribution(context, spectrum_index);
  ETX_ASSERT(entry_index < distribution.spectral_entry_count);
  return distribution.spectral_entries[entry_index].power;
}

ETX_SHARED_INLINE float3 spectrum_access_load_integrated(ETX_IN(SpectrumAccessCPUContext, context), uint32_t spectrum_index) {
  if (spectrum_access_can_evaluate(context, spectrum_index) == false) {
    return make_float3(0.0f, 0.0f, 0.0f);
  }

  return spectrum_access_cpu_integrated(context, spectrum_index);
}

ETX_SHARED_INLINE ::SpectralResponse spectrum_access_evaluate(ETX_IN(SpectrumAccessCPUContext, context), uint32_t spectrum_index, ETX_IN(::SpectralQuery, query)) {
  if (spectrum_access_can_evaluate(context, spectrum_index) == false) {
    return ::spectral_response_zero(query);
  }

  if (::spectral_query_is_spectral(query) == false) {
    return ::spectral_response_make(query, spectrum_access_cpu_integrated(context, spectrum_index));
  }

  uint32_t entry_count = spectrum_access_cpu_entry_count(context, spectrum_index);
  if (entry_count == 0u) {
    return ::spectral_response_make(query, 0.0f);
  }

  uint32_t begin = 0u;
  uint32_t end = entry_count;
  while ((end - begin) > 1u) {
    uint32_t middle = begin + ((end - begin) / 2u);
    float middle_wavelength = spectrum_access_cpu_entry_wavelength(context, spectrum_index, middle);
    if (middle_wavelength > query.wavelength) {
      end = middle;
    } else {
      begin = middle;
    }
  }

  uint32_t i = begin;
  if (i >= entry_count) {
    return ::spectral_response_make(query, 0.0f);
  }

  float wi = spectrum_access_cpu_entry_wavelength(context, spectrum_index, i);
  if ((i == 0u) && (query.wavelength < wi)) {
    return ::spectral_response_make(query, 0.0f);
  }
  if (((i + 1u) == entry_count) && (query.wavelength > wi)) {
    return ::spectral_response_make(query, 0.0f);
  }

  uint32_t j = min(i + 1u, entry_count - 1u);
  float wj = spectrum_access_cpu_entry_wavelength(context, spectrum_index, j);
  float pi = spectrum_access_cpu_entry_power(context, spectrum_index, i);
  float pj = spectrum_access_cpu_entry_power(context, spectrum_index, j);
  float t = (i == j) ? 0.0f : ((query.wavelength - wi) / (wj - wi));
  return ::spectral_response_make(query, lerp(pi, pj, t));
}

ETX_SHARED_INLINE uint32_t spectrum_access_entry_count(ETX_IN(SpectrumAccessCPUContext, context), uint32_t spectrum_index) {
  if (spectrum_access_can_evaluate(context, spectrum_index) == false) {
    return 0u;
  }

  return spectrum_access_cpu_entry_count(context, spectrum_index);
}

ETX_SHARED_INLINE float spectrum_access_entry_wavelength(ETX_IN(SpectrumAccessCPUContext, context), uint32_t spectrum_index, uint32_t entry_index) {
  uint32_t entry_count = spectrum_access_entry_count(context, spectrum_index);
  if (entry_index >= entry_count) {
    return 0.0f;
  }

  return spectrum_access_cpu_entry_wavelength(context, spectrum_index, entry_index);
}

ETX_SHARED_INLINE float spectrum_access_entry_power(ETX_IN(SpectrumAccessCPUContext, context), uint32_t spectrum_index, uint32_t entry_index) {
  uint32_t entry_count = spectrum_access_entry_count(context, spectrum_index);
  if (entry_index >= entry_count) {
    return 0.0f;
  }

  return spectrum_access_cpu_entry_power(context, spectrum_index, entry_index);
}

}  // namespace etx
