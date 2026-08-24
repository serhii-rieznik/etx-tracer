#pragma once

#include <interop/gpu_abi_constants.hxx>
#include <interop/spectrum.hxx>

struct SpectrumAccessGPUContext {
  ByteAddressBuffer buffer;
  uint descriptor_index;
};

float3 spectrum_access_gpu_integrated(SpectrumAccessGPUContext context, uint spectrum_index) {
  uint base_offset = spectrum_index * kSpectralDistributionStride;
  return asfloat(context.buffer.Load3(base_offset + kSpectralDistributionIntegratedOffset));
}

uint spectrum_access_gpu_entry_count_internal(SpectrumAccessGPUContext context, uint spectrum_index) {
  uint base_offset = spectrum_index * kSpectralDistributionStride;
  return context.buffer.Load(base_offset + kSpectralDistributionEntryCountOffset);
}

float spectrum_access_gpu_entry_wavelength_internal(SpectrumAccessGPUContext context, uint spectrum_index, uint entry_index) {
  uint base_offset = spectrum_index * kSpectralDistributionStride + kSpectralDistributionEntriesOffset + entry_index * kSpectralDistributionEntryStride;
  return asfloat(context.buffer.Load(base_offset + 0u));
}

float spectrum_access_gpu_entry_power_internal(SpectrumAccessGPUContext context, uint spectrum_index, uint entry_index) {
  uint base_offset = spectrum_index * kSpectralDistributionStride + kSpectralDistributionEntriesOffset + entry_index * kSpectralDistributionEntryStride;
  return asfloat(context.buffer.Load(base_offset + 4u));
}

SpectrumAccessGPUContext make_spectrum_access_gpu_context(ByteAddressBuffer buffer, uint descriptor_index) {
  SpectrumAccessGPUContext result;
  result.buffer = buffer;
  result.descriptor_index = descriptor_index;
  return result;
}

bool spectrum_access_can_evaluate(SpectrumAccessGPUContext context, uint spectrum_index) {
  return (context.descriptor_index != kInvalidIndex) && (spectrum_index != kInvalidIndex);
}

float3 spectrum_access_load_integrated(SpectrumAccessGPUContext context, uint spectrum_index) {
  if (spectrum_access_can_evaluate(context, spectrum_index) == false) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  return spectrum_access_gpu_integrated(context, spectrum_index);
}

float spectrum_access_evaluate_wavelength(SpectrumAccessGPUContext context, uint spectrum_index, float wavelength) {
  uint entry_count = spectrum_access_gpu_entry_count_internal(context, spectrum_index);
  if (entry_count == 0u) {
    return 0.0f;
  }

#if ETX_SPECTRAL_MODE != ETX_SPECTRAL_MODE_RUNTIME
  if (entry_count == WavelengthCount) {
    if ((wavelength < kShortestWavelength) || (wavelength > kLongestWavelength)) {
      return 0.0f;
    }

    const float wavelength_position = wavelength - kShortestWavelength;
    const float wavelength_floor = floor(wavelength_position);
    const uint i = min(uint(wavelength_floor), WavelengthCount - 1u);
    const uint j = min(i + 1u, WavelengthCount - 1u);
    const float pi = spectrum_access_gpu_entry_power_internal(context, spectrum_index, i);
    const float pj = spectrum_access_gpu_entry_power_internal(context, spectrum_index, j);
    return lerp(pi, pj, wavelength_position - wavelength_floor);
  }
#endif

  uint begin = 0u;
  uint end = entry_count;
  while ((end - begin) > 1u) {
    uint middle = begin + ((end - begin) / 2u);
    float middle_wavelength = spectrum_access_gpu_entry_wavelength_internal(context, spectrum_index, middle);
    if (middle_wavelength > wavelength) {
      end = middle;
    } else {
      begin = middle;
    }
  }

  uint i = begin;
  if (i >= entry_count) {
    return 0.0f;
  }

  float wi = spectrum_access_gpu_entry_wavelength_internal(context, spectrum_index, i);
  if ((i == 0u) && (wavelength < wi)) {
    return 0.0f;
  }
  if (((i + 1u) == entry_count) && (wavelength > wi)) {
    return 0.0f;
  }

  uint j = min(i + 1u, entry_count - 1u);
  float wj = spectrum_access_gpu_entry_wavelength_internal(context, spectrum_index, j);
  float pi = spectrum_access_gpu_entry_power_internal(context, spectrum_index, i);
  float pj = spectrum_access_gpu_entry_power_internal(context, spectrum_index, j);
  float t = (i == j) ? 0.0f : ((wavelength - wi) / (wj - wi));
  return lerp(pi, pj, t);
}

SpectralResponse spectrum_access_evaluate(SpectrumAccessGPUContext context, uint spectrum_index, SpectralQuery spect) {
  if (spectrum_access_can_evaluate(context, spectrum_index) == false) {
    return spectral_response_zero(spect);
  }

  if (spectral_query_is_spectral(spect) == false) {
    return spectral_response_make(spect, spectrum_access_gpu_integrated(context, spectrum_index));
  }

  SpectralResponse result = spectral_response_make(spect, spectrum_access_evaluate_wavelength(context, spectrum_index, spect.wavelength));
  return result;
}

uint spectrum_access_entry_count(SpectrumAccessGPUContext context, uint spectrum_index) {
  if (spectrum_access_can_evaluate(context, spectrum_index) == false) {
    return 0u;
  }

  return spectrum_access_gpu_entry_count_internal(context, spectrum_index);
}

float spectrum_access_entry_wavelength(SpectrumAccessGPUContext context, uint spectrum_index, uint entry_index) {
  uint entry_count = spectrum_access_entry_count(context, spectrum_index);
  if (entry_index >= entry_count) {
    return 0.0f;
  }

  return spectrum_access_gpu_entry_wavelength_internal(context, spectrum_index, entry_index);
}

float spectrum_access_entry_power(SpectrumAccessGPUContext context, uint spectrum_index, uint entry_index) {
  uint entry_count = spectrum_access_entry_count(context, spectrum_index);
  if (entry_index >= entry_count) {
    return 0.0f;
  }

  return spectrum_access_gpu_entry_power_internal(context, spectrum_index, entry_index);
}
