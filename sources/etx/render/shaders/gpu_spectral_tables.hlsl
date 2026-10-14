#pragma once

#include <interop/spectrum.hxx>

#if defined(ETX_SPECTRAL_TABLES_EXTERNAL)
uint spectral_lookup_byte_offset(uint index) {
  uint byte_count = 0u;
  bindless_buffers[NonUniformResourceIndex(constants.scene.spectrums)].GetDimensions(byte_count);
  return (byte_count - kSpectralLookupBufferSize) + index * kSpectralLookupRowStride;
}

float3 spectral_xyz(uint index) {
  return asfloat(bindless_buffers[NonUniformResourceIndex(constants.scene.spectrums)].Load3(spectral_lookup_byte_offset(index)));
}

float spectral_sampling_cdf(uint index) {
  return asfloat(bindless_buffers[NonUniformResourceIndex(constants.scene.spectrums)].Load(spectral_lookup_byte_offset(index) + 12u));
}

float3 spectral_rgb_basis(uint index) {
  return asfloat(bindless_buffers[NonUniformResourceIndex(constants.scene.spectrums)].Load3(spectral_lookup_byte_offset(WavelengthCount + index)));
}
#endif
