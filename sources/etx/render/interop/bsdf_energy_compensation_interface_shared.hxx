#pragma once

#include "material.hxx"

struct BSDFEnergyCompensationInterfaceData {
  uint32_t cls ETX_INIT(MaterialClass::Undefined);
  uint32_t cache_mode ETX_INIT(0u);
  uint32_t spectral_wavelength_count ETX_INIT(0u);
  uint32_t thinfilm_slice_count ETX_INIT(1u);
  uint32_t directional_lut ETX_INIT(kInvalidIndex);
  uint32_t average_lut ETX_INIT(kInvalidIndex);
  uint32_t geometric_lut ETX_INIT(kInvalidIndex);
  uint32_t geometric_average_lut ETX_INIT(kInvalidIndex);
  uint32_t conductor_fms_lut ETX_INIT(kInvalidIndex);
  uint32_t probability_lut ETX_INIT(kInvalidIndex);
  float spectral_shortest_wavelength ETX_INIT(0.0f);
  float spectral_longest_wavelength ETX_INIT(0.0f);
};
