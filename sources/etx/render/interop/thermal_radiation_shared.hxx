#pragma once

#include "interop_base.hxx"

ETX_SHARED_INLINE float thermal_blackbody_radiance_nm(float wavelength_nm, float temperature_kelvin) {
  if (temperature_kelvin == 0.0f) {
    return 0.0f;
  }

  // Vacuum spectral radiance in W / (m^2 sr nm), with wavelength evaluated in micrometres.
  const float wavelength_um = wavelength_nm * 0.001f;
  const float wavelength_squared = wavelength_um * wavelength_um;
  const float wavelength_fifth = wavelength_squared * wavelength_squared * wavelength_um;
  const float exponent = (14387.76877f / wavelength_um) / temperature_kelvin;
  if (exponent < 0.1f) {
    const float denominator = exponent * (1.0f + exponent * (0.5f + exponent * (1.0f / 6.0f + exponent * (1.0f / 24.0f + exponent / 120.0f))));
    return 119104.29724f / (wavelength_fifth * denominator);
  }

  if (exponent > 80.0f) {
    return exp(log(119104.29724f / wavelength_fifth) - exponent);
  }

  const float inverse_exponential = exp(-exponent);
  return (119104.29724f * inverse_exponential) / (wavelength_fifth * (1.0f - inverse_exponential));
}

ETX_SHARED_INLINE float thermal_medium_segment_factor(float extinction, float distance) {
  if (extinction == 0.0f) {
    return distance;
  }

  const float optical_depth = extinction * distance;
  if (optical_depth < 0.1f) {
    return distance * (1.0f - optical_depth * (0.5f - optical_depth * (1.0f / 6.0f - optical_depth * (1.0f / 24.0f - optical_depth / 120.0f))));
  }
  return (1.0f - exp(-optical_depth)) / extinction;
}
