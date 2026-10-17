#pragma once

#include "interop.hxx"

ETX_STATIC_CONST uint32_t kBSDFEnergyCompensationConductorLutSize = 64u;
ETX_STATIC_CONST uint32_t kBSDFEnergyCompensationConductorTerminalLutHeight = kBSDFEnergyCompensationConductorLutSize + 1u;
ETX_STATIC_CONST uint32_t kBSDFEnergyCompensationDielectricLutSize = 64u;
ETX_STATIC_CONST uint32_t kBSDFEnergyCompensationDielectricAzimuthCount = 17u;
ETX_STATIC_CONST uint32_t kBSDFEnergyCompensationDielectricBranchCount = 4u;
ETX_STATIC_CONST uint32_t kBSDFEnergyCompensationDielectricAverageWidth = 8u;
ETX_STATIC_CONST uint32_t kBSDFEnergyCompensationCacheModeIntegratedRGB = 0u;
ETX_STATIC_CONST uint32_t kBSDFEnergyCompensationCacheModeSpectralScalar = 1u;
ETX_STATIC_CONST uint32_t kBSDFEnergyCompensationSpectralWavelengthCount = 128u;
ETX_STATIC_CONST uint32_t kBSDFEnergyCompensationSpectralWavelengthGroupSize = 4u;
ETX_STATIC_CONST uint32_t kBSDFEnergyCompensationSpectralWavelengthGroupCount =
  kBSDFEnergyCompensationSpectralWavelengthCount / kBSDFEnergyCompensationSpectralWavelengthGroupSize;
ETX_STATIC_CONST uint32_t kBSDFEnergyCompensationGpuPassConductorDirectional = 0u;
ETX_STATIC_CONST uint32_t kBSDFEnergyCompensationGpuPassConductorAverage = 1u;
ETX_STATIC_CONST uint32_t kBSDFEnergyCompensationGpuPassDielectricDirectional = 2u;
ETX_STATIC_CONST uint32_t kBSDFEnergyCompensationGpuPassDielectricAverage = 3u;
ETX_STATIC_CONST float kBSDFEnergyCompensationF0Max = 9.99000013e-1f;

struct BSDFEnergyCompensationDielectricLut {
  float2 roughness ETX_INIT({1.0f, 1.0f});
  uint32_t azimuth_count ETX_INIT(1u);
  uint32_t roughness_count ETX_INIT(kBSDFEnergyCompensationDielectricLutSize);
};

ETX_SHARED_INLINE float3 bsdf_energy_compensated_dielectric_lut_direction(float mu, uint32_t azimuth_index, uint32_t azimuth_count) {
  const float radius = sqrt(max(0.0f, 1.0f - mu * mu));
  if (azimuth_count == 1u) {
    return float3(radius, 0.0f, mu);
  }
  const float phi = 0.5f * kPi * float(azimuth_index) / float(azimuth_count - 1u);
  return float3(radius * cos(phi), radius * sin(phi), mu);
}

ETX_SHARED_INLINE float bsdf_energy_compensated_dielectric_azimuth_weight(uint32_t index, uint32_t count) {
  if (count == 1u) {
    return 1.0f;
  }
  // Integrate the piecewise linear azimuth lookup over a symmetry quadrant.
  return ((index == 0u) || ((index + 1u) == count) ? 0.5f : 1.0f) / float(count - 1u);
}

ETX_SHARED_INLINE float bsdf_energy_compensated_dielectric_mu_parameter(uint32_t index, uint32_t size) {
  const float axis = float(index) / float(size - 1u);
  return max(2.0f * kEpsilon, axis * axis);
}

ETX_SHARED_INLINE float2 bsdf_energy_compensated_dielectric_average_weights(uint32_t index, uint32_t size) {
  const float h = 1.0f / float(size - 1u);
  const float axis = float(index) * h;
  const float h2 = h * h;
  const float h3 = h2 * h;
  const float h4 = h3 * h;
  // Integrate linear interpolation in sqrt(mu) against the hemispherical measure 2*mu*dmu.
  return float2(2.0f * h * axis * axis * axis + 2.0f * h2 * axis * axis + h3 * axis + h4 / 5.0f,
    2.0f * h * axis * axis * axis + 4.0f * h2 * axis * axis + 3.0f * h3 * axis + 4.0f * h4 / 5.0f);
}
