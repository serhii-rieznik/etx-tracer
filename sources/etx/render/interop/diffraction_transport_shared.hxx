#pragma once

#include "spectrum.hxx"

// RGB transport cannot represent the wavelength-dependent direction of a
// diffraction event.  RGB scenes therefore use two disjoint Monte Carlo
// branches: integrated RGB paths contribute only while they contain no
// diffraction event, and wavelength-resolved paths contribute only after a
// diffraction event.  Their sum is unbiased for the renderer's RGB transport
// with exact wavelength-resolved diffraction, without splitting a path.
ETX_STATIC_CONST float kDiffractionRGBSpectralBranchProbability = 0.5f;

ETX_SHARED_INLINE bool diffraction_transport_partition_enabled(bool scene_spectral, bool scene_has_diffraction_grating) {
  return (scene_spectral == false) && scene_has_diffraction_grating;
}

ETX_SHARED_INLINE SpectralQuery diffraction_transport_sample_query(bool scene_spectral, bool scene_has_diffraction_grating, float branch_sample, float wavelength_sample) {
  if (scene_spectral) {
    return spectral_query_spectral_sample(wavelength_sample);
  }
  if (diffraction_transport_partition_enabled(scene_spectral, scene_has_diffraction_grating) &&
      (branch_sample < kDiffractionRGBSpectralBranchProbability)) {
    return spectral_query_spectral_sample(wavelength_sample);
  }
  return spectral_query_sample();
}

ETX_SHARED_INLINE float diffraction_transport_branch_pdf(bool partition_enabled, ETX_IN(SpectralQuery, query)) {
  if (partition_enabled == false) {
    return 1.0f;
  }
  return spectral_query_is_spectral(query) ? kDiffractionRGBSpectralBranchProbability : (1.0f - kDiffractionRGBSpectralBranchProbability);
}

ETX_SHARED_INLINE bool diffraction_transport_contribution_enabled(bool partition_enabled, ETX_IN(SpectralQuery, query), bool contains_diffraction) {
  return (partition_enabled == false) || (spectral_query_is_spectral(query) == contains_diffraction);
}
