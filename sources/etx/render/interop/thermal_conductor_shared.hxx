#pragma once

#include "bsdf_conductor_source_shared.hxx"
#include "scene_math_shared.hxx"

ETX_SHARED_NOINLINE SpectralResponse thermal_conductor_radiance(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material), ETX_IN(float2, uv),
  float geometric_cosine, float shading_cosine, ETX_INOUT(Sampler, sampler)) {
  if ((shading_cosine <= kEpsilon) || ((material.two_sided == 0u) && (geometric_cosine <= 0.0f))) {
    return spectral_response_zero(spect);
  }
  const SpectralResponse emission = bsdf_resource_apply_image(context, spect, material.emission, uv);
  if (spectral_response_is_zero(emission) && (material.coating.thermal_spectrum_index == kInvalidIndex)) {
    return spectral_response_zero(spect);
  }
  const BSDFConductorSubstrateEscape escape = bsdf_conductor_substrate_escape(context, spect, material, uv, shading_cosine, sampler);
  const float exponent = scene_math_shared_collimation_to_exponent(material.emission_collimation);
  const SpectralResponse authored =
    spectral_response_mul(spectral_response_mul(escape.source, emission), scene_math_shared_collimated_emission_scale(abs(geometric_cosine), exponent));
  const SpectralResponse thermal = spectral_response_mul(escape.absorption, bsdf_resource_load_spectrum(context, material.coating.thermal_spectrum_index, spect));
  return spectral_response_add(authored, thermal);
}
