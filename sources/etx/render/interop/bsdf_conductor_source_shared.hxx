#pragma once

#include "bsdf_energy_compensated_shared.hxx"
#include "bsdf_coherent_source_shared.hxx"

struct BSDFConductorSubstrateEscape {
  SpectralResponse source ETX_INIT({});
  SpectralResponse absorption ETX_INIT({});
};

// Preparation requires an isotropic or delta surface; its source tables share the reflection model.
ETX_SHARED_NOINLINE BSDFConductorSubstrateEscape bsdf_conductor_substrate_escape(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  ETX_IN(float2, uv), float cosine, ETX_INOUT(Sampler, sampler)) {
  const RefractiveIndexSample exterior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, spect);
  const RefractiveIndexSample interior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, spect);
  RefractiveIndexSample black = exterior;
  black.cls = SpectralDistribution::Conductor;
  black.k = spectral_response_zero(spect);
  const InterfaceOptics optics = bsdf_resource_evaluate_interface_optics(context, spect, material, uv, sampler);
  const SpectralResponse tint = bsdf_resource_apply_image(context, spect, material.reflectance, uv);
  const SpectralResponse black_weight = spectral_response_sub(spectral_response_make(spect, 1.0f), tint);
  const float2 roughness = bsdf_resource_evaluate_roughness(context, material, uv);
  const float mu = saturate(cosine);
  const bool has_layers = bsdf_resource_interface_has_layers(optics);
  SpectralResponse metal_source;
  SpectralResponse metal_escape;
  SpectralResponse black_escape;
  if (max(roughness.x, roughness.y) <= kDeltaAlphaTreshold) {
    const CoherentInterfaceEval coherent = bsdf_fresnel_coherent_interface(optics);
    metal_source = bsdf_coherent_source_ratio(spect, mu, exterior, interior, coherent);
    metal_escape = bsdf_fresnel_terminal_source_calculate(spect, mu, exterior, interior, coherent);
    black_escape = bsdf_fresnel_terminal_source_calculate(spect, mu, exterior, black, coherent);
  } else {
    const float alpha = bsdf_energy_compensated_scalar_roughness_from_value(roughness);
    const float2 lut_value = bsdf_energy_compensated_interface_lut_value(material, optics);
    metal_escape = bsdf_energy_compensated_conductor_terminal_source(context, spect, material.energy_compensation_interface_index, mu, alpha, lut_value);
    black_escape = bsdf_energy_compensated_conductor_terminal_source(context, spect, material.coating.energy_compensation_interface_index, mu, alpha, lut_value);
    const SpectralResponse bare_escape =
      bsdf_energy_compensated_conductor_terminal_source(context, spect, material.conductor_energy_compensation_interface_index, mu, alpha, float2(0.0f, 0.0f));
    metal_source = spectral_response_make(spect, 1.0f);
    if (has_layers) {
      metal_source = spectral_response_div(metal_escape, bare_escape);
    }
  }
  SpectralResponse black_absorption = spectral_response_make(spect, 1.0f);
  if (has_layers) {
    black_absorption = spectral_response_mul(exterior.eta, black_escape);
  }
  BSDFConductorSubstrateEscape result = ETX_ZERO(BSDFConductorSubstrateEscape);
  // Authored emission is the prescribed bare-substrate radiance; thermal emission uses only terminal absorption.
  result.source = spectral_response_add(spectral_response_mul(tint, metal_source), spectral_response_mul(black_weight, black_absorption));
  result.absorption = spectral_response_add(spectral_response_mul(tint, spectral_response_mul(interior.eta, metal_escape)), spectral_response_mul(black_weight, black_absorption));
  return result;
}
