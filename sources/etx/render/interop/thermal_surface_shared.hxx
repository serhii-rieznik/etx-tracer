#pragma once

#include "thermal_radiation_shared.hxx"
#include "bsdf_energy_compensated_shared.hxx"
#include "bsdf_coated_conductor_shared.hxx"
#include "thermal_conductor_shared.hxx"
#include "bsdf_various_shared.hxx"
#include "bsdf_plastic_shared.hxx"
#include "bsdf_velvet_shared.hxx"
#include "scene_math_shared.hxx"

ETX_STATIC_CONST uint32_t kThermalSurfaceDirectionCount = 256u;

ETX_SHARED_INLINE float3 thermal_surface_local_direction(ETX_IN(float3, normal), ETX_IN(float3, tangent), ETX_IN(float3, bitangent), ETX_IN(float3, outgoing)) {
  float3 n, t, b;
  scene_math_shared_build_sampling_frame(normal, tangent, bitangent, n, t, b);
  if (dot(n, outgoing) < 0.0f) {
    n = -n;
    b = -b;
  }
  return float3(dot(t, outgoing), dot(b, outgoing), dot(n, outgoing));
}
ETX_SHARED_INLINE float3 thermal_surface_local_direction(ETX_IN(Material, material), ETX_IN(float3, normal), ETX_IN(float3, tangent), ETX_IN(float3, bitangent),
  ETX_IN(float3, outgoing)) {
  if ((material.cls == MaterialClass::CoatedConductor) && bsdf_coating_transmission_base(material_coating_base_class(material))) {
    float3 n, t, b;
    scene_math_shared_build_sampling_frame(normal, tangent, bitangent, n, t, b);
    return float3(dot(t, outgoing), dot(b, outgoing), dot(n, outgoing));
  }
  return thermal_surface_local_direction(normal, tangent, bitangent, outgoing);
}

ETX_SHARED_INLINE uint32_t thermal_surface_emission_medium(ETX_IN(Material, material), float geometric_cosine) {
  return (material_supports_surface_transmission(material) && (geometric_cosine < 0.0f)) ? material.int_medium : material.ext_medium;
}

ETX_SHARED_INLINE float thermal_surface_equilibrium_radiance(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material)) {
  // Camera transport carries reduced radiance L/n^2 across refractive interfaces.
  return thermal_blackbody_radiance_nm(spect.wavelength, material.temperature_kelvin);
}

ETX_SHARED_INLINE float thermal_surface_directional_reflectance(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material), float mu) {
  if ((material.cls == MaterialClass::Diffuse) || (material.cls == MaterialClass::Plastic)) {
    const float albedo = bsdf_resource_load_spectrum(context, material.scattering.spectrum_index, spect).value;
    const float roughness = saturate(0.5f * (material.roughness.value.x + material.roughness.value.y));
    const float e = bsdf_diffuse_eon_directional_albedo(mu, roughness);
    const float e_avg = bsdf_diffuse_eon_average_albedo(roughness);
    const float rho_ms = (albedo * albedo * e_avg) / (1.0f - albedo * (1.0f - e_avg));
    const float diffuse_albedo = albedo * e + rho_ms * (1.0f - e);
    if (material.cls == MaterialClass::Diffuse) {
      return diffuse_albedo;
    }

    Material spectral_material = material;
    spectral_material.energy_compensation_interface_index = material.thermal_energy_compensation_interface_index;
    BSDFData data = ETX_ZERO(BSDFData);
    data.spectrum_sample = spect;
    const float alpha = bsdf_energy_compensated_scalar_roughness_from_value(bsdf_resource_evaluate_roughness(context, material, float2(0.0f, 0.0f)));
    const BSDFPlasticIncidentTerms terms = bsdf_plastic_prepare_incident_terms(context, data, spectral_material, float3(sqrt(max(0.0f, 1.0f - mu * mu)), 0.0f, mu),
      spectral_response_make(spect, albedo), alpha, float2(0.0f, 0.0f));
    const float cached_single_scattering = bsdf_energy_compensated_dielectric_branch_value(context, spect, spectral_material, mu, alpha, true, true, float2(0.0f, 0.0f)).albedo.x;
    const float compensation = bsdf_plastic_external_reflection_albedo(context, spect, spectral_material, mu, alpha, float2(0.0f, 0.0f)).value - cached_single_scattering;
    const float coating_albedo = bsdf_energy_compensated_conductor_prepared_albedo(context, spect, material, mu).x + compensation;
    const float reflectance = bsdf_resource_load_spectrum(context, material.reflectance.spectrum_index, spect).value;
    return reflectance * coating_albedo + terms.diffuse_scale.value * diffuse_albedo;
  }

  Material spectral_material = material;
  spectral_material.energy_compensation_interface_index = material.thermal_energy_compensation_interface_index;
  const float reflectance = bsdf_resource_load_spectrum(context, material.reflectance.spectrum_index, spect).value;
  const float maximum_roughness = max(material.roughness.value.x, material.roughness.value.y);
  if (maximum_roughness <= kDeltaAlphaTreshold) {
    const RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, spect);
    const RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, spect);
    const ThinfilmEval no_film = ETX_ZERO(ThinfilmEval);
    return reflectance * bsdf_fresnel_calculate(spect, mu, ext_ior, int_ior, no_film).value;
  }

  const float alpha = bsdf_energy_compensated_scalar_roughness_from_value(float2(material.roughness.value.x, material.roughness.value.y));
  const float single_scattering = bsdf_energy_compensated_conductor_prepared_albedo(context, spect, material, mu).x;
  const float geometric_albedo = bsdf_energy_compensated_conductor_geometric_directional_albedo(context, spectral_material, mu, alpha);
  const float average_albedo = bsdf_energy_compensated_conductor_geometric_average_albedo(context, spectral_material, alpha);
  const float f_ms = ((1.0f - average_albedo) > kEpsilon) ? bsdf_energy_compensated_conductor_cached_fms(context, spect, spectral_material, alpha).value : 0.0f;
  return reflectance * (single_scattering + f_ms * (1.0f - geometric_albedo));
}

ETX_SHARED_INLINE SpectralResponse thermal_surface_radiance(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material), float mu) {
  if ((material.temperature_kelvin == 0.0f) || (material.thermal_rgb_image_index == kInvalidIndex)) {
    return spectral_response_zero(spect);
  }
  if (spectral_query_is_spectral(spect)) {
    const float absorptivity = max(0.0f, 1.0f - thermal_surface_directional_reflectance(context, spect, material, mu));
    return spectral_response_make(spect, thermal_surface_equilibrium_radiance(context, spect, material) * absorptivity);
  }
  const float2 uv = float2(bsdf_energy_compensated_lut_uv(mu, kThermalSurfaceDirectionCount), 0.0f);
  const float4 rgb = bsdf_resource_image_evaluate_rgba_no_pdf_or_zero(context, material.thermal_rgb_image_index, uv);
  return spectral_response_make(spect, float3(rgb.x, rgb.y, rgb.z));
}

ETX_SHARED_INLINE bool thermal_surface_uses_substrate_escape(ETX_IN(Material, material)) {
  return (material.cls == MaterialClass::CoatedConductor) || (material.cls == MaterialClass::Velvet) || bsdf_conductor_has_finite_coating(material);
}

ETX_SHARED_INLINE float thermal_surface_sampling_exponent(ETX_IN(Material, material)) {
  return ((material.thermal_rgb_image_index != kInvalidIndex) || thermal_surface_uses_substrate_escape(material) || (material.cls == MaterialClass::Mix) ||
           ((material.cls == MaterialClass::Translucent) && (material.coating.thermal_spectrum_index != kInvalidIndex)))
           ? 1.0f
           : scene_math_shared_collimation_to_exponent(material.emission_collimation);
}

ETX_SHARED_INLINE bool thermal_surface_emits_both_sides(ETX_IN(Material, material)) {
  if ((material.cls == MaterialClass::CoatedConductor) && bsdf_coating_transmission_base(material_coating_base_class(material))) {
    return true;
  }
  if ((material.cls == MaterialClass::Translucent) && (material.coating.thermal_spectrum_index != kInvalidIndex)) {
    return true;
  }
  return ((material.thermal_rgb_image_index != kInvalidIndex) || thermal_surface_uses_substrate_escape(material) || (material.cls == MaterialClass::Mix)) &&
         (material.two_sided != 0u);
}

ETX_SHARED_INLINE float thermal_surface_side_probability(ETX_IN(Material, material)) {
  return thermal_surface_emits_both_sides(material) ? 0.5f : 1.0f;
}

ETX_SHARED_INLINE SpectralResponse thermal_surface_combined_radiance(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  ETX_IN(float2, uv), float geometric_cosine, float shading_cosine);

ETX_SHARED_NOINLINE SpectralResponse thermal_surface_combined_radiance(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  ETX_IN(float2, uv), float geometric_cosine, ETX_IN(float3, local_outgoing), ETX_INOUT(Sampler, sampler)) {
  if (thermal_surface_uses_substrate_escape(material) == false) {
    return thermal_surface_combined_radiance(context, spect, material, uv, geometric_cosine, local_outgoing.z);
  }
  if (bsdf_conductor_has_finite_coating(material)) {
    return thermal_conductor_radiance(context, spect, material, uv, geometric_cosine, local_outgoing.z, sampler);
  }
  const bool transmitting = (material.cls == MaterialClass::CoatedConductor) && bsdf_coating_transmission_base(material_coating_base_class(material));
  if (transmitting) {
    if (abs(local_outgoing.z) <= kEpsilon) {
      return spectral_response_zero(spect);
    }
  } else if ((local_outgoing.z <= kEpsilon) || ((material.two_sided == 0u) && (geometric_cosine <= 0.0f))) {
    return spectral_response_zero(spect);
  }
  if (spectral_response_is_zero(bsdf_resource_apply_image(context, spect, material.emission, uv)) && (material.coating.thermal_spectrum_index == kInvalidIndex)) {
    return spectral_response_zero(spect);
  }
  BSDFData data = ETX_ZERO(BSDFData);
  data.spectrum_sample = spect;
  data.tex = uv;
  data.nrm = float3(0.0f, 0.0f, 1.0f);
  data.tan = float3(1.0f, 0.0f, 0.0f);
  data.btn = float3(0.0f, 1.0f, 0.0f);
  data.w_i = -local_outgoing;
  data.path_source = PathSource::Camera;
  if (material.cls == MaterialClass::Velvet) {
    const BSDFVelvetPrepared prepared = bsdf_velvet_prepare(context, data, material);
    const BSDFVelvetSubstrateEscape escape = bsdf_velvet_substrate_escape(context, spect, prepared, local_outgoing, material.emission_collimation, sampler);
    const SpectralResponse authored = spectral_response_mul(escape.source, bsdf_resource_apply_image(context, spect, material.emission, uv));
    const SpectralResponse thermal = spectral_response_mul(escape.absorption, bsdf_resource_load_spectrum(context, material.coating.thermal_spectrum_index, spect));
    return spectral_response_add(authored, thermal);
  }
  const BSDFCoatedConductor layer = bsdf_coated_conductor_prepare(context, data, material);
  const BSDFCoatingSubstrateEscape escape = bsdf_coated_conductor_substrate_escape(context, layer, sampler);
  const SpectralResponse authored = spectral_response_mul(escape.source, bsdf_resource_apply_image(context, spect, material.emission, uv));
  const SpectralResponse thermal = spectral_response_mul(escape.absorption, bsdf_resource_load_spectrum(context, material.coating.thermal_spectrum_index, spect));
  return spectral_response_add(authored, thermal);
}

ETX_SHARED_INLINE SpectralResponse thermal_surface_combined_radiance(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  ETX_IN(float2, uv), float geometric_cosine, float shading_cosine) {
  const float authored_exponent = scene_math_shared_collimation_to_exponent(material.emission_collimation);
  SpectralResponse authored = spectral_response_zero(spect);
  if (geometric_cosine > 0.0f) {
    authored =
      spectral_response_mul(bsdf_resource_apply_image(context, spect, material.emission, uv), scene_math_shared_collimated_emission_scale(geometric_cosine, authored_exponent));
  }
  if ((material.cls == MaterialClass::Translucent) && (material.coating.thermal_spectrum_index != kInvalidIndex)) {
    BSDFData data = ETX_ZERO(BSDFData);
    data.spectrum_sample = spect;
    data.tex = uv;
    const SpectralResponse thermal =
      spectral_response_mul(bsdf_translucent_absorptance(context, data, material), bsdf_resource_load_spectrum(context, material.coating.thermal_spectrum_index, spect));
    return spectral_response_add(authored, thermal);
  }
  return spectral_response_add(authored, thermal_surface_radiance(context, spect, material, saturate(shading_cosine)));
}
