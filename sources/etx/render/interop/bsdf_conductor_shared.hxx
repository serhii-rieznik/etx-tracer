#pragma once

#include "bsdf_fresnel_shared.hxx"
#include "bsdf_resource_shared.hxx"

ETX_SHARED_INLINE bool bsdf_conductor_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler));
ETX_SHARED_INLINE bool bsdf_conductor_is_delta_with_context(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, tex));

ETX_SHARED_INLINE bool bsdf_conductor_has_finite_coating(ETX_IN(Material, material)) {
#if ETX_ENABLE_COHERENT_COATING
  return (material.cls == MaterialClass::Conductor) && (material.coherent_coating.ior.cls == SpectralDistribution::Conductor);
#else
  return false;
#endif
}

ETX_SHARED_INLINE SpectralResponse bsdf_conductor_fresnel(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), uint32_t interface_index, ETX_IN(SpectralQuery, spect),
  float cosine, ETX_IN(RefractiveIndexSample, exterior), ETX_IN(RefractiveIndexSample, interior), ETX_IN(ThinfilmEval, optics));

ETX_SHARED_INLINE SpectralResponse bsdf_conductor_tinted_fresnel(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(SpectralQuery, spect), float cosine,
  ETX_IN(RefractiveIndexSample, exterior), ETX_IN(RefractiveIndexSample, interior), ETX_IN(ThinfilmEval, optics), ETX_IN(SpectralResponse, reflectance)) {
  const SpectralResponse metal =
    spectral_response_mul(bsdf_conductor_fresnel(context, material, material.energy_compensation_interface_index, spect, cosine, exterior, interior, optics), reflectance);
#if ETX_ENABLE_COHERENT_COATING
  if ((optics.coating_ior.cls == SpectralDistribution::Conductor) && bsdf_resource_interface_has_layers(optics)) {
    RefractiveIndexSample black = exterior;
    black.cls = SpectralDistribution::Conductor;
    black.k = spectral_response_zero(spect);
    const SpectralResponse black_reflection =
      bsdf_conductor_fresnel(context, material, material.coating.energy_compensation_interface_index, spect, cosine, exterior, black, optics);
    return spectral_response_add(metal, spectral_response_mul(black_reflection, spectral_response_sub(spectral_response_make(spect, 1.0f), reflectance)));
  }
#endif
  return metal;
}

ETX_SHARED_INLINE SpectralResponse bsdf_conductor_delta_weight(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material),
  ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, thinfilm)) {
  LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  float3 w_i = local_frame_to_local(frame, -data.w_i);
  if (w_i.z <= kEpsilon) {
    return spectral_response_make(data.spectrum_sample, 0.0f);
  }

  const SpectralResponse reflectance = bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex);
  return bsdf_conductor_tinted_fresnel(context, material, data.spectrum_sample, w_i.z, ext_ior, int_ior, thinfilm, reflectance);
}

ETX_SHARED_INLINE float3 bsdf_conductor_delta_reflect(ETX_IN(BSDFData, data), ETX_IN(Material, material)) {
  LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  return normalize(reflect(data.w_i, frame.nrm));
}

ETX_SHARED_INLINE bool bsdf_conductor_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler)) {
#if (ETX_CPP)
  (void)tex;
  (void)sampler;
#endif
  float2 roughness = float2(material.roughness.value.x, material.roughness.value.y);
  return max(roughness.x, roughness.y) <= kDeltaAlphaTreshold;
}

ETX_SHARED_INLINE bool bsdf_conductor_is_delta_with_context(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, tex)) {
  float2 roughness = bsdf_resource_evaluate_roughness(context, material, tex);
  return max(roughness.x, roughness.y) <= kDeltaAlphaTreshold;
}

ETX_SHARED_INLINE SpectralResponse bsdf_conductor_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
#if (ETX_CPP)
  (void)sampler;
#endif
  return bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex);
}
