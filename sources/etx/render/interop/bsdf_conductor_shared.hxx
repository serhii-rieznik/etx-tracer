#pragma once

#include "bsdf_fresnel_shared.hxx"
#include "bsdf_resource_shared.hxx"

ETX_SHARED_INLINE bool bsdf_conductor_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler));
ETX_SHARED_INLINE bool bsdf_conductor_is_delta_with_context(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, tex));

ETX_SHARED_INLINE SpectralResponse bsdf_conductor_delta_weight(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material),
  ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, thinfilm)) {
  LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  float3 w_i = local_frame_to_local(frame, -data.w_i);
  if (w_i.z <= kEpsilon) {
    return spectral_response_make(data.spectrum_sample, 0.0f);
  }

  SpectralResponse fresnel = bsdf_fresnel_calculate(data.spectrum_sample, w_i.z, ext_ior, int_ior, thinfilm);
  SpectralResponse reflectance = bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex);
  return spectral_response_mul(fresnel, reflectance);
}

ETX_SHARED_INLINE float3 bsdf_conductor_delta_reflect(ETX_IN(BSDFData, data), ETX_IN(Material, material)) {
  LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  return normalize(reflect(data.w_i, frame.nrm));
}

ETX_SHARED_INLINE bool bsdf_conductor_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler)) {
  (void)tex;
  (void)sampler;
  float2 roughness = float2(material.roughness.value.x, material.roughness.value.y);
  return max(roughness.x, roughness.y) <= kDeltaAlphaTreshold;
}

ETX_SHARED_INLINE bool bsdf_conductor_is_delta_with_context(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, tex)) {
  float2 roughness = bsdf_resource_evaluate_roughness(context, material, tex);
  return max(roughness.x, roughness.y) <= kDeltaAlphaTreshold;
}

ETX_SHARED_INLINE SpectralResponse bsdf_conductor_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)sampler;
  return bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex);
}
