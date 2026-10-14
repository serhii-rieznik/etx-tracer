#pragma once

#include "bsdf_energy_compensated_shared.hxx"
#include "bsdf_various_shared.hxx"

#ifndef ETX_ENABLE_COATED_VELVET_SUBSTRATE
# define ETX_ENABLE_COATED_VELVET_SUBSTRATE 1
#endif

#if (ETX_CPP == 1) || ETX_ENABLE_COATED_VELVET_SUBSTRATE
# include "bsdf_velvet_shared.hxx"
#endif

ETX_SHARED_INLINE bool bsdf_coating_substrate_is_delta(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, tex)) {
  if (material.cls == MaterialClass::Dielectric) {
    return bsdf_dielectric_energy_compensated_is_delta_with_context(context, material, tex);
  }
  if (material.cls == MaterialClass::Conductor) {
    return bsdf_conductor_energy_compensated_is_delta_with_context(context, material, tex);
  }
  return material.cls == MaterialClass::Mirror;
}

ETX_SHARED_INLINE BSDFSample bsdf_coating_substrate_sample_non_velvet(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  switch (material.cls) {
    case MaterialClass::Conductor:
      return bsdf_conductor_energy_compensated_sample(context, data, material, sampler);
    case MaterialClass::Diffuse:
      return bsdf_diffuse_sample(context, data, material, sampler);
    case MaterialClass::Mirror:
      return bsdf_mirror_sample(context, data, material, sampler);
    default:
      return bsdf_sample_zero(data.spectrum_sample);
  }
}

ETX_SHARED_INLINE BSDFSample bsdf_coating_substrate_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
#if (ETX_CPP == 1) || ETX_ENABLE_COATED_VELVET_SUBSTRATE
  if (material.cls == MaterialClass::Velvet) {
    return bsdf_velvet_sample(context, data, material, sampler);
  }
#endif
  return bsdf_coating_substrate_sample_non_velvet(context, data, material, sampler);
}

ETX_SHARED_INLINE BSDFEval bsdf_coating_substrate_evaluate_non_velvet(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing),
  ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  switch (material.cls) {
    case MaterialClass::Conductor:
      return bsdf_conductor_energy_compensated_evaluate(context, data, outgoing, material, sampler);
    case MaterialClass::Diffuse:
      return bsdf_diffuse_evaluate(context, data, outgoing, material, sampler);
    default:
      return bsdf_eval_zero(data.spectrum_sample);
  }
}

ETX_SHARED_INLINE BSDFEval bsdf_coating_substrate_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
#if (ETX_CPP == 1) || ETX_ENABLE_COATED_VELVET_SUBSTRATE
  if (material.cls == MaterialClass::Velvet) {
    return bsdf_velvet_evaluate(context, data, outgoing, material, sampler);
  }
#endif
  return bsdf_coating_substrate_evaluate_non_velvet(context, data, outgoing, material, sampler);
}

ETX_SHARED_INLINE float bsdf_coating_substrate_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  switch (material.cls) {
    case MaterialClass::Conductor:
      return bsdf_conductor_energy_compensated_pdf(context, data, outgoing, material, sampler);
    case MaterialClass::Diffuse:
      return bsdf_diffuse_pdf(context, data, outgoing, material, sampler);
#if (ETX_CPP == 1) || ETX_ENABLE_COATED_VELVET_SUBSTRATE
    case MaterialClass::Velvet:
      return bsdf_velvet_pdf(context, data, outgoing, material, sampler);
#endif
    default:
      return 0.0f;
  }
}

ETX_SHARED_INLINE SpectralResponse bsdf_coating_substrate_specular_reflectance(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  ETX_IN(float2, tex), float cosine) {
  if (material.cls == MaterialClass::Mirror) {
    return bsdf_resource_apply_image(context, spect, material.scattering, tex);
  }
  if (material.cls != MaterialClass::Conductor) {
    return spectral_response_zero(spect);
  }
  const RefractiveIndexSample exterior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, spect);
  const RefractiveIndexSample interior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, spect);
  const ThinfilmEval no_film = ETX_ZERO(ThinfilmEval);
  return spectral_response_mul(bsdf_fresnel_calculate(spect, cosine, exterior, interior, no_film), bsdf_resource_apply_image(context, spect, material.reflectance, tex));
}
