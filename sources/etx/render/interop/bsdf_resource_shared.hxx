#pragma once

#include "bsdf_core_shared.hxx"

ETX_SHARED_INLINE SpectralResponse bsdf_resource_apply_rgb(ETX_IN(SpectralQuery, spect), ETX_IN(SpectralResponse, response), ETX_IN(float4, value)) {
  if (spectral_query_is_spectral(spect)) {
    SpectralResponse scale = spectral_rgb_response(spect, float3(value.x, value.y, value.z));
    SpectralResponse result = spectral_response_mul(response, scale);
    return result;
  }

  SpectralResponse result = response;
  result.integrated *= float3(value.x, value.y, value.z);
  result.value = luminance(result.integrated);
  return result;
}

ETX_SHARED_INLINE SpectralResponse bsdf_resource_apply_image(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(SpectralImage, image), ETX_IN(float2, uv),
  ETX_OUT(float, image_pdf)) {
  image_pdf = 0.0f;

  SpectralResponse result = bsdf_resource_load_spectrum(context, image.spectrum_index, spect);
  if (image.image_index == kInvalidIndex) {
    return result;
  }

  float4 image_value;
  if (bsdf_resource_image_try_evaluate_rgba(context, image.image_index, uv, image_pdf, image_value) == false) {
    return result;
  }

  return bsdf_resource_apply_rgb(spect, result, image_value);
}

ETX_SHARED_INLINE SpectralResponse bsdf_resource_apply_image(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(SpectralImage, image), ETX_IN(float2, uv)) {
  SpectralResponse result = bsdf_resource_load_spectrum(context, image.spectrum_index, spect);
  if (image.image_index == kInvalidIndex) {
    return result;
  }

  float4 image_value;
  if (bsdf_resource_image_try_evaluate_rgba_no_pdf(context, image.image_index, uv, image_value) == false) {
    return result;
  }

  return bsdf_resource_apply_rgb(spect, result, image_value);
}

ETX_SHARED_INLINE SpectralResponse bsdf_resource_subsurface_boundary_color(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  ETX_IN(float2, uv)) {
  const SpectralResponse unit = spectral_response_make(spect, 1.0f);
  float4 value;
  if ((material.scattering.image_index == kInvalidIndex) || (bsdf_resource_image_try_evaluate_rgba_no_pdf(context, material.scattering.image_index, uv, value) == false)) {
    return unit;
  }
  return spectral_response_saturate(bsdf_resource_apply_rgb(spect, unit, value));
}

ETX_SHARED_INLINE float bsdf_resource_evaluate_sampled_image(ETX_IN(BSDFResourceContext, context), ETX_IN(SampledImage, image), ETX_IN(float2, uv), float default_value) {
  if (image.image_index == kInvalidIndex) {
    return default_value;
  }

  return bsdf_resource_image_sample_channel_or_default(context, image.image_index, image.channel, uv, default_value);
}

ETX_SHARED_INLINE float2 bsdf_resource_evaluate_base_roughness(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, uv)) {
  float scale = bsdf_resource_evaluate_sampled_image(context, material.roughness, uv, 1.0f);
  return float2(material.roughness.value.x, material.roughness.value.y) * scale;
}

ETX_SHARED_INLINE float2 bsdf_resource_evaluate_roughness(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, uv)) {
  if ((material.cls == MaterialClass::Plastic) && (material.layer_base_class == (MaterialClass::Diffuse + 1u))) {
    const float scale = bsdf_resource_evaluate_sampled_image(context, material.coating.roughness, uv, 1.0f);
    return float2(material.coating.roughness.value.x, material.coating.roughness.value.y) * scale;
  }
  return bsdf_resource_evaluate_base_roughness(context, material, uv);
}

ETX_SHARED_INLINE float bsdf_resource_evaluate_metalness(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, uv)) {
  return material.metalness.value.x * bsdf_resource_evaluate_sampled_image(context, material.metalness, uv, 1.0f);
}

ETX_SHARED_INLINE float bsdf_resource_evaluate_transmission(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, uv)) {
  return material.transmission.value.x * bsdf_resource_evaluate_sampled_image(context, material.transmission, uv, 1.0f);
}

ETX_SHARED_INLINE RefractiveIndexSample bsdf_resource_evaluate_refractive_index(ETX_IN(BSDFResourceContext, context), ETX_IN(RefractiveIndex, refractive_index),
  ETX_IN(SpectralQuery, query)) {
  RefractiveIndexSample result = ETX_ZERO(RefractiveIndexSample);
  result.cls = refractive_index.cls;
  if (refractive_index.eta_index == kInvalidIndex) {
    result.eta = spectral_response_make(query, 1.0f);
  } else {
    result.eta = bsdf_resource_load_spectrum(context, refractive_index.eta_index, query);
  }
  if (refractive_index.k_index == kInvalidIndex) {
    result.k = spectral_response_make(query, 0.0f);
  } else {
    result.k = bsdf_resource_load_spectrum(context, refractive_index.k_index, query);
  }
  return result;
}

ETX_SHARED_INLINE bool bsdf_resource_thinfilm_enabled(ETX_IN(Thinfilm, film)) {
  const float maximum_thickness = max(max(film.min_thickness, 0.0f), max(film.max_thickness, 0.0f));
  return (clamp(film.weight, 0.0f, 1.0f) > 0.0f) && (maximum_thickness > 0.0f) &&
         ((film.ior.cls == SpectralDistribution::Dielectric) || (film.ior.cls == SpectralDistribution::Conductor));
}

ETX_SHARED_INLINE float bsdf_resource_evaluate_interface_thickness(ETX_IN(BSDFResourceContext, context), float minimum, float maximum, uint32_t image_index, ETX_IN(float2, uv)) {
  const float value = image_index == kInvalidIndex ? 1.0f : bsdf_resource_image_sample_channel_or_default(context, image_index, 0u, uv, 1.0f);
  const float lower = max(0.0f, minimum);
  const float upper = max(0.0f, maximum);
  return max(0.0f, lower + (upper - lower) * saturate(value));
}

ETX_SHARED_INLINE bool bsdf_resource_interface_has_layers(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, uv)) {
#if ETX_ENABLE_COHERENT_COATING
  if ((material.coherent_coating.max_thickness_nm > 0.0f) && (bsdf_resource_evaluate_interface_thickness(context, material.coherent_coating.min_thickness_nm,
                                                                material.coherent_coating.max_thickness_nm, material.coherent_coating.thickness_image, uv) > 0.0f)) {
    return true;
  }
#endif
  return bsdf_resource_thinfilm_enabled(material.thinfilm) &&
         (bsdf_resource_evaluate_interface_thickness(context, material.thinfilm.min_thickness, material.thinfilm.max_thickness, material.thinfilm.thinkness_image, uv) > 0.0f);
}

ETX_SHARED_INLINE bool bsdf_resource_interface_has_layers(ETX_IN(InterfaceOptics, optics)) {
  return (optics.coating_thickness_nm > 0.0f) || ((optics.weight > 0.0f) && (optics.thickness > 0.0f));
}

ETX_SHARED_INLINE ThinfilmEval bsdf_resource_evaluate_thinfilm(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Thinfilm, film), ETX_IN(float2, uv),
  ETX_INOUT(Sampler, sampler)) {
#if (ETX_CPP)
  (void)sampler;
#endif

  ThinfilmEval result = ETX_ZERO(ThinfilmEval);
  result.ior.cls = SpectralDistribution::Invalid;
  result.rgb_wavelengths = kRGBWavelengths;
  result.thickness = 0.0f;
  result.weight = 0.0f;

  if (bsdf_resource_thinfilm_enabled(film) == false) {
    return result;
  }

  result.thickness = bsdf_resource_evaluate_interface_thickness(context, film.min_thickness, film.max_thickness, film.thinkness_image, uv);
  result.weight = clamp(film.weight, 0.0f, 1.0f);
  result.ior = bsdf_resource_evaluate_refractive_index(context, film.ior, spect);

  return result;
}

ETX_SHARED_INLINE CoherentInterfaceEval bsdf_resource_evaluate_coherent_interface(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Thinfilm, thinfilm),
  ETX_IN(CoherentCoating, coating), ETX_IN(float2, uv), ETX_INOUT(Sampler, sampler)) {
  CoherentInterfaceEval result = ETX_ZERO(CoherentInterfaceEval);
  result.thinfilm = bsdf_resource_evaluate_thinfilm(context, spect, thinfilm, uv, sampler);
  if ((coating.ior.cls != SpectralDistribution::Dielectric) && (coating.ior.cls != SpectralDistribution::Conductor)) {
    return result;
  }
  result.coating_thickness_nm = bsdf_resource_evaluate_interface_thickness(context, coating.min_thickness_nm, coating.max_thickness_nm, coating.thickness_image, uv);
  result.coating_ior = bsdf_resource_evaluate_refractive_index(context, coating.ior, spect);
  return result;
}

ETX_SHARED_INLINE InterfaceOptics bsdf_resource_evaluate_interface_optics(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  ETX_IN(float2, uv), ETX_INOUT(Sampler, sampler)) {
#if ETX_ENABLE_COHERENT_COATING
  const CoherentInterfaceEval optics = bsdf_resource_evaluate_coherent_interface(context, spect, material.thinfilm, material.coherent_coating, uv, sampler);
  InterfaceOptics result = optics.thinfilm;
  result.coating_ior = optics.coating_ior;
  result.coating_thickness_nm = optics.coating_thickness_nm;
  return result;
#else
  return bsdf_resource_evaluate_thinfilm(context, spect, material.thinfilm, uv, sampler);
#endif
}

#if (ETX_CPP)
ETX_SHARED_INLINE float bsdf_resource_material_coverage(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, uv)) {
  if (material.cls == MaterialClass::Void) {
    return 0.0f;
  }
  const float alpha_mask = bsdf_resource_evaluate_sampled_image(context, material.alpha_mask, uv, 1.0f);
  return saturate(alpha_mask * material.alpha_mask.value.x * material.opacity);
}

ETX_SHARED_INLINE bool bsdf_alpha_test_pass(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, uv), ETX_INOUT(Sampler, sampler)) {
  const float alpha_test_value = bsdf_resource_material_coverage(context, material, uv);
  if (alpha_test_value <= 0.0f) {
    return true;
  }
  if (alpha_test_value >= 1.0f) {
    return false;
  }
  return alpha_test_value <= bsdf_sampler_next(sampler);
}
#endif
