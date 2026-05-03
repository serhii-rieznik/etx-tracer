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

  float4 image_value = float4(1.0f, 1.0f, 1.0f, 1.0f);
  if (bsdf_resource_image_try_evaluate_rgba(context, image.image_index, uv, image_pdf, image_value) == false) {
    return result;
  }

  return bsdf_resource_apply_rgb(spect, result, image_value);
}

ETX_SHARED_INLINE SpectralResponse bsdf_resource_apply_image(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(SpectralImage, image), ETX_IN(float2, uv)) {
  float image_pdf = 0.0f;
  return bsdf_resource_apply_image(context, spect, image, uv, image_pdf);
}

ETX_SHARED_INLINE float bsdf_resource_evaluate_sampled_image(ETX_IN(BSDFResourceContext, context), ETX_IN(SampledImage, image), ETX_IN(float2, uv), float default_value) {
  return bsdf_resource_image_sample_channel_or_default(context, image.image_index, image.channel, uv, default_value);
}

ETX_SHARED_INLINE float2 bsdf_resource_evaluate_roughness(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, uv)) {
  float scale = bsdf_resource_evaluate_sampled_image(context, material.roughness, uv, 1.0f);
  return float2(material.roughness.value.x, material.roughness.value.y) * scale;
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
  return max(film.min_thickness, film.max_thickness) > 0.0f;
}

ETX_SHARED_INLINE ThinfilmEval bsdf_resource_evaluate_thinfilm(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Thinfilm, film), ETX_IN(float2, uv),
  ETX_INOUT(Sampler, sampler)) {
  (void)sampler;

  ThinfilmEval result = ETX_ZERO(ThinfilmEval);
  result.ior.cls = SpectralDistribution::Invalid;
  result.rgb_wavelengths = kRGBWavelengths;
  result.thickness = 0.0f;

  if (bsdf_resource_thinfilm_enabled(film) == false) {
    return result;
  }

  float sampled_thickness = bsdf_resource_image_sample_channel_or_default(context, film.thinkness_image, 0u, uv, 1.0f);
  result.thickness = film.min_thickness + (film.max_thickness - film.min_thickness) * sampled_thickness;
  result.ior = bsdf_resource_evaluate_refractive_index(context, film.ior, spect);

  return result;
}

ETX_SHARED_INLINE bool bsdf_alpha_test_pass(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, uv), ETX_INOUT(Sampler, sampler)) {
  if (material.cls == MaterialClass::Void) {
    return true;
  }

  float material_alpha = material.opacity;
  float alpha_diffuse = 1.0f;
  if ((material.scattering.image_index != kInvalidIndex) && bsdf_resource_image_has_alpha(context, material.scattering.image_index)) {
    alpha_diffuse = bsdf_resource_image_sample_channel_or_default(context, material.scattering.image_index, 3u, uv, 1.0f);
  }

  float alpha_test_value = alpha_diffuse * material_alpha;
  if (alpha_test_value <= 0.0f) {
    return true;
  }
  if (alpha_test_value >= 1.0f) {
    return false;
  }
  return alpha_test_value <= bsdf_sampler_next(sampler);
}
