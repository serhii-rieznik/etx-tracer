#pragma once

#include "bsdf_fresnel_shared.hxx"

// Normalize prescribed substrate radiance against the same bare optical boundary.
ETX_SHARED_NOINLINE float bsdf_coherent_source_multilayer_ratio(float wavelength, float cosine, ETX_IN(complex, exterior), ETX_IN(complex, interior),
  ETX_IN(BSDFFresnelLayerStack, layers)) {
  if (cosine <= 0.0f) {
    return 0.0f;
  }
  bool has_layer = false;
  for (uint32_t index = 0u; index < layers.count; ++index) {
    has_layer = has_layer || (layers.thickness[index] > 0.0f);
  }
  if (has_layer == false) {
    return 1.0f;
  }

  const BSDFFresnelLayerStack no_layers = ETX_ZERO(BSDFFresnelLayerStack);
  const BSDFFresnelPolarizedStack bare = bsdf_fresnel_multilayer_amplitudes(wavelength, cosine, exterior, interior, no_layers);
  const BSDFFresnelPolarizedStack covered = bsdf_fresnel_multilayer_amplitudes(wavelength, cosine, exterior, interior, layers);
  const float eta = bsdf_complex_real(interior);
  const float k = bsdf_complex_imag(interior);
  float s_admittance = 0.0f;
  float p_admittance = 0.0f;
  if (k > 0.0f) {
    const float q_real = bsdf_complex_real(bare.terminal_q);
    const float q_imaginary = bsdf_complex_imag(bare.terminal_q);
    s_admittance = q_real >= q_imaginary ? q_real / eta : k / q_imaginary;
    const float norm_ior = (eta * eta) + (k * k);
    p_admittance = ((s_admittance * ((eta * eta) - (k * k)) / norm_ior) + (2.0f * k * q_imaginary / norm_ior)) / norm_ior;
  } else if (bsdf_complex_real(bare.terminal_q) > 0.0f) {
    s_admittance = bsdf_complex_real(bare.terminal_q);
    p_admittance = bsdf_complex_real(bare.terminal_y);
  } else {
    // The leading lossless-terminal flux is proportional to k (sqrt(k) at critical incidence).
    // Its common factor cancels in the ratio; these weights cover both limits without a zero divisor.
    const float eta_squared = eta * eta;
    const float q_imaginary = bsdf_complex_imag(bare.terminal_q);
    s_admittance = eta_squared * eta_squared;
    p_admittance = eta_squared + (2.0f * q_imaginary * q_imaginary);
  }
  const float exterior_eta = bsdf_complex_real(exterior);
  p_admittance *= exterior_eta * exterior_eta;
  const float bare_response = (s_admittance * bsdf_complex_norm(bare.s.transmission)) + (p_admittance * bsdf_complex_norm(bare.p.transmission));
  const float covered_response = (s_admittance * bsdf_complex_norm(covered.s.transmission)) + (p_admittance * bsdf_complex_norm(covered.p.transmission));
  return covered_response / bare_response;
}

ETX_SHARED_NOINLINE float bsdf_coherent_source_interface_ratio(float wavelength, float cosine, ETX_IN(complex, exterior), ETX_IN(complex, interior), ETX_IN(complex, film_ior),
  ETX_IN(complex, coating_ior), float film_thickness, float film_weight, float coating_thickness, bool incident_inside) {
  BSDFFresnelLayerStack layers = ETX_ZERO(BSDFFresnelLayerStack);
  if (coating_thickness > 0.0f) {
    layers.ior[0] = coating_ior;
    layers.thickness[0] = coating_thickness;
    layers.count = 1u;
  }
  const float coating_ratio = bsdf_coherent_source_multilayer_ratio(wavelength, cosine, exterior, interior, layers);
  if ((film_thickness <= 0.0f) || (film_weight <= 0.0f)) {
    return coating_ratio;
  }
  const uint32_t film_index = incident_inside ? layers.count : 0u;
  if ((incident_inside == false) && (layers.count != 0u)) {
    layers.ior[1] = layers.ior[0];
    layers.thickness[1] = layers.thickness[0];
  }
  layers.ior[film_index] = film_ior;
  layers.thickness[film_index] = film_thickness;
  ++layers.count;
  const float combined_ratio = bsdf_coherent_source_multilayer_ratio(wavelength, cosine, exterior, interior, layers);
  const float weight = saturate(film_weight);
#if ETX_CPP
  const float result = ((1.0f - weight) * coating_ratio) + (weight * combined_ratio);
#else
  precise float result = ((1.0f - weight) * coating_ratio) + (weight * combined_ratio);
#endif
  return result;
}

ETX_SHARED_INLINE SpectralResponse bsdf_coherent_source_ratio(ETX_IN(SpectralQuery, spect), float cosine, ETX_IN(RefractiveIndexSample, exterior),
  ETX_IN(RefractiveIndexSample, interior), ETX_IN(CoherentInterfaceEval, optics)) {
  const ThinfilmEval film = optics.thinfilm;
  const float mu = saturate(abs(cosine));
  const bool incident_inside = optics.incident_inside != 0u;
  if (spectral_query_is_spectral(spect)) {
    return spectral_response_make(spect,
      bsdf_coherent_source_interface_ratio(spect.wavelength, mu, refractive_index_sample_as_complex_spectral(exterior), refractive_index_sample_as_complex_spectral(interior),
        refractive_index_sample_as_complex_spectral(film.ior), refractive_index_sample_as_complex_spectral(optics.coating_ior), film.thickness, film.weight,
        optics.coating_thickness_nm, incident_inside));
  }
  const float r = bsdf_coherent_source_interface_ratio(film.rgb_wavelengths.x, mu, refractive_index_sample_as_complex_x(exterior), refractive_index_sample_as_complex_x(interior),
    refractive_index_sample_as_complex_x(film.ior), refractive_index_sample_as_complex_x(optics.coating_ior), film.thickness, film.weight, optics.coating_thickness_nm,
    incident_inside);
  const float g = bsdf_coherent_source_interface_ratio(film.rgb_wavelengths.y, mu, refractive_index_sample_as_complex_y(exterior), refractive_index_sample_as_complex_y(interior),
    refractive_index_sample_as_complex_y(film.ior), refractive_index_sample_as_complex_y(optics.coating_ior), film.thickness, film.weight, optics.coating_thickness_nm,
    incident_inside);
  const float b = bsdf_coherent_source_interface_ratio(film.rgb_wavelengths.z, mu, refractive_index_sample_as_complex_z(exterior), refractive_index_sample_as_complex_z(interior),
    refractive_index_sample_as_complex_z(film.ior), refractive_index_sample_as_complex_z(optics.coating_ior), film.thickness, film.weight, optics.coating_thickness_nm,
    incident_inside);
  return spectral_response_make(spect, float3(r, g, b));
}
