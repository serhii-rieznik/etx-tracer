#pragma once

#include "bsdf_external_shared.hxx"

ETX_SHARED_INLINE float2 bsdf_terminal_source_cone(ETX_IN(complex, exterior), ETX_IN(complex, interior)) {
  // Guide narrow lossless escape cones; VNDF samples retain support outside this proposal.
  const float exterior_eta = bsdf_complex_real(exterior);
  const float fraction = min(1.0f, bsdf_complex_norm(interior) / (exterior_eta * exterior_eta));
  const float height = fraction / (1.0f + sqrt(max(0.0f, 1.0f - fraction)));
  return float2(fraction, height);
}

ETX_SHARED_INLINE float bsdf_terminal_source_balance(float normal_pdf, float sine_squared, ETX_IN(complex, exterior), ETX_IN(complex, interior)) {
  const float2 cone = bsdf_terminal_source_cone(exterior, interior);
  if ((cone.y <= 0.0f) || (sine_squared > cone.x)) {
    return 1.0f;
  }
  const float scaled_pdf = normal_pdf * kDoublePi * cone.y;
  return scaled_pdf / (1.0f + scaled_pdf);
}

ETX_SHARED_INLINE SpectralResponse bsdf_terminal_source_normal_sample(ETX_IN(SpectralQuery, spect), ETX_IN(RefractiveIndexSample, exterior),
  ETX_IN(RefractiveIndexSample, interior), ETX_IN(ThinfilmEval, optics), ETX_IN(float3, w_i), ETX_IN(float3, m), float alpha, float geometric_weight) {
  const float3 transverse = cross(w_i, m);
  const float sine_squared = dot(transverse, transverse);
  const float normal_pdf = bsdf_external_vndf_pdf(w_i, m, alpha);
  const SpectralResponse source =
    bsdf_fresnel_terminal_source_calculate_with_sine_squared(spect, dot(w_i, m), sine_squared, exterior, interior, bsdf_fresnel_coherent_interface(optics));
  if (spectral_query_is_spectral(spect)) {
    const float balance =
      bsdf_terminal_source_balance(normal_pdf, sine_squared, refractive_index_sample_as_complex_spectral(exterior), refractive_index_sample_as_complex_spectral(interior));
    return spectral_response_mul(source, geometric_weight * balance);
  }
  const float3 balance =
    float3(bsdf_terminal_source_balance(normal_pdf, sine_squared, refractive_index_sample_as_complex_x(exterior), refractive_index_sample_as_complex_x(interior)),
      bsdf_terminal_source_balance(normal_pdf, sine_squared, refractive_index_sample_as_complex_y(exterior), refractive_index_sample_as_complex_y(interior)),
      bsdf_terminal_source_balance(normal_pdf, sine_squared, refractive_index_sample_as_complex_z(exterior), refractive_index_sample_as_complex_z(interior)));
  return spectral_response_mul(source, spectral_response_make(spect, geometric_weight * balance));
}

ETX_SHARED_NOINLINE float bsdf_terminal_source_guided_sample(float wavelength, ETX_IN(complex, exterior), ETX_IN(complex, interior), ETX_IN(complex, film_ior),
  ETX_IN(complex, coating_ior), float film_thickness, float film_weight, float coating_thickness, bool incident_inside, ETX_IN(float3, w_i), float alpha, ETX_IN(float2, rnd)) {
  const float2 cone = bsdf_terminal_source_cone(exterior, interior);
  if (cone.y <= 0.0f) {
    return 0.0f;
  }
  const float decrement = cone.y * rnd.x;
  const float cosine = 1.0f - decrement;
  const float sine_squared = decrement * (2.0f - decrement);
  const float sine = sqrt(sine_squared);
  const float azimuth = kDoublePi * rnd.y;
  // Directional LUTs put the incident ray in the x-z plane.
  const float3 tangent = float3(w_i.z, 0.0f, -w_i.x);
  const float3 m = (cosine * w_i) + (sine * ((cos(azimuth) * tangent) + float3(0.0f, sin(azimuth), 0.0f)));
  const float3 w_o = -w_i + (2.0f * cosine * m);
  if ((m.z <= kEpsilon) || (cosine <= kEpsilon) || (w_o.z <= 0.0f)) {
    return 0.0f;
  }
  const float lambda_i = bsdf_external_ray_info_make(w_i, float2(alpha, alpha)).Lambda;
  const float lambda_o = bsdf_external_ray_info_make(w_o, float2(alpha, alpha)).Lambda;
  const float geometric_weight = (1.0f + lambda_i) / (1.0f + lambda_i + lambda_o);
  const float normal_pdf = bsdf_external_vndf_pdf(w_i, m, alpha);
  const float scaled_pdf = normal_pdf * kDoublePi * cone.y;
  const float balance = scaled_pdf / (1.0f + scaled_pdf);
  const float source = bsdf_fresnel_coherent_terminal_power_with_sine_squared(wavelength, cosine, sine_squared, exterior, interior, film_ior, coating_ior, film_thickness,
    film_weight, coating_thickness, incident_inside)
                         .y;
  return geometric_weight * balance * source;
}

ETX_SHARED_NOINLINE SpectralResponse bsdf_terminal_source_guided_sample(ETX_IN(SpectralQuery, spect), ETX_IN(RefractiveIndexSample, exterior),
  ETX_IN(RefractiveIndexSample, interior), ETX_IN(ThinfilmEval, optics), ETX_IN(float3, w_i), float alpha, ETX_IN(float2, rnd)) {
  const bool incident_inside = optics.incident_inside != 0u;
  if (spectral_query_is_spectral(spect)) {
    const float value = bsdf_terminal_source_guided_sample(spect.wavelength, refractive_index_sample_as_complex_spectral(exterior),
      refractive_index_sample_as_complex_spectral(interior), refractive_index_sample_as_complex_spectral(optics.ior),
      refractive_index_sample_as_complex_spectral(optics.coating_ior), optics.thickness, optics.weight, optics.coating_thickness_nm, incident_inside, w_i, alpha, rnd);
    return spectral_response_make(spect, value);
  }
  const float r = bsdf_terminal_source_guided_sample(optics.rgb_wavelengths.x, refractive_index_sample_as_complex_x(exterior), refractive_index_sample_as_complex_x(interior),
    refractive_index_sample_as_complex_x(optics.ior), refractive_index_sample_as_complex_x(optics.coating_ior), optics.thickness, optics.weight, optics.coating_thickness_nm,
    incident_inside, w_i, alpha, rnd);
  const float g = bsdf_terminal_source_guided_sample(optics.rgb_wavelengths.y, refractive_index_sample_as_complex_y(exterior), refractive_index_sample_as_complex_y(interior),
    refractive_index_sample_as_complex_y(optics.ior), refractive_index_sample_as_complex_y(optics.coating_ior), optics.thickness, optics.weight, optics.coating_thickness_nm,
    incident_inside, w_i, alpha, rnd);
  const float b = bsdf_terminal_source_guided_sample(optics.rgb_wavelengths.z, refractive_index_sample_as_complex_z(exterior), refractive_index_sample_as_complex_z(interior),
    refractive_index_sample_as_complex_z(optics.ior), refractive_index_sample_as_complex_z(optics.coating_ior), optics.thickness, optics.weight, optics.coating_thickness_nm,
    incident_inside, w_i, alpha, rnd);
  return spectral_response_make(spect, float3(r, g, b));
}
