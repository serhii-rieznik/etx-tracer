#pragma once

#include "bsdf_core_shared.hxx"

ETX_SHARED_INLINE float bsdf_dielectric_directional_eta(ETX_IN(RefractiveIndexSample, incident_ior), ETX_IN(RefractiveIndexSample, transmitted_ior)) {
  const float eta_incident = max(kEpsilon, spectral_response_monochromatic(incident_ior.eta));
  const float eta_transmitted = max(kEpsilon, spectral_response_monochromatic(transmitted_ior.eta));
  return eta_transmitted / eta_incident;
}

ETX_SHARED_INLINE float bsdf_complex_real(ETX_IN(complex, value)) {
#if (ETX_CPP)
  return value.real();
#else
  return value.re;
#endif
}

ETX_SHARED_INLINE float bsdf_complex_imag(ETX_IN(complex, value)) {
#if (ETX_CPP)
  return value.imag();
#else
  return value.im;
#endif
}

ETX_SHARED_INLINE complex bsdf_complex_make(float real_value, float imag_value) {
  return make_complex(real_value, imag_value);
}

ETX_SHARED_INLINE complex bsdf_complex_add(ETX_IN(complex, a), ETX_IN(complex, b)) {
#if (ETX_CPP)
  return a + b;
#else
  return bsdf_complex_make(bsdf_complex_real(a) + bsdf_complex_real(b), bsdf_complex_imag(a) + bsdf_complex_imag(b));
#endif
}

ETX_SHARED_INLINE complex bsdf_complex_sub(ETX_IN(complex, a), ETX_IN(complex, b)) {
#if (ETX_CPP)
  return a - b;
#else
  return bsdf_complex_make(bsdf_complex_real(a) - bsdf_complex_real(b), bsdf_complex_imag(a) - bsdf_complex_imag(b));
#endif
}

ETX_SHARED_INLINE complex bsdf_complex_mul(ETX_IN(complex, a), ETX_IN(complex, b)) {
#if (ETX_CPP)
  return a * b;
#else
  return bsdf_complex_make(bsdf_complex_real(a) * bsdf_complex_real(b) - bsdf_complex_imag(a) * bsdf_complex_imag(b),
    bsdf_complex_real(a) * bsdf_complex_imag(b) + bsdf_complex_imag(a) * bsdf_complex_real(b));
#endif
}

ETX_SHARED_INLINE complex bsdf_complex_mul_scalar(ETX_IN(complex, a), float value) {
#if (ETX_CPP)
  return a * value;
#else
  return bsdf_complex_make(bsdf_complex_real(a) * value, bsdf_complex_imag(a) * value);
#endif
}

ETX_SHARED_INLINE complex bsdf_complex_div_scalar(ETX_IN(complex, a), float value) {
#if (ETX_CPP)
  return a / value;
#else
  return bsdf_complex_make(bsdf_complex_real(a) / value, bsdf_complex_imag(a) / value);
#endif
}

ETX_SHARED_INLINE float bsdf_complex_norm(ETX_IN(complex, value)) {
#if (ETX_CPP)
  return complex_norm(value);
#else
  return bsdf_complex_real(value) * bsdf_complex_real(value) + bsdf_complex_imag(value) * bsdf_complex_imag(value);
#endif
}

ETX_SHARED_INLINE float bsdf_complex_abs(ETX_IN(complex, value)) {
#if (ETX_CPP)
  return complex_abs(value);
#else
  return sqrt(bsdf_complex_norm(value));
#endif
}

ETX_SHARED_INLINE bool bsdf_complex_equal(ETX_IN(complex, a), ETX_IN(complex, b)) {
  return (bsdf_complex_real(a) == bsdf_complex_real(b)) && (bsdf_complex_imag(a) == bsdf_complex_imag(b));
}

ETX_SHARED_INLINE bool bsdf_fresnel_equal_indices(ETX_IN(RefractiveIndexSample, a), ETX_IN(RefractiveIndexSample, b)) {
  if (spectral_response_is_spectral(a.eta)) {
    return bsdf_complex_equal(refractive_index_sample_as_complex_spectral(a), refractive_index_sample_as_complex_spectral(b));
  }
  return bsdf_complex_equal(refractive_index_sample_as_complex_x(a), refractive_index_sample_as_complex_x(b)) &&
         bsdf_complex_equal(refractive_index_sample_as_complex_y(a), refractive_index_sample_as_complex_y(b)) &&
         bsdf_complex_equal(refractive_index_sample_as_complex_z(a), refractive_index_sample_as_complex_z(b));
}

ETX_SHARED_INLINE complex bsdf_complex_conjugate(ETX_IN(complex, value)) {
  return bsdf_complex_make(bsdf_complex_real(value), -bsdf_complex_imag(value));
}

ETX_SHARED_INLINE complex bsdf_complex_div(ETX_IN(complex, a), ETX_IN(complex, b)) {
#if (ETX_CPP)
  return a / b;
#else
  complex numerator = bsdf_complex_mul(a, bsdf_complex_conjugate(b));
  float denominator = bsdf_complex_norm(b);
  if (denominator == 0.0f) {
    return bsdf_complex_make(0.0f, 0.0f);
  }
  return bsdf_complex_div_scalar(numerator, denominator);
#endif
}

ETX_SHARED_INLINE complex bsdf_complex_sqrt(ETX_IN(complex, value)) {
#if (ETX_CPP)
  return complex_sqrt(value);
#else
  const float magnitude = bsdf_complex_abs(value);
  if (magnitude == 0.0f) {
    return bsdf_complex_make(0.0f, 0.0f);
  }
  const float real_value = bsdf_complex_real(value);
  const float imaginary_value = bsdf_complex_imag(value);
  const float larger_component = sqrt(0.5f * (magnitude + abs(real_value)));
  // Recover the smaller component from 2 Re(q) Im(q)=Im(q^2), avoiding cancellation.
  if (real_value >= 0.0f) {
    return bsdf_complex_make(larger_component, imaginary_value / (2.0f * larger_component));
  }
  return bsdf_complex_make(abs(imaginary_value) / (2.0f * larger_component), imaginary_value < 0.0f ? -larger_component : larger_component);
#endif
}

struct BSDFFresnelReflectanceResult {
  complex rs ETX_INIT({});
  complex rp ETX_INIT({});
};

struct BSDFFresnelTransmittanceResult {
  complex ts ETX_INIT({});
  complex tp ETX_INIT({});
};

ETX_SHARED_INLINE BSDFFresnelReflectanceResult bsdf_fresnel_reflectance(ETX_IN(complex, ext_ior), ETX_IN(complex, cos_theta_i), ETX_IN(complex, int_ior),
  ETX_IN(complex, cos_theta_j)) {
  if ((bsdf_complex_real(cos_theta_i) == 0.0f) && (bsdf_complex_real(cos_theta_j) == 0.0f) && (bsdf_complex_imag(cos_theta_i) == 0.0f) &&
      (bsdf_complex_imag(cos_theta_j) == 0.0f)) {
    BSDFFresnelReflectanceResult result = ETX_ZERO(BSDFFresnelReflectanceResult);
    result.rs = bsdf_complex_make(1.0f, 0.0f);
    result.rp = bsdf_complex_make(1.0f, 0.0f);
    return result;
  }

  if (bsdf_complex_equal(ext_ior, int_ior)) {
    BSDFFresnelReflectanceResult result = ETX_ZERO(BSDFFresnelReflectanceResult);
    result.rs = bsdf_complex_make(0.0f, 0.0f);
    result.rp = bsdf_complex_make(0.0f, 0.0f);
    return result;
  }

  complex ni_cos_i = bsdf_complex_mul(ext_ior, cos_theta_i);
  complex nj_cos_j = bsdf_complex_mul(int_ior, cos_theta_j);
  complex nj_cos_i = bsdf_complex_mul(int_ior, cos_theta_i);
  complex ni_cos_j = bsdf_complex_mul(ext_ior, cos_theta_j);

  BSDFFresnelReflectanceResult result = ETX_ZERO(BSDFFresnelReflectanceResult);
  result.rs = bsdf_complex_div(bsdf_complex_sub(ni_cos_i, nj_cos_j), bsdf_complex_add(ni_cos_i, nj_cos_j));
  result.rp = bsdf_complex_div(bsdf_complex_sub(nj_cos_i, ni_cos_j), bsdf_complex_add(nj_cos_i, ni_cos_j));
  return result;
}

ETX_SHARED_INLINE BSDFFresnelTransmittanceResult bsdf_fresnel_transmittance(ETX_IN(complex, ext_ior), ETX_IN(complex, cos_theta_i), ETX_IN(complex, int_ior),
  ETX_IN(complex, cos_theta_j)) {
  if ((bsdf_complex_real(cos_theta_i) == 0.0f) && (bsdf_complex_real(cos_theta_j) == 0.0f) && (bsdf_complex_imag(cos_theta_i) == 0.0f) &&
      (bsdf_complex_imag(cos_theta_j) == 0.0f)) {
    BSDFFresnelTransmittanceResult result = ETX_ZERO(BSDFFresnelTransmittanceResult);
    result.ts = bsdf_complex_make(0.0f, 0.0f);
    result.tp = bsdf_complex_make(0.0f, 0.0f);
    return result;
  }

  if (bsdf_complex_equal(ext_ior, int_ior)) {
    BSDFFresnelTransmittanceResult result = ETX_ZERO(BSDFFresnelTransmittanceResult);
    result.ts = bsdf_complex_make(1.0f, 0.0f);
    result.tp = bsdf_complex_make(1.0f, 0.0f);
    return result;
  }

  complex two_ext_cos_i = bsdf_complex_mul_scalar(bsdf_complex_mul(ext_ior, cos_theta_i), 2.0f);
  complex ext_cos_i = bsdf_complex_mul(ext_ior, cos_theta_i);
  complex int_cos_j = bsdf_complex_mul(int_ior, cos_theta_j);
  complex ext_cos_j = bsdf_complex_mul(ext_ior, cos_theta_j);
  complex int_cos_i = bsdf_complex_mul(int_ior, cos_theta_i);

  BSDFFresnelTransmittanceResult result = ETX_ZERO(BSDFFresnelTransmittanceResult);
  result.ts = bsdf_complex_div(two_ext_cos_i, bsdf_complex_add(ext_cos_i, int_cos_j));
  result.tp = bsdf_complex_div(two_ext_cos_i, bsdf_complex_add(ext_cos_j, int_cos_i));
  return result;
}

#if (ETX_CPP)
# define ETX_FRESNEL_PRECISE
#else
# define ETX_FRESNEL_PRECISE precise
#endif

ETX_SHARED_INLINE float2 bsdf_fresnel_product(float a, float b) {
#if (ETX_CPP) && defined(__clang__)
# pragma clang fp contract(off) reassociate(off)
#endif
  // Two 12-bit parts recover the rounded product's residual without a fused operation.
  const float a_high = to_float(to_uint(a) & 0xfffff000u);
  const float b_high = to_float(to_uint(b) & 0xfffff000u);
  ETX_FRESNEL_PRECISE const float a_low = a - a_high;
  ETX_FRESNEL_PRECISE const float b_low = b - b_high;
  ETX_FRESNEL_PRECISE const float product = a * b;
  ETX_FRESNEL_PRECISE const float error = (((a_high * b_high - product) + a_high * b_low) + a_low * b_high) + a_low * b_low;
  return float2(product, error);
}

ETX_SHARED_INLINE float3 bsdf_fresnel_dielectric_longitudinal(float cosine, float incident, float transmitted) {
#if (ETX_CPP) && defined(__clang__)
# pragma clang fp contract(off) reassociate(off)
#endif
  // A common power-of-two scale preserves the index ratio without rounding it.
  const float scale = to_float(max(to_uint(max(incident, transmitted)) & 0x7f800000u, 0x00800000u));
  const float ni = incident / scale;
  const float nt = transmitted / scale;
  const float2 nt_squared = bsdf_fresnel_product(nt, nt);
  ETX_FRESNEL_PRECISE float longitudinal_squared;
  if (cosine <= 0.5f) {
    // Factoring the index-square difference preserves nearly matched grazing interfaces.
    ETX_FRESNEL_PRECISE const float index_sum = nt + ni;
    ETX_FRESNEL_PRECISE const float sum_error = min(nt, ni) - (index_sum - max(nt, ni));
    ETX_FRESNEL_PRECISE const float index_difference = nt - ni;
    ETX_FRESNEL_PRECISE const float virtual_ni = index_difference - nt;
    ETX_FRESNEL_PRECISE const float difference_error = (nt - (index_difference - virtual_ni)) - (ni + virtual_ni);
    const float2 difference_squared = bsdf_fresnel_product(index_sum, index_difference);
    ETX_FRESNEL_PRECISE const float difference_squared_error =
      ((difference_squared.y + index_sum * difference_error) + sum_error * index_difference) + sum_error * difference_error;
    const float2 incident_q = bsdf_fresnel_product(ni, cosine);
    const float2 incident_q_squared = bsdf_fresnel_product(incident_q.x, incident_q.x);
    ETX_FRESNEL_PRECISE const float incident_q_error = (incident_q_squared.y + (2.0f * incident_q.x) * incident_q.y) + incident_q.y * incident_q.y;
    longitudinal_squared = (difference_squared.x + incident_q_squared.x) + (difference_squared_error + incident_q_error);
  } else {
    // Near normal incidence, subtracting two index squares would erase a small transmitted index.
    const float2 ni_squared = bsdf_fresnel_product(ni, ni);
    const float2 cosine_squared = bsdf_fresnel_product(cosine, cosine);
    ETX_FRESNEL_PRECISE const float sine_squared = 1.0f - cosine_squared.x;
    ETX_FRESNEL_PRECISE const float sine_error = ((1.0f - sine_squared) - cosine_squared.x) - cosine_squared.y;
    const float2 transverse_squared = bsdf_fresnel_product(ni_squared.x, sine_squared);
    ETX_FRESNEL_PRECISE const float transverse_error = ((transverse_squared.y + ni_squared.x * sine_error) + ni_squared.y * sine_squared) + ni_squared.y * sine_error;
    // The leading subtraction is exact near the critical angle (Sterbenz's lemma).
    longitudinal_squared = (nt_squared.x - transverse_squared.x) + (nt_squared.y - transverse_error);
  }
  return float3(ni, nt, longitudinal_squared);
}

ETX_SHARED_INLINE float bsdf_fresnel_dielectric_real(float cosine, float incident, float transmitted) {
#if (ETX_CPP) && defined(__clang__)
# pragma clang fp contract(off) reassociate(off)
#endif
  if (incident == transmitted) {
    return 0.0f;
  }
  if (cosine == 0.0f) {
    return 1.0f;
  }
  const float3 state = bsdf_fresnel_dielectric_longitudinal(cosine, incident, transmitted);
  if (state.z <= 0.0f) {
    return 1.0f;
  }
  const float longitudinal = sqrt(state.z);
  const float incident_longitudinal = state.x * cosine;
  const float rs = (incident_longitudinal - longitudinal) / (incident_longitudinal + longitudinal);
  const float incident_admittance = (state.y * state.y) * cosine;
  const float transmitted_admittance = state.x * longitudinal;
  const float rp = (incident_admittance - transmitted_admittance) / (incident_admittance + transmitted_admittance);
  return 0.5f * (rs * rs + rp * rp);
}

#undef ETX_FRESNEL_PRECISE

ETX_SHARED_INLINE float bsdf_fresnel_generic(float cos_theta_i, ETX_IN(complex, ext_ior), ETX_IN(complex, int_ior)) {
  if ((bsdf_complex_imag(ext_ior) == 0.0f) && (bsdf_complex_imag(int_ior) == 0.0f) && (bsdf_complex_real(ext_ior) > 0.0f) && (bsdf_complex_real(int_ior) > 0.0f)) {
    return bsdf_fresnel_dielectric_real(cos_theta_i, bsdf_complex_real(ext_ior), bsdf_complex_real(int_ior));
  }
  complex eta_ratio = bsdf_complex_div(ext_ior, int_ior);
  complex eta_ratio_squared = bsdf_complex_mul(eta_ratio, eta_ratio);
  complex sin_theta_o_squared = bsdf_complex_mul_scalar(eta_ratio_squared, 1.0f - cos_theta_i * cos_theta_i);
  complex cos_theta_o = bsdf_complex_sqrt(bsdf_complex_sub(bsdf_complex_make(1.0f, 0.0f), sin_theta_o_squared));
  BSDFFresnelReflectanceResult reflectance = bsdf_fresnel_reflectance(ext_ior, bsdf_complex_make(cos_theta_i, 0.0f), int_ior, cos_theta_o);
  return 0.5f * (bsdf_complex_norm(reflectance.rs) + bsdf_complex_norm(reflectance.rp));
}

struct BSDFFresnelLayerPhase {
  complex propagation;
  complex cosine;
  complex sine;
};

ETX_SHARED_INLINE complex bsdf_fresnel_longitudinal_component_from_squared(ETX_IN(complex, index_squared), ETX_IN(complex, incident_index_squared),
  ETX_IN(complex, incident_q_squared), float cosine) {
  if (cosine <= 0.5f) {
    return bsdf_complex_sqrt(bsdf_complex_add(bsdf_complex_sub(index_squared, incident_index_squared), incident_q_squared));
  }
  // Near normal incidence, forming n^2-n0^2+q0^2 can erase a small terminal index.
  const float sine_squared = (1.0f - cosine) * (1.0f + cosine);
  return bsdf_complex_sqrt(bsdf_complex_sub(index_squared, bsdf_complex_mul_scalar(incident_index_squared, sine_squared)));
}

ETX_SHARED_INLINE complex bsdf_fresnel_longitudinal_component_with_sine_squared(ETX_IN(complex, index_squared), ETX_IN(complex, incident_index_squared),
  ETX_IN(complex, incident_q_squared), float cosine, float sine_squared) {
  if (cosine <= 0.5f) {
    return bsdf_fresnel_longitudinal_component_from_squared(index_squared, incident_index_squared, incident_q_squared, cosine);
  }
  return bsdf_complex_sqrt(bsdf_complex_sub(index_squared, bsdf_complex_mul_scalar(incident_index_squared, sine_squared)));
}

ETX_SHARED_INLINE complex bsdf_fresnel_longitudinal_component(ETX_IN(complex, index), ETX_IN(complex, incident_index), float cosine) {
  const float incident = bsdf_complex_real(incident_index);
  const float transmitted = bsdf_complex_real(index);
  if ((bsdf_complex_imag(index) == 0.0f) && (bsdf_complex_imag(incident_index) == 0.0f) && (incident > 0.0f) && (transmitted > 0.0f)) {
    const float3 state = bsdf_fresnel_dielectric_longitudinal(cosine, incident, transmitted);
    const float scale = to_float(max(to_uint(max(incident, transmitted)) & 0x7f800000u, 0x00800000u));
    const float magnitude = sqrt(abs(state.z)) * scale;
    if (state.z >= 0.0f) {
      return bsdf_complex_make(magnitude, 0.0f);
    }
    return bsdf_complex_make(0.0f, magnitude);
  }
  const complex incident_q = bsdf_complex_mul_scalar(incident_index, cosine);
  return bsdf_fresnel_longitudinal_component_from_squared(bsdf_complex_mul(index, index), bsdf_complex_mul(incident_index, incident_index),
    bsdf_complex_mul(incident_q, incident_q), cosine);
}

ETX_SHARED_INLINE BSDFFresnelLayerPhase bsdf_fresnel_layer_phase(ETX_IN(complex, q), float optical_distance) {
  const float phase = optical_distance * bsdf_complex_real(q);
  const float attenuation = optical_distance * bsdf_complex_imag(q);
  float sine = sin(phase);
  float cosine = cos(phase);
  if (abs(phase) <= 0.1f) {
    // GPU trigonometric instructions lose relative accuracy near zero; analytic remainders are below 2.8e-15 here.
    const float squared = phase * phase;
    sine = phase * (1.0f - squared * (1.0f / 6.0f - squared * (1.0f / 120.0f - squared / 5040.0f)));
    cosine = 1.0f - squared * (0.5f - squared * (1.0f / 24.0f - squared * (1.0f / 720.0f - squared / 40320.0f)));
  }
  const float propagation_decay = exp(-attenuation);
  const float round_trip_decay = propagation_decay * propagation_decay;
  float half_loss = 0.5f * (1.0f - round_trip_decay);
  if (attenuation <= 0.1f) {
    // Half of -expm1(-2v); the omitted term is below 3.2e-11 on this interval.
    half_loss =
      attenuation *
      (1.0f - attenuation *
                (1.0f - attenuation * (2.0f / 3.0f - attenuation * (1.0f / 3.0f - attenuation * (2.0f / 15.0f - attenuation * (2.0f / 45.0f - attenuation * (4.0f / 315.0f)))))));
  }
  const float sine_real = half_loss + (round_trip_decay * sine * sine);
  const float sine_imaginary = -round_trip_decay * sine * cosine;
  BSDFFresnelLayerPhase result = ETX_ZERO(BSDFFresnelLayerPhase);
  result.propagation = bsdf_complex_make(propagation_decay * cosine, propagation_decay * sine);
  // Scaled characteristic terms: (1 +/- exp(2 i delta))/2, without subtraction near delta=0.
  result.cosine = bsdf_complex_make(half_loss + (round_trip_decay * cosine * cosine), -sine_imaginary);
  result.sine = bsdf_complex_make(sine_real, sine_imaginary);
  return result;
}

// Coherent film between passive half spaces. The incident half space is lossless.
ETX_SHARED_NOINLINE float2 bsdf_fresnel_thinfilm_power(float wavelength, float cos_theta_0, ETX_IN(complex, ext_ior), ETX_IN(complex, film_ior), ETX_IN(complex, int_ior),
  float thickness) {
  if (thickness <= 0.0f) {
    const float reflection = bsdf_fresnel_generic(cos_theta_0, ext_ior, int_ior);
    return float2(reflection, 1.0f - reflection);
  }
  if (cos_theta_0 <= 0.0f) {
    return float2(1.0f, 0.0f);
  }
  const complex n0_squared = bsdf_complex_mul(ext_ior, ext_ior);
  const complex n1_squared = bsdf_complex_mul(film_ior, film_ior);
  const complex n2_squared = bsdf_complex_mul(int_ior, int_ior);
  const complex q0 = bsdf_complex_mul_scalar(ext_ior, cos_theta_0);
  const complex q1 = bsdf_fresnel_longitudinal_component(film_ior, ext_ior, cos_theta_0);
  const complex q2 = bsdf_fresnel_longitudinal_component(int_ior, ext_ior, cos_theta_0);
  const float optical_distance = kDoublePi * thickness / wavelength;
  // Multiply the characteristic matrix by exp(i delta) to avoid overflow in absorbing films.
  const BSDFFresnelLayerPhase phase = bsdf_fresnel_layer_phase(q1, optical_distance);
  const complex propagation = phase.propagation;
  const complex cosine = phase.cosine;
  const complex sine = phase.sine;
  complex sine_over_q = bsdf_complex_make(0.0f, -optical_distance);
  if (bsdf_complex_norm(q1) > 0.0f) {
    sine_over_q = bsdf_complex_div(sine, q1);
  }
  const complex s_product = bsdf_complex_mul(bsdf_complex_mul(q0, q2), sine_over_q);
  const complex s_film = bsdf_complex_mul(q1, sine);
  const complex s_denominator = bsdf_complex_add(bsdf_complex_add(bsdf_complex_mul(bsdf_complex_add(q0, q2), cosine), s_product), s_film);
  const complex rs = bsdf_complex_div(bsdf_complex_sub(bsdf_complex_add(bsdf_complex_mul(bsdf_complex_sub(q0, q2), cosine), s_product), s_film), s_denominator);
  const complex ts = bsdf_complex_div(bsdf_complex_mul_scalar(bsdf_complex_mul(q0, propagation), 2.0f), s_denominator);
  const complex y0 = bsdf_complex_div(q0, n0_squared);
  const complex y1 = bsdf_complex_div(q1, n1_squared);
  const complex y2 = bsdf_complex_div(q2, n2_squared);
  const complex p_product = bsdf_complex_mul(bsdf_complex_mul(bsdf_complex_mul(y0, y2), n1_squared), sine_over_q);
  const complex p_film = bsdf_complex_mul(y1, sine);
  const complex p_denominator = bsdf_complex_add(bsdf_complex_add(bsdf_complex_mul(bsdf_complex_add(y0, y2), cosine), p_product), p_film);
  const complex rp = bsdf_complex_div(bsdf_complex_sub(bsdf_complex_add(bsdf_complex_mul(bsdf_complex_sub(y0, y2), cosine), p_product), p_film), p_denominator);
  const complex tp = bsdf_complex_div(bsdf_complex_mul_scalar(bsdf_complex_mul(y0, propagation), 2.0f), p_denominator);
  const float reflection = 0.5f * (bsdf_complex_norm(rs) + bsdf_complex_norm(rp));
  const float transmission = 0.5f * (bsdf_complex_norm(ts) * bsdf_complex_real(q2) / bsdf_complex_real(q0) + bsdf_complex_norm(tp) * bsdf_complex_real(y2) / bsdf_complex_real(y0));
  return float2(reflection, transmission);
}

ETX_STATIC_CONST uint32_t kBSDFFresnelMaximumCoherentLayers = 3u;

struct BSDFFresnelLayerStack {
  complex ior[kBSDFFresnelMaximumCoherentLayers] ETX_INIT({});
  float thickness[kBSDFFresnelMaximumCoherentLayers] ETX_INIT({});
  uint32_t count ETX_INIT(0u);
};

struct BSDFFresnelStackAmplitudes {
  complex reflection;
  complex transmission;
};

struct BSDFFresnelPolarizedStack {
  BSDFFresnelStackAmplitudes s;
  BSDFFresnelStackAmplitudes p;
  complex incident_q;
  complex incident_y;
  complex terminal_q;
  complex terminal_y;
};

struct BSDFFresnelStackFields {
  complex electric;
  complex magnetic;
  complex transmission;
};

ETX_SHARED_INLINE BSDFFresnelStackFields bsdf_fresnel_prepend_layer(ETX_IN(BSDFFresnelStackFields, stack), ETX_IN(complex, cosine), ETX_IN(complex, upper), ETX_IN(complex, lower),
  ETX_IN(complex, propagation)) {
  BSDFFresnelStackFields result = ETX_ZERO(BSDFFresnelStackFields);
  result.electric = bsdf_complex_add(bsdf_complex_mul(cosine, stack.electric), bsdf_complex_mul(upper, stack.magnetic));
  result.magnetic = bsdf_complex_add(bsdf_complex_mul(lower, stack.electric), bsdf_complex_mul(cosine, stack.magnetic));
  // Normalize the scaled characteristic fields after each slab; opaque propagation remains zero.
  const float scale = max(max(abs(bsdf_complex_real(result.electric)), abs(bsdf_complex_imag(result.electric))),
    max(abs(bsdf_complex_real(result.magnetic)), abs(bsdf_complex_imag(result.magnetic))));
  result.electric = bsdf_complex_div_scalar(result.electric, scale);
  result.magnetic = bsdf_complex_div_scalar(result.magnetic, scale);
  result.transmission = bsdf_complex_div_scalar(bsdf_complex_mul(propagation, stack.transmission), scale);
  return result;
}

ETX_SHARED_INLINE BSDFFresnelStackAmplitudes bsdf_fresnel_enter_stack(ETX_IN(BSDFFresnelStackFields, stack), ETX_IN(complex, incident_admittance)) {
  const complex incident_electric = bsdf_complex_mul(incident_admittance, stack.electric);
  const complex denominator = bsdf_complex_add(incident_electric, stack.magnetic);
  BSDFFresnelStackAmplitudes result = ETX_ZERO(BSDFFresnelStackAmplitudes);
  result.reflection = bsdf_complex_div(bsdf_complex_sub(incident_electric, stack.magnetic), denominator);
  result.transmission = bsdf_complex_div(bsdf_complex_mul(bsdf_complex_mul_scalar(incident_admittance, 2.0f), stack.transmission), denominator);
  return result;
}

ETX_SHARED_INLINE BSDFFresnelPolarizedStack bsdf_fresnel_multilayer_amplitudes_internal(float wavelength, float cos_theta_0, float sine_squared, bool cosine_derived,
  ETX_IN(complex, ext_ior), ETX_IN(complex, int_ior), ETX_IN(BSDFFresnelLayerStack, layers)) {
  const complex one = bsdf_complex_make(1.0f, 0.0f);
  const complex n0_squared = bsdf_complex_mul(ext_ior, ext_ior);
  const complex terminal_squared = bsdf_complex_mul(int_ior, int_ior);
  const complex q0 = bsdf_complex_mul_scalar(ext_ior, cos_theta_0);
  const complex q0_squared = bsdf_complex_mul(q0, q0);
  complex terminal_q;
  if (cosine_derived) {
    terminal_q = bsdf_fresnel_longitudinal_component(int_ior, ext_ior, cos_theta_0);
  } else {
    terminal_q = bsdf_fresnel_longitudinal_component_with_sine_squared(terminal_squared, n0_squared, q0_squared, cos_theta_0, sine_squared);
  }
  const complex y0 = bsdf_complex_div(q0, n0_squared);
  const complex terminal_y = bsdf_complex_div(terminal_q, terminal_squared);
  BSDFFresnelStackFields s_stack = ETX_ZERO(BSDFFresnelStackFields);
  s_stack.electric = one;
  s_stack.magnetic = terminal_q;
  s_stack.transmission = one;
  BSDFFresnelStackFields p_stack = ETX_ZERO(BSDFFresnelStackFields);
  p_stack.electric = one;
  p_stack.magnetic = terminal_y;
  p_stack.transmission = one;
  for (uint32_t cursor = layers.count; cursor > 0u; --cursor) {
    const uint32_t index = cursor - 1u;
    if (layers.thickness[index] <= 0.0f) {
      continue;
    }
    const complex n_squared = bsdf_complex_mul(layers.ior[index], layers.ior[index]);
    complex q;
    if (cosine_derived) {
      q = bsdf_fresnel_longitudinal_component(layers.ior[index], ext_ior, cos_theta_0);
    } else {
      q = bsdf_fresnel_longitudinal_component_with_sine_squared(n_squared, n0_squared, q0_squared, cos_theta_0, sine_squared);
    }
    const float optical_distance = kDoublePi * layers.thickness[index] / wavelength;
    // Each scaled matrix stays finite when a conducting layer becomes opaque.
    const BSDFFresnelLayerPhase phase = bsdf_fresnel_layer_phase(q, optical_distance);
    const complex layer_propagation = phase.propagation;
    const complex cosine = phase.cosine;
    const complex sine = phase.sine;
    complex sine_over_q = bsdf_complex_make(0.0f, -optical_distance);
    if (bsdf_complex_norm(q) > 0.0f) {
      sine_over_q = bsdf_complex_div(sine, q);
    }
    s_stack = bsdf_fresnel_prepend_layer(s_stack, cosine, sine_over_q, bsdf_complex_mul(q, sine), layer_propagation);
    p_stack = bsdf_fresnel_prepend_layer(p_stack, cosine, bsdf_complex_mul(n_squared, sine_over_q), bsdf_complex_mul(bsdf_complex_div(q, n_squared), sine), layer_propagation);
  }
  BSDFFresnelPolarizedStack result = ETX_ZERO(BSDFFresnelPolarizedStack);
  result.s = bsdf_fresnel_enter_stack(s_stack, q0);
  result.p = bsdf_fresnel_enter_stack(p_stack, y0);
  result.incident_q = q0;
  result.incident_y = y0;
  result.terminal_q = terminal_q;
  result.terminal_y = terminal_y;
  return result;
}

ETX_SHARED_INLINE BSDFFresnelPolarizedStack bsdf_fresnel_multilayer_amplitudes_with_sine_squared(float wavelength, float cos_theta_0, float sine_squared, ETX_IN(complex, ext_ior),
  ETX_IN(complex, int_ior), ETX_IN(BSDFFresnelLayerStack, layers)) {
  // Explicit quadrature coordinates retain their tangential momentum even when cosine rounds to one.
  return bsdf_fresnel_multilayer_amplitudes_internal(wavelength, cos_theta_0, sine_squared, false, ext_ior, int_ior, layers);
}

ETX_SHARED_INLINE BSDFFresnelPolarizedStack bsdf_fresnel_multilayer_amplitudes(float wavelength, float cos_theta_0, ETX_IN(complex, ext_ior), ETX_IN(complex, int_ior),
  ETX_IN(BSDFFresnelLayerStack, layers)) {
  return bsdf_fresnel_multilayer_amplitudes_internal(wavelength, cos_theta_0, (1.0f - cos_theta_0) * (1.0f + cos_theta_0), true, ext_ior, int_ior, layers);
}

// Ordered from the incident side: optional film, finite coating, existing standalone film.
// The incident half space is lossless; finite layers and the terminal half space are passive.
ETX_SHARED_NOINLINE float2 bsdf_fresnel_multilayer_power(float wavelength, float cos_theta_0, ETX_IN(complex, ext_ior), ETX_IN(complex, int_ior),
  ETX_IN(BSDFFresnelLayerStack, layers)) {
  bool has_layer = false;
  bool lossless_layers = true;
  for (uint32_t index = 0u; index < layers.count; ++index) {
    if (layers.thickness[index] > 0.0f) {
      has_layer = true;
      lossless_layers = lossless_layers && ((bsdf_complex_real(layers.ior[index]) * bsdf_complex_imag(layers.ior[index])) == 0.0f);
    }
  }
  if (has_layer == false) {
    const float reflection = bsdf_fresnel_generic(cos_theta_0, ext_ior, int_ior);
    return float2(reflection, 1.0f - reflection);
  }
  if (cos_theta_0 <= 0.0f) {
    return float2(1.0f, 0.0f);
  }
  const BSDFFresnelPolarizedStack stack = bsdf_fresnel_multilayer_amplitudes(wavelength, cos_theta_0, ext_ior, int_ior, layers);
  const float2 s_power =
    float2(bsdf_complex_norm(stack.s.reflection), bsdf_complex_norm(stack.s.transmission) * bsdf_complex_real(stack.terminal_q) / bsdf_complex_real(stack.incident_q));
  const float2 p_power =
    float2(bsdf_complex_norm(stack.p.reflection), bsdf_complex_norm(stack.p.transmission) * bsdf_complex_real(stack.terminal_y) / bsdf_complex_real(stack.incident_y));
  float2 power = 0.5f * (s_power + p_power);
  if (lossless_layers) {
    // Flux conservation determines the larger probability without grazing-angle cancellation.
    if (power.x >= power.y) {
      power.x = 1.0f - power.y;
    } else {
      power.y = 1.0f - power.x;
    }
  }
  return power;
}

// T/eta measures the terminal source response without losing the eta-to-zero conductor limit.
// Requires a real positive incident IOR and a nonzero passive terminal IOR.
ETX_SHARED_INLINE float bsdf_fresnel_terminal_response_from_stack(ETX_IN(BSDFFresnelPolarizedStack, stack), ETX_IN(complex, int_ior)) {
  const float eta = bsdf_complex_real(int_ior);
  const float k = bsdf_complex_imag(int_ior);
  float s_admittance = 0.0f;
  float p_admittance = 0.0f;
  if (k > 0.0f) {
    const float q_imaginary = bsdf_complex_imag(stack.terminal_q);
    // Im(q^2)=2 eta k gives Re(q)/eta=k/Im(q), also at eta=0 by continuity.
    const float q_real = bsdf_complex_real(stack.terminal_q);
    s_admittance = q_real >= q_imaginary ? q_real / eta : k / q_imaginary;
    const float norm_ior = (eta * eta) + (k * k);
    p_admittance = ((s_admittance * ((eta * eta) - (k * k)) / norm_ior) + (2.0f * k * q_imaginary / norm_ior)) / norm_ior;
  } else {
    s_admittance = bsdf_complex_real(stack.terminal_q) / eta;
    p_admittance = bsdf_complex_real(stack.terminal_y) / eta;
  }
  return 0.5f * ((bsdf_complex_norm(stack.s.transmission) * s_admittance / bsdf_complex_real(stack.incident_q)) +
                  (bsdf_complex_norm(stack.p.transmission) * p_admittance / bsdf_complex_real(stack.incident_y)));
}

ETX_SHARED_NOINLINE float bsdf_fresnel_multilayer_terminal_response(float wavelength, float cosine, ETX_IN(complex, ext_ior), ETX_IN(complex, int_ior),
  ETX_IN(BSDFFresnelLayerStack, layers)) {
  if (cosine <= 0.0f) {
    return 0.0f;
  }
  return bsdf_fresnel_terminal_response_from_stack(bsdf_fresnel_multilayer_amplitudes(wavelength, cosine, ext_ior, int_ior, layers), int_ior);
}

ETX_SHARED_INLINE float bsdf_fresnel_thinfilm(float wavelength, float cos_theta_0, ETX_IN(complex, ext_ior), ETX_IN(complex, film_ior), ETX_IN(complex, int_ior), float thickness) {
  return bsdf_fresnel_thinfilm_power(wavelength, cos_theta_0, ext_ior, film_ior, int_ior, thickness).x;
}

ETX_SHARED_INLINE CoherentInterfaceEval bsdf_fresnel_orient_coherent_interface(ETX_IN(CoherentInterfaceEval, optics), bool incident_outside) {
  CoherentInterfaceEval result = optics;
  result.incident_inside = incident_outside ? 0u : 1u;
  return result;
}

ETX_SHARED_INLINE ThinfilmEval bsdf_fresnel_orient_coherent_interface(ETX_IN(ThinfilmEval, optics), bool incident_outside) {
  ThinfilmEval result = optics;
  result.incident_inside = incident_outside ? 0u : 1u;
  return result;
}

ETX_SHARED_INLINE CoherentInterfaceEval bsdf_fresnel_coherent_interface(ETX_IN(ThinfilmEval, optics)) {
  CoherentInterfaceEval result = ETX_ZERO(CoherentInterfaceEval);
  result.thinfilm = optics;
  result.coating_ior = optics.coating_ior;
  result.coating_thickness_nm = optics.coating_thickness_nm;
  result.incident_inside = optics.incident_inside;
  return result;
}

// Transmission is power entering the terminal medium, including substrate absorption for an opaque conductor.
ETX_SHARED_NOINLINE float2 bsdf_fresnel_coherent_interface_power(float wavelength, float cosine, ETX_IN(complex, exterior), ETX_IN(complex, interior), ETX_IN(complex, film_ior),
  ETX_IN(complex, coating_ior), float film_thickness, float film_weight, float coating_thickness, bool incident_inside) {
  BSDFFresnelLayerStack layers = ETX_ZERO(BSDFFresnelLayerStack);
  if (coating_thickness > 0.0f) {
    layers.ior[0] = coating_ior;
    layers.thickness[0] = coating_thickness;
    layers.count = 1u;
  }
  const float2 coating_power = bsdf_fresnel_multilayer_power(wavelength, cosine, exterior, interior, layers);
  if ((film_thickness <= 0.0f) || (film_weight <= 0.0f)) {
    return coating_power;
  }
  const uint32_t film_index = incident_inside ? layers.count : 0u;
  if ((incident_inside == false) && (layers.count != 0u)) {
    layers.ior[1] = layers.ior[0];
    layers.thickness[1] = layers.thickness[0];
  }
  layers.ior[film_index] = film_ior;
  layers.thickness[film_index] = film_thickness;
  ++layers.count;
  const float2 combined_power = bsdf_fresnel_multilayer_power(wavelength, cosine, exterior, interior, layers);
  const float weight = saturate(film_weight);
#if ETX_CPP
  const float2 result = (1.0f - weight) * coating_power + weight * combined_power;
#else
  precise float2 result = (1.0f - weight) * coating_power + weight * combined_power;
#endif
  return result;
}

ETX_SHARED_NOINLINE float2 bsdf_fresnel_coherent_terminal_power_with_sine_squared(float wavelength, float cosine, float sine_squared, ETX_IN(complex, exterior),
  ETX_IN(complex, interior), ETX_IN(complex, film_ior), ETX_IN(complex, coating_ior), float film_thickness, float film_weight, float coating_thickness, bool incident_inside) {
  if (cosine <= 0.0f) {
    return float2(1.0f, 0.0f);
  }
  BSDFFresnelLayerStack layers = ETX_ZERO(BSDFFresnelLayerStack);
  if (coating_thickness > 0.0f) {
    layers.ior[0] = coating_ior;
    layers.thickness[0] = coating_thickness;
    layers.count = 1u;
  }
  const BSDFFresnelPolarizedStack coating_stack = bsdf_fresnel_multilayer_amplitudes_with_sine_squared(wavelength, cosine, sine_squared, exterior, interior, layers);
  const float2 coating_response = float2(0.5f * (bsdf_complex_norm(coating_stack.s.reflection) + bsdf_complex_norm(coating_stack.p.reflection)),
    bsdf_fresnel_terminal_response_from_stack(coating_stack, interior));
  if ((film_thickness <= 0.0f) || (film_weight <= 0.0f)) {
    return coating_response;
  }
  const uint32_t film_index = incident_inside ? layers.count : 0u;
  if ((incident_inside == false) && (layers.count != 0u)) {
    layers.ior[1] = layers.ior[0];
    layers.thickness[1] = layers.thickness[0];
  }
  layers.ior[film_index] = film_ior;
  layers.thickness[film_index] = film_thickness;
  ++layers.count;
  const BSDFFresnelPolarizedStack combined_stack = bsdf_fresnel_multilayer_amplitudes_with_sine_squared(wavelength, cosine, sine_squared, exterior, interior, layers);
  const float2 combined_response = float2(0.5f * (bsdf_complex_norm(combined_stack.s.reflection) + bsdf_complex_norm(combined_stack.p.reflection)),
    bsdf_fresnel_terminal_response_from_stack(combined_stack, interior));
  const float weight = saturate(film_weight);
#if ETX_CPP
  const float2 result = ((1.0f - weight) * coating_response) + (weight * combined_response);
#else
  precise float2 result = ((1.0f - weight) * coating_response) + (weight * combined_response);
#endif
  return result;
}

ETX_SHARED_NOINLINE float bsdf_fresnel_coherent_terminal_response(float wavelength, float cosine, ETX_IN(complex, exterior), ETX_IN(complex, interior), ETX_IN(complex, film_ior),
  ETX_IN(complex, coating_ior), float film_thickness, float film_weight, float coating_thickness, bool incident_inside) {
  return bsdf_fresnel_coherent_terminal_power_with_sine_squared(wavelength, cosine, (1.0f - cosine) * (1.0f + cosine), exterior, interior, film_ior, coating_ior, film_thickness,
    film_weight, coating_thickness, incident_inside)
    .y;
}

ETX_SHARED_NOINLINE SpectralResponse bsdf_fresnel_terminal_source_calculate_with_sine_squared(ETX_IN(SpectralQuery, spect), float cosine, float sine_squared,
  ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(CoherentInterfaceEval, optics)) {
  const ThinfilmEval film = optics.thinfilm;
  const float mu = saturate(abs(cosine));
  const bool incident_inside = optics.incident_inside != 0u;
  if (spectral_query_is_spectral(spect)) {
    const float response = bsdf_fresnel_coherent_terminal_power_with_sine_squared(spect.wavelength, mu, sine_squared, refractive_index_sample_as_complex_spectral(ext_ior),
      refractive_index_sample_as_complex_spectral(int_ior), refractive_index_sample_as_complex_spectral(film.ior), refractive_index_sample_as_complex_spectral(optics.coating_ior),
      film.thickness, film.weight, optics.coating_thickness_nm, incident_inside)
                             .y;
    return spectral_response_make(spect, response);
  }
  const float r = bsdf_fresnel_coherent_terminal_power_with_sine_squared(film.rgb_wavelengths.x, mu, sine_squared, refractive_index_sample_as_complex_x(ext_ior),
    refractive_index_sample_as_complex_x(int_ior), refractive_index_sample_as_complex_x(film.ior), refractive_index_sample_as_complex_x(optics.coating_ior), film.thickness,
    film.weight, optics.coating_thickness_nm, incident_inside)
                    .y;
  const float g = bsdf_fresnel_coherent_terminal_power_with_sine_squared(film.rgb_wavelengths.y, mu, sine_squared, refractive_index_sample_as_complex_y(ext_ior),
    refractive_index_sample_as_complex_y(int_ior), refractive_index_sample_as_complex_y(film.ior), refractive_index_sample_as_complex_y(optics.coating_ior), film.thickness,
    film.weight, optics.coating_thickness_nm, incident_inside)
                    .y;
  const float b = bsdf_fresnel_coherent_terminal_power_with_sine_squared(film.rgb_wavelengths.z, mu, sine_squared, refractive_index_sample_as_complex_z(ext_ior),
    refractive_index_sample_as_complex_z(int_ior), refractive_index_sample_as_complex_z(film.ior), refractive_index_sample_as_complex_z(optics.coating_ior), film.thickness,
    film.weight, optics.coating_thickness_nm, incident_inside)
                    .y;
  return spectral_response_make(spect, float3(r, g, b));
}

ETX_SHARED_NOINLINE SpectralResponse bsdf_fresnel_terminal_source_calculate(ETX_IN(SpectralQuery, spect), float cosine, ETX_IN(RefractiveIndexSample, ext_ior),
  ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(CoherentInterfaceEval, optics)) {
  const float mu = saturate(abs(cosine));
  return bsdf_fresnel_terminal_source_calculate_with_sine_squared(spect, mu, (1.0f - mu) * (1.0f + mu), ext_ior, int_ior, optics);
}

ETX_SHARED_NOINLINE SpectralResponse bsdf_fresnel_coherent_interface_response(ETX_IN(SpectralQuery, spect), float cosine, ETX_IN(RefractiveIndexSample, ext_ior),
  ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(CoherentInterfaceEval, optics), bool transmission) {
  const ThinfilmEval film = optics.thinfilm;
  const float mu = saturate(abs(cosine));
  const bool incident_inside = optics.incident_inside != 0u;
  if (spectral_query_is_spectral(spect)) {
    const float2 power = bsdf_fresnel_coherent_interface_power(spect.wavelength, mu, refractive_index_sample_as_complex_spectral(ext_ior),
      refractive_index_sample_as_complex_spectral(int_ior), refractive_index_sample_as_complex_spectral(film.ior), refractive_index_sample_as_complex_spectral(optics.coating_ior),
      film.thickness, film.weight, optics.coating_thickness_nm, incident_inside);
    return spectral_response_make(spect, transmission ? power.y : power.x);
  }
  const float2 power_r = bsdf_fresnel_coherent_interface_power(film.rgb_wavelengths.x, mu, refractive_index_sample_as_complex_x(ext_ior),
    refractive_index_sample_as_complex_x(int_ior), refractive_index_sample_as_complex_x(film.ior), refractive_index_sample_as_complex_x(optics.coating_ior), film.thickness,
    film.weight, optics.coating_thickness_nm, incident_inside);
  const float2 power_g = bsdf_fresnel_coherent_interface_power(film.rgb_wavelengths.y, mu, refractive_index_sample_as_complex_y(ext_ior),
    refractive_index_sample_as_complex_y(int_ior), refractive_index_sample_as_complex_y(film.ior), refractive_index_sample_as_complex_y(optics.coating_ior), film.thickness,
    film.weight, optics.coating_thickness_nm, incident_inside);
  const float2 power_b = bsdf_fresnel_coherent_interface_power(film.rgb_wavelengths.z, mu, refractive_index_sample_as_complex_z(ext_ior),
    refractive_index_sample_as_complex_z(int_ior), refractive_index_sample_as_complex_z(film.ior), refractive_index_sample_as_complex_z(optics.coating_ior), film.thickness,
    film.weight, optics.coating_thickness_nm, incident_inside);
  const float3 value = transmission ? float3(power_r.y, power_g.y, power_b.y) : float3(power_r.x, power_g.x, power_b.x);
  return spectral_response_make(spect, value);
}

ETX_SHARED_INLINE SpectralResponse bsdf_fresnel_calculate(ETX_IN(SpectralQuery, spect), float cosine, ETX_IN(RefractiveIndexSample, ext_ior),
  ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(CoherentInterfaceEval, optics)) {
  return bsdf_fresnel_coherent_interface_response(spect, cosine, ext_ior, int_ior, optics, false);
}

ETX_SHARED_INLINE SpectralResponse bsdf_fresnel_transmission_calculate(ETX_IN(SpectralQuery, spect), float cosine, ETX_IN(RefractiveIndexSample, ext_ior),
  ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(CoherentInterfaceEval, optics)) {
  return bsdf_fresnel_coherent_interface_response(spect, cosine, ext_ior, int_ior, optics, true);
}

ETX_SHARED_INLINE bool bsdf_fresnel_thinfilm_absorbs(ETX_IN(ThinfilmEval, film)) {
  return (film.weight > 0.0f) && (film.thickness > 0.0f) && (spectral_response_is_zero(spectral_response_mul(film.ior.eta, film.ior.k)) == false);
}

ETX_SHARED_INLINE bool bsdf_fresnel_film_absorbs(ETX_IN(ThinfilmEval, film)) {
#if ETX_ENABLE_COHERENT_COATING
  return (bsdf_fresnel_thinfilm_absorbs(film)) ||
         ((film.coating_thickness_nm > 0.0f) && (spectral_response_is_zero(spectral_response_mul(film.coating_ior.eta, film.coating_ior.k)) == false));
#else
  return bsdf_fresnel_thinfilm_absorbs(film);
#endif
}

ETX_SHARED_NOINLINE SpectralResponse bsdf_fresnel_thinfilm_calculate(ETX_IN(SpectralQuery, spect), float cos_theta, ETX_IN(RefractiveIndexSample, ext_ior),
  ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, thinfilm)) {
  const float abs_cos_theta = saturate(abs(cos_theta));
  const float film_weight = saturate(thinfilm.weight);
  const bool film_enabled =
    (film_weight > 0.0f) && (thinfilm.thickness > 0.0f) && ((spectral_response_is_zero(thinfilm.ior.eta) == false) || (spectral_response_is_zero(thinfilm.ior.k) == false));
  SpectralResponse result = spectral_response_make(spect, 0.0f);

  if (spectral_query_is_spectral(spect)) {
    const float bare_value = bsdf_fresnel_generic(abs_cos_theta, refractive_index_sample_as_complex_spectral(ext_ior), refractive_index_sample_as_complex_spectral(int_ior));
    float value = bare_value;
    if (film_enabled) {
      const float coated_value = bsdf_fresnel_thinfilm(spect.wavelength, abs_cos_theta, refractive_index_sample_as_complex_spectral(ext_ior),
        refractive_index_sample_as_complex_spectral(thinfilm.ior), refractive_index_sample_as_complex_spectral(int_ior), thinfilm.thickness);
      value = bare_value + (coated_value - bare_value) * film_weight;
    }

    result.value = saturate(value);
    return result;
  }

  float3 bare_values = float3(0.0f, 0.0f, 0.0f);
  bare_values.x = bsdf_fresnel_generic(abs_cos_theta, refractive_index_sample_as_complex_x(ext_ior), refractive_index_sample_as_complex_x(int_ior));
  bare_values.y = bsdf_fresnel_generic(abs_cos_theta, refractive_index_sample_as_complex_y(ext_ior), refractive_index_sample_as_complex_y(int_ior));
  bare_values.z = bsdf_fresnel_generic(abs_cos_theta, refractive_index_sample_as_complex_z(ext_ior), refractive_index_sample_as_complex_z(int_ior));

  float3 values = bare_values;
  if (film_enabled) {
    float3 coated_values = float3(0.0f, 0.0f, 0.0f);
    coated_values.x = bsdf_fresnel_thinfilm(thinfilm.rgb_wavelengths.x, abs_cos_theta, refractive_index_sample_as_complex_x(ext_ior),
      refractive_index_sample_as_complex_x(thinfilm.ior), refractive_index_sample_as_complex_x(int_ior), thinfilm.thickness);
    coated_values.y = bsdf_fresnel_thinfilm(thinfilm.rgb_wavelengths.y, abs_cos_theta, refractive_index_sample_as_complex_y(ext_ior),
      refractive_index_sample_as_complex_y(thinfilm.ior), refractive_index_sample_as_complex_y(int_ior), thinfilm.thickness);
    coated_values.z = bsdf_fresnel_thinfilm(thinfilm.rgb_wavelengths.z, abs_cos_theta, refractive_index_sample_as_complex_z(ext_ior),
      refractive_index_sample_as_complex_z(thinfilm.ior), refractive_index_sample_as_complex_z(int_ior), thinfilm.thickness);
    values = lerp(bare_values, coated_values, film_weight);
  }

  result.integrated = saturate(values);
  result.value = luminance(result.integrated);
  return result;
}

ETX_SHARED_INLINE SpectralResponse bsdf_fresnel_calculate(ETX_IN(SpectralQuery, spect), float cos_theta, ETX_IN(RefractiveIndexSample, ext_ior),
  ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, thinfilm)) {
#if ETX_ENABLE_COHERENT_COATING
  if (thinfilm.coating_thickness_nm > 0.0f) {
    return bsdf_fresnel_calculate(spect, cos_theta, ext_ior, int_ior, bsdf_fresnel_coherent_interface(thinfilm));
  }
#endif
  return bsdf_fresnel_thinfilm_calculate(spect, cos_theta, ext_ior, int_ior, thinfilm);
}

ETX_SHARED_NOINLINE SpectralResponse bsdf_fresnel_thinfilm_transmission_calculate(ETX_IN(SpectralQuery, spect), float cos_theta, ETX_IN(RefractiveIndexSample, ext_ior),
  ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, thinfilm)) {
  const float mu = saturate(abs(cos_theta));
  const SpectralResponse one = spectral_response_make(spect, 1.0f);
  if (bsdf_fresnel_thinfilm_absorbs(thinfilm) == false) {
    return spectral_response_sub(one, bsdf_fresnel_thinfilm_calculate(spect, mu, ext_ior, int_ior, thinfilm));
  }
  const float weight = saturate(thinfilm.weight);
  if (spectral_query_is_spectral(spect)) {
    const complex exterior = refractive_index_sample_as_complex_spectral(ext_ior);
    const complex interior = refractive_index_sample_as_complex_spectral(int_ior);
    const float bare = 1.0f - bsdf_fresnel_generic(mu, exterior, interior);
    const float coated = bsdf_fresnel_thinfilm_power(spect.wavelength, mu, exterior, refractive_index_sample_as_complex_spectral(thinfilm.ior), interior, thinfilm.thickness).y;
    return spectral_response_make(spect, saturate(bare + weight * (coated - bare)));
  }
  const float3 bare = float3(1.0f - bsdf_fresnel_generic(mu, refractive_index_sample_as_complex_x(ext_ior), refractive_index_sample_as_complex_x(int_ior)),
    1.0f - bsdf_fresnel_generic(mu, refractive_index_sample_as_complex_y(ext_ior), refractive_index_sample_as_complex_y(int_ior)),
    1.0f - bsdf_fresnel_generic(mu, refractive_index_sample_as_complex_z(ext_ior), refractive_index_sample_as_complex_z(int_ior)));
  const float3 coated = float3(bsdf_fresnel_thinfilm_power(thinfilm.rgb_wavelengths.x, mu, refractive_index_sample_as_complex_x(ext_ior),
                                 refractive_index_sample_as_complex_x(thinfilm.ior), refractive_index_sample_as_complex_x(int_ior), thinfilm.thickness)
                                 .y,
    bsdf_fresnel_thinfilm_power(thinfilm.rgb_wavelengths.y, mu, refractive_index_sample_as_complex_y(ext_ior), refractive_index_sample_as_complex_y(thinfilm.ior),
      refractive_index_sample_as_complex_y(int_ior), thinfilm.thickness)
      .y,
    bsdf_fresnel_thinfilm_power(thinfilm.rgb_wavelengths.z, mu, refractive_index_sample_as_complex_z(ext_ior), refractive_index_sample_as_complex_z(thinfilm.ior),
      refractive_index_sample_as_complex_z(int_ior), thinfilm.thickness)
      .y);
  return spectral_response_make(spect, saturate(lerp(bare, coated, weight)));
}

ETX_SHARED_INLINE SpectralResponse bsdf_fresnel_transmission_calculate(ETX_IN(SpectralQuery, spect), float cos_theta, ETX_IN(RefractiveIndexSample, ext_ior),
  ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, thinfilm)) {
#if ETX_ENABLE_COHERENT_COATING
  if (thinfilm.coating_thickness_nm > 0.0f) {
    return bsdf_fresnel_transmission_calculate(spect, cos_theta, ext_ior, int_ior, bsdf_fresnel_coherent_interface(thinfilm));
  }
#endif
  return bsdf_fresnel_thinfilm_transmission_calculate(spect, cos_theta, ext_ior, int_ior, thinfilm);
}

ETX_SHARED_NOINLINE float2 bsdf_fresnel_coherent_terminal_average(float wavelength, ETX_IN(complex, exterior), ETX_IN(complex, interior), ETX_IN(complex, film_ior),
  ETX_IN(complex, coating_ior), float film_thickness, float film_weight, float coating_thickness, bool incident_inside) {
  const float nodes[8] = {1.98550718e-2f, 1.01666761e-1f, 2.37233795e-1f, 4.08282679e-1f, 5.91717321e-1f, 7.62766205e-1f, 8.98333239e-1f, 9.80144928e-1f};
  const float weights[8] = {5.06142681e-2f, 1.11190517e-1f, 1.56853323e-1f, 1.81341892e-1f, 1.81341892e-1f, 1.56853323e-1f, 1.11190517e-1f, 5.06142681e-2f};
  const float eta = bsdf_complex_real(interior) / bsdf_complex_real(exterior);
  const float cone_fraction = min(1.0f, eta * eta);
  float2 result = float2(0.0f, 0.0f);
  for (uint32_t region = 0u; region < 2u; ++region) {
    const float fraction = region == 0u ? cone_fraction : (1.0f - cone_fraction);
    if (fraction <= 0.0f) {
      continue;
    }
    for (uint32_t segment = 0u; segment < 4u; ++segment) {
      for (uint32_t index = 0u; index < 8u; ++index) {
        const float ordinate = 0.25f * (float(segment) + nodes[index]);
        float sine_squared = 0.0f;
        float cosine_squared = 0.0f;
        float measure = fraction * ordinate * ordinate * ordinate;
        if (region == 0u) {
          // Integrate the whole transmission cone even when its cosine rounds to one.
          const float transverse = ordinate * ordinate;
          sine_squared = cone_fraction * (1.0f - transverse) * (1.0f + transverse);
          cosine_squared = 1.0f - sine_squared;
        } else {
          // Smooth both the critical-angle continuation and the grazing endpoint.
          const float angle = kHalfPi * ordinate;
          const float sine = sin(angle);
          const float cosine = cos(angle);
          const float transverse = sine * sine;
          sine_squared = cone_fraction + (fraction * transverse * transverse);
          cosine_squared = fraction * cosine * cosine * (1.0f + transverse);
          measure = kHalfPi * fraction * sine * transverse * cosine;
        }
        const float cosine = sqrt(max(0.0f, cosine_squared));
        const float2 value = bsdf_fresnel_coherent_terminal_power_with_sine_squared(wavelength, cosine, sine_squared, exterior, interior, film_ior, coating_ior, film_thickness,
          film_weight, coating_thickness, incident_inside);
        result += value * (weights[index] * measure);
      }
    }
  }
  return result;
}

ETX_SHARED_NOINLINE SpectralResponse bsdf_fresnel_coherent_terminal_average(ETX_IN(SpectralQuery, spect), ETX_IN(RefractiveIndexSample, ext_ior),
  ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, optics), bool terminal) {
  const bool incident_inside = optics.incident_inside != 0u;
  if (spectral_query_is_spectral(spect)) {
    const float2 value = bsdf_fresnel_coherent_terminal_average(spect.wavelength, refractive_index_sample_as_complex_spectral(ext_ior),
      refractive_index_sample_as_complex_spectral(int_ior), refractive_index_sample_as_complex_spectral(optics.ior),
      refractive_index_sample_as_complex_spectral(optics.coating_ior), optics.thickness, optics.weight, optics.coating_thickness_nm, incident_inside);
    return spectral_response_make(spect, terminal ? value.y : value.x);
  }
  const float2 r = bsdf_fresnel_coherent_terminal_average(optics.rgb_wavelengths.x, refractive_index_sample_as_complex_x(ext_ior), refractive_index_sample_as_complex_x(int_ior),
    refractive_index_sample_as_complex_x(optics.ior), refractive_index_sample_as_complex_x(optics.coating_ior), optics.thickness, optics.weight, optics.coating_thickness_nm,
    incident_inside);
  const float2 g = bsdf_fresnel_coherent_terminal_average(optics.rgb_wavelengths.y, refractive_index_sample_as_complex_y(ext_ior), refractive_index_sample_as_complex_y(int_ior),
    refractive_index_sample_as_complex_y(optics.ior), refractive_index_sample_as_complex_y(optics.coating_ior), optics.thickness, optics.weight, optics.coating_thickness_nm,
    incident_inside);
  const float2 b = bsdf_fresnel_coherent_terminal_average(optics.rgb_wavelengths.z, refractive_index_sample_as_complex_z(ext_ior), refractive_index_sample_as_complex_z(int_ior),
    refractive_index_sample_as_complex_z(optics.ior), refractive_index_sample_as_complex_z(optics.coating_ior), optics.thickness, optics.weight, optics.coating_thickness_nm,
    incident_inside);
  return spectral_response_make(spect, terminal ? float3(r.y, g.y, b.y) : float3(r.x, g.x, b.x));
}

ETX_SHARED_INLINE SpectralResponse bsdf_fresnel_average(ETX_IN(SpectralQuery, spect), ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior),
  ETX_IN(ThinfilmEval, thinfilm)) {
#if ETX_ENABLE_COHERENT_COATING
  if ((int_ior.cls == SpectralDistribution::Conductor) &&
      ((thinfilm.coating_ior.cls == SpectralDistribution::Dielectric) || (thinfilm.coating_ior.cls == SpectralDistribution::Conductor))) {
    return bsdf_fresnel_coherent_terminal_average(spect, ext_ior, int_ior, thinfilm, false);
  }
#endif
  const float nodes[8] = {
    1.98550718e-2f,
    1.01666761e-1f,
    2.37233795e-1f,
    4.08282679e-1f,
    5.91717321e-1f,
    7.62766205e-1f,
    8.98333239e-1f,
    9.80144928e-1f,
  };
  const float weights[8] = {
    5.06142681e-2f,
    1.11190517e-1f,
    1.56853323e-1f,
    1.81341892e-1f,
    1.81341892e-1f,
    1.56853323e-1f,
    1.11190517e-1f,
    5.06142681e-2f,
  };

  SpectralResponse result = spectral_response_make(spect, 0.0f);
  for (uint32_t i = 0u; i < 8u; ++i) {
    const float mu = nodes[i];
    const SpectralResponse fresnel = bsdf_fresnel_calculate(spect, mu, ext_ior, int_ior, thinfilm);
    result = spectral_response_add(result, spectral_response_mul(fresnel, 2.0f * weights[i] * mu));
  }
  return result;
}

ETX_SHARED_INLINE float3 bsdf_fresnel_longitudinal_turning_cosines(ETX_IN(SpectralQuery, spect), ETX_IN(RefractiveIndexSample, incident),
  ETX_IN(RefractiveIndexSample, transmitted)) {
  const SpectralResponse eta = spectral_response_div(transmitted.eta, incident.eta);
  const SpectralResponse k = spectral_response_div(transmitted.k, incident.eta);
  const SpectralResponse real_index_squared = spectral_response_sub(spectral_response_mul(eta, eta), spectral_response_mul(k, k));
  const SpectralResponse value =
    spectral_response_sqrt(spectral_response_min(spectral_response_max(spectral_response_sub(spectral_response_make(spect, 1.0f), real_index_squared), 0.0f), 1.0f));
  return spectral_query_is_spectral(spect) ? float3(value.value, value.value, value.value) : value.integrated;
}

struct BSDFFresnelAverageDomain {
  float boundaries[11];
};

ETX_SHARED_INLINE BSDFFresnelAverageDomain bsdf_fresnel_average_domain(ETX_IN(SpectralQuery, spect), ETX_IN(RefractiveIndexSample, incident),
  ETX_IN(RefractiveIndexSample, transmitted), ETX_IN(ThinfilmEval, film), bool transmission) {
  const float3 terminal = bsdf_fresnel_longitudinal_turning_cosines(spect, incident, transmitted);
  const float minimum = transmission ? min(terminal.x, min(terminal.y, terminal.z)) : 0.0f;
  // Finite-layer Re(q^2)=0 cuts partition quadrature without removing any absorption or transmission.
  float3 film_turning = float3(0.0f, 0.0f, 0.0f);
  if ((film.thickness > 0.0f) && (film.weight > 0.0f)) {
    film_turning = bsdf_fresnel_longitudinal_turning_cosines(spect, incident, film.ior);
  }
  float3 coating_turning = float3(0.0f, 0.0f, 0.0f);
#if ETX_ENABLE_COHERENT_COATING
  if (film.coating_thickness_nm > 0.0f) {
    coating_turning = bsdf_fresnel_longitudinal_turning_cosines(spect, incident, film.coating_ior);
  }
#endif
  BSDFFresnelAverageDomain result = ETX_ZERO(BSDFFresnelAverageDomain);
  result.boundaries[0] = minimum;
  result.boundaries[1] = terminal.x;
  result.boundaries[2] = terminal.y;
  result.boundaries[3] = terminal.z;
  result.boundaries[4] = max(minimum, film_turning.x);
  result.boundaries[5] = max(minimum, film_turning.y);
  result.boundaries[6] = max(minimum, film_turning.z);
  result.boundaries[7] = max(minimum, coating_turning.x);
  result.boundaries[8] = max(minimum, coating_turning.y);
  result.boundaries[9] = max(minimum, coating_turning.z);
  result.boundaries[10] = 1.0f;
  for (uint32_t i = 1u; i < 10u; ++i) {
    const float value = result.boundaries[i];
    uint32_t j = i;
    while (j > 0u) {
      if (result.boundaries[j - 1u] <= value) {
        break;
      }
      result.boundaries[j] = result.boundaries[j - 1u];
      --j;
    }
    result.boundaries[j] = value;
  }
  return result;
}

ETX_SHARED_NOINLINE SpectralResponse bsdf_fresnel_transmission_average(ETX_IN(SpectralQuery, spect), ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior),
  ETX_IN(ThinfilmEval, thinfilm)) {
  const float nodes[8] = {1.98550718e-2f, 1.01666761e-1f, 2.37233795e-1f, 4.08282679e-1f, 5.91717321e-1f, 7.62766205e-1f, 8.98333239e-1f, 9.80144928e-1f};
  const float weights[8] = {5.06142681e-2f, 1.11190517e-1f, 1.56853323e-1f, 1.81341892e-1f, 1.81341892e-1f, 1.56853323e-1f, 1.11190517e-1f, 5.06142681e-2f};
  SpectralResponse result = spectral_response_make(spect, 0.0f);
  const BSDFFresnelAverageDomain domain = bsdf_fresnel_average_domain(spect, ext_ior, int_ior, thinfilm, true);
  const uint32_t segment_count = domain.boundaries[9] > 0.0f ? 4u : 1u;
  for (uint32_t region = 0u; region < 10u; ++region) {
    const float low = domain.boundaries[region];
    const float high = domain.boundaries[region + 1u];
    if (high <= low) {
      continue;
    }
    const float width = (high - low) / float(segment_count);
    for (uint32_t segment = 0u; segment < segment_count; ++segment) {
      for (uint32_t i = 0u; i < 8u; ++i) {
        const float mu = low + width * (float(segment) + nodes[i]);
        result = spectral_response_add(result, spectral_response_mul(bsdf_fresnel_transmission_calculate(spect, mu, ext_ior, int_ior, thinfilm), 2.0f * width * weights[i] * mu));
      }
    }
  }
  return result;
}

ETX_SHARED_NOINLINE SpectralResponse bsdf_fresnel_terminal_source_average(ETX_IN(SpectralQuery, spect), ETX_IN(RefractiveIndexSample, ext_ior),
  ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, optics)) {
  return bsdf_fresnel_coherent_terminal_average(spect, ext_ior, int_ior, optics, true);
}

ETX_SHARED_NOINLINE SpectralResponse bsdf_fresnel_film_internal_average(ETX_IN(SpectralQuery, spect), ETX_IN(RefractiveIndexSample, exterior),
  ETX_IN(RefractiveIndexSample, interior), ETX_IN(ThinfilmEval, film)) {
  const SpectralResponse eta = spectral_response_div(interior.eta, exterior.eta);
  if (bsdf_fresnel_film_absorbs(film) == false) {
    // Reciprocity includes the solid-angle Jacobian and avoids quadrature across the TIR discontinuity.
    return spectral_response_sub(spectral_response_make(spect, 1.0f),
      spectral_response_div(bsdf_fresnel_transmission_average(spect, exterior, interior, film), spectral_response_mul(eta, eta)));
  }
  const float nodes[8] = {1.98550718e-2f, 1.01666761e-1f, 2.37233795e-1f, 4.08282679e-1f, 5.91717321e-1f, 7.62766205e-1f, 8.98333239e-1f, 9.80144928e-1f};
  const float weights[8] = {5.06142681e-2f, 1.11190517e-1f, 1.56853323e-1f, 1.81341892e-1f, 1.81341892e-1f, 1.56853323e-1f, 1.11190517e-1f, 5.06142681e-2f};
  const BSDFFresnelAverageDomain domain = bsdf_fresnel_average_domain(spect, interior, exterior, film, false);
  SpectralResponse result = spectral_response_make(spect, 0.0f);
  for (uint32_t region = 0u; region < 10u; ++region) {
    const float region_low = domain.boundaries[region];
    const float region_high = domain.boundaries[region + 1u];
    if (region_high <= region_low) {
      continue;
    }
    for (uint32_t segment = 0u; segment < 4u; ++segment) {
      const float low = region_low + (region_high - region_low) * float(segment) / 4.0f;
      const float high = region_low + (region_high - region_low) * float(segment + 1u) / 4.0f;
      for (uint32_t i = 0u; i < 8u; ++i) {
        const float mu = low + (high - low) * nodes[i];
        const SpectralResponse reflection = bsdf_fresnel_calculate(spect, mu, interior, exterior, bsdf_fresnel_orient_coherent_interface(film, false));
        result = spectral_response_add(result, spectral_response_mul(reflection, 2.0f * (high - low) * weights[i] * mu));
      }
    }
  }
  return result;
}
