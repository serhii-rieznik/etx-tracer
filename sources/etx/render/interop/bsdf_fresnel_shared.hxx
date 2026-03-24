#pragma once

#include "bsdf_core_shared.hxx"

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

ETX_SHARED_INLINE complex bsdf_complex_conjugate(ETX_IN(complex, value)) {
  return bsdf_complex_make(bsdf_complex_real(value), -bsdf_complex_imag(value));
}

ETX_SHARED_INLINE complex bsdf_complex_div(ETX_IN(complex, a), ETX_IN(complex, b)) {
#if (ETX_CPP)
  return a / b;
#else
  complex numerator = bsdf_complex_mul(a, bsdf_complex_conjugate(b));
  float denominator = bsdf_complex_norm(b);
  return bsdf_complex_div_scalar(numerator, denominator);
#endif
}

ETX_SHARED_INLINE complex bsdf_complex_sqrt(ETX_IN(complex, value)) {
#if (ETX_CPP)
  return complex_sqrt(value);
#else
  float magnitude = bsdf_complex_abs(value);
  float real_part = sqrt(max(0.0f, 0.5f * (magnitude + bsdf_complex_real(value))));
  float imag_sign = (bsdf_complex_imag(value) < 0.0f) ? -1.0f : 1.0f;
  float imag_part = imag_sign * sqrt(max(0.0f, 0.5f * (magnitude - bsdf_complex_real(value))));
  return bsdf_complex_make(real_part, imag_part);
#endif
}

ETX_SHARED_INLINE complex bsdf_complex_exp(ETX_IN(complex, value)) {
#if (ETX_CPP)
  return complex_exp(value);
#else
  float exp_real = exp(bsdf_complex_real(value));
  return bsdf_complex_make(exp_real * cos(bsdf_complex_imag(value)), exp_real * sin(bsdf_complex_imag(value)));
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

ETX_SHARED_INLINE float bsdf_fresnel_generic(float cos_theta_i, ETX_IN(complex, ext_ior), ETX_IN(complex, int_ior)) {
  complex eta_ratio = bsdf_complex_div(ext_ior, int_ior);
  complex eta_ratio_squared = bsdf_complex_mul(eta_ratio, eta_ratio);
  complex sin_theta_o_squared = bsdf_complex_mul_scalar(eta_ratio_squared, 1.0f - cos_theta_i * cos_theta_i);
  complex cos_theta_o = bsdf_complex_sqrt(bsdf_complex_sub(bsdf_complex_make(1.0f, 0.0f), sin_theta_o_squared));
  BSDFFresnelReflectanceResult reflectance = bsdf_fresnel_reflectance(ext_ior, bsdf_complex_make(cos_theta_i, 0.0f), int_ior, cos_theta_o);
  return 0.5f * (bsdf_complex_norm(reflectance.rs) + bsdf_complex_norm(reflectance.rp));
}

ETX_SHARED_INLINE float bsdf_fresnel_thinfilm(float wavelength, float cos_theta_0, ETX_IN(complex, ext_ior), ETX_IN(complex, film_ior), ETX_IN(complex, int_ior), float thickness) {
  complex i = bsdf_complex_make(0.0f, 1.0f);

  if (cos_theta_0 == 0.0f) {
    return 0.0f;
  }

  complex ext_to_film = bsdf_complex_div(ext_ior, film_ior);
  complex sin_theta_1_squared = bsdf_complex_mul_scalar(bsdf_complex_mul(ext_to_film, ext_to_film), 1.0f - cos_theta_0 * cos_theta_0);
  if (bsdf_complex_real(sin_theta_1_squared) >= 1.0f) {
    return 1.0f;
  }

  complex cos_theta_1 = bsdf_complex_sqrt(bsdf_complex_sub(bsdf_complex_make(1.0f, 0.0f), sin_theta_1_squared));

  complex film_to_int = bsdf_complex_div(film_ior, int_ior);
  complex cos_theta_1_squared = bsdf_complex_mul(cos_theta_1, cos_theta_1);
  complex sin_theta_2_squared = bsdf_complex_mul(bsdf_complex_mul(film_to_int, film_to_int), bsdf_complex_sub(bsdf_complex_make(1.0f, 0.0f), cos_theta_1_squared));
  if (bsdf_complex_real(sin_theta_2_squared) >= 1.0f) {
    return 1.0f;
  }

  complex cos_theta_2 = bsdf_complex_sqrt(bsdf_complex_sub(bsdf_complex_make(1.0f, 0.0f), sin_theta_2_squared));
  complex ratio = bsdf_complex_div(bsdf_complex_mul(int_ior, cos_theta_2), bsdf_complex_mul(ext_ior, bsdf_complex_make(cos_theta_0, 0.0f)));

  float delta_10 = (bsdf_complex_real(ext_ior) < bsdf_complex_real(film_ior)) ? kPi : 0.0f;
  float delta_21 = (bsdf_complex_real(film_ior) < bsdf_complex_real(int_ior)) ? kPi : 0.0f;
  float phase_shift = delta_10 + delta_21;

  BSDFFresnelReflectanceResult r01 = bsdf_fresnel_reflectance(ext_ior, bsdf_complex_make(cos_theta_0, 0.0f), film_ior, cos_theta_1);
  BSDFFresnelTransmittanceResult t01 = bsdf_fresnel_transmittance(ext_ior, bsdf_complex_make(cos_theta_0, 0.0f), film_ior, cos_theta_1);
  BSDFFresnelReflectanceResult r12 = bsdf_fresnel_reflectance(film_ior, cos_theta_1, int_ior, cos_theta_2);
  BSDFFresnelTransmittanceResult t12 = bsdf_fresnel_transmittance(film_ior, cos_theta_1, int_ior, cos_theta_2);

  complex phase_term = bsdf_complex_add(bsdf_complex_mul_scalar(cos_theta_1, kDoublePi * 2.0f * thickness), bsdf_complex_mul_scalar(film_ior, phase_shift));
  complex phi = bsdf_complex_div_scalar(phase_term, wavelength);
  complex exp_i_phi = bsdf_complex_exp(bsdf_complex_mul(i, phi));

  complex tp_numerator = bsdf_complex_mul(t01.tp, t12.tp);
  complex tp_denominator = bsdf_complex_sub(bsdf_complex_make(1.0f, 0.0f), bsdf_complex_mul(bsdf_complex_mul(r01.rp, r12.rp), exp_i_phi));
  complex tp_base = bsdf_complex_div(tp_numerator, tp_denominator);
  complex tp = bsdf_complex_mul(tp_base, tp_base);

  complex ts_numerator = bsdf_complex_mul(t01.ts, t12.ts);
  complex ts_denominator = bsdf_complex_sub(bsdf_complex_make(1.0f, 0.0f), bsdf_complex_mul(bsdf_complex_mul(r01.rs, r12.rs), exp_i_phi));
  complex ts_base = bsdf_complex_div(ts_numerator, ts_denominator);
  complex ts = bsdf_complex_mul(ts_base, ts_base);

  complex transmission = bsdf_complex_mul_scalar(bsdf_complex_add(tp, ts), 0.5f);
  complex result = bsdf_complex_sub(bsdf_complex_make(1.0f, 0.0f), bsdf_complex_mul(ratio, transmission));
  return bsdf_complex_abs(result);
}

ETX_SHARED_INLINE SpectralResponse bsdf_fresnel_calculate(ETX_IN(SpectralQuery, spect), float cos_theta, ETX_IN(RefractiveIndexSample, ext_ior),
  ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, thinfilm)) {
  float abs_cos_theta = abs(cos_theta);
  SpectralResponse result = spectral_response_make(spect, 0.0f);

  if (spectral_query_is_spectral(spect)) {
    float value = 0.0f;
    if ((thinfilm.thickness == 0.0f) || spectral_response_is_zero(thinfilm.ior.eta)) {
      value = bsdf_fresnel_generic(abs_cos_theta, refractive_index_sample_as_complex_spectral(ext_ior), refractive_index_sample_as_complex_spectral(int_ior));
    } else {
      value = bsdf_fresnel_thinfilm(spect.wavelength, abs_cos_theta, refractive_index_sample_as_complex_spectral(ext_ior),
        refractive_index_sample_as_complex_spectral(thinfilm.ior), refractive_index_sample_as_complex_spectral(int_ior), thinfilm.thickness);
    }

    result.value = saturate(value);
    result.integrated = make_float3(result.value, result.value, result.value);
    return result;
  }

  float3 values = float3(0.0f, 0.0f, 0.0f);
  if ((thinfilm.thickness == 0.0f) || spectral_response_is_zero(thinfilm.ior.eta)) {
    values.x = bsdf_fresnel_generic(abs_cos_theta, refractive_index_sample_as_complex_x(ext_ior), refractive_index_sample_as_complex_x(int_ior));
    values.y = bsdf_fresnel_generic(abs_cos_theta, refractive_index_sample_as_complex_y(ext_ior), refractive_index_sample_as_complex_y(int_ior));
    values.z = bsdf_fresnel_generic(abs_cos_theta, refractive_index_sample_as_complex_z(ext_ior), refractive_index_sample_as_complex_z(int_ior));
    if (int_ior.cls == SpectralDistribution::Conductor) {
      values = spectral_xyz_to_rgb(values) * kSpectralDistributionRGBLuminanceScale;
    }
  } else {
    values.x = bsdf_fresnel_thinfilm(thinfilm.rgb_wavelengths.x, abs_cos_theta, refractive_index_sample_as_complex_x(ext_ior), refractive_index_sample_as_complex_x(thinfilm.ior),
      refractive_index_sample_as_complex_x(int_ior), thinfilm.thickness);
    values.y = bsdf_fresnel_thinfilm(thinfilm.rgb_wavelengths.y, abs_cos_theta, refractive_index_sample_as_complex_y(ext_ior), refractive_index_sample_as_complex_y(thinfilm.ior),
      refractive_index_sample_as_complex_y(int_ior), thinfilm.thickness);
    values.z = bsdf_fresnel_thinfilm(thinfilm.rgb_wavelengths.z, abs_cos_theta, refractive_index_sample_as_complex_z(ext_ior), refractive_index_sample_as_complex_z(thinfilm.ior),
      refractive_index_sample_as_complex_z(int_ior), thinfilm.thickness);
  }

  result.integrated = saturate(values);
  result.value = luminance(result.integrated);
  return result;
}
