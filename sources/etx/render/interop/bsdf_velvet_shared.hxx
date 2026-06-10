#pragma once

#include "bsdf_various_shared.hxx"

ETX_STATIC_CONST uint32_t kVelvetDirectionalAlbedoLutSize = 16u;
ETX_STATIC_CONST float kVelvetDirectionalAlbedoLut[kVelvetDirectionalAlbedoLutSize * kVelvetDirectionalAlbedoLutSize] = {
  0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
  0.00000000f, 0.66526577f, 0.45551903f, 0.32191254f, 0.22922222f, 0.16254533f, 0.11390004f, 0.07835285f, 0.05260917f, 0.03436360f, 0.02156526f, 0.01276981f, 0.00693857f, 0.00328658f, 0.00120032f, 0.00019185f,
  0.00000000f, 0.62559566f, 0.47108369f, 0.36609501f, 0.28777431f, 0.22660942f, 0.17769124f, 0.13810356f, 0.10605283f, 0.08054848f, 0.06004458f, 0.04345862f, 0.03007712f, 0.01940255f, 0.01106496f, 0.00476644f,
  0.00000000f, 0.62647779f, 0.49173446f, 0.39807856f, 0.32630442f, 0.26845107f, 0.22049081f, 0.18009804f, 0.14597210f, 0.11770350f, 0.09386545f, 0.07343666f, 0.05577803f, 0.04047969f, 0.02727031f, 0.01595222f,
  0.00000000f, 0.64177665f, 0.51577087f, 0.42732485f, 0.35865934f, 0.30241446f, 0.25490908f, 0.21405475f, 0.17877255f, 0.14898691f, 0.12329155f, 0.10065142f, 0.08042708f, 0.06222009f, 0.04578085f, 0.03093751f,
  0.00000000f, 0.66310572f, 0.54095038f, 0.45485461f, 0.38756175f, 0.33193952f, 0.28444920f, 0.24310710f, 0.20694964f, 0.17611689f, 0.14919348f, 0.12511571f, 0.10322860f, 0.08312821f, 0.06456834f, 0.04738643f,
  0.00000000f, 0.68661643f, 0.56579301f, 0.48054041f, 0.41367567f, 0.35811462f, 0.31036526f, 0.26848763f, 0.23158196f, 0.19993469f, 0.17211211f, 0.14702232f, 0.12399481f, 0.10261924f, 0.08265118f, 0.06393797f,
  0.00000000f, 0.71012269f, 0.58927656f, 0.50405517f, 0.43711012f, 0.38131260f, 0.33316854f, 0.29075069f, 0.25319477f, 0.22088879f, 0.19237871f, 0.16654865f, 0.14271578f, 0.12046560f, 0.09955706f, 0.07985027f,
  0.00000000f, 0.73219916f, 0.61065377f, 0.52506787f, 0.45780748f, 0.40165393f, 0.35308689f, 0.31017598f, 0.27207616f, 0.23924682f, 0.21021486f, 0.18384511f, 0.15944572f, 0.13660023f, 0.11507214f, 0.09473604f,
  0.00000000f, 0.75183251f, 0.62935875f, 0.54329104f, 0.47566847f, 0.41916754f, 0.37023266f, 0.32692352f, 0.28840530f, 0.25518855f, 0.22578410f, 0.19904191f, 0.17426390f, 0.15103309f, 0.12911808f, 0.10840816f,
  0.00000000f, 0.76826661f, 0.64495659f, 0.55849014f, 0.49059612f, 0.43385268f, 0.38467039f, 0.34109827f, 0.30230769f, 0.26884756f, 0.23921657f, 0.21225394f, 0.18725759f, 0.16381103f, 0.14168797f, 0.12079191f,
  0.00000000f, 0.78092658f, 0.65711396f, 0.57048188f, 0.50251075f, 0.44570315f, 0.39644487f, 0.35277799f, 0.31387970f, 0.28033003f, 0.25062028f, 0.22358273f, 0.19851322f, 0.17499650f, 0.15281133f, 0.13187456f,
  0.00000000f, 0.78937843f, 0.66558053f, 0.57912920f, 0.51135369f, 0.45471615f, 0.40559255f, 0.36202510f, 0.32319920f, 0.28972314f, 0.26008605f, 0.23311692f, 0.20811127f, 0.18465578f, 0.16253422f, 0.14167600f,
  0.00000000f, 0.79330626f, 0.67017657f, 0.58433592f, 0.51708653f, 0.46089465f, 0.41214568f, 0.36889149f, 0.33033016f, 0.29709864f, 0.26768953f, 0.24093210f, 0.21612318f, 0.19285211f, 0.17090776f, 0.15023143f,
  0.00000000f, 0.79249834f, 0.67078431f, 0.58604193f, 0.51968922f, 0.46424737f, 0.41613348f, 0.37342032f, 0.33532448f, 0.30251440f, 0.27349208f, 0.24709050f, 0.22260948f, 0.19964167f, 0.17798131f, 0.15758106f,
  0.00000000f, 0.78683825f, 0.66734201f, 0.58421965f, 0.51915813f, 0.46478816f, 0.41758242f, 0.37564673f, 0.33822292f, 0.30601521f, 0.27754118f, 0.25164091f, 0.22761874f, 0.20507131f, 0.18379852f, 0.16376397f,
};

ETX_SHARED_INLINE float bsdf_velvet_lambda_l(float roughness, float x) {
  const float clamped_x = max(x, 0.0f);
  const float inverse_lerp = 1.0f - roughness;
  const float inverse_lerp_squared = inverse_lerp * inverse_lerp;
  const float a = inverse_lerp_squared * 25.3245f + (1.0f - inverse_lerp_squared) * 21.5473f;
  const float b = inverse_lerp_squared * 3.32435f + (1.0f - inverse_lerp_squared) * 3.82987f;
  const float c = inverse_lerp_squared * 0.16801f + (1.0f - inverse_lerp_squared) * 0.19823f;
  const float d = inverse_lerp_squared * (-1.27393f) + (1.0f - inverse_lerp_squared) * (-1.97760f);
  const float e = inverse_lerp_squared * (-4.85967f) + (1.0f - inverse_lerp_squared) * (-4.32054f);
  return a / (1.0f + b * pow(clamped_x, c)) + d * clamped_x + e;
}

ETX_SHARED_INLINE float bsdf_velvet_lambda(float roughness, float cos_theta) {
  if (cos_theta < 0.5f) {
    return exp(bsdf_velvet_lambda_l(roughness, cos_theta));
  }

  return exp(2.0f * bsdf_velvet_lambda_l(roughness, 0.5f) - bsdf_velvet_lambda_l(roughness, 1.0f - cos_theta));
}

ETX_SHARED_INLINE float bsdf_velvet_distribution(float roughness, float m_dot_n) {
  if (roughness <= kEpsilon) {
    return 0.0f;
  }

  const float inv_alpha = 1.0f / max(kEpsilon, roughness);
  const float sin_theta = sqrt(max(0.0f, 1.0f - m_dot_n * m_dot_n));
  return (2.0f + inv_alpha) * pow(sin_theta, inv_alpha) / kDoublePi;
}

ETX_SHARED_INLINE float bsdf_velvet_directional_albedo(float roughness, float mu) {
  const float alpha = saturate(roughness) * float(kVelvetDirectionalAlbedoLutSize - 1u);
  const float theta = saturate(mu) * float(kVelvetDirectionalAlbedoLutSize - 1u);
  const uint32_t alpha_index = min(kVelvetDirectionalAlbedoLutSize - 2u, uint32_t(alpha));
  const uint32_t theta_index = min(kVelvetDirectionalAlbedoLutSize - 2u, uint32_t(theta));
  const float alpha_t = alpha - float(alpha_index);
  const float theta_t = theta - float(theta_index);

  const uint32_t row_0 = alpha_index * kVelvetDirectionalAlbedoLutSize;
  const uint32_t row_1 = row_0 + kVelvetDirectionalAlbedoLutSize;
  const float v00 = kVelvetDirectionalAlbedoLut[row_0 + theta_index];
  const float v10 = kVelvetDirectionalAlbedoLut[row_0 + theta_index + 1u];
  const float v01 = kVelvetDirectionalAlbedoLut[row_1 + theta_index];
  const float v11 = kVelvetDirectionalAlbedoLut[row_1 + theta_index + 1u];
  const float row_value_0 = v00 + (v10 - v00) * theta_t;
  const float row_value_1 = v01 + (v11 - v01) * theta_t;
  return row_value_0 + (row_value_1 - row_value_0) * alpha_t;
}

ETX_SHARED_INLINE float bsdf_velvet_single_scatter_bsdf_cos(ETX_IN(float3, local_w_i), ETX_IN(float3, local_w_o), float roughness) {
  if ((local_w_i.z <= kEpsilon) || (local_w_o.z <= kEpsilon)) {
    return 0.0f;
  }

  const float3 half_vector = normalize(local_w_i + local_w_o);
  const float i_dot_m = dot(local_w_i, half_vector);
  const float o_dot_m = dot(local_w_o, half_vector);
  if ((i_dot_m <= kEpsilon) || (o_dot_m <= kEpsilon)) {
    return 0.0f;
  }

  const float d = bsdf_velvet_distribution(roughness, half_vector.z);
  const float lambda_i = bsdf_velvet_lambda(roughness, local_w_i.z);
  const float lambda_o = bsdf_velvet_lambda(roughness, local_w_o.z);
  const float g = 1.0f / (1.0f + lambda_i + lambda_o);
  return 0.25f * d * g / local_w_i.z;
}

ETX_SHARED_INLINE BSDFEval bsdf_velvet_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler));

ETX_SHARED_INLINE BSDFSample bsdf_velvet_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  const LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  const float2 rnd = bsdf_sampler_has_fixed(sampler) ? float2(sampler.fixed_u, sampler.fixed_v) : bsdf_sampler_next_2d(sampler);
  const float3 local_w_o = sample_cosine_distribution(rnd, 0.0f);
  const float3 sampled_direction = local_frame_from_local(frame, local_w_o);
  const BSDFEval eval = bsdf_velvet_evaluate(context, data, sampled_direction, material, sampler);

  BSDFSample result = ETX_ZERO(BSDFSample);
  result.w_o = sampled_direction;
  result.properties = BSDFSample::Reflection | BSDFSample::Diffuse;
  result.medium_index = data.current_medium;
  result.eta = 1.0f;
  result.pdf = eval.pdf;
  if (result.pdf > 0.0f) {
    result.weight = spectral_response_div(eval.bsdf, result.pdf);
  }
  return result;
}

ETX_SHARED_INLINE BSDFEval bsdf_velvet_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  const LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  const float3 local_w_i = local_frame_to_local(frame, -data.w_i);
  const float3 local_w_o = local_frame_to_local(frame, outgoing_direction);
  if ((local_w_i.z <= kEpsilon) || (local_w_o.z <= kEpsilon)) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  const float2 roughness = bsdf_resource_evaluate_roughness(context, material, data.tex);
  const float alpha = saturate(0.5f * (roughness.x + roughness.y));
  const SpectralResponse sheen = bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex);
  const float single_scatter_bsdf = bsdf_velvet_single_scatter_bsdf_cos(local_w_i, local_w_o, alpha);
  const float single_scatter_albedo = bsdf_velvet_directional_albedo(alpha, local_w_i.z);
  const SpectralResponse base_scale = spectral_response_max(spectral_response_sub(spectral_response_make(data.spectrum_sample, 1.0f), spectral_response_mul(sheen, single_scatter_albedo)), 0.0f);
  const BSDFEval diffuse = bsdf_diffuse_layer(context, data, local_w_i, local_w_o, material, sampler);
  const SpectralResponse diffuse_bsdf = spectral_response_mul(diffuse.bsdf, base_scale);
  const SpectralResponse sheen_bsdf = spectral_response_mul(sheen, single_scatter_bsdf);

  BSDFEval result = ETX_ZERO(BSDFEval);
  result.bsdf = spectral_response_add(diffuse_bsdf, sheen_bsdf);
  result.func = spectral_response_div(result.bsdf, local_w_o.z);
  result.pdf = 1.0f / kDoublePi;
  result.eta = 1.0f;
  result.properties = BSDFSample::Reflection | BSDFSample::Diffuse;
  result.medium_index = data.current_medium;
  return result;
}

ETX_SHARED_INLINE float bsdf_velvet_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)material;
  (void)sampler;

  const LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  const float3 local_w_i = local_frame_to_local(frame, -data.w_i);
  const float3 local_w_o = local_frame_to_local(frame, outgoing_direction);
  if ((local_w_i.z <= kEpsilon) || (local_w_o.z <= kEpsilon)) {
    return 0.0f;
  }

  return 1.0f / kDoublePi;
}

ETX_SHARED_INLINE bool bsdf_velvet_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler)) {
  (void)material;
  (void)tex;
  (void)sampler;
  return false;
}

ETX_SHARED_INLINE SpectralResponse bsdf_velvet_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)sampler;
  const SpectralResponse base = bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
  const SpectralResponse sheen = bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex);
  return spectral_response_min(spectral_response_add(base, sheen), 1.0f);
}
