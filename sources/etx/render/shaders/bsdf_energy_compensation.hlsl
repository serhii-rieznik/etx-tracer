#include "bindless.hlsl"

#include <interop/bsdf_energy_compensation_constants_shared.hxx>
#include <interop/bsdf_external_shared.hxx>

struct EnergyCompensationPushConstants {
  uint params_buffer_index;
};

struct EnergyCompensationParams {
  uint output_directional_index;
  uint output_average_index;
  uint output_geometric_index;
  uint output_geometric_average_index;
  uint output_conductor_fms_index;
  uint output_probability_index;
  uint output_total_index;
  uint cache_mode;
  uint wavelength_group_index;
  uint thinfilm_slice_index;
  uint thinfilm_slice_count;
  uint pass_kind;
  uint sample_count;
  uint multisample_count;
  uint spectral_channel;
  uint pad1;
  float4 ext_eta;
  float4 ext_k;
  float4 int_eta;
  float4 int_k;
  float4 film_eta;
  float4 film_k;
  float4 wavelengths;
  float thinfilm_thickness;
  uint film_cls;
  float thinfilm_weight;
  uint pad3;
};

struct EnergyCompensationDirectionalResult {
  float4 albedo;
  float visible_probability;
  float geometric_albedo;
};

struct EnergyCompensationDielectricResult {
  float4 branch_albedo[2];
  float branch_visible_probability[2];
  float visible_probability;
};

struct EnergyCompensationLobe {
  SpectralResponse bsdf;
  float pdf;
};

[[vk::push_constant]] EnergyCompensationPushConstants constants;

uint load_u32(ByteAddressBuffer buffer, uint byte_offset) {
  return buffer.Load(byte_offset);
}

float load_f32(ByteAddressBuffer buffer, uint byte_offset) {
  return asfloat(buffer.Load(byte_offset));
}

float4 load_f32x4(ByteAddressBuffer buffer, uint byte_offset) {
  return asfloat(buffer.Load4(byte_offset));
}

EnergyCompensationParams load_params() {
  ByteAddressBuffer buffer = bindless_buffers[NonUniformResourceIndex(constants.params_buffer_index)];
  EnergyCompensationParams result;
  result.output_directional_index = load_u32(buffer, 0u);
  result.output_average_index = load_u32(buffer, 4u);
  result.output_geometric_index = load_u32(buffer, 8u);
  result.output_geometric_average_index = load_u32(buffer, 12u);
  result.output_conductor_fms_index = load_u32(buffer, 16u);
  result.output_probability_index = load_u32(buffer, 20u);
  result.output_total_index = load_u32(buffer, 24u);
  result.cache_mode = load_u32(buffer, 28u);
  result.wavelength_group_index = load_u32(buffer, 32u);
  result.thinfilm_slice_index = load_u32(buffer, 36u);
  result.thinfilm_slice_count = load_u32(buffer, 40u);
  result.pass_kind = load_u32(buffer, 44u);
  result.sample_count = load_u32(buffer, 48u);
  result.multisample_count = load_u32(buffer, 52u);
  result.spectral_channel = load_u32(buffer, 56u);
  result.pad1 = load_u32(buffer, 60u);
  result.ext_eta = load_f32x4(buffer, 64u);
  result.ext_k = load_f32x4(buffer, 80u);
  result.int_eta = load_f32x4(buffer, 96u);
  result.int_k = load_f32x4(buffer, 112u);
  result.film_eta = load_f32x4(buffer, 128u);
  result.film_k = load_f32x4(buffer, 144u);
  result.wavelengths = load_f32x4(buffer, 160u);
  result.thinfilm_thickness = load_f32(buffer, 176u);
  result.film_cls = load_u32(buffer, 180u);
  result.thinfilm_weight = load_f32(buffer, 184u);
  result.pad3 = load_u32(buffer, 188u);
  return result;
}

void store_float4(uint buffer_index, uint element_index, float4 value) {
  RWByteAddressBuffer buffer = bindless_rw_buffers[NonUniformResourceIndex(buffer_index)];
  buffer.Store4(element_index * 16u, asuint(value));
}

float4 load_float4(uint buffer_index, uint element_index) {
  ByteAddressBuffer buffer = bindless_buffers[NonUniformResourceIndex(buffer_index)];
  return asfloat(buffer.Load4(element_index * 16u));
}

float ec_saturate(float value) {
  return min(1.0f, max(0.0f, value));
}

float4 ec_saturate4(float4 value) {
  return min(float4(1.0f, 1.0f, 1.0f, 1.0f), max(float4(0.0f, 0.0f, 0.0f, 0.0f), value));
}

float ec_lut_parameter(uint index, uint size) {
  return float(index) / float(size - 1u);
}

float ec_mu_parameter(uint index, uint size) {
  if (index == 0u) {
    return 0.5f / float(size - 1u);
  }
  return ec_lut_parameter(index, size);
}

float ec_alpha_parameter(uint index, uint size) {
  float axis = ec_lut_parameter(index, size);
  return kBSDFNormalDistributionMinAlpha + (1.0f - kBSDFNormalDistributionMinAlpha) * axis * axis;
}

float ec_radical_inverse_vdc(uint bits) {
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xaaaaaaaau) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xccccccccu) >> 2u);
  bits = ((bits & 0x0f0f0f0fu) << 4u) | ((bits & 0xf0f0f0f0u) >> 4u);
  bits = ((bits & 0x00ff00ffu) << 8u) | ((bits & 0xff00ff00u) >> 8u);
  return float(bits) * 2.3283064365386963e-10f;
}

float2 ec_hammersley(uint index, uint count) {
  return float2((float(index) + 0.5f) / float(count), ec_radical_inverse_vdc(index));
}

float ec_quasi_random(uint index, uint dimension) {
  uint seed = index + 1u + dimension * 0x9e3779b9u;
  return min(1.0f - kEpsilon, max(kEpsilon, ec_radical_inverse_vdc(seed)));
}

float2 ec_quasi_random_2d(uint index, uint dimension) {
  return float2(ec_quasi_random(index, dimension), ec_quasi_random(index, dimension + 1u));
}

float3 ec_incident_direction_from_mu(float mu) {
  return float3(sqrt(max(0.0f, 1.0f - mu * mu)), 0.0f, mu);
}

SpectralQuery ec_query(EnergyCompensationParams params, uint channel) {
  SpectralQuery result = (SpectralQuery)0;
  if (params.cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
    result.flags = SpectralFlags::Spectral;
    result.wavelength = params.wavelengths[channel];
  }
  return result;
}

float ec_channel(float4 value, uint channel) {
  if (channel == 0u) {
    return value.x;
  }
  if (channel == 1u) {
    return value.y;
  }
  if (channel == 2u) {
    return value.z;
  }
  return value.w;
}

void ec_set_channel(inout float4 value, uint channel, float scalar) {
  if (channel == 0u) {
    value.x = scalar;
  } else if (channel == 1u) {
    value.y = scalar;
  } else if (channel == 2u) {
    value.z = scalar;
  } else {
    value.w = scalar;
  }
}

RefractiveIndexSample ec_refractive_index(float4 eta, float4 k, uint cls, SpectralQuery spect, uint channel) {
  RefractiveIndexSample result = (RefractiveIndexSample)0;
  result.cls = cls;
  if (spectral_query_is_spectral(spect)) {
    result.eta = spectral_response_make(spect, ec_channel(eta, channel));
    result.k = spectral_response_make(spect, ec_channel(k, channel));
  } else {
    result.eta = spectral_response_make(eta.xyz);
    result.k = spectral_response_make(k.xyz);
  }
  return result;
}

RefractiveIndexSample ec_select_refractive_index(bool use_first, RefractiveIndexSample first, RefractiveIndexSample second) {
  if (use_first) {
    return first;
  }
  return second;
}

ThinfilmEval ec_thinfilm(EnergyCompensationParams params, SpectralQuery spect, uint channel) {
  ThinfilmEval result = (ThinfilmEval)0;
  result.ior = ec_refractive_index(params.film_eta, params.film_k, params.film_cls, spect, channel);
  result.ior.k = spectral_response_make(spect, 0.0f);
  result.rgb_wavelengths = kRGBWavelengths;
  result.thickness = params.thinfilm_thickness;
  result.weight = params.thinfilm_weight;
  return result;
}

bool ec_spectral_response_finite(SpectralResponse value) {
  return isfinite(value.integrated.x) && isfinite(value.integrated.y) && isfinite(value.integrated.z) && isfinite(value.value);
}

SpectralResponse ec_average_fresnel(SpectralQuery spect, RefractiveIndexSample ext_ior, RefractiveIndexSample int_ior, ThinfilmEval thinfilm) {
  const float nodes[8] = {
    1.98550718e-2f, 1.01666761e-1f, 2.37233795e-1f, 4.08282679e-1f,
    5.91717321e-1f, 7.62766205e-1f, 8.98333239e-1f, 9.80144928e-1f,
  };
  const float weights[8] = {
    5.06142681e-2f, 1.11190517e-1f, 1.56853323e-1f, 1.81341892e-1f,
    1.81341892e-1f, 1.56853323e-1f, 1.11190517e-1f, 5.06142681e-2f,
  };
  SpectralResponse result = spectral_response_make(spect, 0.0f);
  for (uint i = 0u; i < 8u; ++i) {
    float mu = nodes[i];
    SpectralResponse fresnel = bsdf_fresnel_calculate(spect, mu, ext_ior, int_ior, thinfilm);
    result = spectral_response_add(result, spectral_response_mul(fresnel, 2.0f * weights[i] * mu));
  }
  return result;
}

SpectralResponse ec_conductor_fms(SpectralQuery spect, RefractiveIndexSample ext_ior, RefractiveIndexSample int_ior, ThinfilmEval thinfilm, float average_albedo) {
  SpectralResponse fresnel_average = ec_average_fresnel(spect, ext_ior, int_ior, thinfilm);
  if (ec_spectral_response_finite(fresnel_average) == false) {
    return spectral_response_make(spect, 0.0f);
  }
  SpectralResponse numerator = spectral_response_mul(spectral_response_mul(fresnel_average, fresnel_average), average_albedo);
  SpectralResponse denominator = spectral_response_sub(spectral_response_make(spect, 1.0f), spectral_response_mul(fresnel_average, 1.0f - average_albedo));
  if ((ec_spectral_response_finite(numerator) == false) || (ec_spectral_response_finite(denominator) == false)) {
    return spectral_response_make(spect, 0.0f);
  }
  return spectral_response_div(numerator, spectral_response_max(denominator, kEpsilon));
}

EnergyCompensationLobe ec_conductor_base_lobe(SpectralQuery spect, float3 w_i, float3 w_o, float alpha, RefractiveIndexSample ext_ior,
  RefractiveIndexSample int_ior, ThinfilmEval thinfilm) {
  EnergyCompensationLobe result = (EnergyCompensationLobe)0;
  result.bsdf = spectral_response_make(spect, 0.0f);
  if ((w_i.z <= kEpsilon) || (w_o.z <= kEpsilon)) {
    return result;
  }
  float3 half_vector_sum = w_i + w_o;
  if (dot(half_vector_sum, half_vector_sum) <= kEpsilon) {
    return result;
  }
  float3 m = normalize(half_vector_sum);
  if ((m.z <= kEpsilon) || (dot(w_i, m) <= kEpsilon) || (dot(w_o, m) <= kEpsilon)) {
    return result;
  }
  SpectralResponse fresnel = bsdf_fresnel_calculate(spect, dot(w_i, m), ext_ior, int_ior, thinfilm);
  float lambda_i = bsdf_external_ray_info_make(w_i, float2(alpha, alpha)).Lambda;
  float lambda_o = bsdf_external_ray_info_make(w_o, float2(alpha, alpha)).Lambda;
  float d = bsdf_external_d_ggx(m, float2(alpha, alpha));
  float g2 = 1.0f / (1.0f + lambda_i + lambda_o);
  result.bsdf = spectral_response_mul(fresnel, d * g2 / (4.0f * w_i.z));
  float vndf_pdf = bsdf_external_vndf_pdf(w_i, m, alpha);
  result.pdf = vndf_pdf / max(kEpsilon, 4.0f * dot(w_o, m));
  return result;
}

EnergyCompensationLobe ec_dielectric_base_lobe(SpectralQuery spect, float3 w_i_local, float3 w_o_local, float alpha, RefractiveIndexSample ext_ior,
  RefractiveIndexSample int_ior, ThinfilmEval thinfilm) {
  EnergyCompensationLobe result = (EnergyCompensationLobe)0;
  result.bsdf = spectral_response_make(spect, 0.0f);
  if ((abs(w_i_local.z) <= kEpsilon) || (abs(w_o_local.z) <= kEpsilon)) {
    return result;
  }
  bool outside = w_i_local.z > 0.0f;
  float direction_scale = outside ? 1.0f : -1.0f;
  float3 w_i = direction_scale * w_i_local;
  float3 w_o = direction_scale * w_o_local;
  bool reflection = w_o.z > 0.0f;
  RefractiveIndexSample phase_ext_ior = ec_select_refractive_index(outside, ext_ior, int_ior);
  RefractiveIndexSample phase_int_ior = ec_select_refractive_index(outside, int_ior, ext_ior);
  float eta = spectral_response_monochromatic(spectral_response_div(phase_int_ior.eta, phase_ext_ior.eta));
  float lambda_i = bsdf_external_ray_info_make(w_i, float2(alpha, alpha)).Lambda;

  if (reflection) {
    float3 half_vector_sum = w_i + w_o;
    if (dot(half_vector_sum, half_vector_sum) <= kEpsilon) {
      return result;
    }
    float3 m = normalize(half_vector_sum);
    if ((m.z <= kEpsilon) || (dot(w_i, m) <= kEpsilon) || (dot(w_o, m) <= kEpsilon)) {
      return result;
    }
    SpectralResponse fresnel = bsdf_fresnel_calculate(spect, dot(w_i, m), phase_ext_ior, phase_int_ior, thinfilm);
    float lambda_o = bsdf_external_ray_info_make(w_o, float2(alpha, alpha)).Lambda;
    float d = bsdf_external_d_ggx(m, float2(alpha, alpha));
    float g2 = 1.0f / (1.0f + lambda_i + lambda_o);
    result.bsdf = spectral_response_mul(fresnel, d * g2 / (4.0f * w_i.z));
    float vndf_pdf = bsdf_external_vndf_pdf(w_i, m, alpha);
    float fresnel_probability = spectral_response_monochromatic(fresnel);
    result.pdf = fresnel_probability * vndf_pdf / max(kEpsilon, 4.0f * dot(w_o, m));
    return result;
  }

  float3 m = normalize(w_i + w_o * eta);
  m *= (m.z >= 0.0f) ? 1.0f : -1.0f;
  float i_dot_m = dot(w_i, m);
  float o_dot_m = dot(w_o, m);
  float denominator = i_dot_m + eta * o_dot_m;
  if ((m.z <= kEpsilon) || (i_dot_m <= kEpsilon) || (o_dot_m >= -kEpsilon) || (abs(denominator) <= kEpsilon)) {
    return result;
  }
  SpectralResponse fresnel = bsdf_fresnel_calculate(spect, i_dot_m, phase_ext_ior, phase_int_ior, thinfilm);
  SpectralResponse one_minus_fresnel = spectral_response_sub(spectral_response_make(spect, 1.0f), fresnel);
  float3 oriented_w_o = -w_o;
  float lambda_o = bsdf_external_ray_info_make(oriented_w_o, float2(alpha, alpha)).Lambda;
  float d = bsdf_external_d_ggx(m, float2(alpha, alpha));
  float g2 = bsdf_external_beta(1.0f + lambda_i, 1.0f + lambda_o);
  if (isfinite(g2) == false) {
    return result;
  }
  float scalar = i_dot_m * max(0.0f, -o_dot_m) * d * g2 / (w_i.z * denominator * denominator);
  if (isfinite(scalar) == false) {
    return result;
  }
  result.bsdf = spectral_response_mul(one_minus_fresnel, scalar * eta * eta);
  float vndf_pdf = bsdf_external_vndf_pdf(w_i, m, alpha);
  float fresnel_probability = 1.0f - spectral_response_monochromatic(fresnel);
  float dwh_dwo = (eta * eta) * abs(o_dot_m) / (denominator * denominator);
  result.pdf = fresnel_probability * vndf_pdf * dwh_dwo;
  return result;
}

EnergyCompensationDirectionalResult ec_integrate_conductor_directional(EnergyCompensationParams params, uint channel, float mu_i, float alpha) {
  EnergyCompensationDirectionalResult result = (EnergyCompensationDirectionalResult)0;
  if (mu_i <= kEpsilon) {
    return result;
  }
  SpectralQuery spect = ec_query(params, channel);
  RefractiveIndexSample ext_ior = ec_refractive_index(params.ext_eta, params.ext_k, SpectralDistribution::Dielectric, spect, channel);
  RefractiveIndexSample int_ior = ec_refractive_index(params.int_eta, params.int_k, SpectralDistribution::Conductor, spect, channel);
  ThinfilmEval thinfilm = ec_thinfilm(params, spect, channel);
  float3 w_i = ec_incident_direction_from_mu(mu_i);
  float2 alpha2 = float2(alpha, alpha);
  float lambda_i = bsdf_external_ray_info_make(w_i, alpha2).Lambda;
  float g1_i = 1.0f / (1.0f + lambda_i);

  for (uint sample_index = 0u; sample_index < params.sample_count; ++sample_index) {
    float3 m = bsdf_external_sample_vndf_local(w_i, alpha, ec_hammersley(sample_index, params.sample_count));
    float i_dot_m = dot(w_i, m);
    if ((m.z <= kEpsilon) || (i_dot_m <= kEpsilon)) {
      continue;
    }
    float3 w_o = -w_i + 2.0f * m * i_dot_m;
    if (w_o.z > 0.0f) {
      float vndf_pdf = bsdf_external_vndf_pdf(w_i, m, alpha);
      float raw_specular_pdf = vndf_pdf / max(kEpsilon, 4.0f * dot(w_o, m));
      EnergyCompensationLobe lobe = ec_conductor_base_lobe(spect, w_i, w_o, alpha, ext_ior, int_ior, thinfilm);
      if ((raw_specular_pdf > kEpsilon) && (lobe.pdf > kEpsilon)) {
        result.albedo += float4(lobe.bsdf.integrated / raw_specular_pdf, 0.0f);
        float lambda_o = bsdf_external_ray_info_make(w_o, alpha2).Lambda;
        float d = bsdf_external_d_ggx(m, alpha2);
        float g2 = 1.0f / (1.0f + lambda_i + lambda_o);
        result.geometric_albedo += (d * g2 / (4.0f * w_i.z)) / raw_specular_pdf;
      }
      if (g1_i > kEpsilon) {
        result.visible_probability += 1.0f;
      }
    }
  }

  float inv_sample_count = 1.0f / float(params.sample_count);
  result.albedo = ec_saturate4(result.albedo * inv_sample_count);
  result.geometric_albedo = ec_saturate(result.geometric_albedo * inv_sample_count);
  result.visible_probability = ec_saturate(result.visible_probability * inv_sample_count);
  return result;
}

uint ec_dielectric_side(bool outside) {
  return outside ? 0u : 1u;
}

uint ec_dielectric_branch_index(uint incident_side, uint outgoing_side) {
  return incident_side * 2u + outgoing_side;
}

EnergyCompensationDielectricResult ec_integrate_dielectric_directional(EnergyCompensationParams params, uint channel, bool total_scatter, bool incident_outside, float mu_i, float alpha) {
  EnergyCompensationDielectricResult result = (EnergyCompensationDielectricResult)0;
  if (mu_i <= kEpsilon) {
    return result;
  }
  SpectralQuery spect = ec_query(params, channel);
  RefractiveIndexSample ext_ior = ec_refractive_index(params.ext_eta, params.ext_k, SpectralDistribution::Dielectric, spect, channel);
  RefractiveIndexSample int_ior = ec_refractive_index(params.int_eta, params.int_k, SpectralDistribution::Dielectric, spect, channel);
  ThinfilmEval thinfilm = ec_thinfilm(params, spect, channel);
  RefractiveIndexSample source_ior = ec_select_refractive_index(incident_outside, ext_ior, int_ior);
  RefractiveIndexSample target_ior = ec_select_refractive_index(incident_outside, int_ior, ext_ior);
  float eta = spectral_response_monochromatic(spectral_response_div(target_ior.eta, source_ior.eta));
  uint incident_side = ec_dielectric_side(incident_outside);
  uint opposite_side = 1u - incident_side;
  bool no_thinfilm = (thinfilm.weight <= 0.0f) || (thinfilm.thickness <= 0.0f) || spectral_response_is_zero(thinfilm.ior.eta);
  if ((no_thinfilm) && (abs(eta - 1.0f) <= (16.0f * kEpsilon))) {
    result.branch_albedo[opposite_side] = float4(1.0f, 1.0f, 1.0f, 1.0f);
    result.branch_visible_probability[opposite_side] = 1.0f;
    result.visible_probability = 1.0f;
    return result;
  }

  float3 w_i = ec_incident_direction_from_mu(mu_i);
  if (total_scatter == false) {
    for (uint sample_index = 0u; sample_index < params.sample_count; ++sample_index) {
      float3 m = bsdf_external_sample_vndf_local(w_i, alpha, ec_hammersley(sample_index, params.sample_count));
      float i_dot_m = dot(w_i, m);
      if (i_dot_m <= kEpsilon) {
        continue;
      }
      SpectralResponse fresnel = bsdf_fresnel_calculate(spect, i_dot_m, source_ior, target_ior, thinfilm);
      float fresnel_probability = spectral_response_monochromatic(fresnel);
      float cos_theta_t2 = 1.0f - (1.0f - i_dot_m * i_dot_m) / (eta * eta);
      float3 w_o_r = -w_i + 2.0f * m * i_dot_m;
      if (w_o_r.z > 0.0f) {
        EnergyCompensationLobe lobe = ec_dielectric_base_lobe(spect, w_i, w_o_r, alpha, source_ior, target_ior, thinfilm);
        if ((fresnel_probability > kEpsilon) && (lobe.pdf > kEpsilon)) {
          result.branch_albedo[incident_side] += float4(lobe.bsdf.integrated * (fresnel_probability / lobe.pdf), 0.0f);
          result.branch_visible_probability[incident_side] += fresnel_probability;
        }
      }
      if ((fresnel_probability < 1.0f) && (cos_theta_t2 > 0.0f)) {
        float3 w_o_t = normalize(bsdf_external_refract(w_i, m, eta));
        w_o_t.z = -abs(w_o_t.z);
        if (w_o_t.z < 0.0f) {
          EnergyCompensationLobe lobe = ec_dielectric_base_lobe(spect, w_i, w_o_t, alpha, source_ior, target_ior, thinfilm);
          float transmission_probability = 1.0f - fresnel_probability;
          if ((transmission_probability > kEpsilon) && (lobe.pdf > kEpsilon)) {
            result.branch_albedo[opposite_side] += float4(lobe.bsdf.integrated * (transmission_probability / lobe.pdf), 0.0f);
            result.branch_visible_probability[opposite_side] += transmission_probability;
          }
        }
      }
    }
  } else {
    float2 alpha2 = float2(alpha, alpha);
    for (uint sample_index = 0u; sample_index < params.multisample_count; ++sample_index) {
      BSDFExternalRayInfo ray = bsdf_external_ray_info_make(-w_i, alpha2);
      ray = bsdf_external_ray_info_update_height(ray, 1.0f);
      bool ray_outside = true;
      bool valid = true;
      uint scattering_order = 0u;
      uint dimension = 0u;
      while (valid) {
        float sampled_height = bsdf_external_sample_height(ray, ec_quasi_random(sample_index, dimension));
        dimension += 1u;
        if (sampled_height == kMaxFloat) {
          break;
        }
        ray = bsdf_external_ray_info_update_height(ray, sampled_height);
        float2 rnd_slope = ec_quasi_random_2d(sample_index, dimension);
        dimension += 2u;
        float rnd_reflection = ec_quasi_random(sample_index, dimension);
        dimension += 1u;
        RefractiveIndexSample phase_ext_ior = ec_select_refractive_index(ray_outside, source_ior, target_ior);
        RefractiveIndexSample phase_int_ior = ec_select_refractive_index(ray_outside, target_ior, source_ior);
        BSDFExternalDielectricSample sample = bsdf_external_sample_phase_function_dielectric(spect, rnd_slope, rnd_reflection, -ray.w, alpha2, phase_ext_ior, phase_int_ior, thinfilm);
        if (sample.reflection) {
          ray = bsdf_external_ray_info_update_direction(ray, sample.w_o, alpha2);
          ray = bsdf_external_ray_info_update_height(ray, ray.h);
        } else {
          ray_outside = (ray_outside == false);
          ray = bsdf_external_ray_info_update_direction(ray, -sample.w_o, alpha2);
          ray = bsdf_external_ray_info_update_height(ray, -ray.h);
        }
        scattering_order += 1u;
        if ((scattering_order > kBSDFExternalScatteringOrderMax) || (ray.h != ray.h) || (ray.w.x != ray.w.x)) {
          valid = false;
        }
      }
      if (valid == false) {
        continue;
      }
      float3 local_w_o = ray_outside ? ray.w : -ray.w;
      uint outgoing_side = (local_w_o.z > 0.0f) ? incident_side : opposite_side;
      result.branch_albedo[outgoing_side] += float4(1.0f, 1.0f, 1.0f, 1.0f);
      result.visible_probability += 1.0f;
    }
  }

  float inv_sample_count = total_scatter ? (1.0f / float(params.multisample_count)) : (1.0f / float(params.sample_count));
  result.branch_albedo[0] = ec_saturate4(result.branch_albedo[0] * inv_sample_count);
  result.branch_albedo[1] = ec_saturate4(result.branch_albedo[1] * inv_sample_count);
  result.branch_visible_probability[0] = ec_saturate(result.branch_visible_probability[0] * inv_sample_count);
  result.branch_visible_probability[1] = ec_saturate(result.branch_visible_probability[1] * inv_sample_count);
  result.visible_probability = total_scatter ? ec_saturate(result.visible_probability * inv_sample_count)
                                             : ec_saturate(result.branch_visible_probability[0] + result.branch_visible_probability[1]);
  return result;
}

float4 ec_integrate_average(uint buffer_index, uint alpha_index, uint lut_size, uint side_offset) {
  float4 total = float4(0.0f, 0.0f, 0.0f, 0.0f);
  float h = 1.0f / float(lut_size - 1u);
  for (uint mu_index = 0u; mu_index < (lut_size - 1u); ++mu_index) {
    float mu = float(mu_index) * h;
    float weight_0 = h * mu + h * h / 3.0f;
    float weight_1 = h * mu + 2.0f * h * h / 3.0f;
    float4 value_0 = load_float4(buffer_index, side_offset + alpha_index * lut_size + mu_index);
    float4 value_1 = load_float4(buffer_index, side_offset + alpha_index * lut_size + mu_index + 1u);
    total += value_0 * weight_0 + value_1 * weight_1;
  }
  return ec_saturate4(total);
}

float4 ec_integrate_dielectric_branch_average(uint buffer_index, uint alpha_index, uint branch) {
  float4 total = float4(0.0f, 0.0f, 0.0f, 0.0f);
  float h = 1.0f / float(kBSDFEnergyCompensationDielectricLutSize - 1u);
  uint width = kBSDFEnergyCompensationDielectricBranchCount * kBSDFEnergyCompensationDielectricLutSize;
  uint branch_offset = branch * kBSDFEnergyCompensationDielectricLutSize;
  uint row_offset = alpha_index * width;
  for (uint mu_index = 0u; mu_index < (kBSDFEnergyCompensationDielectricLutSize - 1u); ++mu_index) {
    float mu = float(mu_index) * h;
    float weight_0 = h * mu + h * h / 3.0f;
    float weight_1 = h * mu + 2.0f * h * h / 3.0f;
    float4 value_0 = load_float4(buffer_index, row_offset + branch_offset + mu_index);
    float4 value_1 = load_float4(buffer_index, row_offset + branch_offset + mu_index + 1u);
    total += value_0 * weight_0 + value_1 * weight_1;
  }
  return ec_saturate4(total);
}

float ec_integrate_geometric_average(uint buffer_index, uint alpha_index) {
  float total = 0.0f;
  float h = 1.0f / float(kBSDFEnergyCompensationConductorLutSize - 1u);
  for (uint mu_index = 0u; mu_index < (kBSDFEnergyCompensationConductorLutSize - 1u); ++mu_index) {
    float mu = float(mu_index) * h;
    float weight_0 = h * mu + h * h / 3.0f;
    float weight_1 = h * mu + 2.0f * h * h / 3.0f;
    float value_0 = load_float4(buffer_index, alpha_index * kBSDFEnergyCompensationConductorLutSize + mu_index).x;
    float value_1 = load_float4(buffer_index, alpha_index * kBSDFEnergyCompensationConductorLutSize + mu_index + 1u).x;
    total += value_0 * weight_0 + value_1 * weight_1;
  }
  return ec_saturate(total);
}

float4 ec_residual_coefficient(float4 residual_average, float4 side_residual_a, float4 side_residual_b) {
  return max(float4(0.0f, 0.0f, 0.0f, 0.0f), residual_average / max(float4(kEpsilon, kEpsilon, kEpsilon, kEpsilon), side_residual_a * side_residual_b));
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  EnergyCompensationParams params = load_params();
  if (params.pass_kind == kBSDFEnergyCompensationGpuPassConductorDirectional) {
    if ((id.x >= kBSDFEnergyCompensationConductorLutSize) || (id.y >= kBSDFEnergyCompensationConductorLutSize)) {
      return;
    }
    uint index = id.y * kBSDFEnergyCompensationConductorLutSize + id.x;
    float mu = ec_mu_parameter(id.x, kBSDFEnergyCompensationConductorLutSize);
    float alpha = ec_alpha_parameter(id.y, kBSDFEnergyCompensationConductorLutSize);
    float4 image = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 geometric = float4(0.0f, 0.0f, 0.0f, 1.0f);
    if (params.cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
      uint channel = params.spectral_channel;
      EnergyCompensationDirectionalResult value = ec_integrate_conductor_directional(params, channel, mu, alpha);
      if (channel > 0u) {
        image = load_float4(params.output_directional_index, index);
      }
      ec_set_channel(image, channel, value.albedo.x);
      if (channel == 0u) {
        geometric.x = value.geometric_albedo;
        geometric.y = value.visible_probability;
      }
    } else {
      EnergyCompensationDirectionalResult value = ec_integrate_conductor_directional(params, 0u, mu, alpha);
      image = float4(value.albedo.x, value.albedo.y, value.albedo.z, value.visible_probability);
      geometric = float4(value.geometric_albedo, value.visible_probability, 0.0f, 1.0f);
    }
    store_float4(params.output_directional_index, index, image);
    if ((params.cache_mode != kBSDFEnergyCompensationCacheModeSpectralScalar) || (params.spectral_channel == 0u)) {
      store_float4(params.output_geometric_index, index, geometric);
    }
    return;
  }

  if (params.pass_kind == kBSDFEnergyCompensationGpuPassConductorAverage) {
    if ((id.x >= kBSDFEnergyCompensationConductorLutSize) || (id.y != 0u)) {
      return;
    }
    uint alpha_index = id.x;
    float4 average = ec_integrate_average(params.output_directional_index, alpha_index, kBSDFEnergyCompensationConductorLutSize, 0u);
    float geometric_average = ec_integrate_geometric_average(params.output_geometric_index, alpha_index);
    float alpha = ec_alpha_parameter(alpha_index, kBSDFEnergyCompensationConductorLutSize);
    float4 conductor_fms = float4(0.0f, 0.0f, 0.0f, 1.0f);
    if (params.cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
      for (uint channel = 0u; channel < kBSDFEnergyCompensationSpectralWavelengthGroupSize; ++channel) {
        SpectralQuery spect = ec_query(params, channel);
        RefractiveIndexSample ext_ior = ec_refractive_index(params.ext_eta, params.ext_k, SpectralDistribution::Dielectric, spect, channel);
        RefractiveIndexSample int_ior = ec_refractive_index(params.int_eta, params.int_k, SpectralDistribution::Conductor, spect, channel);
        ThinfilmEval thinfilm = ec_thinfilm(params, spect, channel);
        SpectralResponse fms = ec_conductor_fms(spect, ext_ior, int_ior, thinfilm, geometric_average);
        ec_set_channel(conductor_fms, channel, spectral_response_monochromatic(fms));
      }
    } else {
      SpectralQuery spect = ec_query(params, 0u);
      RefractiveIndexSample ext_ior = ec_refractive_index(params.ext_eta, params.ext_k, SpectralDistribution::Dielectric, spect, 0u);
      RefractiveIndexSample int_ior = ec_refractive_index(params.int_eta, params.int_k, SpectralDistribution::Conductor, spect, 0u);
      ThinfilmEval thinfilm = ec_thinfilm(params, spect, 0u);
      SpectralResponse fms = ec_conductor_fms(spect, ext_ior, int_ior, thinfilm, geometric_average);
      conductor_fms = float4(fms.integrated, 1.0f);
    }
    if (params.cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
      store_float4(params.output_average_index, alpha_index, average);
    } else {
      store_float4(params.output_average_index, alpha_index, float4(average.x, average.y, average.z, 1.0f));
    }
    store_float4(params.output_geometric_average_index, alpha_index, float4(geometric_average, 0.0f, 0.0f, 1.0f));
    store_float4(params.output_conductor_fms_index, alpha_index, conductor_fms);
    return;
  }

  if (params.pass_kind == kBSDFEnergyCompensationGpuPassDielectricDirectional) {
    if ((id.x >= kBSDFEnergyCompensationDielectricLutSize) || (id.y >= (2u * kBSDFEnergyCompensationDielectricLutSize))) {
      return;
    }
    uint side = id.y / kBSDFEnergyCompensationDielectricLutSize;
    uint alpha_index = id.y - side * kBSDFEnergyCompensationDielectricLutSize;
    uint mu_index = id.x;
    bool incident_outside = side == 0u;
    float mu = ec_mu_parameter(mu_index, kBSDFEnergyCompensationDielectricLutSize);
    float alpha = ec_alpha_parameter(alpha_index, kBSDFEnergyCompensationDielectricLutSize);
    float4 branch_albedo[2] = {float4(0.0f, 0.0f, 0.0f, 0.0f), float4(0.0f, 0.0f, 0.0f, 0.0f)};
    float4 total_albedo[2] = {float4(0.0f, 0.0f, 0.0f, 0.0f), float4(0.0f, 0.0f, 0.0f, 0.0f)};
    float4 probability[2] = {float4(0.0f, 0.0f, 0.0f, 0.0f), float4(0.0f, 0.0f, 0.0f, 0.0f)};
    if (params.cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
      uint channel = params.spectral_channel;
      EnergyCompensationDielectricResult single_value = ec_integrate_dielectric_directional(params, channel, false, incident_outside, mu, alpha);
      EnergyCompensationDielectricResult total_value = ec_integrate_dielectric_directional(params, channel, true, incident_outside, mu, alpha);
      uint incident_side = side;
      for (uint outgoing_side = 0u; outgoing_side < 2u; ++outgoing_side) {
        uint branch = ec_dielectric_branch_index(incident_side, outgoing_side);
        uint x = branch * kBSDFEnergyCompensationDielectricLutSize + mu_index;
        uint out_index = alpha_index * (kBSDFEnergyCompensationDielectricBranchCount * kBSDFEnergyCompensationDielectricLutSize) + x;
        if (channel > 0u) {
          branch_albedo[outgoing_side] = load_float4(params.output_directional_index, out_index);
          total_albedo[outgoing_side] = load_float4(params.output_total_index, out_index);
          probability[outgoing_side] = load_float4(params.output_probability_index, out_index);
        }
        ec_set_channel(branch_albedo[outgoing_side], channel, single_value.branch_albedo[outgoing_side].x);
        ec_set_channel(total_albedo[outgoing_side], channel, total_value.branch_albedo[outgoing_side].x);
        ec_set_channel(probability[outgoing_side], channel, single_value.branch_visible_probability[outgoing_side]);
      }
    } else {
      EnergyCompensationDielectricResult single_value = ec_integrate_dielectric_directional(params, 0u, false, incident_outside, mu, alpha);
      EnergyCompensationDielectricResult total_value = ec_integrate_dielectric_directional(params, 0u, true, incident_outside, mu, alpha);
      branch_albedo[0] = float4(single_value.branch_albedo[0].xyz, single_value.branch_visible_probability[0]);
      branch_albedo[1] = float4(single_value.branch_albedo[1].xyz, single_value.branch_visible_probability[1]);
      total_albedo[0] = float4(total_value.branch_albedo[0].xyz, total_value.visible_probability);
      total_albedo[1] = float4(total_value.branch_albedo[1].xyz, total_value.visible_probability);
      probability[0] = float4(single_value.branch_visible_probability[0], single_value.branch_visible_probability[0], single_value.branch_visible_probability[0], single_value.branch_visible_probability[0]);
      probability[1] = float4(single_value.branch_visible_probability[1], single_value.branch_visible_probability[1], single_value.branch_visible_probability[1], single_value.branch_visible_probability[1]);
    }
    uint incident_side = side;
    for (uint outgoing_side = 0u; outgoing_side < 2u; ++outgoing_side) {
      uint branch = ec_dielectric_branch_index(incident_side, outgoing_side);
      uint x = branch * kBSDFEnergyCompensationDielectricLutSize + mu_index;
      uint out_index = alpha_index * (kBSDFEnergyCompensationDielectricBranchCount * kBSDFEnergyCompensationDielectricLutSize) + x;
      store_float4(params.output_directional_index, out_index, branch_albedo[outgoing_side]);
      store_float4(params.output_total_index, out_index, total_albedo[outgoing_side]);
      store_float4(params.output_probability_index, out_index, probability[outgoing_side]);
    }
    return;
  }

  if (params.pass_kind == kBSDFEnergyCompensationGpuPassDielectricAverage) {
    if ((id.x >= kBSDFEnergyCompensationDielectricLutSize) || (id.y != 0u)) {
      return;
    }
    uint alpha_index = id.x;
    float4 single_average[kBSDFEnergyCompensationDielectricBranchCount];
    float4 total_average[kBSDFEnergyCompensationDielectricBranchCount];
    float4 residual_average[kBSDFEnergyCompensationDielectricBranchCount];
    float4 residual_coefficient[kBSDFEnergyCompensationDielectricBranchCount];
    for (uint incident_side = 0u; incident_side < 2u; ++incident_side) {
      for (uint outgoing_side = 0u; outgoing_side < 2u; ++outgoing_side) {
        uint branch = ec_dielectric_branch_index(incident_side, outgoing_side);
        single_average[branch] = ec_integrate_dielectric_branch_average(params.output_directional_index, alpha_index, branch);
        total_average[branch] = ec_integrate_dielectric_branch_average(params.output_total_index, alpha_index, branch);
      }
    }
    float4 side_residual[2];
    for (uint side = 0u; side < 2u; ++side) {
      uint branch_0 = ec_dielectric_branch_index(side, 0u);
      uint branch_1 = ec_dielectric_branch_index(side, 1u);
      float4 row_single = ec_saturate4(single_average[branch_0] + single_average[branch_1]);
      side_residual[side] = max(float4(0.0f, 0.0f, 0.0f, 0.0f), float4(1.0f, 1.0f, 1.0f, 1.0f) - row_single);
    }
    for (uint incident_side = 0u; incident_side < 2u; ++incident_side) {
      for (uint outgoing_side = 0u; outgoing_side < 2u; ++outgoing_side) {
        uint branch = ec_dielectric_branch_index(incident_side, outgoing_side);
        residual_average[branch] = max(float4(0.0f, 0.0f, 0.0f, 0.0f), total_average[branch] - single_average[branch]);
      }
    }
    for (uint incident_side = 0u; incident_side < 2u; ++incident_side) {
      uint branch_0 = ec_dielectric_branch_index(incident_side, 0u);
      uint branch_1 = ec_dielectric_branch_index(incident_side, 1u);
      float4 residual_sum = residual_average[branch_0] + residual_average[branch_1];
      float4 scale = min(float4(1.0f, 1.0f, 1.0f, 1.0f), side_residual[incident_side] / max(float4(kEpsilon, kEpsilon, kEpsilon, kEpsilon), residual_sum));
      residual_average[branch_0] *= scale;
      residual_average[branch_1] *= scale;
    }
    for (uint incident_side = 0u; incident_side < 2u; ++incident_side) {
      for (uint outgoing_side = 0u; outgoing_side < 2u; ++outgoing_side) {
        uint branch = ec_dielectric_branch_index(incident_side, outgoing_side);
        residual_coefficient[branch] = ec_residual_coefficient(residual_average[branch], side_residual[incident_side], side_residual[outgoing_side]);
      }
    }
    for (uint branch = 0u; branch < kBSDFEnergyCompensationDielectricBranchCount; ++branch) {
      if (params.cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
        store_float4(params.output_average_index, alpha_index * kBSDFEnergyCompensationDielectricAverageWidth + branch, ec_saturate4(single_average[branch]));
        store_float4(params.output_average_index, alpha_index * kBSDFEnergyCompensationDielectricAverageWidth + 4u + branch, max(float4(0.0f, 0.0f, 0.0f, 0.0f), residual_coefficient[branch]));
      } else {
        float4 single_value = ec_saturate4(single_average[branch]);
        float4 residual_value = max(float4(0.0f, 0.0f, 0.0f, 0.0f), residual_coefficient[branch]);
        store_float4(params.output_average_index, alpha_index * kBSDFEnergyCompensationDielectricAverageWidth + branch, float4(single_value.x, single_value.y, single_value.z, 1.0f));
        store_float4(params.output_average_index, alpha_index * kBSDFEnergyCompensationDielectricAverageWidth + 4u + branch, float4(residual_value.x, residual_value.y, residual_value.z, 1.0f));
      }
    }
  }
}
