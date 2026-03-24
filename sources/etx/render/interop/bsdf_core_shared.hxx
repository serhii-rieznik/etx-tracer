#pragma once

#include "bsdf_sampler_shared.hxx"
#include "material.hxx"
#include "math_shared.hxx"

ETX_SHARED_INLINE SpectralResponse bsdf_spectral_response_make(ETX_IN(SpectralQuery, query), float value) {
  return spectral_response_make(query, value);
}

ETX_SHARED_INLINE SpectralResponse bsdf_spectral_response_make(ETX_IN(SpectralQuery, query), ETX_IN(float3, value)) {
  return spectral_response_make(query, value);
}

struct PathSource {
  enum : uint32_t {
    Undefined = 0u,
    Camera = 1u,
    Light = 2u,
  };
};

struct BSDFData {
  float3 pos ETX_INIT({});
  float3 nrm ETX_INIT({});
  float3 tan ETX_INIT({});
  float3 btn ETX_INIT({});
  float2 tex ETX_INIT({});
  float3 w_i ETX_INIT({});
  SpectralQuery spectrum_sample ETX_INIT({});
  uint32_t path_source ETX_INIT(PathSource::Undefined);
  uint32_t current_medium ETX_INIT(kInvalidIndex);
};

ETX_SHARED_INLINE BSDFData bsdf_data_make(ETX_IN(Vertex, vertex), ETX_IN(SpectralQuery, spect), uint32_t medium_index, uint32_t source, ETX_IN(float3, incoming_direction)) {
  BSDFData result = ETX_ZERO(BSDFData);
  result.pos = vertex.pos;
  result.nrm = vertex.nrm;
  result.tan = vertex.tan;
  result.btn = vertex.btn;
  result.tex = vertex.tex;
  result.w_i = incoming_direction;
  result.spectrum_sample = spect;
  result.path_source = source;
  result.current_medium = medium_index;
  return result;
}

ETX_SHARED_INLINE float3 bsdf_data_front_facing_normal(ETX_IN(BSDFData, data)) {
  return (dot(data.nrm, data.w_i) < 0.0f) ? data.nrm : -data.nrm;
}

ETX_SHARED_INLINE LocalFrame bsdf_local_frame_make(ETX_IN(float3, tangent_hint), ETX_IN(float3, bitangent_hint), ETX_IN(float3, normal), bool entering_material) {
  float3 frame_normal = normal;
  float3 tangent = tangent_hint;
  float3 bitangent = bitangent_hint;

#if (ETX_CPP == 0)
  float3 normalized_normal = normalize(normal);
  frame_normal = normalized_normal;
  float tangent_hint_length_sq = dot(tangent_hint, tangent_hint);
  float bitangent_hint_length_sq = dot(bitangent_hint, bitangent_hint);
  if ((tangent_hint_length_sq > 0.0f) && (bitangent_hint_length_sq > 0.0f)) {
    tangent = orthogonalize(tangent_hint, normalized_normal);
    bitangent = normalize(cross(normalized_normal, tangent));
    if (dot(bitangent, bitangent_hint) < 0.0f) {
      bitangent = -bitangent;
    }
  } else {
    OrthonormalBasis basis = orthonormal_basis(normalized_normal);
    tangent = basis.u;
    bitangent = basis.v;
  }
#endif

  LocalFrame result = ETX_ZERO(LocalFrame);
  result.tan = entering_material ? tangent : -tangent;
  result.btn = entering_material ? bitangent : -bitangent;
  result.nrm = entering_material ? frame_normal : -frame_normal;
  result.flags = entering_material ? LocalFrame::EnteringMaterial : 0u;
  return result;
}

ETX_SHARED_INLINE LocalFrame bsdf_data_get_normal_frame(ETX_IN(BSDFData, data)) {
  bool entering_material = dot(data.nrm, data.w_i) < 0.0f;
  return bsdf_local_frame_make(data.tan, data.btn, data.nrm, entering_material);
}

ETX_SHARED_INLINE LocalFrame bsdf_data_get_normal_frame(ETX_IN(BSDFData, data), ETX_IN(Material, material)) {
  bool entering_material = dot(data.nrm, data.w_i) < 0.0f;
  bool force_entering = (material.two_sided != 0u) && (entering_material == false);
  bool use_entering_frame = entering_material || force_entering;
  return bsdf_local_frame_make(data.tan, data.btn, data.nrm, use_entering_frame);
}

struct BSDFEval {
  SpectralResponse func ETX_INIT({});
  SpectralResponse bsdf ETX_INIT({});
  float pdf ETX_INIT(0.0f);
  float eta ETX_INIT(1.0f);
};

ETX_SHARED_INLINE BSDFEval bsdf_eval_zero(ETX_IN(SpectralQuery, query)) {
  BSDFEval result = ETX_ZERO(BSDFEval);
  result.func = bsdf_spectral_response_make(query, 0.0f);
  result.bsdf = bsdf_spectral_response_make(query, 0.0f);
  result.pdf = 0.0f;
  result.eta = 1.0f;
  return result;
}

ETX_SHARED_INLINE bool bsdf_eval_valid(ETX_IN(BSDFEval, value)) {
  return value.pdf > 0.0f;
}

struct BSDFSample {
  enum Properties : uint32_t {
    Diffuse = 1u << 0u,
    Reflection = 1u << 1u,
    Transmission = 1u << 2u,
    MediumChanged = 1u << 3u,
    Delta = 1u << 4u,
  };

  SpectralResponse weight ETX_INIT({});
  float3 w_o ETX_INIT({});
  float pdf ETX_INIT(0.0f);
  float eta ETX_INIT(1.0f);
  uint32_t properties ETX_INIT(0u);
  uint32_t medium_index ETX_INIT(kInvalidIndex);
  uint32_t pad ETX_INIT(0u);
};

ETX_SHARED_INLINE BSDFSample bsdf_sample_zero(ETX_IN(SpectralQuery, query)) {
  BSDFSample result = ETX_ZERO(BSDFSample);
  result.weight = bsdf_spectral_response_make(query, 0.0f);
  result.eta = 1.0f;
  result.medium_index = kInvalidIndex;
  return result;
}

ETX_SHARED_INLINE bool bsdf_sample_valid(ETX_IN(BSDFSample, value)) {
  return value.pdf > 0.0f;
}

ETX_SHARED_INLINE bool bsdf_sample_invalid(ETX_IN(BSDFSample, value)) {
  return value.pdf <= 0.0f;
}

ETX_SHARED_INLINE bool bsdf_sample_is_diffuse(ETX_IN(BSDFSample, value)) {
  return (value.properties & BSDFSample::Diffuse) != 0u;
}

ETX_SHARED_INLINE bool bsdf_sample_is_delta(ETX_IN(BSDFSample, value)) {
  return (value.properties & BSDFSample::Delta) == BSDFSample::Delta;
}

struct BSDFNormalDistributionEval {
  float ndf ETX_INIT(0.0f);
  float g1_in ETX_INIT(0.0f);
  float visibility ETX_INIT(0.0f);
  float pdf ETX_INIT(0.0f);
};

ETX_STATIC_CONST float kBSDFNormalDistributionMinAlpha = 1.0f / 256.0f;

ETX_SHARED_INLINE float bsdf_normal_distribution_visibility_local(ETX_IN(float2, alpha), ETX_IN(float3, m), ETX_IN(float3, w)) {
  float xy_alpha_2 = (alpha.x * w.x) * (alpha.x * w.x) + (alpha.y * w.y) * (alpha.y * w.y);
  if (xy_alpha_2 == 0.0f) {
    return 1.0f;
  }

  if ((dot(w, m) * w.z) <= 0.0f) {
    return 0.0f;
  }

  float tan_theta_alpha_2 = xy_alpha_2 / (w.z * w.z);
  float result = 2.0f / (1.0f + sqrt(1.0f + tan_theta_alpha_2));
  return result;
}

ETX_SHARED_INLINE float bsdf_normal_distribution_visibility_term_local(ETX_IN(float2, alpha), ETX_IN(float3, m), ETX_IN(float3, w_i), ETX_IN(float3, w_o)) {
  return bsdf_normal_distribution_visibility_local(alpha, m, w_i) * bsdf_normal_distribution_visibility_local(alpha, m, w_o);
}

ETX_SHARED_INLINE float bsdf_normal_distribution_local(ETX_IN(float2, alpha), ETX_IN(float3, m)) {
  float alpha_uv = alpha.x * alpha.y;
  float mx = m.x / alpha.x;
  float my = m.y / alpha.y;
  float mz = m.z;
  float result = 1.0f / (kPi * alpha_uv * ((mx * mx + my * my + mz * mz) * (mx * mx + my * my + mz * mz)));
  return result;
}

ETX_SHARED_INLINE float3 bsdf_normal_distribution_sample(ETX_IN(LocalFrame, frame), ETX_IN(float2, alpha_value), ETX_INOUT(Sampler, sampler), ETX_IN(float3, incoming_direction)) {
  float2 alpha = float2(max(kBSDFNormalDistributionMinAlpha, alpha_value.x), max(kBSDFNormalDistributionMinAlpha, alpha_value.y));
  float3 w_i = local_frame_to_local(frame, -incoming_direction);
  float3 v_h = normalize(float3(alpha.x * w_i.x, alpha.y * w_i.y, w_i.z));

  float v_h_len = v_h.x * v_h.x + v_h.y * v_h.y;
  float3 u = (v_h_len > 0.0f) ? float3(-v_h.y, v_h.x, 0.0f) / sqrt(v_h_len) : float3(1.0f, 0.0f, 0.0f);
  float3 v = cross(v_h, u);

  float r = sqrt(bsdf_sampler_next(sampler));
  float phi = kDoublePi * bsdf_sampler_next(sampler);
  float t1 = r * cos(phi);
  float t2 = r * sin(phi);
  float s = 0.5f * (1.0f + v_h.z);
  t2 = (1.0f - s) * sqrt(1.0f - t1 * t1) + s * t2;
  float3 n_h = t1 * u + t2 * v + sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * v_h;
  float3 local_m = normalize(float3(alpha.x * n_h.x, alpha.y * n_h.y, n_h.z));
  return local_frame_from_local(frame, local_m);
}

ETX_SHARED_INLINE BSDFNormalDistributionEval bsdf_normal_distribution_evaluate(ETX_IN(LocalFrame, frame), ETX_IN(float2, alpha_value), ETX_IN(float3, in_m),
  ETX_IN(float3, incoming_direction), ETX_IN(float3, outgoing_direction)) {
  float2 alpha = float2(max(kBSDFNormalDistributionMinAlpha, alpha_value.x), max(kBSDFNormalDistributionMinAlpha, alpha_value.y));
  float3 local_w_i = local_frame_to_local(frame, -incoming_direction);
  if (local_w_i.z <= kEpsilon) {
    return ETX_ZERO(BSDFNormalDistributionEval);
  }

  float3 local_w_o = local_frame_to_local(frame, outgoing_direction);
  float3 local_m = local_frame_to_local(frame, in_m);

  BSDFNormalDistributionEval result = ETX_ZERO(BSDFNormalDistributionEval);
  result.visibility = bsdf_normal_distribution_visibility_term_local(alpha, local_m, local_w_i, local_w_o);
  result.ndf = bsdf_normal_distribution_local(alpha, local_m);
  result.g1_in = bsdf_normal_distribution_visibility_local(alpha, local_m, local_w_i);
  float s = abs(dot(local_w_i, local_m)) / local_w_i.z;
  result.pdf = result.ndf * result.g1_in * s;
  return result;
}

ETX_SHARED_INLINE float bsdf_normal_distribution_pdf(ETX_IN(LocalFrame, frame), ETX_IN(float2, alpha_value), ETX_IN(float3, in_m), ETX_IN(float3, incoming_direction),
  ETX_IN(float3, outgoing_direction)) {
  float2 alpha = float2(max(kBSDFNormalDistributionMinAlpha, alpha_value.x), max(kBSDFNormalDistributionMinAlpha, alpha_value.y));
  float3 local_w_i = local_frame_to_local(frame, -incoming_direction);
  if (local_w_i.z <= kEpsilon) {
    return 0.0f;
  }

  float3 local_m = local_frame_to_local(frame, in_m);
  float g1 = bsdf_normal_distribution_visibility_local(alpha, local_m, local_w_i);
  float d = bsdf_normal_distribution_local(alpha, local_m);
  float s = abs(dot(local_w_i, local_m)) / local_w_i.z;
  return g1 * d * s;
}

ETX_SHARED_INLINE float bsdf_fix_shading_normal(ETX_IN(float3, geo_normal), ETX_IN(float3, shading_normal), ETX_IN(float3, incoming_direction),
  ETX_IN(float3, outgoing_direction)) {
  float incoming_geo = dot(incoming_direction, geo_normal);
  float incoming_shading = dot(incoming_direction, shading_normal);
  float outgoing_geo = dot(outgoing_direction, geo_normal);
  float outgoing_shading = dot(outgoing_direction, shading_normal);
  float denom = max(kInvMaxHalf, abs(outgoing_shading * incoming_geo));
  return abs(outgoing_geo * incoming_shading) / denom;
}
