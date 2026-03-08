#include <interop/geometry.hxx>
#include <shaders/bindless.hlsl>

struct OceanPushConstants {
  uint vbIndex;
  uint samplerIndex;
  uint instanceBufferIndex;
  uint settingsBufferIndex;
  uint dispMapIndex[4];
  float4 cameraPos;
  column_major float4x4 vpMatrix;
};

[[vk::push_constant]]
OceanPushConstants pushConstants;

struct VSOutput {
  float4 position : SV_Position;
  float3 worldPos : WORLD_POS;
  float3 geomNormal : TEXCOORD2;
  float2 uv : TEXCOORD;
  float2 surfaceXZ : TEXCOORD3;
  nointerpolation float mipLevel : TEXCOORD1;
};

struct ClipmapInstance {
  float x_offset;
  float z_offset;
  float scale;
  float level;
};

struct OceanRenderSettings {
  uint envmapIndex;
  uint sceneColorIndex;
  uint envSamplerIndex;
  uint sceneSamplerIndex;
  float stitchTransitionCells;
  float mipColorMix;
  float mipColorEnable;
  float _padding0;
  float4 cascadeLengths;
  float4 cascadeWeights;
  float4 clipmapCenterOffset;
  float4 debugView;
  float4 surfaceNormalControls;
  float4 wireframeColor;
  float4 screenSize;
  column_major float4x4 invViewProj;
  uint4 surfaceDerivUIndex;
  uint4 surfaceDerivVIndex;
  uint4 slopeMetricIndex;
  float4 waterOptics0;
  float4 waterOptics1;
  float4 waterAbsorption;
  float4 waterScattering;
  float4 sunDirectionEnable;
  float4 sunRadiance;
  column_major float4x4 prevViewProj;
  uint waveThicknessMinIndex;
  uint waveThicknessMaxIndex;
  uint waveThicknessSamplerIndex;
  uint _padding1;
  uint foamHistoryIndex;
  uint foamDetailIndex;
  uint foamSamplerIndex;
  uint aerationDetailIndex;
  float4 waveThicknessControls;
  float4 foamControls0;
  float4 foamControls1;
  float4 foamControls2;
  float4 foamColor;
  float4 foamTemporalControls;
  float4 foamDetailControls;
  float4 foamFlowControls;
};

float3 sample_displacement_field(float2 world_xz, float cascade_lengths[3], float cascade_weights[3]) {
  float3 disp = float3(0.0f, 0.0f, 0.0f);
  SamplerState envSampler = bindless_samplers[NonUniformResourceIndex(pushConstants.samplerIndex)];
  for (int i = 0; i < 3; ++i) {
    if ((pushConstants.dispMapIndex[i] != 0u) && (cascade_weights[i] > 0.0f)) {
      Texture2D dispTex = bindless_textures[NonUniformResourceIndex(pushConstants.dispMapIndex[i])];
      float2 uv = world_xz / cascade_lengths[i];
      disp += dispTex.SampleLevel(envSampler, uv, 0).xyz * cascade_weights[i];
    }
  }
  return disp;
}

float displacement_sample_step(float cascade_lengths[3], float cascade_weights[3]) {
  bool has_step = false;
  float min_step = 0.0f;
  for (int i = 0; i < 3; ++i) {
    if ((pushConstants.dispMapIndex[i] != 0u) && (cascade_weights[i] > 0.0f)) {
      Texture2D dispTex = bindless_textures[NonUniformResourceIndex(pushConstants.dispMapIndex[i])];
      uint tex_width = 0u;
      uint tex_height = 0u;
      dispTex.GetDimensions(tex_width, tex_height);
      float texel_world = cascade_lengths[i] / max((float)tex_width, 1.0f);
      if ((has_step == false) || (texel_world < min_step)) {
        min_step = texel_world;
        has_step = true;
      }
    }
  }
  if (has_step == false) {
    min_step = 0.25f;
  }
  return max(min_step, 1.0e-3f);
}

float3 top_surface_normal_from_tangent_basis(float3 dPdu, float3 dPdv) {
  float det_xz = (dPdu.x * dPdv.z) - (dPdv.x * dPdu.z);
  if (abs(det_xz) > 1.0e-8f) {
    float inv_det_xz = 1.0f / det_xz;
    float dYdX = ((dPdu.y * dPdv.z) - (dPdv.y * dPdu.z)) * inv_det_xz;
    float dYdZ = ((dPdv.y * dPdu.x) - (dPdu.y * dPdv.x)) * inv_det_xz;
    float3 n = float3(-dYdX, 1.0f, -dYdZ);
    float n2 = dot(n, n);
    if (n2 > 1.0e-12f) {
      n *= rsqrt(n2);
      if (n.y < 0.0f) {
        n = -n;
      }
      return n;
    }
  }

  float3 n = cross(dPdv, dPdu);
  float n2 = dot(n, n);
  if (n2 > 1.0e-12f) {
    n *= rsqrt(n2);
    if (n.y < 0.0f) {
      n = -n;
    }
    return n;
  }

  return float3(0.0f, 1.0f, 0.0f);
}

float3 displacement_geometric_normal(float2 world_xz, float cascade_lengths[3], float cascade_weights[3]) {
  float h = displacement_sample_step(cascade_lengths, cascade_weights);
  float2 offset_x = float2(h, 0.0f);
  float2 offset_z = float2(0.0f, h);

  float3 p_x_plus = float3(world_xz.x + h, 0.0f, world_xz.y) + sample_displacement_field(world_xz + offset_x, cascade_lengths, cascade_weights);
  float3 p_x_minus = float3(world_xz.x - h, 0.0f, world_xz.y) + sample_displacement_field(world_xz - offset_x, cascade_lengths, cascade_weights);
  float3 p_z_plus = float3(world_xz.x, 0.0f, world_xz.y + h) + sample_displacement_field(world_xz + offset_z, cascade_lengths, cascade_weights);
  float3 p_z_minus = float3(world_xz.x, 0.0f, world_xz.y - h) + sample_displacement_field(world_xz - offset_z, cascade_lengths, cascade_weights);

  float3 tangent_x = p_x_plus - p_x_minus;
  float3 tangent_z = p_z_plus - p_z_minus;
  return top_surface_normal_from_tangent_basis(tangent_x, tangent_z);
}

float3 reconstruct_surface_normal(
  float2 surface_xz,
  float2 surface_xz_ddx,
  float2 surface_xz_ddy,
  OceanRenderSettings settings,
  float cascade_lengths[3],
  float cascade_weights[3],
  int cascade_filter,
  bool use_cascade_weights,
  float surface_normal_strength,
  bool sample_mip0,
  out bool has_surface_data) {
  float3 base_dPdu = float3(1.0f, 0.0f, 0.0f);
  float3 base_dPdv = float3(0.0f, 0.0f, 1.0f);
  float3 dPdu = base_dPdu;
  float3 dPdv = base_dPdv;
  has_surface_data = false;

  SamplerState envSampler = bindless_samplers[NonUniformResourceIndex(pushConstants.samplerIndex)];
  float gain = saturate(surface_normal_strength);

  for (int i = 0; i < 3; ++i) {
    if ((cascade_filter >= 0) && (i != cascade_filter)) {
      continue;
    }
    if ((settings.surfaceDerivUIndex[i] == 0u) || (settings.surfaceDerivVIndex[i] == 0u)) {
      continue;
    }

    float cascade_weight = use_cascade_weights ? cascade_weights[i] : 1.0f;
    if (cascade_weight <= 0.0f) {
      continue;
    }

    float2 uv = surface_xz / cascade_lengths[i];
    float2 uv_ddx = surface_xz_ddx / cascade_lengths[i];
    float2 uv_ddy = surface_xz_ddy / cascade_lengths[i];
    Texture2D derivUTex = bindless_textures[NonUniformResourceIndex(settings.surfaceDerivUIndex[i])];
    Texture2D derivVTex = bindless_textures[NonUniformResourceIndex(settings.surfaceDerivVIndex[i])];
    float3 sample_dPdu = float3(1.0f, 0.0f, 0.0f);
    float3 sample_dPdv = float3(0.0f, 0.0f, 1.0f);
    if (sample_mip0) {
      sample_dPdu = derivUTex.SampleLevel(envSampler, uv, 0.0f).xyz;
      sample_dPdv = derivVTex.SampleLevel(envSampler, uv, 0.0f).xyz;
    } else {
      sample_dPdu = derivUTex.SampleGrad(envSampler, uv, uv_ddx, uv_ddy).xyz;
      sample_dPdv = derivVTex.SampleGrad(envSampler, uv, uv_ddx, uv_ddy).xyz;
    }

    dPdu += ((sample_dPdu - base_dPdu) * cascade_weight * gain);
    dPdv += ((sample_dPdv - base_dPdv) * cascade_weight * gain);
    has_surface_data = true;
  }

  return top_surface_normal_from_tangent_basis(dPdu, dPdv);
}

float smoothstep_range(float edge0, float edge1, float x) {
  float delta = edge1 - edge0;
  if (abs(delta) <= 1.0e-8f) {
    return (x >= edge1) ? 1.0f : 0.0f;
  }
  float t = saturate((x - edge0) / delta);
  return t * t * (3.0f - (2.0f * t));
}

float3 sample_filtered_slope_metrics(
  float2 surface_xz,
  float2 surface_xz_ddx,
  float2 surface_xz_ddy,
  OceanRenderSettings settings,
  float cascade_lengths[3],
  float cascade_weights[3]) {
  SamplerState repeat_sampler = bindless_samplers[NonUniformResourceIndex(pushConstants.samplerIndex)];
  float slope_accum = 0.0f;
  float area_accum = 0.0f;
  float ny_accum = 0.0f;
  float weight_accum = 0.0f;
  for (int i = 0; i < 3; ++i) {
    if (settings.slopeMetricIndex[i] == 0u) {
      continue;
    }
    float weight = max(cascade_weights[i], 0.0f);
    if (weight <= 0.0f) {
      continue;
    }
    float2 uv = surface_xz / cascade_lengths[i];
    float2 uv_ddx = surface_xz_ddx / cascade_lengths[i];
    float2 uv_ddy = surface_xz_ddy / cascade_lengths[i];
    Texture2D slopeTex = bindless_textures[NonUniformResourceIndex(settings.slopeMetricIndex[i])];
    float4 metric_sample = slopeTex.SampleGrad(repeat_sampler, uv, uv_ddx, uv_ddy);
    float weight_sq = weight * weight;
    slope_accum += max(metric_sample.x, 0.0f) * weight_sq;
    area_accum += max(metric_sample.y, 0.0f) * weight;
    ny_accum += metric_sample.z * weight;
    weight_accum += weight;
  }

  float area_avg = 1.0f;
  float ny_avg = 1.0f;
  if (weight_accum > 1.0e-6f) {
    area_avg = area_accum / weight_accum;
    ny_avg = ny_accum / weight_accum;
  }
  return float3(max(slope_accum, 0.0f), area_avg, ny_avg);
}

float sample_filtered_slope_energy(
  float2 surface_xz,
  float2 surface_xz_ddx,
  float2 surface_xz_ddy,
  OceanRenderSettings settings,
  float cascade_lengths[3],
  float cascade_weights[3]) {
  return sample_filtered_slope_metrics(surface_xz, surface_xz_ddx, surface_xz_ddy, settings, cascade_lengths, cascade_weights).x;
}

float normal_angle_error_radians(float3 a, float3 b) {
  float a2 = dot(a, a);
  float b2 = dot(b, b);
  if ((a2 <= 1.0e-12f) || (b2 <= 1.0e-12f)) {
    return 0.0f;
  }
  float3 na = a * rsqrt(a2);
  float3 nb = b * rsqrt(b2);
  return acos(clamp(dot(na, nb), -1.0f, 1.0f));
}

float3 heat_color(float t) {
  float x = saturate(t);
  float3 c0 = float3(0.0f, 0.05f, 0.2f);
  float3 c1 = float3(0.0f, 0.75f, 1.0f);
  float3 c2 = float3(1.0f, 0.95f, 0.2f);
  float3 c3 = float3(1.0f, 0.1f, 0.0f);
  if (x < 0.3333f) {
    return lerp(c0, c1, x / 0.3333f);
  }
  if (x < 0.6666f) {
    return lerp(c1, c2, (x - 0.3333f) / 0.3333f);
  }
  return lerp(c2, c3, (x - 0.6666f) / 0.3334f);
}

float ior_to_f0(float ior) {
  float safe_ior = max(ior, 1.0f + 1.0e-4f);
  float f = (safe_ior - 1.0f) / (safe_ior + 1.0f);
  return f * f;
}

float schlick_fresnel(float f0, float ndotv) {
  float m = saturate(1.0f - ndotv);
  float m2 = m * m;
  float m5 = m2 * m2 * m;
  return f0 + ((1.0f - f0) * m5);
}

float3 beer_lambert(float3 sigma_a, float distance_m) {
  float safe_distance = max(distance_m, 0.0f);
  return exp(-max(sigma_a, 0.0f) * safe_distance);
}

float2 sample_env_uv(float3 dir, bool equal_area_mapping) {
  float inv_pi = 1.0f / 3.14159265f;
  float inv_two_pi = 0.5f * inv_pi;
  float u = (atan2(dir.z, dir.x) * inv_two_pi) + 0.5f;
  float y = clamp(dir.y, -1.0f, 1.0f);
  float v = equal_area_mapping ? (0.5f * (1.0f - y)) : (acos(y) * inv_pi);
  return float2(u, v);
}

float transmitted_cos_theta(float ndotv, float water_ior) {
  float cos_i = saturate(ndotv);
  float safe_ior = max(water_ior, 1.0f + 1.0e-4f);
  float eta = 1.0f / safe_ior;
  float sin2_t = (eta * eta) * max(1.0f - (cos_i * cos_i), 0.0f);
  float cos2_t = max(1.0f - sin2_t, 0.0f);
  return sqrt(cos2_t);
}

float ggx_d(float alpha, float ndoth) {
  float a = max(alpha, 1.0e-4f);
  float a2 = a * a;
  float nh = saturate(ndoth);
  float nh2 = nh * nh;
  float denom = max(((nh2 * (a2 - 1.0f)) + 1.0f), 1.0e-4f);
  return a2 / max(3.14159265f * denom * denom, 1.0e-4f);
}

float smith_g1_ggx(float alpha, float ndotx) {
  float a = max(alpha, 1.0e-4f);
  float nx = saturate(ndotx);
  float a2 = a * a;
  float nx2 = nx * nx;
  float denom = nx + sqrt(max(a2 + ((1.0f - a2) * nx2), 1.0e-6f));
  return (2.0f * nx) / max(denom, 1.0e-6f);
}

float3 eval_ggx_specular_dielectric(float3 N, float3 V, float3 L, float f0, float alpha) {
  float3 H = normalize(V + L);
  float ndotv = saturate(dot(N, V));
  float ndotl = saturate(dot(N, L));
  float ndoth = saturate(dot(N, H));
  float vdoth = saturate(dot(V, H));
  if ((ndotv <= 0.0f) || (ndotl <= 0.0f)) {
    return float3(0.0f, 0.0f, 0.0f);
  }
  float D = ggx_d(alpha, ndoth);
  float G = smith_g1_ggx(alpha, ndotv) * smith_g1_ggx(alpha, ndotl);
  float F = schlick_fresnel(f0, vdoth);
  float spec = (D * G * F) / max(4.0f * ndotv * ndotl, 1.0e-6f);
  return float3(spec, spec, spec);
}

float2 screen_uv_from_svpos(float4 sv_pos, float4 screen_size) {
  float2 inv_size = max(screen_size.zw, float2(0.0f, 0.0f));
  return sv_pos.xy * inv_size;
}

float3 reconstruct_world_position_from_uv_depth(float2 uv, float depth, float4x4 inv_view_proj) {
  float2 ndc_xy = float2((uv.x * 2.0f) - 1.0f, 1.0f - (uv.y * 2.0f));
  float4 world = mul(inv_view_proj, float4(ndc_xy, depth, 1.0f));
  float inv_w = (abs(world.w) > 1.0e-6f) ? (1.0f / world.w) : 0.0f;
  return world.xyz * inv_w;
}

float sample_wave_thickness_meters(float2 screen_uv, OceanRenderSettings settings, out bool has_wave_thickness) {
  has_wave_thickness = false;
  if ((settings.waveThicknessMinIndex == 0u) || (settings.waveThicknessMaxIndex == 0u)) {
    return 0.0f;
  }

  SamplerState thickness_sampler = bindless_samplers[NonUniformResourceIndex(settings.waveThicknessSamplerIndex)];
  Texture2D min_tex = bindless_textures[NonUniformResourceIndex(settings.waveThicknessMinIndex)];
  Texture2D max_tex = bindless_textures[NonUniformResourceIndex(settings.waveThicknessMaxIndex)];
  float min_depth_m = min_tex.SampleLevel(thickness_sampler, screen_uv, 0.0f).x;
  float max_depth_m = max_tex.SampleLevel(thickness_sampler, screen_uv, 0.0f).x;
  bool valid_min = (min_depth_m < 1.0e30f);
  bool valid_max = (max_depth_m > 0.0f);
  float thickness_m = max(max_depth_m - min_depth_m, 0.0f);
  if ((valid_min) && (valid_max) && (thickness_m > 1.0e-5f)) {
    has_wave_thickness = true;
  } else {
    thickness_m = 0.0f;
  }
  return min(thickness_m, 256.0f);
}

bool clip_to_screen_uv(float4 clip_pos, out float2 out_uv) {
  out_uv = float2(0.0f, 0.0f);
  if (abs(clip_pos.w) <= 1.0e-6f) {
    return false;
  }
  float inv_w = 1.0f / clip_pos.w;
  float2 ndc_xy = clip_pos.xy * inv_w;
  if ((ndc_xy.x < -1.0f) || (ndc_xy.x > 1.0f) || (ndc_xy.y < -1.0f) || (ndc_xy.y > 1.0f)) {
    return false;
  }
  out_uv = float2((ndc_xy.x * 0.5f) + 0.5f, 0.5f - (ndc_xy.y * 0.5f));
  return true;
}

float sample_reprojected_foam_history(float3 world_pos, OceanRenderSettings settings, out bool has_history) {
  has_history = false;
  if (settings.foamHistoryIndex == 0u) {
    return 0.0f;
  }

  float2 prev_uv = float2(0.0f, 0.0f);
  float4 prev_clip = mul(settings.prevViewProj, float4(world_pos, 1.0f));
  if (clip_to_screen_uv(prev_clip, prev_uv) == false) {
    return 0.0f;
  }

  SamplerState foam_sampler = bindless_samplers[NonUniformResourceIndex(settings.foamSamplerIndex)];
  Texture2D foam_history_tex = bindless_textures[NonUniformResourceIndex(settings.foamHistoryIndex)];
  float history = saturate(foam_history_tex.SampleLevel(foam_sampler, prev_uv, 0.0f).x);
  has_history = true;
  return history;
}

float sample_detail_texture(float2 world_xz, float2 world_xz_ddx, float2 world_xz_ddy, uint texture_index, OceanRenderSettings settings) {
  if (texture_index == 0u) {
    return 1.0f;
  }
  SamplerState repeat_sampler = bindless_samplers[NonUniformResourceIndex(pushConstants.samplerIndex)];
  Texture2D detail_tex = bindless_textures[NonUniformResourceIndex(texture_index)];
  float2 wind_dir = settings.foamFlowControls.xy;
  float wind_len2 = dot(wind_dir, wind_dir);
  if (wind_len2 > 1.0e-8f) {
    wind_dir *= rsqrt(wind_len2);
  } else {
    wind_dir = float2(1.0f, 0.0f);
  }
  float time_s = settings.foamFlowControls.z;
  float scroll_speed = max(settings.foamFlowControls.w, 0.0f);
  float2 flow_offset = wind_dir * time_s * scroll_speed;

  float scale_1 = max(settings.foamDetailControls.x, 1.0e-4f);
  float scale_2 = max(settings.foamDetailControls.y, 1.0e-4f);
  float mix_amount = saturate(settings.foamDetailControls.z);
  float contrast = max(settings.foamDetailControls.w, 0.0f);

  float2 uv0 = (world_xz * scale_1) + flow_offset;
  float2 uv1 = (float2(-world_xz.y, world_xz.x) * scale_2) - (flow_offset * 1.73f);
  float2 uv0_ddx = world_xz_ddx * scale_1;
  float2 uv0_ddy = world_xz_ddy * scale_1;
  float2 uv1_ddx = float2(-world_xz_ddx.y, world_xz_ddx.x) * scale_2;
  float2 uv1_ddy = float2(-world_xz_ddy.y, world_xz_ddy.x) * scale_2;
  float n0 = detail_tex.SampleGrad(repeat_sampler, uv0, uv0_ddx, uv0_ddy).x;
  float n1 = detail_tex.SampleGrad(repeat_sampler, uv1, uv1_ddx, uv1_ddy).x;
  float detail = lerp(n0, n1, mix_amount);
  float centered = (detail * 2.0f) - 1.0f;
  float shaped = saturate((centered * contrast * 0.5f) + 0.5f);
  return shaped;
}

float sample_foam_detail_texture(float2 world_xz, float2 world_xz_ddx, float2 world_xz_ddy, OceanRenderSettings settings) {
  return sample_detail_texture(world_xz, world_xz_ddx, world_xz_ddy, settings.foamDetailIndex, settings);
}

float sample_aeration_detail_texture(float2 world_xz, float2 world_xz_ddx, float2 world_xz_ddy, OceanRenderSettings settings) {
  return sample_detail_texture(world_xz, world_xz_ddx, world_xz_ddy, settings.aerationDetailIndex, settings);
}

float evaluate_foam_mask(
  float wave_thickness_m,
  bool has_wave_thickness,
  float slope_energy,
  float area_mag,
  float slope_metric_ny,
  OceanRenderSettings settings) {
  if (settings.foamControls0.x <= 0.5f) {
    return 0.0f;
  }

  float foam_strength = max(settings.foamControls0.y, 0.0f);
  float slope_start = max(settings.foamControls0.z, 0.0f);
  float slope_end = max(settings.foamControls0.w, slope_start + 1.0e-5f);
  float thickness_start = max(settings.foamControls1.x, 0.0f);
  float thickness_end = max(settings.foamControls1.y, thickness_start + 1.0e-5f);

  float slope_term = smoothstep_range(slope_start, slope_end, slope_energy);
  float thickness_term = has_wave_thickness ? smoothstep_range(thickness_start, thickness_end, wave_thickness_m) : 0.0f;
  float compression_term = smoothstep_range(1.00f, 0.65f, area_mag);
  float foldover_term = smoothstep_range(0.20f, -0.05f, slope_metric_ny);
  float crest_term = smoothstep_range(0.85f, 0.35f, slope_metric_ny);
  float structural_term = max(max(compression_term, foldover_term), crest_term * 0.75f);
  float thickness_gate = max(thickness_term, structural_term * 0.5f);
  float foam_seed = max(slope_term * thickness_gate, structural_term * 0.65f);
  return saturate(foam_seed * foam_strength);
}

float accumulate_foam_history(float current_seed, float previous_history, bool has_previous_history, OceanRenderSettings settings) {
  float decay = saturate(settings.foamTemporalControls.x);
  float gain = max(settings.foamTemporalControls.y, 0.0f);
  float bias = max(settings.foamTemporalControls.z, 0.0f);
  float prev_term = has_previous_history ? saturate((previous_history * decay) - bias) : 0.0f;
  float curr_term = saturate(current_seed * gain);
  return saturate(max(curr_term, prev_term));
}

VSOutput VSMain(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID) {
  ByteAddressBuffer vb = bindless_buffers[NonUniformResourceIndex(pushConstants.vbIndex)];
  Vertex v = vb.Load<Vertex>(vertexId * 56);
  
  ByteAddressBuffer instanceBuffer = bindless_buffers[NonUniformResourceIndex(pushConstants.instanceBufferIndex)];
  ClipmapInstance inst = instanceBuffer.Load<ClipmapInstance>(instanceId * 16);
  ByteAddressBuffer settingsBuffer = bindless_buffers[NonUniformResourceIndex(pushConstants.settingsBufferIndex)];
  OceanRenderSettings settings = settingsBuffer.Load<OceanRenderSettings>(0);

  float patchResolution = max(settings.debugView.y, 1.0f);
  float gridSize = inst.scale / patchResolution;
  float levelScale = exp2(inst.level);
  float baseGridSize = gridSize / max(levelScale, 1.0f);
  float snappedCamX = floor(pushConstants.cameraPos.x / baseGridSize) * baseGridSize;
  float snappedCamZ = floor(pushConstants.cameraPos.z / baseGridSize) * baseGridSize;
  float snappedOriginX = snappedCamX + settings.clipmapCenterOffset.x;
  float snappedOriginZ = snappedCamZ + settings.clipmapCenterOffset.y;

  float3 worldPos = float3(v.pos.x * inst.scale, 0.0f, v.pos.z * inst.scale);
  worldPos.x += inst.x_offset + snappedOriginX;
  worldPos.z += inst.z_offset + snappedOriginZ;

  // Proper clipmap stitching: morph boundary strips to the parent grid.
  if ((inst.level < settings.debugView.w) && (settings.stitchTransitionCells > 0.0f)) {
    float parentGridSize = gridSize * 2.0f;
    float patchEps = inst.scale * 1e-5f;
    float transitionCells = clamp(settings.stitchTransitionCells, 1.0f, 16.0f);
    float transitionWidth = min(inst.scale * 0.5f, transitionCells * gridSize);

    bool isLeftOuterPatch = (abs(inst.x_offset - (-2.0f * inst.scale)) <= patchEps);
    bool isRightOuterPatch = (abs(inst.x_offset - (1.0f * inst.scale)) <= patchEps);
    bool isTopOuterPatch = (abs(inst.z_offset - (-2.0f * inst.scale)) <= patchEps);
    bool isBottomOuterPatch = (abs(inst.z_offset - (1.0f * inst.scale)) <= patchEps);

    if (isLeftOuterPatch) {
      float edgeDist = v.tex.x * inst.scale;
      float morph = saturate((transitionWidth - edgeDist) / transitionWidth);
      float snappedZ = (round((worldPos.z - snappedOriginZ) / parentGridSize) * parentGridSize) + snappedOriginZ;
      worldPos.z = lerp(worldPos.z, snappedZ, morph);
    } else if (isRightOuterPatch) {
      float edgeDist = (1.0f - v.tex.x) * inst.scale;
      float morph = saturate((transitionWidth - edgeDist) / transitionWidth);
      float snappedZ = (round((worldPos.z - snappedOriginZ) / parentGridSize) * parentGridSize) + snappedOriginZ;
      worldPos.z = lerp(worldPos.z, snappedZ, morph);
    }
    if (isTopOuterPatch) {
      float edgeDist = v.tex.y * inst.scale;
      float morph = saturate((transitionWidth - edgeDist) / transitionWidth);
      float snappedX = (round((worldPos.x - snappedOriginX) / parentGridSize) * parentGridSize) + snappedOriginX;
      worldPos.x = lerp(worldPos.x, snappedX, morph);
    } else if (isBottomOuterPatch) {
      float edgeDist = (1.0f - v.tex.y) * inst.scale;
      float morph = saturate((transitionWidth - edgeDist) / transitionWidth);
      float snappedX = (round((worldPos.x - snappedOriginX) / parentGridSize) * parentGridSize) + snappedOriginX;
      worldPos.x = lerp(worldPos.x, snappedX, morph);
    }
  }

  float cascade_lengths[3] = {settings.cascadeLengths.x, settings.cascadeLengths.y, settings.cascadeLengths.z};
  float cascade_weights[3] = {settings.cascadeWeights.x, settings.cascadeWeights.y, settings.cascadeWeights.z};
  float2 base_xz = worldPos.xz;
  float3 geom_normal = displacement_geometric_normal(base_xz, cascade_lengths, cascade_weights);
  float3 disp = sample_displacement_field(base_xz, cascade_lengths, cascade_weights);

  worldPos += disp;
  
  VSOutput output;
  output.position = mul(pushConstants.vpMatrix, float4(worldPos, 1.0f));
  output.worldPos = worldPos;
  output.geomNormal = geom_normal;
  output.uv = v.tex;
  output.surfaceXZ = base_xz;
  output.mipLevel = inst.level;
  return output;
}

float3 get_mip_color(uint mip_level) {
  if (mip_level == 0u) return float3(1.0f, 0.25f, 0.2f);
  if (mip_level == 1u) return float3(1.0f, 0.65f, 0.2f);
  if (mip_level == 2u) return float3(0.95f, 0.9f, 0.25f);
  if (mip_level == 3u) return float3(0.2f, 0.85f, 0.3f);
  if (mip_level == 4u) return float3(0.2f, 0.55f, 1.0f);
  return float3(0.75f, 0.3f, 1.0f);
}

float4 PSMain(VSOutput input) : SV_Target0 {
  ByteAddressBuffer settingsBuffer = bindless_buffers[NonUniformResourceIndex(pushConstants.settingsBufferIndex)];
  OceanRenderSettings settings = settingsBuffer.Load<OceanRenderSettings>(0);

  float3 N = float3(0.0, 1.0, 0.0);
  float cascade_lengths[3] = {settings.cascadeLengths.x, settings.cascadeLengths.y, settings.cascadeLengths.z};
  float cascade_weights[3] = {settings.cascadeWeights.x, settings.cascadeWeights.y, settings.cascadeWeights.z};
  bool surface_normal_shading_enable = (settings.surfaceNormalControls.x > 0.5f);
  float surface_normal_strength = saturate(settings.surfaceNormalControls.y);
  uint surface_normal_debug_mode = min((uint)round(max(settings.surfaceNormalControls.z, 0.0f)), 7u);

  float3 geom_n = input.geomNormal;
  float geom_n2 = dot(geom_n, geom_n);
  if (geom_n2 > 1e-12f) {
    geom_n *= rsqrt(geom_n2);
  } else {
    geom_n = float3(0.0f, 1.0f, 0.0f);
  }

  float2 surface_xz_ddx = ddx(input.surfaceXZ);
  float2 surface_xz_ddy = ddy(input.surfaceXZ);
  bool has_combined_surface_normal = false;
  float3 mapped_n = geom_n;
  if (surface_normal_strength > 0.0f) {
    mapped_n = reconstruct_surface_normal(
      input.surfaceXZ, surface_xz_ddx, surface_xz_ddy, settings, cascade_lengths, cascade_weights, -1, true, surface_normal_strength,
      false, has_combined_surface_normal);
    if (has_combined_surface_normal == false) {
      mapped_n = geom_n;
    }
  }

  if (surface_normal_debug_mode > 0u) {
    bool has_debug_surface_normal = false;
    float3 debug_normal = reconstruct_surface_normal(
      input.surfaceXZ, surface_xz_ddx, surface_xz_ddy, settings, cascade_lengths, cascade_weights, -1, true, 1.0f, false, has_debug_surface_normal);
    if (has_debug_surface_normal == false) {
      debug_normal = geom_n;
    }
    if ((surface_normal_debug_mode >= 2u) && (surface_normal_debug_mode <= 4u)) {
      uint cascade_index = surface_normal_debug_mode - 2u;
      debug_normal = reconstruct_surface_normal(
        input.surfaceXZ, surface_xz_ddx, surface_xz_ddy, settings, cascade_lengths, cascade_weights, (int)cascade_index, false, 1.0f,
        false, has_debug_surface_normal);
      if (has_debug_surface_normal == false) {
        debug_normal = float3(0.0f, 1.0f, 0.0f);
      }
      float3 normal_vis = (normalize(debug_normal) * 0.5f) + 0.5f;
      return float4(normal_vis, 1.0f);
    }

    if (surface_normal_debug_mode == 1u) {
      float3 normal_vis = (normalize(debug_normal) * 0.5f) + 0.5f;
      return float4(normal_vis, 1.0f);
    }

    if (surface_normal_debug_mode == 5u) {
      float3 reference_n = displacement_geometric_normal(input.surfaceXZ, cascade_lengths, cascade_weights);
      float angle_rad = normal_angle_error_radians(debug_normal, reference_n);
      float angle_deg = angle_rad * (180.0f / 3.14159265f);
      float3 color = heat_color(angle_deg / 10.0f);
      return float4(color, 1.0f);
    }

    if (surface_normal_debug_mode == 6u) {
      bool has_mip0_surface_normal = false;
      float3 mip0_normal = reconstruct_surface_normal(
        input.surfaceXZ, surface_xz_ddx, surface_xz_ddy, settings, cascade_lengths, cascade_weights, -1, true, 1.0f, true, has_mip0_surface_normal);
      if (has_mip0_surface_normal == false) {
        mip0_normal = geom_n;
      }
      float angle_rad = normal_angle_error_radians(debug_normal, mip0_normal);
      float angle_deg = angle_rad * (180.0f / 3.14159265f);
      float3 color = heat_color(angle_deg / 5.0f);
      return float4(color, 1.0f);
    }

    if (surface_normal_debug_mode == 7u) {
      float3 dPdu_dbg = float3(1.0f, 0.0f, 0.0f);
      float3 dPdv_dbg = float3(0.0f, 0.0f, 1.0f);
      bool has_basis = false;
      SamplerState envSampler = bindless_samplers[NonUniformResourceIndex(pushConstants.samplerIndex)];
      for (int i = 0; i < 3; ++i) {
        if ((settings.surfaceDerivUIndex[i] == 0u) || (settings.surfaceDerivVIndex[i] == 0u) || (cascade_weights[i] <= 0.0f)) {
          continue;
        }
        float2 uv = input.surfaceXZ / cascade_lengths[i];
        float2 uv_ddx = surface_xz_ddx / cascade_lengths[i];
        float2 uv_ddy = surface_xz_ddy / cascade_lengths[i];
        Texture2D derivUTex = bindless_textures[NonUniformResourceIndex(settings.surfaceDerivUIndex[i])];
        Texture2D derivVTex = bindless_textures[NonUniformResourceIndex(settings.surfaceDerivVIndex[i])];
        float3 sample_dPdu = derivUTex.SampleGrad(envSampler, uv, uv_ddx, uv_ddy).xyz;
        float3 sample_dPdv = derivVTex.SampleGrad(envSampler, uv, uv_ddx, uv_ddy).xyz;
        dPdu_dbg += (sample_dPdu - float3(1.0f, 0.0f, 0.0f)) * cascade_weights[i];
        dPdv_dbg += (sample_dPdv - float3(0.0f, 0.0f, 1.0f)) * cascade_weights[i];
        has_basis = true;
      }
      float3 area_vec = cross(dPdv_dbg, dPdu_dbg);
      float area_len = length(area_vec);
      float sign_y = area_vec.y;
      float3 color = heat_color(saturate(area_len * 0.5f));
      if ((has_basis == false) || (sign_y <= 0.0f)) {
        color = float3(1.0f, 0.0f, 1.0f);
      }
      return float4(color, 1.0f);
    }

    float3 fallback_vis = (normalize(debug_normal) * 0.5f) + 0.5f;
    return float4(fallback_vis, 1.0f);
  }

  float2 screen_uv = screen_uv_from_svpos(input.position, settings.screenSize);
  bool has_wave_thickness = false;
  float wave_thickness_m = sample_wave_thickness_meters(screen_uv, settings, has_wave_thickness);
  float wave_thickness_path_scale = max(settings.waveThicknessControls.x, 0.0f);
  uint water_debug_visualize_mode = min((uint)round(max(settings.waveThicknessControls.y, 0.0f)), 2u);
  if (water_debug_visualize_mode == 1u) {
    float debug_max_m = max(settings.waveThicknessControls.z, 1.0e-3f);
    float vis_t = has_wave_thickness ? (wave_thickness_m / debug_max_m) : 0.0f;
    float3 color = has_wave_thickness ? heat_color(vis_t) : float3(0.0f, 0.0f, 0.0f);
    return float4(color, 1.0f);
  }

  if (surface_normal_shading_enable == false) {
    N = geom_n;
  } else {
    N = mapped_n;
  }

  float3 V = normalize(pushConstants.cameraPos.xyz - input.worldPos);
  
  float NdotV = saturate(dot(N, V));
  bool physical_render_mode = (settings.waterOptics0.w > 0.5f);
  float water_ior = settings.waterOptics0.x;
  float env_reflection_intensity = max(settings.waterOptics0.y, 0.0f);
  float refract_distortion_scale = max(settings.waterOptics0.z, 0.0f);
  float f0 = ior_to_f0(water_ior);
  float fresnel = schlick_fresnel(f0, NdotV);
  float3 sigma_a_base = max(settings.waterAbsorption.xyz, 0.0f);
  float3 sigma_s_base = max(settings.waterScattering.xyz, 0.0f);
  float base_unresolved_roughness = min(max(settings.waterOptics1.y, 0.0f), 1.0f);
  float specular_aa_strength = max(settings.waterOptics1.z, 0.0f);
  float3 slope_metrics = sample_filtered_slope_metrics(input.surfaceXZ, surface_xz_ddx, surface_xz_ddy, settings, cascade_lengths, cascade_weights);
  float slope_energy = slope_metrics.x;
  float slope_area_mag = slope_metrics.y;
  float slope_metric_ny = slope_metrics.z;
  float foam_mask = evaluate_foam_mask(wave_thickness_m, has_wave_thickness, slope_energy, slope_area_mag, slope_metric_ny, settings);
  if (water_debug_visualize_mode == 2u) {
    return float4(heat_color(foam_mask), 1.0f);
  }
  float foam_surface_coverage = saturate(settings.foamControls1.z);
  float2 foam_world_xz_ddx = ddx(input.worldPos.xz);
  float2 foam_world_xz_ddy = ddy(input.worldPos.xz);
  float foam_detail = sample_foam_detail_texture(input.worldPos.xz, foam_world_xz_ddx, foam_world_xz_ddy, settings);
  float aeration_detail = sample_aeration_detail_texture(input.worldPos.xz, foam_world_xz_ddx, foam_world_xz_ddy, settings);
  float foam_detail_mix = saturate(settings.foamDetailControls.z);
  float bubble_tex = foam_detail;
  float bubble_fill = smoothstep_range(0.28f, 0.72f, bubble_tex);
  float bubble_rim = smoothstep_range(0.70f, 0.97f, bubble_tex);
  float bubble_alpha = saturate((bubble_fill * 0.35f) + (bubble_rim * 0.95f));
  float bubble_alpha_shaped = bubble_alpha * bubble_alpha;
  bubble_alpha_shaped = lerp(bubble_alpha_shaped, sqrt(max(bubble_alpha_shaped, 0.0f)), foam_detail_mix);
  float foam_surface_seed = saturate(foam_mask * foam_surface_coverage);
  float bubble_alpha_aa = max(fwidth(bubble_alpha_shaped), 1.0f / 255.0f);
  float foam_alpha_threshold = saturate(settings.foamColor.w);
  float bubble_threshold_hi = min(max(foam_alpha_threshold + 0.28f, 0.0f), 0.995f);
  float bubble_threshold_lo = min(max(foam_alpha_threshold - 0.32f, 0.0f), 0.995f);
  float bubble_threshold = lerp(bubble_threshold_hi, bubble_threshold_lo, foam_surface_seed);
  float bubble_coverage = smoothstep(bubble_threshold - bubble_alpha_aa, bubble_threshold + bubble_alpha_aa, bubble_alpha_shaped);
  float foam_surface_intensity = foam_surface_seed * foam_surface_seed;
  foam_surface_intensity = foam_surface_intensity * (3.0f - (2.0f * foam_surface_intensity));
  float foam_surface_mask = saturate(bubble_coverage * foam_surface_intensity);
  float3 foam_albedo = max(settings.foamColor.xyz, 0.0f);
  float foam_aeration_scatter_scale = max(settings.foamControls2.z, 0.0f);
  float foam_aeration_absorption_scale = max(settings.foamControls2.w, 0.0f);
  float foam_thickness_norm = has_wave_thickness ? saturate(wave_thickness_m / max(settings.foamControls1.y, 1.0e-3f)) : 0.0f;
  float aeration_pattern = smoothstep_range(0.15f, 0.95f, aeration_detail);
  float foam_optical_mask = max((foam_mask * lerp(0.10f, 0.45f, aeration_pattern)), foam_surface_mask);
  float aeration_density = foam_optical_mask * lerp(0.4f, 1.0f, foam_thickness_norm);
  float3 sigma_a = sigma_a_base + (foam_aeration_absorption_scale * aeration_density);
  float3 sigma_s = sigma_s_base + (foam_albedo * (foam_aeration_scatter_scale * aeration_density));
  float3 sigma_t = sigma_a + sigma_s;
  float3 scatter_albedo = sigma_s / max(sigma_t, 1.0e-4f);
  float slope_alpha = sqrt(max(slope_energy * 0.02f, 0.0f));
  float3 dNdx = ddx(N);
  float3 dNdy = ddy(N);
  float normal_grad2 = max(dot(dNdx, dNdx), dot(dNdy, dNdy));
  float spec_aa_roughness = sqrt(saturate(normal_grad2 * specular_aa_strength));
  float base_alpha = max(base_unresolved_roughness * base_unresolved_roughness, 1.0e-6f);
  float spec_aa_alpha = max(spec_aa_roughness * spec_aa_roughness, 1.0e-6f);
  float ggx_alpha = min(max(max(base_alpha, slope_alpha), spec_aa_alpha), 0.36f);
  float effective_roughness = sqrt(ggx_alpha);
  
  float3 waterRefractColor = float3(0.01f, 0.05f, 0.1f);
  bool has_scene_refraction = false;
  if (physical_render_mode) {
    // Keep volume refraction dark until we have an actual lighting source (scene color or env sample).
    waterRefractColor = float3(0.0f, 0.0f, 0.0f);
  }
  float3 waterReflectColor = float3(0.5f, 0.6f, 0.8f) * env_reflection_intensity;
  float3 direct_specular = float3(0.0f, 0.0f, 0.0f);

  if ((physical_render_mode) && (settings.sceneColorIndex != 0u)) {
    Texture2D sceneTex = bindless_textures[NonUniformResourceIndex(settings.sceneColorIndex)];
    SamplerState sceneSampler = bindless_samplers[NonUniformResourceIndex(settings.sceneSamplerIndex)];
    float4 scene_center_sample = sceneTex.Sample(sceneSampler, screen_uv);
    float scene_depth = saturate(scene_center_sample.a);
    float water_depth = saturate(input.position.z);
    bool has_scene_hit = (scene_depth < 0.999999f) && ((scene_depth - water_depth) > 1.0e-6f);
    if (has_scene_hit) {
      float3 hit_world_center = reconstruct_world_position_from_uv_depth(screen_uv, scene_depth, settings.invViewProj);
      float3 hit_delta_center = hit_world_center - input.worldPos;
      float thickness_center_m = length(hit_delta_center);
      bool hit_below_surface_center = (dot(hit_delta_center, N) < 0.0f);
      if ((thickness_center_m > 0.0f) && hit_below_surface_center) {
        float refract_blend = saturate((thickness_center_m - 0.05f) / 0.75f);
        float2 refract_offset = N.xz * refract_distortion_scale * refract_blend;
        float2 sample_uv = saturate(screen_uv + refract_offset);
        float4 scene_refract_sample = sceneTex.Sample(sceneSampler, sample_uv);
        float refract_depth = saturate(scene_refract_sample.a);
        bool has_refract_hit = (refract_depth < 0.999999f) && ((refract_depth - water_depth) > 1.0e-6f);

        float3 scene_color = scene_center_sample.rgb;
        float thickness_m = thickness_center_m;
        float extra_wave_path_m = has_wave_thickness ? (wave_thickness_m * wave_thickness_path_scale) : 0.0f;
        if (has_refract_hit) {
          float3 hit_world_refract = reconstruct_world_position_from_uv_depth(sample_uv, refract_depth, settings.invViewProj);
          float3 hit_delta_refract = hit_world_refract - input.worldPos;
          float thickness_refract_m = length(hit_delta_refract);
          bool hit_below_surface_refract = (dot(hit_delta_refract, N) < 0.0f);
          if ((thickness_refract_m > 0.0f) && hit_below_surface_refract) {
            // Use distorted depth/hit for true refracted silhouette, but keep a soft transition near the waterline.
            float depth_consistency = 1.0f - saturate(abs(refract_depth - scene_depth) / 0.02f);
            float distortion_confidence = max(refract_blend, depth_consistency * 0.25f);
            scene_color = lerp(scene_center_sample.rgb, scene_refract_sample.rgb, distortion_confidence);
            thickness_m = lerp(thickness_center_m, thickness_refract_m, distortion_confidence);
          }
        }
        thickness_m += extra_wave_path_m;

        float3 beam_transmittance = beer_lambert(sigma_t, thickness_m);
        float3 scatter_source = max(scene_color, 0.0f);
        float3 inscatter_color = scatter_albedo * scatter_source * (1.0f - beam_transmittance);
        waterRefractColor = max((scene_color * beam_transmittance) + inscatter_color, 0.0f);
        has_scene_refraction = true;
      }
    }
  }

  if (settings.envmapIndex != 0) {
    Texture2D envTex = bindless_textures[NonUniformResourceIndex(settings.envmapIndex)];
    bool envmap_equal_area_mapping = (settings._padding0 > 0.5f);
    SamplerState envSampler = bindless_samplers[NonUniformResourceIndex(settings.envSamplerIndex)];
    float3 R = reflect(-V, N);
    float2 uv = sample_env_uv(R, envmap_equal_area_mapping);
    waterReflectColor = envTex.Sample(envSampler, uv).rgb * env_reflection_intensity;
    if (physical_render_mode) {
      float safe_ior = max(water_ior, 1.0f + 1.0e-4f);
      float eta = 1.0f / safe_ior;
      float3 Tdir = refract(-V, N, eta);
      float Tdir2 = dot(Tdir, Tdir);
      if (Tdir2 > 1.0e-8f) {
        float vertical_optical_depth_m = max(settings.waterOptics1.x, 0.0f);
        float cos_theta_t = max(transmitted_cos_theta(NdotV, water_ior), 1.0e-3f);
        float optical_path_m = vertical_optical_depth_m / cos_theta_t;
        if (has_wave_thickness) {
          optical_path_m += wave_thickness_m * wave_thickness_path_scale;
        }
        float3 beam_transmittance = beer_lambert(sigma_t, optical_path_m);
        float2 refract_uv = sample_env_uv(normalize(Tdir), envmap_equal_area_mapping);
        if (has_scene_refraction == false) {
          float3 transmitted_source = envTex.Sample(envSampler, refract_uv).rgb;
          float3 scatter_source = max(transmitted_source, 0.0f);
          float3 inscatter_color = scatter_albedo * scatter_source * (1.0f - beam_transmittance);
          waterRefractColor = max((transmitted_source * beam_transmittance) + inscatter_color, 0.0f);
        }
      }
    }
  }
  
  if (settings.sunDirectionEnable.w > 0.5f) {
    float3 L = settings.sunDirectionEnable.xyz;
    float L2 = dot(L, L);
    if (L2 > 1.0e-8f) {
      L *= rsqrt(L2);
      float NdotL = saturate(dot(N, L));
      if (NdotL > 0.0f) {
        float3 sun_radiance = max(settings.sunRadiance.xyz, 0.0f);
        float3 ggx_spec = eval_ggx_specular_dielectric(N, V, L, f0, ggx_alpha);
        direct_specular = sun_radiance * ggx_spec * NdotL;
      }
    }
  }

  float foam_specular_suppression = saturate(settings.foamControls1.w);
  float foam_specular_keep = 1.0f - (foam_surface_mask * foam_specular_suppression);
  waterReflectColor *= foam_specular_keep;
  direct_specular *= foam_specular_keep;

  float3 color = (waterRefractColor * (1.0f - fresnel)) + (waterReflectColor * fresnel) + direct_specular;
  if (foam_surface_mask > 0.0f) {
    float foam_diffuse_gain = max(settings.foamControls2.x, 0.0f);
    float foam_backlight_gain = max(settings.foamControls2.y, 0.0f);
    float3 foam_lighting = float3(0.15f, 0.18f, 0.20f);
    if (settings.envmapIndex != 0u) {
      Texture2D envTexFoam = bindless_textures[NonUniformResourceIndex(settings.envmapIndex)];
      SamplerState envSamplerFoam = bindless_samplers[NonUniformResourceIndex(settings.envSamplerIndex)];
      bool envmap_equal_area_mapping = (settings._padding0 > 0.5f);
      float2 foam_env_uv = sample_env_uv(N, envmap_equal_area_mapping);
      foam_lighting = envTexFoam.Sample(envSamplerFoam, foam_env_uv).rgb * max(settings.waterOptics0.y, 0.0f);
    }

    float3 foam_diffuse = foam_lighting * foam_diffuse_gain;
    float3 foam_backlight = float3(0.0f, 0.0f, 0.0f);
    if (settings.sunDirectionEnable.w > 0.5f) {
      float3 L_foam = settings.sunDirectionEnable.xyz;
      float L2_foam = dot(L_foam, L_foam);
      if (L2_foam > 1.0e-8f) {
        L_foam *= rsqrt(L2_foam);
        float NdotL_foam = saturate(dot(N, L_foam));
        float VdotNegL = saturate(dot(V, -L_foam));
        float3 sun_radiance_foam = max(settings.sunRadiance.xyz, 0.0f);
        foam_diffuse += sun_radiance_foam * (NdotL_foam * foam_diffuse_gain * (1.0f / 3.14159265f));
        float backlight_gate = saturate((1.0f - NdotL_foam) * (0.35f + (0.65f * foam_thickness_norm)));
        foam_backlight = sun_radiance_foam * (VdotNegL * backlight_gate * foam_backlight_gain);
      }
    }

    float bubble_density_tint = lerp(0.82f, 1.05f, bubble_fill);
    float bubble_rim_boost = lerp(0.95f, 1.35f, bubble_rim);
    float3 bubble_albedo = foam_albedo * bubble_density_tint;
    float3 foam_surface_color = ((bubble_albedo * foam_diffuse) * bubble_rim_boost) + (bubble_albedo * foam_backlight * (1.0f - fresnel));
    float foam_blend_mask = foam_surface_mask;
    color = lerp(color, foam_surface_color, foam_blend_mask);
  }

  uint mip_level = min((uint)input.mipLevel, 5u);
  float3 mip_color = get_mip_color(mip_level);
  float mip_mix = saturate(settings.mipColorMix * settings.mipColorEnable);
  color = lerp(color, mip_color, mip_mix);

  if (settings.debugView.x > 0.5f) {
    color = settings.wireframeColor.xyz;
  }

  return float4(color, 1.0);
}

float4 PSFoamHistory(VSOutput input) : SV_Target0 {
  ByteAddressBuffer settingsBuffer = bindless_buffers[NonUniformResourceIndex(pushConstants.settingsBufferIndex)];
  OceanRenderSettings settings = settingsBuffer.Load<OceanRenderSettings>(0);

  float cascade_lengths[3] = {settings.cascadeLengths.x, settings.cascadeLengths.y, settings.cascadeLengths.z};
  float cascade_weights[3] = {settings.cascadeWeights.x, settings.cascadeWeights.y, settings.cascadeWeights.z};
  float2 surface_xz_ddx = ddx(input.surfaceXZ);
  float2 surface_xz_ddy = ddy(input.surfaceXZ);
  float2 screen_uv = screen_uv_from_svpos(input.position, settings.screenSize);

  bool has_wave_thickness = false;
  float wave_thickness_m = sample_wave_thickness_meters(screen_uv, settings, has_wave_thickness);
  float3 slope_metrics = sample_filtered_slope_metrics(input.surfaceXZ, surface_xz_ddx, surface_xz_ddy, settings, cascade_lengths, cascade_weights);
  float current_seed = evaluate_foam_mask(wave_thickness_m, has_wave_thickness, slope_metrics.x, slope_metrics.y, slope_metrics.z, settings);

  float history = current_seed;
  return float4(history, 0.0f, 0.0f, 1.0f);
}

float4 PSThickness(VSOutput input) : SV_Target0 {
  float depth_m = length(pushConstants.cameraPos.xyz - input.worldPos);
  return float4(depth_m, 0.0f, 0.0f, 1.0f);
}
