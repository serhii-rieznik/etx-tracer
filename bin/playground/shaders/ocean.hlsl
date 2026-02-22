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
  float3 n = cross(tangent_z, tangent_x);
  float n2 = dot(n, n);
  if (n2 > 1.0e-12f) {
    n *= rsqrt(n2);
  } else {
    n = float3(0.0f, 1.0f, 0.0f);
  }
  return n;
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

  float3 n = cross(dPdv, dPdu);
  float n2 = dot(n, n);
  if (n2 > 1.0e-12f) {
    return n * rsqrt(n2);
  }

  return float3(0.0f, 1.0f, 0.0f);
}

float sample_filtered_slope_energy(
  float2 surface_xz,
  float2 surface_xz_ddx,
  float2 surface_xz_ddy,
  OceanRenderSettings settings,
  float cascade_lengths[3],
  float cascade_weights[3]) {
  SamplerState repeat_sampler = bindless_samplers[NonUniformResourceIndex(pushConstants.samplerIndex)];
  float accum = 0.0f;
  float weight_sum = 0.0f;
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
    float slope_energy = slopeTex.SampleGrad(repeat_sampler, uv, uv_ddx, uv_ddy).x;
    accum += max(slope_energy, 0.0f) * weight;
    weight_sum += weight;
  }
  if (weight_sum > 1.0e-6f) {
    return accum / weight_sum;
  }
  return 0.0f;
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

float2 sample_equirect_uv(float3 dir) {
  float inv_pi = 1.0f / 3.14159265f;
  float inv_two_pi = 0.5f * inv_pi;
  float u = (atan2(dir.z, dir.x) * inv_two_pi) + 0.5f;
  float v = acos(clamp(dir.y, -1.0f, 1.0f)) * inv_pi;
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
  float3 sigma_a = max(settings.waterAbsorption.xyz, 0.0f);
  float3 sigma_s = max(settings.waterScattering.xyz, 0.0f);
  float3 sigma_t = sigma_a + sigma_s;
  float3 scatter_albedo = sigma_s / max(sigma_t, 1.0e-4f);
  float base_unresolved_roughness = min(max(settings.waterOptics1.y, 0.0f), 1.0f);
  float specular_aa_strength = max(settings.waterOptics1.z, 0.0f);
  float slope_energy = sample_filtered_slope_energy(input.surfaceXZ, surface_xz_ddx, surface_xz_ddy, settings, cascade_lengths, cascade_weights);
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
    float vertical_optical_depth_m = max(settings.waterOptics1.x, 0.0f);
    float cos_theta_t = transmitted_cos_theta(NdotV, water_ior);
    float optical_path_m = vertical_optical_depth_m / max(cos_theta_t, 1.0e-3f);
    float3 beam_transmittance = beer_lambert(sigma_t, optical_path_m);
    float3 inscatter_color = scatter_albedo * (1.0f - beam_transmittance);
    waterRefractColor = saturate(inscatter_color);
  }
  float3 waterReflectColor = float3(0.5f, 0.6f, 0.8f) * env_reflection_intensity;
  float3 direct_specular = float3(0.0f, 0.0f, 0.0f);

  if ((physical_render_mode) && (settings.sceneColorIndex != 0u)) {
    Texture2D sceneTex = bindless_textures[NonUniformResourceIndex(settings.sceneColorIndex)];
    SamplerState sceneSampler = bindless_samplers[NonUniformResourceIndex(settings.sceneSamplerIndex)];
    float2 screen_uv = screen_uv_from_svpos(input.position, settings.screenSize);
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

        float3 beam_transmittance = beer_lambert(sigma_t, thickness_m);
        waterRefractColor = saturate((scene_color * beam_transmittance) + (scatter_albedo * (1.0f - beam_transmittance)));
        has_scene_refraction = true;
      }
    }
  }

  if (settings.envmapIndex != 0) {
    Texture2D envTex = bindless_textures[NonUniformResourceIndex(settings.envmapIndex)];
    SamplerState envSampler = bindless_samplers[NonUniformResourceIndex(settings.envSamplerIndex)];
    float3 R = reflect(-V, N);
    float2 uv = sample_equirect_uv(R);
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
        float3 beam_transmittance = beer_lambert(sigma_t, optical_path_m);
        float2 refract_uv = sample_equirect_uv(normalize(Tdir));
        if (has_scene_refraction == false) {
          float3 transmitted_source = envTex.Sample(envSampler, refract_uv).rgb;
          waterRefractColor = saturate((transmitted_source * beam_transmittance) + (scatter_albedo * (1.0f - beam_transmittance)));
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
  
  float3 color = (waterRefractColor * (1.0f - fresnel)) + (waterReflectColor * fresnel) + direct_specular;
  uint mip_level = min((uint)input.mipLevel, 5u);
  float3 mip_color = get_mip_color(mip_level);
  float mip_mix = saturate(settings.mipColorMix * settings.mipColorEnable);
  color = lerp(color, mip_color, mip_mix);

  if (settings.debugView.x > 0.5f) {
    color = settings.wireframeColor.xyz;
  }

  return float4(color, 1.0);
}
