#include <interop/geometry.hxx>
#include <shaders/bindless.hlsl>

struct OceanPushConstants {
  uint vbIndex;
  uint samplerIndex;
  uint instanceBufferIndex;
  uint settingsBufferIndex;
  uint dispMapIndex[4];
  uint normalMapIndex[4];
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
  float stitchTransitionCells;
  float mipColorMix;
  float mipColorEnable;
  float4 cascadeLengths;
  float4 cascadeWeights;
  float4 clipmapCenterOffset;
  float4 debugView;
  float4 normalControls;
  float4 wireframeColor;
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
  bool normal_map_shading_enable = (settings.normalControls.x > 0.5f);
  float normal_map_scale = saturate(settings.normalControls.y);
  uint normal_debug_mode = min((uint)round(max(settings.normalControls.z, 0.0f)), 4u);

  float3 geom_n = input.geomNormal;
  float geom_n2 = dot(geom_n, geom_n);
  if (geom_n2 > 1e-12f) {
    geom_n *= rsqrt(geom_n2);
  } else {
    geom_n = float3(0.0f, 1.0f, 0.0f);
  }

  float2 accumulated_slope = float2(0.0f, 0.0f);
  bool has_slope = false;
  float2 surface_xz_ddx = ddx(input.surfaceXZ);
  float2 surface_xz_ddy = ddy(input.surfaceXZ);

  for(int i = 0; i < 3; ++i) {
    if ((pushConstants.normalMapIndex[i] != 0) && (cascade_weights[i] > 0.0f)) {
      Texture2D normTex = bindless_textures[NonUniformResourceIndex(pushConstants.normalMapIndex[i])];
      SamplerState envSampler = bindless_samplers[NonUniformResourceIndex(pushConstants.samplerIndex)];
      float2 uv = input.surfaceXZ / cascade_lengths[i];
      float2 uv_ddx = surface_xz_ddx / cascade_lengths[i];
      float2 uv_ddy = surface_xz_ddy / cascade_lengths[i];
      float3 sampledN = normTex.SampleGrad(envSampler, uv, uv_ddx, uv_ddy).xyz;
      float sampled_n2 = dot(sampledN, sampledN);
      if (sampled_n2 > 1e-8f) {
        sampledN *= rsqrt(sampled_n2);
        if (sampledN.y > 1.0e-4f) {
          float2 sampled_slope = -sampledN.xz / sampledN.y;
          accumulated_slope += sampled_slope * cascade_weights[i];
          has_slope = true;
        }
      }
    }
  }

  float3 mapped_n = geom_n;
  if (has_slope) {
    mapped_n = normalize(float3(-accumulated_slope.x, 1.0f, -accumulated_slope.y));
  }

  if (normal_debug_mode > 0u) {
    float3 debug_normal = mapped_n;
    if ((normal_debug_mode >= 2u) && (normal_debug_mode <= 4u)) {
      uint cascade_index = normal_debug_mode - 2u;
      if (pushConstants.normalMapIndex[cascade_index] != 0u) {
        Texture2D normTex = bindless_textures[NonUniformResourceIndex(pushConstants.normalMapIndex[cascade_index])];
        SamplerState envSampler = bindless_samplers[NonUniformResourceIndex(pushConstants.samplerIndex)];
        float2 uv = input.surfaceXZ / cascade_lengths[cascade_index];
        float2 uv_ddx = surface_xz_ddx / cascade_lengths[cascade_index];
        float2 uv_ddy = surface_xz_ddy / cascade_lengths[cascade_index];
        float3 sampledN = normTex.SampleGrad(envSampler, uv, uv_ddx, uv_ddy).xyz;
        float sampled_n2 = dot(sampledN, sampledN);
        if (sampled_n2 > 1e-8f) {
          debug_normal = sampledN * rsqrt(sampled_n2);
        } else {
          debug_normal = float3(0.0f, 1.0f, 0.0f);
        }
      }
    }
    float3 normal_vis = (normalize(debug_normal) * 0.5f) + 0.5f;
    return float4(normal_vis, 1.0f);
  }

  if (normal_map_shading_enable == false) {
    N = geom_n;
  } else {
    N = normalize(geom_n + ((mapped_n - geom_n) * normal_map_scale));
  }

  float3 V = normalize(pushConstants.cameraPos.xyz - input.worldPos);
  
  float NdotV = saturate(dot(N, V));
  float fresnel = 0.02 + 0.98 * pow(1.0 - NdotV, 5.0);
  
  float3 waterRefractColor = float3(0.01, 0.05, 0.1);
  float3 waterReflectColor = float3(0.5, 0.6, 0.8);

  if (settings.envmapIndex != 0) {
    Texture2D envTex = bindless_textures[NonUniformResourceIndex(settings.envmapIndex)];
    SamplerState envSampler = bindless_samplers[NonUniformResourceIndex(pushConstants.samplerIndex)];
    float3 R = reflect(-V, N);
    
    // Sample spherical environment map
    float2 uv = float2(atan2(R.z, R.x) / (2.0 * 3.14159265) + 0.5, acos(R.y) / 3.14159265);
    waterReflectColor = envTex.Sample(envSampler, uv).rgb;
  }
  
  float3 color = lerp(waterRefractColor, waterReflectColor, fresnel);
  uint mip_level = min((uint)input.mipLevel, 5u);
  float3 mip_color = get_mip_color(mip_level);
  float mip_mix = saturate(settings.mipColorMix * settings.mipColorEnable);
  color = lerp(color, mip_color, mip_mix);

  if (settings.debugView.x > 0.5f) {
    color = settings.wireframeColor.xyz;
  }

  return float4(color, 1.0);
}
