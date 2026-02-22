#include <interop/geometry.hxx>
#include <shaders/bindless.hlsl>

// Bindless resource passing through push constants
struct PushConstants {
  uint vertexBufferIndex;
  uint testTextureIndex;
  uint testSamplerIndex;
  uint oceanSamplerIndex;
  uint4 dispMapIndex01;
  float4 cascadeLenWeight01;
  float4 buoyCenterDraft;
  column_major float4x4 vp;
};

[[vk::push_constant]]
PushConstants pushConstants;

struct VSOutput {
  float4 position : SV_Position;
  float2 uv : TEXCOORD;
};

float3 sample_displacement_field(float2 world_xz) {
  float3 disp = float3(0.0f, 0.0f, 0.0f);
  SamplerState ocean_sampler = bindless_samplers[NonUniformResourceIndex(pushConstants.oceanSamplerIndex)];
  float cascade_lengths[2] = {pushConstants.cascadeLenWeight01.x, pushConstants.cascadeLenWeight01.y};
  float cascade_weights[2] = {pushConstants.cascadeLenWeight01.z, pushConstants.cascadeLenWeight01.w};
  uint cascade_indices[2] = {pushConstants.dispMapIndex01.x, pushConstants.dispMapIndex01.y};
  for (int i = 0; i < 2; ++i) {
    if ((cascade_indices[i] != 0u) && (cascade_weights[i] > 0.0f)) {
      Texture2D disp_tex = bindless_textures[NonUniformResourceIndex(cascade_indices[i])];
      float2 uv = world_xz / max(cascade_lengths[i], 1.0e-3f);
      disp += disp_tex.SampleLevel(ocean_sampler, uv, 0.0f).xyz * cascade_weights[i];
    }
  }
  return disp;
}

void compute_buoy_frame(out float3 out_center, out float3 out_x_axis, out float3 out_y_axis, out float3 out_z_axis) {
  float2 center_xz = pushConstants.buoyCenterDraft.xy;
  float draft = pushConstants.buoyCenterDraft.z;
  float sample_radius_scale = max(pushConstants.buoyCenterDraft.w, 0.1f);
  float3 buoy_half_extents = float3(0.55f, 0.33f, 0.95f);
  float2 footprint_half = max(buoy_half_extents.xz, 0.05f) * sample_radius_scale;

  float2 offset_a = float2(-footprint_half.x, -footprint_half.y);
  float2 offset_b = float2(footprint_half.x, -footprint_half.y);
  float2 offset_c = float2(-footprint_half.x, footprint_half.y);
  float2 offset_d = float2(footprint_half.x, footprint_half.y);

  float3 p_a = float3(center_xz.x + offset_a.x, 0.0f, center_xz.y + offset_a.y) + sample_displacement_field(center_xz + offset_a);
  float3 p_b = float3(center_xz.x + offset_b.x, 0.0f, center_xz.y + offset_b.y) + sample_displacement_field(center_xz + offset_b);
  float3 p_c = float3(center_xz.x + offset_c.x, 0.0f, center_xz.y + offset_c.y) + sample_displacement_field(center_xz + offset_c);
  float3 p_d = float3(center_xz.x + offset_d.x, 0.0f, center_xz.y + offset_d.y) + sample_displacement_field(center_xz + offset_d);

  float3 avg_surface = (p_a + p_b + p_c + p_d) * 0.25f;
  float3 x_dir = ((p_b + p_d) - (p_a + p_c)) * 0.5f;
  float3 z_dir = ((p_c + p_d) - (p_a + p_b)) * 0.5f;

  float x_len2 = dot(x_dir, x_dir);
  if (x_len2 > 1.0e-12f) {
    x_dir *= rsqrt(x_len2);
  } else {
    x_dir = float3(1.0f, 0.0f, 0.0f);
  }

  float3 y_dir = cross(z_dir, x_dir);
  float y_len2 = dot(y_dir, y_dir);
  if (y_len2 > 1.0e-12f) {
    y_dir *= rsqrt(y_len2);
  } else {
    y_dir = float3(0.0f, 1.0f, 0.0f);
  }

  float3 z_ortho = cross(x_dir, y_dir);
  float z_len2 = dot(z_ortho, z_ortho);
  if (z_len2 > 1.0e-12f) {
    z_ortho *= rsqrt(z_len2);
  } else {
    z_ortho = float3(0.0f, 0.0f, 1.0f);
  }

  out_center = avg_surface - (y_dir * draft);
  out_x_axis = x_dir;
  out_y_axis = y_dir;
  out_z_axis = z_ortho;
}

VSOutput VSMain(uint vertexId : SV_VertexID) {
  VSOutput output;
  
  // Access data through bindless buffer array
  ByteAddressBuffer vb = bindless_buffers[NonUniformResourceIndex(pushConstants.vertexBufferIndex)];
  
  // Load using shared ABI 
  Vertex v = vb.Load<Vertex>(vertexId * 56);

  float3 center = float3(0.0f, 0.0f, 0.0f);
  float3 x_axis = float3(1.0f, 0.0f, 0.0f);
  float3 y_axis = float3(0.0f, 1.0f, 0.0f);
  float3 z_axis = float3(0.0f, 0.0f, 1.0f);
  compute_buoy_frame(center, x_axis, y_axis, z_axis);

  float3 buoy_half_extents = float3(0.55f, 0.33f, 0.95f);
  float3 local_pos = v.pos * (buoy_half_extents * 2.0f);
  float3 world_pos = center + (x_axis * local_pos.x) + (y_axis * local_pos.y) + (z_axis * local_pos.z);

  output.position = mul(pushConstants.vp, float4(world_pos, 1.0f));
  output.uv = v.tex;
  
  return output;
}

float4 PSMain(VSOutput input) : SV_Target0 {
  float4 color = float4(0.55f, 0.55f, 0.55f, 1.0f);
  color.a = saturate(input.position.z);
  return color;
}
