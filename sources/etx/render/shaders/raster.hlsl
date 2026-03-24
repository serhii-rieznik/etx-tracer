struct VSInput {
  float3 pos : POSITION;
  float3 nrm : NORMAL;
};

struct VSOutput {
  float4 pos      : SV_Position;
  float3 nrm      : NORMAL;
  float3 view_dir : TEXCOORD0;
};

struct RasterConstants {
  float4x4 view_proj;
  float3 camera_pos;
  uint pad;
};

[[vk::push_constant]] RasterConstants params;

VSOutput vertex_main(VSInput input) {
  VSOutput output;
  output.pos = mul(params.view_proj, float4(input.pos, 1.0f));
  output.nrm = input.nrm;
  output.view_dir = params.camera_pos - input.pos;
  return output;
}

float4 fragment_main(VSOutput input) : SV_Target0 {
  float3 n = normalize(input.nrm);
  float3 v = normalize(input.view_dir);
  float ndotv = saturate(dot(n, v));
  return float4(ndotv.xxx, 1.0f);
}
