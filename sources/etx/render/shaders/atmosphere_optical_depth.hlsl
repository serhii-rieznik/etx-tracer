#include "bindless.hlsl"

#include <interop/atmosphere_scattering_shared.hxx>

struct AtmosphereOpticalDepthPushConstants {
  uint output_texture_index;
  uint width;
  uint height;
  uint pad0;
};

[[vk::push_constant]] AtmosphereOpticalDepthPushConstants constants;

[numthreads(8, 8, 1)] void optical_depth_main(uint3 dtid : SV_DispatchThreadID) {
  if ((dtid.x >= constants.width) || (dtid.y >= constants.height)) {
    return;
  }

  float2 uv = float2(float(dtid.x) / float(constants.width), float(dtid.y) / float(constants.height));
  float2 params = scattering_uv_to_precomputed_params(uv);
  float dir_x = sqrt(max(0.0f, 1.0f - (params.x * params.x)));
  float3 direction = float3(dir_x, params.x, 0.0f);
  float3 origin = float3(0.0f, kScatteringPlanetRadius + params.y, 0.0f);
  float total_distance = scattering_distance_to_sphere(origin, direction, float3(0.0f, 0.0f, 0.0f), kScatteringOuterSphereRadius);
  float3 value = scattering_optical_length_direct(origin, direction, total_distance);

  bindless_storage_textures[NonUniformResourceIndex(constants.output_texture_index)][int2(dtid.xy)] = float4(value, 0.0f);
}
