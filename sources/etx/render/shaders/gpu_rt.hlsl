#include "bindless.hlsl"
#include <interop/render_options.hxx>
#include <interop/gpu_rt_shared.hxx>

[[vk::push_constant]] GPURTConstants constants;

[numthreads(8, 8, 1)] void compute_main(uint3 dtid : SV_DispatchThreadID) {
  if (any(dtid.xy >= constants.camera.film_size))
    return;

  float2 pixel = float2(dtid.xy) + 0.5f;
  float2 uv = pixel / float2(constants.camera.film_size);
  float2 ndc = uv * 2.0f - 1.0f;
  ndc.y = -ndc.y;

  // Generate ray from camera
  // Match etx::generate_ray logic from scene_camera.hxx
  float3 s = ndc.x * constants.camera.side;
  float3 u = ndc.y * constants.camera.up / constants.camera.aspect;
  float3 ray_dir = normalize(constants.camera.tan_half_fov * (s + u) + constants.camera.direction);

  RayDesc ray;
  ray.Origin = constants.camera.position;
  ray.Direction = ray_dir;
  ray.TMin = 0.001f;
  ray.TMax = 10000.0f;

  RayQuery<RAY_FLAG_NONE> q;
  RaytracingAccelerationStructure as = bindless_accel_structs[NonUniformResourceIndex(constants.as_index)];

  q.TraceRayInline(as, RAY_FLAG_NONE, 0xFF, ray);
  q.Proceed();

  float4 color = float4(0.0f, 0.0f, 0.0f, 1.0f);

  if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
    float2 bary = q.CommittedTriangleBarycentrics();
    color = float4(bary.x, bary.y, 1.0f - bary.x - bary.y, 1.0f);
  } else {
    // Gradient sky
    float t = 0.5f * (ray_dir.y + 1.0f);
    color = lerp(float4(1.0f, 1.0f, 1.0f, 1.0f), float4(0.5f, 0.7f, 1.0f, 1.0f), t);
  }

  bindless_storage_textures[constants.output_image_index][dtid.xy] = color;
}
