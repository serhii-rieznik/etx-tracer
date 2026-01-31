struct GPUConstants {
  uint as_index;
  uint output_image_index;
  uint2 pad;
};

[[vk::push_constant]]
GPUConstants params;

RaytracingAccelerationStructure GlobalAS[] : register(t4, space0);
RWTexture2D<float4> GlobalStorageImages[] : register(u3, space0);

[numthreads(16, 16, 1)]
void compute_main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
  RWTexture2D<float4> output_image = GlobalStorageImages[params.output_image_index];
  uint2 size;
  output_image.GetDimensions(size.x, size.y);
  if (any(dispatch_thread_id.xy >= size)) return;

  float2 uv = (float2(dispatch_thread_id.xy) + 0.5f) / float2(size);
  
  // Basic ray generation (placeholder)
  float3 origin = float3(0, 0, -5);
  float3 direction = normalize(float3(uv * 2.0f - 1.0f, 1.0f));

  RayDesc ray;
  ray.Origin = origin;
  ray.Direction = direction;
  ray.TMin = 0.001f;
  ray.TMax = 1000.0f;

  RaytracingAccelerationStructure scene = GlobalAS[params.as_index];
  RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
  query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, ray);
  query.Proceed();

  float3 color = float3(0, 0, 0);
  if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
    float2 barycentrics = query.CommittedTriangleBarycentrics();
    color = float3(barycentrics, 1.0f - barycentrics.x - barycentrics.y);
  }

  output_image[dispatch_thread_id.xy] = float4(color, 1.0f);
}
