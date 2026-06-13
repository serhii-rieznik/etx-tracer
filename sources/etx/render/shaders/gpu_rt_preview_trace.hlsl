#include <interop/gpu_rt_shared.hxx>

[[vk::push_constant]] GPURTConstants constants;

#include "gpu_rt_shared.hlsl"

uint2 preview_render_window_size() {
  return uint2(constants.render_window_width, constants.render_window_height);
}

bool preview_render_window_contains(uint2 local_pixel) {
  return all(local_pixel < preview_render_window_size());
}

uint2 preview_output_pixel(uint2 local_pixel) {
  return uint2(constants.render_window_origin_x, constants.render_window_origin_y) + local_pixel;
}

float3 preview_miss_color(float3 direction) {
  float t = saturate(direction.y * 0.5f + 0.5f);
  float3 ground = float3(0.36f, 0.38f, 0.36f);
  float3 sky = float3(0.52f, 0.66f, 0.88f);
  return lerp(ground, sky, t);
}

float3 preview_hit_color(float3 normal, float hit_t, float3 miss_color, Camera camera) {
  float3 light_dir = normalize(float3(0.35f, 0.75f, 0.55f));
  float diffuse = saturate(dot(normal, light_dir) * 0.5f + 0.5f);
  float3 normal_tint = abs(normal) * 0.45f + float3(0.30f, 0.34f, 0.38f);
  float3 color = normal_tint * (0.45f + 0.55f * diffuse);
  float depth_range = (camera.clip_far > 0.0f) ? max(1.0f, camera.clip_far * 0.12f) : 128.0f;
  float fog = saturate(hit_t / depth_range) * 0.35f;
  return lerp(color, miss_color, fog);
}

[numthreads(8, 8, 1)] void gpu_preview_trace_main(uint3 dtid : SV_DispatchThreadID) {
  if ((constants.camera_buffer_index == kInvalidIndex) || (constants.output_image_index == kInvalidIndex) || (constants.as_index == kInvalidIndex)) {
    return;
  }
  if ((constants.scene.triangles == kInvalidIndex) || (constants.scene.scene_globals == kInvalidIndex)) {
    return;
  }
  if (preview_render_window_contains(dtid.xy) == false) {
    return;
  }

  Camera camera = load_camera(bindless_buffers[NonUniformResourceIndex(constants.camera_buffer_index)]);
  uint2 output_pixel = preview_output_pixel(dtid.xy);
  float2 uv = camera_sample_film_uv(output_pixel, camera.film_size, float2(0.5f, 0.5f));
  Ray camera_ray = camera_generate_primary_ray(camera, uv, float2(0.5f, 0.5f));

  RayDesc ray = (RayDesc)0;
  ray.Origin = camera_ray.o;
  ray.Direction = camera_ray.d;
  ray.TMin = max(kRayEpsilon, camera_ray.min_t);
  ray.TMax = max(ray.TMin + kRayEpsilon, camera_ray.max_t);

  float3 miss_color = preview_miss_color(ray.Direction);
  float3 color = miss_color;

  RayQuery<RAY_FLAG_FORCE_OPAQUE> ray_query;
  ray_query.TraceRayInline(bindless_accel_structs[NonUniformResourceIndex(constants.as_index)], RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
  while (ray_query.Proceed()) {
  }

  if (ray_query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
    SceneGPUSharedGlobals scene_globals_data = scene_gpu_load_globals(bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)]);
    uint triangle_index = ray_query.CommittedPrimitiveIndex();
    if (triangle_index < scene_globals_data.triangle_count) {
      TriangleData tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], triangle_index);
      float3 normal = normalize(tri.geo_n);
      if (dot(normal, ray.Direction) > 0.0f) {
        normal = -normal;
      }
      color = preview_hit_color(normal, ray_query.CommittedRayT(), miss_color, camera);
    }
  }

  bindless_storage_textures[NonUniformResourceIndex(constants.output_image_index)][output_pixel] = float4(color, 1.0f);
}
