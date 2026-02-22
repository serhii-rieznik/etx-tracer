#pragma once

#include "camera_shared.hxx"

#ifndef ETX_CAMERA_PRIMARY_RAY_SHARED_CONTEXT_TYPE
# error "ETX_CAMERA_PRIMARY_RAY_SHARED_CONTEXT_TYPE must be defined before including camera_primary_ray_shared.hxx"
#endif

ETX_SHARED_INLINE float2 camera_primary_ray_shared_primary_uv(ETX_IN(uint2, pixel), ETX_IN(uint2, film_size)) {
  return camera_shared_flip_y(camera_shared_center_uv(pixel, film_size));
}

ETX_SHARED_INLINE float2 camera_primary_ray_shared_sensor_sample(ETX_IN(ETX_CAMERA_PRIMARY_RAY_SHARED_CONTEXT_TYPE, context), ETX_IN(Camera, camera),
  ETX_IN(float2, sensor_sample_rnd)) {
  return camera_lens_sample_shared(context, camera.lens_radius, camera.focal_distance, camera.lens_image, sensor_sample_rnd);
}

ETX_SHARED_INLINE float3 camera_primary_ray_shared_lens_point(ETX_IN(ETX_CAMERA_PRIMARY_RAY_SHARED_CONTEXT_TYPE, context), ETX_IN(Camera, camera),
  ETX_IN(float2, sensor_sample_rnd)) {
  float2 sensor_sample = camera_primary_ray_shared_sensor_sample(context, camera, sensor_sample_rnd);
  sensor_sample *= camera.lens_radius;
  return camera_film_shared_lens_point(camera, sensor_sample);
}

ETX_SHARED_INLINE Ray camera_primary_ray_shared_generate(ETX_IN(ETX_CAMERA_PRIMARY_RAY_SHARED_CONTEXT_TYPE, context), ETX_IN(Camera, camera), ETX_IN(float2, uv),
  ETX_IN(float2, sensor_sample_rnd)) {
  float2 sensor_sample = camera_primary_ray_shared_sensor_sample(context, camera, sensor_sample_rnd);
  return camera_generate_ray(camera, uv, sensor_sample);
}
