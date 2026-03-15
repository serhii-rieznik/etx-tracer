#pragma once

#include "camera_shared.hxx"

struct ETX_ALIGNED CameraFilmSampleShared {
  float3 position ETX_INIT({});
  float3 normal ETX_INIT({});
  float3 direction ETX_INIT({});
  float2 uv ETX_INIT({});
  float weight ETX_INIT({});
  float pdf_dir ETX_INIT({});
  float pdf_area ETX_INIT({});
  float pdf_dir_out ETX_INIT({});
};

struct ETX_ALIGNED CameraFilmEvalShared {
  float3 normal ETX_INIT({});
  float pdf_dir ETX_INIT({});
};

ETX_SHARED_INLINE float4 camera_film_shared_project(ETX_IN(float4x4, matrix), ETX_IN(float4, projected_point)) {
#if defined(__cplusplus)
  return float4(
    matrix.col[0].x * projected_point.x + matrix.col[1].x * projected_point.y + matrix.col[2].x * projected_point.z + matrix.col[3].x * projected_point.w,
    matrix.col[0].y * projected_point.x + matrix.col[1].y * projected_point.y + matrix.col[2].y * projected_point.z + matrix.col[3].y * projected_point.w,
    matrix.col[0].z * projected_point.x + matrix.col[1].z * projected_point.y + matrix.col[2].z * projected_point.z + matrix.col[3].z * projected_point.w,
    matrix.col[0].w * projected_point.x + matrix.col[1].w * projected_point.y + matrix.col[2].w * projected_point.z + matrix.col[3].w * projected_point.w);
#else
  return mul(matrix, projected_point);
#endif
}

ETX_SHARED_INLINE float3 camera_film_shared_lens_point(ETX_IN(Camera, camera), ETX_IN(float2, sensor_sample)) {
  return camera.position + sensor_sample.x * camera.side + sensor_sample.y * camera.up;
}

ETX_SHARED_INLINE CameraFilmSampleShared camera_film_shared_evaluate(ETX_IN(Camera, camera), ETX_IN(float3, world_point), ETX_IN(float3, lens_point)) {
  ETX_ZERO_INIT(CameraFilmSampleShared, result);
  if (camera.cls == Camera::Class::Equirectangular) {
    result.position = camera.position;
    result.normal = camera.direction;
    result.direction = result.position - world_point;

    float distance_squared = dot(result.direction, result.direction);
    if (distance_squared <= kEpsilon) {
      return result;
    }

    float inv_distance = 1.0f / sqrt(distance_squared);
    result.direction *= inv_distance;

    float3 out_direction = -result.direction;
    float2 uv = direction_to_uv(out_direction, float2(0.0f, 0.0f), 1.0f, Projection::Equirectangular);
    result.uv = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
    result.pdf_area = 1.0f;
    result.pdf_dir = distance_squared;
    result.pdf_dir_out = projection_environment_image_pdf_to_solid_angle(1.0f, uv, Projection::Equirectangular);
    result.weight = result.pdf_dir_out / result.pdf_dir;
    return result;
  }

  result.position = lens_point;
  result.direction = result.position - world_point;
  result.normal = camera.direction;

  float cos_t = -dot(result.direction, result.normal);
  if (cos_t < 0.0f) {
    ETX_ZERO_INIT(CameraFilmSampleShared, zero_result);
    return zero_result;
  }

  float distance_squared = dot(result.direction, result.direction);
  if (distance_squared <= kEpsilon) {
    ETX_ZERO_INIT(CameraFilmSampleShared, zero_result);
    return zero_result;
  }

  float distance = sqrt(distance_squared);
  result.direction /= distance;
  cos_t /= distance;

  float focal_plane_distance = ((camera.lens_radius > kEpsilon) && (camera.focal_distance > kEpsilon)) ? camera.focal_distance : 1.0f;
  float3 focus_point = result.position - result.direction * (focal_plane_distance / cos_t);
  float4 projected = camera_film_shared_project(camera.view_proj, float4(focus_point.x, focus_point.y, focus_point.z, 1.0f));
  result.uv = float2(projected.x / projected.w, projected.y / projected.w);
  if ((projected.w <= 0.0f) || (result.uv.x < -1.0f) || (result.uv.y < -1.0f) || (result.uv.x > 1.0f) || (result.uv.y > 1.0f)) {
    ETX_ZERO_INIT(CameraFilmSampleShared, zero_result);
    return zero_result;
  }

  float lens_area = (camera.lens_radius > kEpsilon) ? kPi * camera.lens_radius * camera.lens_radius : 1.0f;
  result.pdf_area = 1.0f / lens_area;
  result.pdf_dir = result.pdf_area * distance_squared / cos_t;
  result.pdf_dir_out = 1.0f / (camera.area * lens_area * cos_t * cos_t * cos_t);

  float importance = result.pdf_dir_out / cos_t;
  result.weight = importance / result.pdf_dir;
  return result;
}

ETX_SHARED_INLINE CameraFilmEvalShared camera_film_shared_evaluate_out(ETX_IN(Camera, camera), ETX_IN(Ray, out_ray)) {
  ETX_ZERO_INIT(CameraFilmEvalShared, result);
  result.normal = camera.direction;
  if (camera.cls == Camera::Class::Equirectangular) {
    float2 uv = direction_to_uv(normalize(out_ray.d), float2(0.0f, 0.0f), 1.0f, Projection::Equirectangular);
    result.pdf_dir = projection_environment_image_pdf_to_solid_angle(1.0f, uv, Projection::Equirectangular);
    return result;
  }

  float cos_t = dot(out_ray.d, camera.direction);
  result.pdf_dir = 1.0f / (camera.area * cos_t * cos_t * cos_t);
  return result;
}
