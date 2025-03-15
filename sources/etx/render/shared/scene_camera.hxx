#pragma once

namespace etx {

ETX_GPU_CODE float2 get_center_uv(const uint2& pixel, const uint2& dim) {
  return {
    (float(pixel.x) + 0.5f) / float(dim.x) * 2.0f - 1.0f,
    (float(pixel.y) + 0.5f) / float(dim.y) * 2.0f - 1.0f,
  };
}

ETX_GPU_CODE float2 get_jittered_uv(Sampler& smp, const uint2& pixel, const uint2& dim) {
  float sample_radius = 0.5f;
  return {
    (float(pixel.x) + 0.5f + sample_radius * (smp.next() * 2.0f - 1.0f)) / float(dim.x) * 2.0f - 1.0f,
    (float(pixel.y) + 0.5f + sample_radius * (smp.next() * 2.0f - 1.0f)) / float(dim.y) * 2.0f - 1.0f,
  };
}

ETX_GPU_CODE float film_pdf_out(const Camera& camera, const float3& to_point) {
  auto w_i = normalize(to_point - camera.position);
  float cos_t = dot(w_i, camera.direction);
  return 1.0f / fabsf(camera.area * cos_t * cos_t * cos_t);
}

ETX_GPU_CODE Ray generate_ray(const Scene& scene, const Camera& camera, const float2& uv, const float2& sensor_sample_rnd) {
  ETX_CHECK_FINITE(uv);

  if (camera.cls == Camera::Class::Equirectangular) {
    return {camera.position, from_spherical(uv.x * kPi, uv.y * kHalfPi), kRayEpsilon, kMaxFloat};
  }

  float3 origin = camera.position;
  float3 direction = camera.direction;
  ETX_CHECK_FINITE(direction);
  float3 s = (uv.x * camera.aspect) * camera.side;
  ETX_CHECK_FINITE(s);
  float3 u = uv.y * camera.up;
  ETX_CHECK_FINITE(u);
  float3 w_o = normalize(camera.tan_half_fov * (s + u) + direction);
  ETX_CHECK_FINITE(w_o);

  if ((camera.lens_radius > kEpsilon) && (camera.focal_distance > kEpsilon)) {
    float2 sensor_sample = {};
    if (camera.lens_image == kInvalidIndex) {
      sensor_sample = sample_disk(sensor_sample_rnd);
    } else {
      sensor_sample = scene.images[camera.lens_image].sample(sensor_sample_rnd) * 2.0f - 1.0f;
    }
    sensor_sample *= camera.lens_radius;
    origin = origin + camera.side * sensor_sample.x + camera.up * sensor_sample.y;
    float focal_plane_distance = camera.focal_distance / dot(w_o, direction);
    float3 p = camera.position + focal_plane_distance * w_o;
    w_o = normalize(p - origin);
    ETX_CHECK_FINITE(w_o);
  }

  return {origin, w_o, kRayEpsilon, kMaxFloat};
}

ETX_GPU_CODE CameraSample sample_film(Sampler& smp, const Scene& scene, const Camera& camera, const float3& from_point) {
  if (camera.cls == Camera::Class::Equirectangular) {
    // TODO : implelemt for Equirectangular camera
    return {};
  }

  float2 sensor_sample = {};
  if ((camera.lens_radius > kEpsilon) && (camera.focal_distance > kEpsilon)) {
    if (camera.lens_image == kInvalidIndex) {
      sensor_sample = sample_disk(smp.next_2d());
    } else {
      float pdf = {};
      uint2 location = {};
      float4 value = {};
      sensor_sample = scene.images[camera.lens_image].sample(smp.next_2d(), pdf, location, value);
      sensor_sample = sensor_sample * 2.0f - 1.0f;
    }
    sensor_sample *= camera.lens_radius;
  }

  CameraSample result;
  result.position = camera.position + sensor_sample.x * camera.side + sensor_sample.y * camera.up;
  result.direction = result.position - from_point;
  result.normal = camera.direction;

  float cos_t = -dot(result.direction, result.normal);
  if (cos_t < 0.0f) {
    return {};
  }

  float distance_squared = dot(result.direction, result.direction);
  float distance = sqrtf(distance_squared);
  result.direction /= distance;
  cos_t /= distance;

  float focal_plane_distance = ((camera.lens_radius > kEpsilon) && (camera.focal_distance > kEpsilon)) ? camera.focal_distance : 1.0f;
  float3 focus_point = result.position - result.direction * (focal_plane_distance / cos_t);

  auto projected = camera.view_proj * float4{focus_point.x, focus_point.y, focus_point.z, 1.0f};
  result.uv = {projected.x / projected.w, projected.y / projected.w};
  if ((projected.w <= 0.0f) || (result.uv.x < -1.0f) || (result.uv.y < -1.0f) || (result.uv.x > 1.0f) || (result.uv.y > 1.0f)) {
    return {};
  }

  float lens_area = (camera.lens_radius > kEpsilon) ? kPi * sqr(camera.lens_radius) : 1.0f;

  result.pdf_area = 1.0f / lens_area;
  result.pdf_dir = result.pdf_area * distance_squared / cos_t;
  result.pdf_dir_out = 1.0f / (camera.area * lens_area * cos_t * cos_t * cos_t);

  float importance = result.pdf_dir_out / cos_t;
  result.weight = importance / result.pdf_dir;

  return result;
}

ETX_GPU_CODE CameraEval film_evaluate_out(SpectralQuery spect, const Camera& camera, const Ray& out_ray) {
  float cos_t = dot(out_ray.d, camera.direction);
  CameraEval result = {};
  result.normal = camera.direction;
  result.pdf_dir = (camera.cls == Camera::Class::Equirectangular) ? 1.0f : 1.0f / (camera.area * cos_t * cos_t * cos_t);
  return result;
}

}  // namespace etx
