#pragma once

namespace etx {

ETX_GPU_CODE float2 get_jittered_uv(Sampler& smp, const uint2& pixel, const uint2& dim) {
  return {
    (float(pixel.x) + smp.next()) / float(dim.x) * 2.0f - 1.0f,
    (float(pixel.y) + smp.next()) / float(dim.y) * 2.0f - 1.0f,
  };
}

ETX_GPU_CODE float film_pdf_out(const Camera& camera, const float3& to_point) {
  auto w_i = normalize(to_point - camera.position);
  float cos_t = dot(w_i, camera.direction);
  return 1.0f / fabsf(camera.area * cos_t * cos_t * cos_t);
}

ETX_GPU_CODE Ray generate_ray(Sampler& smp, const Scene& scene, const float2& uv) {
  if (scene.camera.cls == Camera::Class::Equirectangular) {
    return {scene.camera.position, phi_theta_to_direction(uv.x * kPi, uv.y * kHalfPi), kRayEpsilon, kMaxFloat};
  }

  float3 origin = scene.camera.position;
  float3 s = (uv.x * scene.camera.aspect) * scene.camera.side;
  float3 u = (uv.y) * scene.camera.up;
  float3 w_o = normalize(scene.camera.tan_half_fov * (s + u) + scene.camera.direction);
  if (scene.camera.lens_radius > 0.0f) {
    float2 sensor_sample = {};
    if (scene.camera_lens_shape_image_index == kInvalidIndex) {
      sensor_sample = sample_disk(smp.next(), smp.next());
    } else {
      float pdf = {};
      uint2 location = {};
      sensor_sample = scene.images[scene.camera_lens_shape_image_index].sample(smp.next(), smp.next(), pdf, location);
      sensor_sample = sensor_sample * 2.0f - 1.0f;
    }
    sensor_sample *= scene.camera.lens_radius;
    origin = origin + scene.camera.side * sensor_sample.x + scene.camera.up * sensor_sample.y;
    float focal_plane_distance = scene.camera.focal_distance / dot(w_o, scene.camera.direction);
    float3 p = scene.camera.position + focal_plane_distance * w_o;
    w_o = normalize(p - origin);
  }

  return {origin, w_o, kRayEpsilon, kMaxFloat};
}

ETX_GPU_CODE CameraSample sample_film(Sampler& smp, const Scene& scene, const float3& from_point) {
  if (scene.camera.cls == Camera::Class::Equirectangular) {
    // TODO : implelemt for Equirectangular camera
    return {};
  }

  float2 sensor_sample = {};

  if (scene.camera.lens_radius > 0.0f) {
    if (scene.camera_lens_shape_image_index == kInvalidIndex) {
      sensor_sample = sample_disk(smp.next(), smp.next());
    } else {
      float pdf = {};
      uint2 location = {};
      sensor_sample = scene.images[scene.camera_lens_shape_image_index].sample(smp.next(), smp.next(), pdf, location);
      sensor_sample = sensor_sample * 2.0f - 1.0f;
    }
    sensor_sample *= scene.camera.lens_radius;
  }

  CameraSample result;
  result.position = scene.camera.position + sensor_sample.x * scene.camera.side + sensor_sample.y * scene.camera.up;
  result.direction = result.position - from_point;
  result.normal = scene.camera.direction;

  float cos_t = -dot(result.direction, result.normal);
  if (cos_t < 0.0f) {
    return {};
  }

  float distance_squared = dot(result.direction, result.direction);
  float distance = sqrtf(distance_squared);
  result.direction /= distance;
  cos_t /= distance;

  float focal_plane_distance = (scene.camera.lens_radius > 0.0f) ? scene.camera.focal_distance : 1.0f;
  float3 focus_point = result.position - result.direction * (focal_plane_distance / cos_t);

  auto projected = scene.camera.view_proj * float4{focus_point.x, focus_point.y, focus_point.z, 1.0f};
  result.uv = {projected.x / projected.w, -projected.y / projected.w};
  if ((projected.w <= 0.0f) || (result.uv.x < -1.0f) || (result.uv.y < -1.0f) || (result.uv.x > 1.0f) || (result.uv.y > 1.0f)) {
    return {};
  }

  float lens_area = (scene.camera.lens_radius > 0.0) ? kPi * sqr(scene.camera.lens_radius) : 1.0f;

  result.pdf_area = 1.0f / lens_area;
  result.pdf_dir = result.pdf_area * distance_squared / cos_t;
  result.pdf_dir_out = 1.0f / (scene.camera.area * lens_area * cos_t * cos_t * cos_t);

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
