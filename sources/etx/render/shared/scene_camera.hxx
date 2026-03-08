#pragma once
#include <etx/render/interop/camera_shared.hxx>
#include <etx/render/interop/camera_film_shared.hxx>

namespace etx {

ETX_SHARED_INLINE bool camera_lens_sampling_enabled(const Camera& camera) {
  return (camera.lens_radius > kEpsilon) && (camera.focal_distance > kEpsilon);
}

ETX_SHARED_INLINE float2 camera_sample_lens_uv(const Scene& scene, const Camera& camera, const float2& sensor_sample_rnd) {
  if (camera_lens_sampling_enabled(camera) == false) {
    return float2(0.0f, 0.0f);
  }

  if ((camera.lens_image == kInvalidIndex) || (camera.lens_image >= scene.images.count)) {
    return sample_disk(sensor_sample_rnd);
  }

  float2 image_uv = scene.images[camera.lens_image].sample(sensor_sample_rnd);
  return image_uv * 2.0f - 1.0f;
}

ETX_SHARED_INLINE float3 camera_lens_point(const Scene& scene, const Camera& camera, const float2& sensor_sample_rnd) {
  float2 sensor_sample = camera_sample_lens_uv(scene, camera, sensor_sample_rnd) * camera.lens_radius;
  return camera_film_shared_lens_point(camera, sensor_sample);
}

ETX_SHARED_INLINE Ray camera_generate_primary_ray(const Scene& scene, const Camera& camera, const float2& uv, const float2& sensor_sample_rnd) {
  float2 sensor_sample = camera_sample_lens_uv(scene, camera, sensor_sample_rnd);
  return camera_generate_ray(camera, uv, sensor_sample);
}

ETX_SHARED_INLINE float2 get_center_uv(const uint2& pixel, const uint2& dim) {
  return camera_shared_center_uv(pixel, dim);
}

ETX_SHARED_INLINE float2 get_jittered_uv(Sampler& smp, const uint2& pixel, const uint2& dim) {
  float2 jitter = {smp.next(), smp.next()};
  return camera_shared_jittered_uv(pixel, dim, jitter);
}

ETX_SHARED_INLINE float film_pdf_out(const Camera& camera, const float3& to_point) {
  return camera_shared_film_pdf_out(camera, to_point);
}

ETX_SHARED_INLINE float camera_clip_direction_scale(const Camera& camera, const float3& direction_to_camera) {
  return camera_shared_clip_direction_scale(camera, direction_to_camera);
}

ETX_SHARED_INLINE Ray generate_ray(const Scene& scene, const Camera& camera, const float2& uv, const float2& sensor_sample_rnd) {
  ETX_CHECK_FINITE(uv);

  Ray ray = camera_generate_primary_ray(scene, camera, uv, sensor_sample_rnd);
  ETX_CHECK_FINITE(ray.o);
  ETX_CHECK_FINITE(ray.d);
  return ray;
}

ETX_SHARED_INLINE CameraSample evaluate_film(const Scene& scene, const Camera& camera, const float3& world_point, const float3& lens_point) {
  (void)scene;
  CameraFilmSampleShared shared = camera_film_shared_evaluate(camera, world_point, lens_point);
  CameraSample result = {};
  camera_film_shared_unpack_sample(
    shared, result.position, result.normal, result.direction, result.uv, result.weight, result.pdf_dir, result.pdf_area, result.pdf_dir_out);
  return result;
}

ETX_SHARED_INLINE CameraSample sample_film(Sampler& smp, const Scene& scene, const Camera& camera, const float3& from_point) {
  if (camera.cls == Camera::Class::Equirectangular) {
    (void)smp;
    return evaluate_film(scene, camera, from_point, camera.position);
  }

  float3 lens_point = camera.position;
  if (camera_lens_sampling_enabled(camera)) {
    float2 sensor_sample_rnd = smp.next_2d();
    lens_point = camera_lens_point(scene, camera, sensor_sample_rnd);
  }
  return evaluate_film(scene, camera, from_point, lens_point);
}

ETX_SHARED_INLINE CameraEval film_evaluate_out(SpectralQuery spect, const Camera& camera, const Ray& out_ray) {
  (void)spect;
  CameraFilmEvalShared shared = camera_film_shared_evaluate_out(camera, out_ray);
  CameraEval result = {};
  camera_film_shared_unpack_eval(shared, result.normal, result.pdf_dir);
  return result;
}

ETX_SHARED_INLINE float3 clamp_view_direction_away_from_up(const float3& view_direction, const float3& up_vector, float min_cosine_threshold) {
  return camera_shared_clamp_view_direction_away_from_up(view_direction, up_vector, kWorldRight, min_cosine_threshold);
}

}  // namespace etx
