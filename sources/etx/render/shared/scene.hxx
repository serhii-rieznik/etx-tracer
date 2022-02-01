#pragma once

#include <etx/render/shared/spectrum.hxx>
#include <etx/render/shared/camera.hxx>
#include <etx/render/shared/image.hxx>
#include <etx/render/shared/medium.hxx>
#include <etx/render/shared/material.hxx>
#include <etx/render/shared/emitter.hxx>

namespace etx {

struct alignas(16) EnvironmentEmitters {
  constexpr static const uint32_t kMaxCount = 7;
  uint32_t emitters[kMaxCount];
  uint32_t count;
};

struct alignas(16) Scene {
  ArrayView<Vertex> vertices ETX_EMPTY_INIT;
  ArrayView<Triangle> triangles ETX_EMPTY_INIT;
  ArrayView<Material> materials ETX_EMPTY_INIT;
  ArrayView<Emitter> emitters ETX_EMPTY_INIT;
  ArrayView<Image> images ETX_EMPTY_INIT;
  ArrayView<Medium> mediums ETX_EMPTY_INIT;
  Distribution emitters_distribution ETX_EMPTY_INIT;
  EnvironmentEmitters environment_emitters ETX_EMPTY_INIT;
  Spectrums* spectrums ETX_EMPTY_INIT;
  float3 bounding_sphere_center ETX_EMPTY_INIT;
  float bounding_sphere_radius ETX_EMPTY_INIT;
  uint64_t acceleration_structure ETX_EMPTY_INIT;
  uint32_t camera_medium_index ETX_INIT_WITH(kInvalidIndex);
  uint32_t camera_lens_shape_image_index ETX_INIT_WITH(kInvalidIndex);
};

ETX_GPU_CODE Ray generate_ray(Sampler& smp, const Scene& scene, const Camera& camera, const float2& uv) {
  float3 s = (uv.x * camera.aspect) * camera.side;
  float3 u = (uv.y) * camera.up;
  float3 w_o = normalize(camera.tan_half_fov * (s + u) + camera.direction);

  float3 origin = camera.position;
  if (camera.lens_radius > 0.0f) {
    float2 sensor_sample = {};
    if (scene.camera_lens_shape_image_index == kInvalidIndex) {
      sensor_sample = sample_disk(smp.next(), smp.next());
    } else {
      float pdf = {};
      uint2 location = {};
      sensor_sample = scene.images[scene.camera_lens_shape_image_index].sample(smp.next(), smp.next(), pdf, location);
      sensor_sample = sensor_sample * 2.0f - 1.0f;
    }
    sensor_sample *= camera.lens_radius;
    origin = origin + camera.side * sensor_sample.x + camera.up * sensor_sample.y;
    float focal_plane_distance = camera.focus_distance / dot(w_o, camera.direction);
    float3 p = camera.position + focal_plane_distance * w_o;
    w_o = normalize(p - origin);
  }

  return {origin, w_o, kRayEpsilon, kMaxFloat};
}
}  // namespace etx