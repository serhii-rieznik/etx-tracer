#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/shared/scene.hxx>

namespace etx {

struct SceneRepresentation {
  SceneRepresentation();
  ~SceneRepresentation();

  bool load_from_file(const char* filename);

  const Scene& scene() const;
  Camera& camera();

  operator bool() const;

  ETX_DECLARE_PIMPL(SceneRepresentation, 2048);
};

Camera build_camera(const float3& origin, const float3& target, const float3& up, const float2& viewport, float fov, float lens_radius, float focal_distance);
void update_camera(Camera& camera, const float3& origin, const float3& target, const float3& up, const float2& viewport, float fov);
float get_camera_fov(const Camera& camera);

}  // namespace etx
