#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/host/tasks.hxx>
#include <etx/render/host/film.hxx>
#include <etx/render/shared/scene.hxx>

#include <unordered_map>

namespace etx {

struct SceneRepresentation {
  using MaterialMapping = std::unordered_map<std::string, uint32_t>;
  using MediumMapping = std::unordered_map<std::string, uint32_t>;

  enum : uint32_t {
    LoadGeometry = 0u,
    SetupCamera = 1u << 0u,
    LoadEverything = LoadGeometry | SetupCamera,
  };

  SceneRepresentation(TaskScheduler&);
  ~SceneRepresentation();

  bool load_from_file(const char* filename, uint32_t options);

  Scene& mutable_scene();
  Scene* mutable_scene_pointer();

  const Scene& scene() const;
  const MaterialMapping& material_mapping() const;
  const MediumMapping& medium_mapping() const;

  Camera& camera();

  bool valid() const;

  ETX_DECLARE_PIMPL(SceneRepresentation, 24u * 1024u);
};

void build_camera(Camera& camera, const float3& origin, const float3& target, const float3& up, const uint2& viewport, float fov);

float get_camera_fov(const Camera& camera);
float get_camera_focal_length(const Camera& camera);
float fov_to_focal_length(float fov);
float focal_length_to_fov(float focal_len);

Material::Class material_string_to_class(const char* s);
const char* material_class_to_string(Material::Class cls);
void material_class_to_string(Material::Class cls, const char** str);

void build_emitters_distribution(Scene& scene);
float emitter_weight(const Emitter&);

}  // namespace etx
