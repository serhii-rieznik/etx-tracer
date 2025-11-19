#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/host/tasks.hxx>
#include <etx/render/host/film.hxx>
#include <etx/render/shared/scene.hxx>
#include <etx/rt/integrators/integrator.hxx>
#include <etx/util/options.hxx>

#include <unordered_map>
#include <string>

namespace etx {

struct IORDatabase;

struct SceneRepresentation {
  using MaterialMapping = std::unordered_map<std::string, uint32_t>;
  using MediumMapping = std::unordered_map<std::string, uint32_t>;

  enum : uint32_t {
    LoadGeometry = 0u,
    SetupCamera = 1u << 0u,
    LoadEverything = LoadGeometry | SetupCamera,
  };

  SceneRepresentation(TaskScheduler&, const IORDatabase&);
  ~SceneRepresentation();

  struct IntegratorData {
    Integrator::Type selected = Integrator::Type::Invalid;
    std::unordered_map<Integrator::Type, Options> settings;
  };

  bool load_from_file(const char* filename, uint32_t options, IntegratorData* out_integrator = nullptr);
  std::string save_to_file(const char* filename, Integrator::Type selected_type = Integrator::Type::Invalid, Integrator* integrator_array[] = nullptr, size_t integrator_count = 0);

  Scene& mutable_scene();
  Camera& mutable_camera();

  const Scene& scene() const;
  const MaterialMapping& material_mapping() const;
  const MediumMapping& medium_mapping() const;

  uint32_t add_medium(const char* name = nullptr);
  void rebuild_area_emitters();

  Camera& camera();
  const Camera& camera() const;

  bool valid() const;

  ETX_DECLARE_PIMPL(SceneRepresentation, 5120);
};

void build_camera(Camera& camera, const float3& origin, const float3& target, const float3& up, const uint2& viewport, const float fov);

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
