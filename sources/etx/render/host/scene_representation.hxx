#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/host/tasks.hxx>
#include <etx/render/host/film.hxx>
#include <etx/render/host/scene_data.hxx>
#include <etx/render/shared/scattering.hxx>
#include <etx/rt/integrators/integrator.hxx>
#include <etx/engine/options.hxx>
namespace etx {

struct IORDatabase;
struct SceneData;
struct RHIContext;

struct SceneRepresentation {
  using MaterialMapping = std::unordered_map<std::string, uint32_t>;
  using MediumMapping = std::unordered_map<std::string, uint32_t>;
  using MeshMapping = std::unordered_map<std::string, uint32_t>;
  using CameraMapping = std::unordered_map<std::string, uint32_t>;

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

  Camera& mutable_camera();
  SceneData& data();
  const SceneData& data() const;
  const MaterialMapping& material_mapping() const;
  const MediumMapping& medium_mapping() const;
  const MeshMapping& mesh_mapping() const;
  const CameraMapping& camera_mapping() const;

  uint32_t add_material(const char* name = nullptr);
  std::string rename_material(uint32_t index, const char* name);
  uint32_t add_medium(const char* name = nullptr);
  std::string rename_medium(uint32_t index, const char* name);
  std::string rename_mesh(uint32_t index, const char* name);
  void set_mesh_material(uint32_t mesh_index, uint32_t material_index);
  void update_medium_bounds();
  void update_active_camera();
  void store_active_camera();

  uint32_t add_environment_emitter(const float3& color, uint32_t medium_index);
  uint32_t add_directional_emitter(const float3& direction, const float3& color, float angular_diameter_degrees, uint32_t medium_index);
  void add_atmosphere_emitter(const AtmosphereEmitterParameters& params);
  void rebuild_atmosphere_emitter(uint32_t emitter_index);
  void set_scattering_rhi(RHIContext& rhi);
  void create_area_emitters_from_materials();
  bool delete_emitter(uint32_t emitter_index);

  Camera& camera();
  const Camera& camera() const;

  bool valid() const;

  ETX_DECLARE_PIMPL(SceneRepresentation, 32768);
};

void build_camera(Camera& camera, const float3& position, const float3& direction, const float3& up, const uint2& viewport, const float fov);
void compute_camera_position_to_fit_scene(const SceneData& scene_data, const Camera& camera, const float3& view_direction, float3& out_position, float3& out_target);

float get_camera_fov(const Camera& camera);
float get_camera_focal_length(const Camera& camera);
float fov_to_focal_length(float fov);
float focal_length_to_fov(float focal_len);
float horizontal_fov_to_vertical_fov(float horizontal_fov);
float vertical_fov_to_horizontal_fov(float vertical_fov);

const char* material_class_to_string(Material::Class cls);
void material_class_to_string(Material::Class cls, const char** str);

float emitter_weight(const Emitter&);

}  // namespace etx
