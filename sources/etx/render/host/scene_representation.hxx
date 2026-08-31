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

enum class EnergyCompensationPreparationState : uint32_t {
  Ready,
  Preparing,
  Failed,
};

struct EnergyCompensationPreparationStatus {
  EnergyCompensationPreparationState state = EnergyCompensationPreparationState::Ready;
  uint32_t completed_steps = 0u;
  uint32_t total_steps = 0u;
  double elapsed_seconds = 0.0;
  double remaining_seconds = 0.0;
  bool remaining_available = false;
};

enum class NodeGeometryOperation : uint32_t {
  BakeLocalTransform,
  CenterPivot,
};

enum class NodeGeometryEditResult : uint32_t {
  Success,
  InvalidNode,
  NoMeshAttachments,
  NonMeshAttachments,
  InvalidGeometry,
  SingularTransform,
  DegenerateGeometry,
  HierarchyUpdateFailed,
};

const char* node_geometry_edit_result_message(NodeGeometryEditResult result);

enum class ScenePrimitive : uint32_t {
  Sphere,
  Box,
  Plane,
  Disk,
  Cylinder,
  Cone,
  Capsule,
  Torus,
  Ring,
  Tube,
  Tetrahedron,
  Cube,
  Octahedron,
  Dodecahedron,
  Icosahedron,
};

enum class SceneEditStatus : uint32_t {
  Success,
  InvalidNode,
  InvalidParent,
  InvalidResource,
  DuplicateAttachment,
  ResourceAlreadyAttached,
  UnsupportedAttachments,
  ActiveCameraProtected,
  InvalidTransform,
  GeometryGenerationFailed,
  HierarchyUpdateFailed,
};

struct SceneEditResult {
  SceneEditStatus status = SceneEditStatus::Success;
  uint32_t node_index = kInvalidIndex;
  uint32_t mesh_index = kInvalidIndex;
  std::vector<uint32_t> node_remapping;

  bool succeeded() const {
    return status == SceneEditStatus::Success;
  }
};

enum class SceneResourceEditStatus : uint32_t {
  Success,
  InvalidResource,
  ResourceInUse,
  ActiveResource,
  ManagedResource,
  ResourceUpdateFailed,
};

struct SceneResourceEditResult {
  SceneResourceEditStatus status = SceneResourceEditStatus::Success;
  uint32_t resource_index = kInvalidIndex;
  std::vector<uint32_t> resource_remapping;

  bool succeeded() const {
    return status == SceneResourceEditStatus::Success;
  }
};

const char* scene_edit_status_message(SceneEditStatus status);
const char* scene_resource_edit_status_message(SceneResourceEditStatus status);
const char* scene_primitive_name(ScenePrimitive primitive);

struct SceneRepresentation {
  using MaterialMapping = std::unordered_map<std::string, uint32_t>;
  using MediumMapping = std::unordered_map<std::string, uint32_t>;
  using MeshMapping = std::unordered_map<std::string, uint32_t>;

  enum : uint32_t {
    LoadGeometry = 0u,
    SetupCamera = 1u << 0u,
    PreferRecoveredSave = 1u << 1u,
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
  void replace_loaded_scene(SceneRepresentation& source);

  Camera& mutable_camera();
  SceneData& data();
  const SceneData& data() const;
  const MaterialMapping& material_mapping() const;
  const MediumMapping& medium_mapping() const;
  const MeshMapping& mesh_mapping() const;

  uint32_t add_material(const char* name = nullptr);
  SceneResourceEditResult create_material(const char* name);
  SceneResourceEditResult duplicate_material(uint32_t index);
  SceneResourceEditResult delete_material(uint32_t index);
  std::string rename_material(uint32_t index, const char* name);
  uint32_t add_medium(const char* name = nullptr);
  SceneResourceEditResult create_medium(const char* name);
  SceneResourceEditResult duplicate_medium(uint32_t index);
  SceneResourceEditResult delete_medium(uint32_t index);
  std::string rename_medium(uint32_t index, const char* name);
  SceneResourceEditResult create_camera(const char* name);
  SceneResourceEditResult duplicate_camera(uint32_t index);
  SceneResourceEditResult delete_camera(uint32_t index);
  std::string rename_camera(uint32_t index, const char* name);
  SceneResourceEditResult duplicate_emitter(uint32_t index);
  SceneResourceEditResult delete_emitter_profile(uint32_t index);
  std::string rename_emitter(uint32_t index, const char* name);
  const std::vector<std::string>& emitter_names() const;
  std::string rename_mesh(uint32_t index, const char* name);
  void set_mesh_material(uint32_t mesh_index, uint32_t material_index);
  SceneEditResult create_empty_node();
  SceneEditResult create_primitive(ScenePrimitive primitive);
  SceneEditResult duplicate_node_subtree(uint32_t node_index);
  SceneEditResult delete_node_subtree(uint32_t node_index);
  SceneEditResult reparent_node(uint32_t node_index, uint32_t parent_index);
  SceneEditResult set_node_enabled(uint32_t node_index, bool enabled);
  SceneEditResult set_node_local_transform(uint32_t node_index, const AffineTransform& transform);
  SceneEditResult attach_node_resource(uint32_t node_index, SceneAttachment::Type type, uint32_t resource_index);
  SceneEditResult detach_node_resource(uint32_t node_index, uint32_t local_attachment_index);
  std::string rename_node(uint32_t node_index, const char* name);
  NodeGeometryEditResult validate_node_geometry_edit(uint32_t node_index, NodeGeometryOperation operation) const;
  NodeGeometryEditResult edit_node_geometry(uint32_t node_index, NodeGeometryOperation operation);
  void update_medium_bounds();
  void update_active_camera();
  bool store_active_camera();

  uint32_t add_environment_emitter(const float3& color, uint32_t medium_index);
  uint32_t add_directional_emitter(const float3& direction, const float3& color, float angular_diameter_degrees, uint32_t medium_index);
  void add_atmosphere_emitter(const AtmosphereEmitterParameters& params);
  void rebuild_atmosphere_emitter(uint32_t emitter_index);
  void set_scattering_rhi(RHIContext& rhi);
  bool ensure_energy_compensation_interfaces();
  bool begin_energy_compensation_interface_preparation();
  void cancel_energy_compensation_interface_preparation();
  EnergyCompensationPreparationState poll_energy_compensation_interface_preparation();
  EnergyCompensationPreparationStatus energy_compensation_interface_preparation_status() const;
  void create_area_emitters_from_materials();
  bool delete_emitter(uint32_t emitter_index);

  Camera& camera();
  const Camera& camera() const;
  const IntegratorData& integrator_data() const;
  void set_integrator_data(const IntegratorData&);

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
