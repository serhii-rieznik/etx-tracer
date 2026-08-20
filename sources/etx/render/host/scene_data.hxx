#pragma once

#include <etx/render/shared/scene.hxx>
#include <etx/render/shared/camera.hxx>
#include <etx/render/shared/spectrum.hxx>
#include <etx/render/shared/image.hxx>
#include <etx/render/shared/scattering.hxx>
#include <etx/render/shared/medium.hxx>
#include <etx/render/shared/distribution.hxx>

#include <etx/render/host/tasks.hxx>
#include <etx/render/host/buffer_pool.hxx>
#include <etx/render/host/image_pool.hxx>
#include <etx/render/host/medium_pool.hxx>
#include <etx/render/host/scene_hierarchy.hxx>
namespace etx {

struct AtmosphereEmitterParameters {
  scattering::Parameters scattering = {};
  float quality = 0.125f;
  SpectralDistribution env_spectrum = SpectralDistribution::rgb_luminance({1.0f, 1.0f, 1.0f});
};

struct UpdateFlags {
  enum : uint32_t {
    VerticesPos,
    Triangles,
    VerticesNrm,
    VerticesTan,
    VerticesBtn,
    VerticesTex,
    Meshes,
    Hierarchy,
    Transforms,
    Attachments,
    Materials,
    Spectra,
    Emitters,
    Images,
    Mediums,
    EnergyCompensationInterfaces,
    PixelFilter,
    Defaults,
    Options,

    AnyGeometry,
    AnyGeometryStructure,
    AnyGeometryAttributes,
    AnyMaterials,

    EmbreeScene,

    Count
  };

  bool flags[Count] = {};

  bool& operator[](uint32_t flag) {
    return flags[flag];
  }

  bool operator[](uint32_t flag) const {
    return flags[flag];
  }

  bool any() const {
    for (uint32_t i = 0; i < Count; ++i) {
      if (flags[i]) {
        return true;
      }
    }
    return false;
  }
};

struct SceneHashes {
  uint64_t vertices_pos_hash = 0;
  uint64_t vertices_nrm_hash = 0;
  uint64_t vertices_tan_hash = 0;
  uint64_t vertices_btn_hash = 0;
  uint64_t vertices_tex_hash = 0;
  uint64_t triangles_hash = 0;
  uint64_t triangle_indices_hash = 0;
  uint64_t meshes_hash = 0;
  uint64_t hierarchy_hash = 0;
  uint64_t transforms_hash = 0;
  uint64_t attachments_hash = 0;
  uint64_t materials_hash = 0;
  uint64_t spectra_hash = 0;
  uint64_t emitter_profiles_hash = 0;
  uint64_t images_hash = 0;
  uint64_t mediums_hash = 0;
  uint64_t energy_compensation_interfaces_hash = 0;
  uint64_t pixel_filter_hash = 0;
  uint64_t defaults_hash = 0;
  uint64_t options_hash = 0;

  UpdateFlags compare(const SceneHashes& existing) const {
    UpdateFlags result = {};

    result[UpdateFlags::VerticesPos] = (vertices_pos_hash != existing.vertices_pos_hash);
    result[UpdateFlags::VerticesNrm] = (vertices_nrm_hash != existing.vertices_nrm_hash);
    result[UpdateFlags::VerticesTan] = (vertices_tan_hash != existing.vertices_tan_hash);
    result[UpdateFlags::VerticesBtn] = (vertices_btn_hash != existing.vertices_btn_hash);
    result[UpdateFlags::VerticesTex] = (vertices_tex_hash != existing.vertices_tex_hash);
    result[UpdateFlags::Triangles] = (triangles_hash != existing.triangles_hash);
    result[UpdateFlags::Meshes] = (meshes_hash != existing.meshes_hash);
    result[UpdateFlags::Hierarchy] = (hierarchy_hash != existing.hierarchy_hash);
    result[UpdateFlags::Transforms] = (transforms_hash != existing.transforms_hash);
    result[UpdateFlags::Attachments] = (attachments_hash != existing.attachments_hash);
    result[UpdateFlags::Materials] = (materials_hash != existing.materials_hash);
    result[UpdateFlags::Spectra] = (spectra_hash != existing.spectra_hash);
    result[UpdateFlags::Emitters] = (emitter_profiles_hash != existing.emitter_profiles_hash);
    result[UpdateFlags::Images] = (images_hash != existing.images_hash);
    result[UpdateFlags::Mediums] = (mediums_hash != existing.mediums_hash);
    result[UpdateFlags::EnergyCompensationInterfaces] = (energy_compensation_interfaces_hash != existing.energy_compensation_interfaces_hash);
    result[UpdateFlags::PixelFilter] = (pixel_filter_hash != existing.pixel_filter_hash);
    result[UpdateFlags::Defaults] = (defaults_hash != existing.defaults_hash);
    result[UpdateFlags::Options] = (options_hash != existing.options_hash);

    result[UpdateFlags::AnyGeometry] = result[UpdateFlags::VerticesPos] || result[UpdateFlags::VerticesNrm] || result[UpdateFlags::VerticesTan] ||
                                       result[UpdateFlags::VerticesBtn] || result[UpdateFlags::VerticesTex] || result[UpdateFlags::Triangles] || result[UpdateFlags::Meshes] ||
                                       result[UpdateFlags::Hierarchy] || result[UpdateFlags::Transforms] || result[UpdateFlags::Attachments];

    result[UpdateFlags::AnyGeometryStructure] =
      result[UpdateFlags::VerticesPos] || (triangle_indices_hash != existing.triangle_indices_hash) || result[UpdateFlags::Hierarchy] || result[UpdateFlags::Attachments];

    result[UpdateFlags::AnyGeometryAttributes] =
      result[UpdateFlags::VerticesNrm] || result[UpdateFlags::VerticesTan] || result[UpdateFlags::VerticesBtn] || result[UpdateFlags::VerticesTex];

    result[UpdateFlags::AnyMaterials] = result[UpdateFlags::Materials] || result[UpdateFlags::Spectra];

    result[UpdateFlags::EmbreeScene] = result[UpdateFlags::AnyGeometryStructure] || result[UpdateFlags::Transforms];

    return result;
  }
};

struct SceneData {
  using MaterialMapping = std::unordered_map<std::string, uint32_t>;

  struct CameraInfo {
    Camera cam;
    std::string id;
    bool active = false;
  };

  struct {
    std::vector<float3> pos;
    std::vector<float3> nrm;
    std::vector<float3> tan;
    std::vector<float3> btn;
    std::vector<float2> tex;
  } vertices;

  std::vector<Triangle> triangles;
  std::vector<Material> materials;
  std::vector<Mesh> meshes;
  std::vector<EmitterProfile> emitter_profiles;
  std::vector<SpectralDistribution> spectrum_values;
  std::vector<Image> images_vector;
  std::vector<Medium> mediums_vector;
  std::vector<Scene::EnergyCompensationInterface> energy_compensation_interfaces;
  SceneHierarchy hierarchy;

  BufferPool buffer_pool;
  ImagePool images;
  MediumPool mediums;
  MaterialMapping material_mapping;
  MaterialMapping mesh_mapping;
  std::vector<std::string> spectrum_names;
  std::unordered_map<uint32_t, uint32_t> material_to_emitter_profile;
  std::unordered_map<uint32_t, uint32_t> gltf_image_mapping;
  std::unordered_map<int32_t, uint32_t> gltf_material_mapping;
  std::vector<CameraInfo> cameras;
  std::string json_file_name;
  std::string geometry_file_name;
  std::string materials_file_name;
  TaskScheduler& scheduler;

  PixelFilter pixel_filter = {};
  Scene::Defaults defaults = {};
  Scene::Options options = {};

  SceneData(TaskScheduler& s);
  SceneData(const SceneData&) = delete;
  SceneData operator=(const SceneData&) = delete;

  BoundingBox compute_bounding_volumes() const;

  SceneHashes compute_hashes() const;

  void clear(TaskScheduler& scheduler);

  uint32_t add_spectrum(const char* source_id, const SpectralDistribution& spd);
  uint32_t add_spectrum(const char* id);
  uint32_t add_spectrum(const SpectralDistribution& spd);
  uint32_t add_spectrum();
  uint32_t find_spectrum(const char* id) const;

  bool has_material(const char* name) const;
  uint32_t add_material(const char* name);
  uint32_t clone_material(const Material& src, const char* name);

  uint32_t add_mesh(const char* name, uint32_t triangle_offset, uint32_t triangle_count, const float3& bbox_min, const float3& bbox_max);
  uint32_t add_mesh_asset(const char* name, uint32_t triangle_offset, uint32_t triangle_count, const float3& bbox_min, const float3& bbox_max);

  bool resolve_hierarchy();

  uint32_t add_image(const char* path, uint32_t options, const float2& offset, const float2& scale);
  uint32_t add_image(const float4* data, const uint2& dim, uint32_t options, const float2& offset, const float2& scale);
  uint32_t add_image(const Image& img);
  uint32_t add_image(const char* path, uint32_t options);
  void add_image_options(uint32_t index, uint32_t options);

  uint32_t add_energy_compensation_interface(const Scene::EnergyCompensationInterface& interface_data);

  uint32_t add_medium(Medium::Class cls, const char* name, const char* volume_file, const SpectralDistribution& s_a, const SpectralDistribution& s_t, float g,
    bool explicit_connections);

  uint32_t add_atmosphere_emitter(const AtmosphereEmitterParameters& params);
  void build_atmosphere_and_sun_images(uint32_t atmosphere_emitter_index, RHIContext& rhi, scattering::GpuContext& gpu_context);
  void rebuild_sun_images_for_atmosphere(uint32_t atmosphere_emitter_index, const std::vector<uint32_t>& sun_emitter_indices, RHIContext& rhi, scattering::GpuContext& gpu_context);
  void rebuild_atmosphere_emitter(uint32_t emitter_index, RHIContext& rhi, scattering::GpuContext& gpu_context);

 private:
  std::vector<uint32_t> _camera_attachment_nodes_scratch;
  std::vector<uint32_t> _medium_attachment_nodes_scratch;
};

}  // namespace etx
