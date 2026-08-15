#include <etx/render/interop/interop.hxx>

#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/core/json.hxx>

#include <etx/render/shared/scene.hxx>
#include <etx/render/shared/scattering.hxx>
#include <etx/render/shared/ior_database.hxx>
#include <etx/render/shared/vertex_utils.hxx>
#include <etx/render/shared/sampler.hxx>

#include <etx/render/host/scene_representation.hxx>
#include <etx/render/host/image_pool.hxx>
#include <etx/render/host/medium_pool.hxx>
#include <etx/render/host/bsdf_energy_compensation_lut.hxx>
#include <etx/render/host/scene_data.hxx>
#include <etx/render/host/scene_serialization.hxx>
#include <etx/render/host/scene_loader_utils.hxx>
#include <etx/rt/integrators/integrator.hxx>

#include <etx/render/host/scene_obj_loader.hxx>
#include <etx/render/host/scene_gltf_loader.hxx>
#include <etx/render/host/scene_tungsten_loader.hxx>

#include <tinyexr.hxx>

#include <mikktspace.h>
namespace etx {

namespace {

constexpr float kDefaultCameraClipNear = 0.1f;
constexpr float kDefaultCameraClipFar = 1000.0f;
constexpr uint2 kDefaultModelCameraFilmSize = {1280u, 720u};
constexpr float kDefaultModelSunAngularDiameter = 0.53f;
constexpr float kDefaultModelAtmosphereQuality = 0.125f;

void sanitize_camera_clip_planes(Camera& camera) {
  camera.clip_near = (camera.clip_near > 0.0f) ? camera.clip_near : kDefaultCameraClipNear;
  camera.clip_far = (camera.clip_far > camera.clip_near) ? camera.clip_far : max(camera.clip_near + 0.001f, kDefaultCameraClipFar);
}

Integrator::Type legacy_integrator_selection_to_type(const std::string& type_id) {
  if (type_id == "bdpt_distilled") {
    return Integrator::Type::Bidirectional;
  }

  return integrator_id_to_type(type_id.c_str());
}

std::string rename_entry(std::unordered_map<std::string, uint32_t>& mapping, uint32_t index, const char* desired_name, const char* fallback_prefix) {
  auto current = mapping.end();
  for (auto it = mapping.begin(); it != mapping.end(); ++it) {
    if (it->second == index) {
      current = it;
      break;
    }
  }
  if (current == mapping.end()) {
    return {};
  }

  std::string base = (desired_name != nullptr) ? desired_name : "";
  auto strip_prefix = [](const std::string& s) -> std::string {
    if (s.starts_with("etx::")) {
      return s.substr(5);
    }
    if (s.starts_with("et::")) {
      return s.substr(4);
    }
    return s;
  };
  base = strip_prefix(base);
  if (base.empty()) {
    base = current->first;
  }
  if (base.empty()) {
    base = std::string(fallback_prefix) + std::to_string(index);
  }

  std::string final = base;
  uint32_t suffix = 1;
  while (true) {
    auto found = mapping.find(final);
    if ((found == mapping.end()) || (found->second == index)) {
      break;
    }
    final = base + "#" + std::to_string(suffix++);
  }

  if (final != current->first) {
    mapping.erase(current);
    mapping.emplace(final, index);
  }
  return final;
}

bool scene_has_environment_emitter(const SceneData& data) {
  for (const auto& profile : data.emitter_profiles) {
    if (profile.cls == EmitterProfile::Class::Environment) {
      return true;
    }
  }
  return false;
}

void add_default_raw_model_lighting(SceneData& data) {
  scattering::Parameters scattering_params = {};
  scattering_params.altitude = 1000.0f;
  scattering_params.anisotropy = 0.825f;
  scattering_params.rayleigh_scale = 1.0f;
  scattering_params.mie_scale = 1.0f;
  scattering_params.ozone_scale = 1.0f;

  const uint32_t atmosphere_index = data.add_atmosphere_emitter({scattering_params, kDefaultModelAtmosphereQuality});

  auto& sun = data.emitter_profiles.emplace_back(EmitterProfile::Class::Directional);
  sun.emission.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_luminance({1.0f, 1.0f, 1.0f}));
  sun.emission.image_index = kInvalidIndex;
  sun.directional.direction = normalize(float3{0.0f, 1.0f, 1.0f});
  sun.directional.angular_size = kDefaultModelSunAngularDiameter * kPi / 180.0f;
  sun.directional.equivalent_disk_size = 2.0f * std::tan(sun.directional.angular_size * 0.5f);
  sun.directional.angular_size_cosine = std::cos(sun.directional.angular_size * 0.5f);
  sun.reference_emitter_index = atmosphere_index;
  sun.medium_index = kInvalidIndex;
}

}  // namespace

void material_class_to_string(Material::Class cls, const char** str) {
  static const char* names[] = {
    "diffuse",
    "translucent",
    "plastic",
    "conductor",
    "dielectric",
    "thinfilm",
    "mirror",
    "boundary",
    "velvet",
    "openpbr",
    "void",
    "undefined",
  };
  static_assert(sizeof(names) / sizeof(names[0]) == uint32_t(MaterialClass::Count) + 1);
  *str = cls < MaterialClass::Count ? names[uint32_t(cls)] : "undefined";
}

const char* material_class_to_string(Material::Class cls) {
  const char* result = nullptr;
  material_class_to_string(cls, &result);
  return result;
}

struct SceneRepresentationImpl {
  static constexpr float kDefaultDielectricEta = 1.5f;
  static constexpr float kDefaultConductorK = 1000000.0f;

  TaskScheduler& scheduler;
  SceneData data;
  Camera active_camera;
  std::mutex mt;
  RHIContext* rhi = nullptr;
  scattering::GpuContext scattering_gpu = {};
  bool scattering_gpu_ready = false;

  const IORDatabase& ior_database;
  SceneRepresentation::IntegratorData integrator_data = {};

  bool load_illuminant_from_identifier(const char* identifier, SpectralDistribution& spd) const {
    if ((identifier == nullptr) || (identifier[0] == 0))
      return false;

    if (const IORDefinition* def = ior_database.find_by_name(identifier, SpectralDistribution::Illuminant)) {
      spd = def->eta;
      return true;
    }

    std::filesystem::path candidate = locate_spectrum_file(identifier, {"emission"});
    if (candidate.empty())
      return false;

    std::string title;
    auto cls = SpectralDistribution::load_from_file(candidate.string().c_str(), spd, nullptr, false, title);
    return cls != SpectralDistribution::Invalid;
  }

  SceneRepresentationImpl(TaskScheduler& s, const IORDatabase& db)
    : scheduler(s)
    , data(s)
    , ior_database(db) {
    data.images.init(1024u);
    data.mediums.init(1024u);
    build_camera(active_camera, {5.0f, 5.0f, 5.0f}, normalize(float3{0.0f, 0.0f, 0.0f} - float3{5.0f, 5.0f, 5.0f}), kWorldUp, {1280u, 720u}, 26.99f);
  }

  ~SceneRepresentationImpl() {
    cleanup();
    if ((rhi != nullptr) && scattering_gpu.initialized) {
      scattering::gpu_cleanup(*rhi, scattering_gpu);
    }
    data.images.cleanup();
    data.mediums.cleanup();
  }

  void init_default_values() {
    data.defaults.black_spectrum = data.add_spectrum(SpectralDistribution::rgb_reflectance({0.0f, 0.0f, 0.0f}));
    data.defaults.white_spectrum = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
    data.defaults.rayleigh_spectrum = data.add_spectrum(scattering::rayleigh_spectrum());
    data.defaults.mie_spectrum = data.add_spectrum(scattering::mie_spectrum());
    data.defaults.ozone_spectrum = data.add_spectrum(scattering::ozone_spectrum());
    data.defaults.dielectric_eta = data.add_spectrum(SpectralDistribution::constant(kDefaultDielectricEta));
    data.defaults.conductor_eta = data.add_spectrum(SpectralDistribution::constant(0.0f));
    data.defaults.conductor_k = data.add_spectrum(SpectralDistribution::constant(kDefaultConductorK));

    data.options.properties[Scene::Properties::Spectral] = false;
    data.options.properties[Scene::Properties::MultipleImportanceSampling] = true;
    data.options.properties[Scene::Properties::BlueNoise] = true;

    data.defaults.subsurface_scatter_material = data.add_material("etx::subsurface-scatter");
    data.materials[data.defaults.subsurface_scatter_material].reflectance = {.spectrum_index = data.defaults.black_spectrum};
    data.materials[data.defaults.subsurface_scatter_material].scattering = {.spectrum_index = data.defaults.white_spectrum};
    data.materials[data.defaults.subsurface_scatter_material].cls = MaterialClass::Translucent;

    data.defaults.subsurface_exit_material = data.add_material("etx::subsurface-exit");
    data.materials[data.defaults.subsurface_exit_material].reflectance = {.spectrum_index = data.defaults.white_spectrum};
    data.materials[data.defaults.subsurface_exit_material].scattering = {.spectrum_index = data.defaults.white_spectrum};
    data.materials[data.defaults.subsurface_exit_material].cls = MaterialClass::Diffuse;

    data.defaults.missing_material = data.add_material("etx::missing");
    data.materials[data.defaults.missing_material].reflectance = {.spectrum_index = data.defaults.white_spectrum};
    data.materials[data.defaults.missing_material].scattering = {.spectrum_index = data.defaults.white_spectrum};
    data.materials[data.defaults.missing_material].cls = MaterialClass::Diffuse;
  }

  void cleanup() {
    data.clear(scheduler);
    integrator_data = {};

    active_camera = {};
    active_camera.lens_image = kInvalidIndex;
    active_camera.medium_index = kInvalidIndex;
    active_camera.up = kWorldUp;

    build_camera(active_camera, {5.0f, 5.0f, 5.0f}, normalize(float3{0.0f, 0.0f, 0.0f} - float3{5.0f, 5.0f, 5.0f}), kWorldUp, {1280u, 720u}, 26.99f);

    init_default_values();
  }

  float triangle_area(const Triangle& t) {
    return 0.5f * length(cross(data.vertices.pos[t.i[1]] - data.vertices.pos[t.i[0]], data.vertices.pos[t.i[2]] - data.vertices.pos[t.i[0]]));
  }

  void validate_materials() {
    std::mutex mt;
    scheduler.execute(data.materials.size(), [this, &mt](uint32_t begin, uint32_t end, uint32_t) {
      for (uint32_t i = begin; i < end; ++i) {
        auto& mtl = data.materials[i];
        mtl.thinfilm.weight = clamp(mtl.thinfilm.weight, 0.0f, 1.0f);
        mtl.thinfilm.min_thickness = max(0.0f, mtl.thinfilm.min_thickness);
        mtl.thinfilm.max_thickness = max(0.0f, mtl.thinfilm.max_thickness);
        if (mtl.reflectance.spectrum_index == kInvalidIndex) {
          std::unique_lock lock(mt);
          mtl.reflectance.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
        }
        if (mtl.scattering.spectrum_index == kInvalidIndex) {
          std::unique_lock lock(mt);
          mtl.scattering.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
        }
        if (mtl.subsurface.spectrum_index == kInvalidIndex) {
          std::unique_lock lock(mt);
          mtl.subsurface.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 0.2f, 0.04f}));
        }
        if ((mtl.subsurface_cls != SubsurfaceMaterial::Disabled) && (mtl.subsurface_cls != SubsurfaceMaterial::RandomWalk)) {
          mtl.subsurface_cls = SubsurfaceMaterial::RandomWalk;
        }
        if (mtl.emission.spectrum_index == kInvalidIndex) {
          std::unique_lock lock(mt);
          mtl.emission.spectrum_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
        }
        if ((mtl.roughness.value.x > 0.0f) || (mtl.roughness.value.y > 0.0f)) {
          mtl.roughness.value.x = max(kEpsilon, mtl.roughness.value.x);
          mtl.roughness.value.y = max(kEpsilon, mtl.roughness.value.y);
        }
        if (mtl.int_ior.eta_index == kInvalidIndex) {
          std::unique_lock lock(mt);
          if (mtl.cls == MaterialClass::Conductor) {
            mtl.int_ior.cls = SpectralDistribution::Conductor;
            mtl.int_ior.eta_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
          } else {
            mtl.int_ior.cls = SpectralDistribution::Dielectric;
            mtl.int_ior.eta_index = data.add_spectrum(SpectralDistribution::constant(kDefaultDielectricEta));
          }
        }
        if (mtl.int_ior.k_index == kInvalidIndex) {
          std::unique_lock lock(mt);
          if (mtl.cls == MaterialClass::Conductor) {
            mtl.int_ior.k_index = data.add_spectrum(SpectralDistribution::constant(kDefaultConductorK));
          } else {
            mtl.int_ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
          }
        }
        if (mtl.ext_ior.eta_index == kInvalidIndex) {
          std::unique_lock lock(mt);
          mtl.ext_ior.cls = SpectralDistribution::Dielectric;
          mtl.ext_ior.eta_index = data.add_spectrum(SpectralDistribution::constant(1.0f));
        }
        if (mtl.ext_ior.k_index == kInvalidIndex) {
          std::unique_lock lock(mt);
          mtl.ext_ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
        }
        const bool thinfilm_requested =
          (mtl.thinfilm.weight > 0.0f) && (max(mtl.thinfilm.min_thickness, mtl.thinfilm.max_thickness) > 0.0f);
        if (thinfilm_requested && (mtl.thinfilm.ior.cls != SpectralDistribution::Dielectric)) {
          log::warning("Material %u uses a non-dielectric thin-film IOR; disabling its unsupported thin film", i);
          mtl.thinfilm.weight = 0.0f;
        }
        {
          std::unique_lock lock(mt);
          if (mtl.thinfilm.ior.k_index >= data.spectrum_values.size()) {
            mtl.thinfilm.ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
          }
          if (mtl.thinfilm.ior.eta_index >= data.spectrum_values.size()) {
            mtl.thinfilm.ior.eta_index = data.add_spectrum(SpectralDistribution::constant(1.0f));
          }
          if (thinfilm_requested && (mtl.thinfilm.ior.cls == SpectralDistribution::Dielectric) &&
              (data.spectrum_values[mtl.thinfilm.ior.k_index].is_zero() == false)) {
            log::warning("Material %u uses absorption in its thin film; forcing extinction to zero for the supported lossless-film model", i);
            mtl.thinfilm.ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
          }
        }
      }
    });
  }

  void validate_mediums() {
    // Clamp medium densities to prevent extremely small mean free paths
    for (uint32_t i = 0; i < data.mediums.array_size(); ++i) {
      const Medium& medium = data.mediums.get(i);

      if (medium.absorption_index == kInvalidIndex || medium.absorption_index >= data.spectrum_values.size()) {
        continue;
      }

      if (medium.scattering_index == kInvalidIndex || medium.scattering_index >= data.spectrum_values.size()) {
        continue;
      }

      SpectralDistribution& absorption = data.spectrum_values[medium.absorption_index];
      SpectralDistribution& scattering = data.spectrum_values[medium.scattering_index];

      float max_absorption = absorption.maximum_spectral_power();
      float max_scattering = scattering.maximum_spectral_power();

      float max_extinction = max_absorption + max_scattering;
      if (max_extinction <= 0.0f) {
        continue;
      }

      constexpr float kMinMeanFreePathAbsolute = 0.01f;  // Minimum mean free path in absolute units
      float max_allowed_extinction = 1.0f / kMinMeanFreePathAbsolute;

      if (max_extinction <= max_allowed_extinction) {
        continue;
      }

      float scale_factor = max_allowed_extinction / max_extinction;

      // Scale both spectra by the same factor to preserve ratios
      absorption.scale(scale_factor);
      scattering.scale(scale_factor);
    }
  }

  void validate_normals(std::vector<bool>& referenced_vertices, bool& has_invalid_tangents) {
    std::vector<bool> init_normals(data.vertices.nrm.size(), false);
    referenced_vertices.resize(data.vertices.nrm.size());

    bool has_tangents = data.vertices.tan.size() == data.vertices.nrm.size();
    if (has_tangents == false)
      has_invalid_tangents = true;

    scheduler.execute(data.triangles.size(), [&](uint32_t begin, uint32_t end, uint32_t) {
      for (uint32_t t = begin; t < end; ++t) {
        const auto& tri = data.triangles[t];
        const float tri_area = triangle_area(tri);
        for (uint32_t i = 0; i < 3; ++i) {
          uint32_t index = tri.i[i];
          ETX_CRITICAL(is_valid_vector(tri.geo_n));
          referenced_vertices[index] = true;

          if (has_tangents && (is_valid_vector(data.vertices.tan[index]) == false)) {
            has_invalid_tangents = true;
          }

          if (is_valid_vector(data.vertices.nrm[index]))
            continue;

          if (init_normals[index]) {
            data.vertices.nrm[index] += tri.geo_n * tri_area;
          } else {
            init_normals[index] = true;
            data.vertices.nrm[index] = tri.geo_n * tri_area;
          }
        }
      }
    });

    scheduler.execute(data.vertices.nrm.size(), [this](uint32_t begin, uint32_t end, uint32_t) {
      for (uint32_t i = begin; i < end; ++i) {
        data.vertices.nrm[i] = normalize(data.vertices.nrm[i]);
      }
    });
  }

  void build_tangents() {
    const uint64_t normal_count = data.vertices.nrm.size();
    if (data.vertices.tan.size() != normal_count)
      data.vertices.tan.resize(normal_count);
    if (data.vertices.btn.size() != normal_count)
      data.vertices.btn.resize(normal_count);

    TimeMeasure uv_timer = {};
    float2 min_uv = {kMaxFloat, kMaxFloat};
    float2 max_uv = {-kMaxFloat, -kMaxFloat};
    for (const auto& v : data.vertices.tex) {
      min_uv = min(min_uv, v);
      max_uv = max(max_uv, v);
    }
    auto uv_span = max_uv - min_uv;
    if (dot(uv_span, uv_span) <= kEpsilon) {
      log::warning("No texture coordinates: tangents will be computed automatically");
      return;
    }
    log::info("UV validation: %.4f sec", uv_timer.lap());

    TimeMeasure total_timer = {};

    // Pre-resolve vertex data to eliminate index lookups during computation
    // Use SoA (Structure of Arrays) for better cache performance
    TimeMeasure resolve_timer = {};
    const size_t total_vertices = data.triangles.size() * 3;
    std::vector<float3> resolved_positions(total_vertices);
    std::vector<float3> resolved_normals(total_vertices);
    std::vector<float2> resolved_texcoords(total_vertices);

    for (size_t tri_idx = 0; tri_idx < data.triangles.size(); ++tri_idx) {
      const auto& tri = data.triangles[tri_idx];
      const size_t base_idx = tri_idx * 3;
      for (uint32_t i = 0; i < 3; ++i) {
        uint32_t vertex_index = tri.i[i];
        resolved_positions[base_idx + i] = data.vertices.pos[vertex_index];
        resolved_normals[base_idx + i] = data.vertices.nrm[vertex_index];
        resolved_texcoords[base_idx + i] = data.vertices.tex[vertex_index];
      }
    }
    log::info("Vertex data resolution: %.4f sec", resolve_timer.lap());

    struct MikkTSpaceUserData {
      const std::vector<float3>& positions;
      const std::vector<float3>& normals;
      const std::vector<float2>& texcoords;
      SceneData& data;
      std::vector<bool> computed_flags;
    };
    MikkTSpaceUserData user_data = {resolved_positions, resolved_normals, resolved_texcoords, data, std::vector<bool>(normal_count, false)};

    TimeMeasure interface_timer = {};
    SMikkTSpaceInterface contextInterface = {};
    contextInterface.m_getNumFaces = [](const SMikkTSpaceContext* pContext) -> int {
      const auto& user_data = *reinterpret_cast<MikkTSpaceUserData*>(pContext->m_pUserData);
      return static_cast<int>(user_data.data.triangles.size());
    };
    contextInterface.m_getNumVerticesOfFace = [](const SMikkTSpaceContext* pContext, const int iFace) -> int {
      return 3;
    };
    contextInterface.m_getPosition = [](const SMikkTSpaceContext* pContext, float fvPosOut[], const int iFace, const int iVert) {
      const auto& user_data = *reinterpret_cast<MikkTSpaceUserData*>(pContext->m_pUserData);
      const auto& pos = user_data.positions[iFace * 3 + iVert];
      fvPosOut[0] = pos.x;
      fvPosOut[1] = pos.y;
      fvPosOut[2] = pos.z;
    };
    contextInterface.m_getNormal = [](const SMikkTSpaceContext* pContext, float fvNormOut[], const int iFace, const int iVert) {
      const auto& user_data = *reinterpret_cast<MikkTSpaceUserData*>(pContext->m_pUserData);
      const auto& nrm = user_data.normals[iFace * 3 + iVert];
      fvNormOut[0] = nrm.x;
      fvNormOut[1] = nrm.y;
      fvNormOut[2] = nrm.z;
    };
    contextInterface.m_getTexCoord = [](const SMikkTSpaceContext* pContext, float fvTexcOut[], const int iFace, const int iVert) {
      const auto& user_data = *reinterpret_cast<MikkTSpaceUserData*>(pContext->m_pUserData);
      const auto& tex = user_data.texcoords[iFace * 3 + iVert];
      fvTexcOut[0] = tex.x;
      fvTexcOut[1] = tex.y;
    };
    contextInterface.m_setTSpaceBasic = [](const SMikkTSpaceContext* pContext, const float fvTangent[], const float fSign, const int iFace, const int iVert) {
      auto& user_data = *reinterpret_cast<MikkTSpaceUserData*>(pContext->m_pUserData);
      const auto& tri = user_data.data.triangles[iFace];
      uint32_t vertex_index = tri.i[iVert];
      auto& nrm = user_data.data.vertices.nrm[vertex_index];
      auto& tan = user_data.data.vertices.tan[vertex_index];
      auto& btn = user_data.data.vertices.btn[vertex_index];

      // Let MikkTSpace set tangents for vertices it hasn't touched yet
      if (user_data.computed_flags[vertex_index] == false) {
        tan = normalize(float3{fvTangent[0], fvTangent[1], fvTangent[2]});
        btn = normalize(cross(tan, nrm) * fSign);
        user_data.computed_flags[vertex_index] = true;
      }
    };

    SMikkTSpaceContext context = {};
    context.m_pUserData = &user_data;
    context.m_pInterface = &contextInterface;

    log::info("MikkTSpace interface setup: %.4f sec", interface_timer.lap());

    TimeMeasure compute_timer = {};
    genTangSpaceDefault(&context);
    log::info("MikkTSpace computation: %.4f sec", compute_timer.lap());

    log::info("Total tangent building: %.4f sec", total_timer.lap());
  }

  void validate_tangents(std::vector<bool>& referenced_vertices, bool force) {
    for (uint64_t vertex_index = 0, e = data.vertices.tan.size(); vertex_index < e; ++vertex_index) {
      auto& v_nrm = data.vertices.nrm[vertex_index];
      auto& v_tan = data.vertices.tan[vertex_index];
      auto& v_btn = data.vertices.btn[vertex_index];

      bool tan_valid = is_valid_vector(v_tan);
      bool btn_valid = is_valid_vector(v_btn);

      if (tan_valid && btn_valid) {
        continue;
      }

      if (force || referenced_vertices[vertex_index]) {
        ETX_ASSERT(is_valid_vector(v_nrm));
        auto [t, b] = orthonormal_basis(v_nrm);
        v_tan = t;
        v_btn = b;
      }
    }
  }

  void update_medium_bounds();
  void set_mesh_material(uint32_t mesh_index, uint32_t material_index);

  void set_mesh_material_impl(uint32_t mesh_index, uint32_t material_index);
  void add_atmosphere_emitter(const AtmosphereEmitterParameters& params);
  void rebuild_atmosphere_emitter(uint32_t emitter_index);
  void set_scattering_rhi(RHIContext& rhi_context);
  bool ensure_scattering_gpu_context();
  void generate_pixel_sampler_image();

  void create_area_emitters_from_materials();
  bool delete_emitter(uint32_t emitter_index);
  void setup_atmosphere_references();

  bool finalize_scene_loading(uint32_t options, const char* base_folder, uint32_t load_result, float camera_fov, bool use_focal_len, float camera_focal_len, bool force_tangents,
    bool spectral_scene, bool create_default_camera_entry);
};

void build_camera(Camera& camera, const float3& position, const float3& direction, const float3& up, const uint2& viewport, const float fov) {
  sanitize_camera_clip_planes(camera);

  float3 target = position + direction;

  float4x4 view = look_at(position, target, up);
  float4x4 proj = perspective(fov * kPi / 180.0f, viewport.x, viewport.y, camera.clip_near, camera.clip_far);

  camera.position = position;
  camera.direction = normalize(direction);
  camera.side = {view.col[0].x, view.col[1].x, view.col[2].x};
  camera.up = {view.col[0].y, view.col[1].y, view.col[2].y};
  camera.tan_half_fov = 1.0f / std::abs(proj.col[0].x);
  camera.aspect = proj.col[1].y / proj.col[0].x;
  camera.view_proj = proj * view;

  float plane_w = 2.0f * camera.tan_half_fov;
  float plane_h = 2.0f * camera.tan_half_fov / camera.aspect;
  camera.area = plane_w * plane_h;
  camera.film_size = viewport;
  camera.image_plane = float(camera.film_size.x) / (2.0f * camera.tan_half_fov);
}

float get_camera_fov(const Camera& camera) {
  return 2.0f * atanf(camera.tan_half_fov) * 180.0f / kPi;
}

float get_camera_focal_length(const Camera& camera) {
  return 0.5f * Film::kFilmHorizontalSize / camera.tan_half_fov;
}

float fov_to_focal_length(float fov) {
  return 0.5f * Film::kFilmHorizontalSize / tanf(0.5f * fov);
}

float focal_length_to_fov(float focal_len) {
  return 2.0f * atanf(Film::kFilmHorizontalSize / (2.0f * focal_len));
}

float horizontal_fov_to_vertical_fov(float horizontal_fov) {
  float aspect_ratio = Film::kFilmHorizontalSize / Film::kFilmVerticalSize;
  return 2.0f * atanf(tanf(0.5f * horizontal_fov) / aspect_ratio);
}

float vertical_fov_to_horizontal_fov(float vertical_fov) {
  float aspect_ratio = Film::kFilmHorizontalSize / Film::kFilmVerticalSize;
  return 2.0f * atanf(tanf(0.5f * vertical_fov) * aspect_ratio);
}

void compute_camera_position_to_fit_scene(const Scene& scene_data, const Camera& camera, const float3& view_direction, float3& out_position, float3& out_target) {
  const float3 bbox_min = scene_data.bounding_box_min;
  const float3 bbox_max = scene_data.bounding_box_max;
  const float3 center = 0.5f * (bbox_min + bbox_max);
  constexpr float kMinCosineThreshold = 0.99f;
  const float3 view_dir = clamp_view_direction_away_from_up(view_direction, kWorldUp, kMinCosineThreshold);

  float distance = 3.0f * scene_data.bounding_sphere_radius;

  if ((camera.cls == Camera::Class::Perspective) && (camera.tan_half_fov > kEpsilon)) {
    const float3 bbox_size = bbox_max - bbox_min;
    float3 right = cross(view_dir, kWorldUp);
    if (length(right) < kEpsilon) {
      right = cross(view_dir, kWorldRight);
    }
    right = normalize(right);
    const float3 up = normalize(cross(right, view_dir));

    const float3 bbox_half_size = 0.5f * bbox_size;
    const float3 bbox_corners[8] = {
      center + float3{-bbox_half_size.x, -bbox_half_size.y, -bbox_half_size.z},
      center + float3{+bbox_half_size.x, -bbox_half_size.y, -bbox_half_size.z},
      center + float3{-bbox_half_size.x, +bbox_half_size.y, -bbox_half_size.z},
      center + float3{+bbox_half_size.x, +bbox_half_size.y, -bbox_half_size.z},
      center + float3{-bbox_half_size.x, -bbox_half_size.y, +bbox_half_size.z},
      center + float3{+bbox_half_size.x, -bbox_half_size.y, +bbox_half_size.z},
      center + float3{-bbox_half_size.x, +bbox_half_size.y, +bbox_half_size.z},
      center + float3{+bbox_half_size.x, +bbox_half_size.y, +bbox_half_size.z},
    };

    const float tan_half_fov = camera.tan_half_fov;
    const float aspect = camera.aspect;
    const float margin = 1.1f;

    float min_distance = 0.0f;

    for (uint32_t i = 0; i < 8; ++i) {
      const float3 corner_rel = bbox_corners[i] - center;
      const float right_proj = dot(corner_rel, right);
      const float up_proj = dot(corner_rel, up);
      const float forward_proj = dot(corner_rel, view_dir);

      const float corner_dist_for_width = margin * fabsf(right_proj) / tan_half_fov;
      const float corner_dist_for_height = margin * fabsf(up_proj) * aspect / tan_half_fov;
      const float corner_required_dist = max(corner_dist_for_width, corner_dist_for_height);

      const float corner_distance = forward_proj + corner_required_dist;
      min_distance = max(min_distance, corner_distance);
    }

    distance = min_distance;
  }

  out_position = center + distance * view_dir;
  out_target = center;
}

void compute_camera_position_to_fit_scene(const SceneData& scene_data, const Camera& camera, const float3& view_direction, float3& out_position, float3& out_target) {
  auto bbox = scene_data.compute_bounding_volumes();
  const float3 bbox_size = bbox.p_max - bbox.p_min;
  const float3 center = 0.5f * (bbox.p_min + bbox.p_max);

  constexpr float kMinCosineThreshold = 0.99f;
  const float3 view_dir = clamp_view_direction_away_from_up(view_direction, kWorldUp, kMinCosineThreshold);

  float distance = 3.0f * length(bbox_size);

  if ((camera.cls == Camera::Class::Perspective) && (camera.tan_half_fov > kEpsilon)) {
    float3 right = cross(view_dir, kWorldUp);
    if (length(right) < kEpsilon) {
      right = cross(view_dir, kWorldRight);
    }
    right = normalize(right);
    const float3 up = normalize(cross(right, view_dir));

    const float3 bbox_half_size = 0.5f * bbox_size;
    const float3 bbox_corners[8] = {
      center + float3{-bbox_half_size.x, -bbox_half_size.y, -bbox_half_size.z},
      center + float3{+bbox_half_size.x, -bbox_half_size.y, -bbox_half_size.z},
      center + float3{-bbox_half_size.x, +bbox_half_size.y, -bbox_half_size.z},
      center + float3{+bbox_half_size.x, +bbox_half_size.y, -bbox_half_size.z},
      center + float3{-bbox_half_size.x, -bbox_half_size.y, +bbox_half_size.z},
      center + float3{+bbox_half_size.x, -bbox_half_size.y, +bbox_half_size.z},
      center + float3{-bbox_half_size.x, +bbox_half_size.y, +bbox_half_size.z},
      center + float3{+bbox_half_size.x, +bbox_half_size.y, +bbox_half_size.z},
    };

    const float tan_half_fov = camera.tan_half_fov;
    const float aspect = camera.aspect;
    const float margin = 1.1f;

    float min_distance = 0.0f;

    for (uint32_t i = 0; i < 8; ++i) {
      const float3 corner_rel = bbox_corners[i] - center;
      const float right_proj = dot(corner_rel, right);
      const float up_proj = dot(corner_rel, up);
      const float forward_proj = dot(corner_rel, view_dir);

      const float corner_dist_for_width = margin * fabsf(right_proj) / tan_half_fov;
      const float corner_dist_for_height = margin * fabsf(up_proj) * aspect / tan_half_fov;
      const float corner_required_dist = max(corner_dist_for_width, corner_dist_for_height);

      const float corner_distance = forward_proj + corner_required_dist;
      min_distance = max(min_distance, corner_distance);
    }

    distance = max(distance, min_distance);
  }

  out_position = center + distance * view_dir;
  out_target = center;
}

ETX_PIMPL_IMPLEMENT(SceneRepresentation, Impl);

SceneRepresentation::SceneRepresentation(TaskScheduler& s, const IORDatabase& db) {
  ETX_PIMPL_INIT(SceneRepresentation, s, db);
}

SceneRepresentation::~SceneRepresentation() {
  ETX_PIMPL_CLEANUP(SceneRepresentation);
}

SceneData& SceneRepresentation::data() {
  return _private->data;
}

const SceneData& SceneRepresentation::data() const {
  return _private->data;
}

Camera& SceneRepresentation::mutable_camera() {
  return _private->active_camera;
}

const SceneRepresentation::MaterialMapping& SceneRepresentation::material_mapping() const {
  return _private->data.material_mapping;
}

const SceneRepresentation::MediumMapping& SceneRepresentation::medium_mapping() const {
  return _private->data.mediums.mapping();
}

const SceneRepresentation::MeshMapping& SceneRepresentation::mesh_mapping() const {
  return _private->data.mesh_mapping;
}

const SceneRepresentation::CameraMapping& SceneRepresentation::camera_mapping() const {
  static CameraMapping camera_mapping_cache;
  camera_mapping_cache.clear();
  for (size_t i = 0; i < _private->data.cameras.size(); ++i) {
    camera_mapping_cache[_private->data.cameras[i].id] = static_cast<uint32_t>(i);
  }
  return camera_mapping_cache;
}

uint32_t SceneRepresentation::add_material(const char* name) {
  uint32_t index = _private->data.add_material(name);
  auto& mat = _private->data.materials[index];
  mat.cls = MaterialClass::Diffuse;
  mat.reflectance.spectrum_index = _private->data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
  mat.scattering.spectrum_index = _private->data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
  mat.emission.spectrum_index = _private->data.add_spectrum(SpectralDistribution::constant(0.0f));
  mat.subsurface.spectrum_index = _private->data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 0.2f, 0.04f}));
  mat.int_ior.cls = SpectralDistribution::Dielectric;
  mat.int_ior.eta_index = _private->data.add_spectrum(SpectralDistribution::constant(kDefaultDielectricEta));
  mat.int_ior.k_index = _private->data.add_spectrum(SpectralDistribution::constant(0.0f));
  mat.ext_ior.cls = SpectralDistribution::Dielectric;
  mat.ext_ior.eta_index = _private->data.add_spectrum(SpectralDistribution::constant(1.0f));
  mat.ext_ior.k_index = _private->data.add_spectrum(SpectralDistribution::constant(0.0f));
  return index;
}

std::string SceneRepresentation::rename_material(uint32_t index, const char* name) {
  return rename_entry(_private->data.material_mapping, index, name, "material-");
}

uint32_t SceneRepresentation::add_medium(const char* name) {
  SpectralDistribution absorption_spectrum = SpectralDistribution::constant(0.0f);
  SpectralDistribution scattering_spectrum = SpectralDistribution::constant(1.0f);
  uint32_t absorption_index = _private->data.add_spectrum(absorption_spectrum);
  uint32_t scattering_index = _private->data.add_spectrum(scattering_spectrum);
  std::string id = name && name[0] ? name : ("medium-" + std::to_string(_private->data.mediums.array_size()));
  return _private->data.mediums.add(Medium::Homogeneous, id, nullptr, absorption_index, scattering_index, 0.0f, true);
}

std::string SceneRepresentation::rename_medium(uint32_t index, const char* name) {
  return _private->data.mediums.rename(index, (name != nullptr) ? name : "");
}

void SceneRepresentation::update_medium_bounds() {
  _private->update_medium_bounds();
}

void SceneRepresentation::update_active_camera() {
  auto it = std::find_if(_private->data.cameras.begin(), _private->data.cameras.end(), [](const auto& e) {
    return e.active;
  });
  if (it != _private->data.cameras.end()) {
    _private->active_camera = it->cam;
  }
}

void SceneRepresentation::store_active_camera() {
  auto it = std::find_if(_private->data.cameras.begin(), _private->data.cameras.end(), [](const auto& e) {
    return e.active;
  });
  if (it != _private->data.cameras.end()) {
    it->cam = _private->active_camera;
  }
}

std::string SceneRepresentation::rename_mesh(uint32_t index, const char* name) {
  return rename_entry(_private->data.mesh_mapping, index, name, "mesh-");
}

void SceneRepresentation::set_mesh_material(uint32_t mesh_index, uint32_t material_index) {
  _private->set_mesh_material_impl(mesh_index, material_index);
}

Camera& SceneRepresentation::camera() {
  return _private->active_camera;
}

const Camera& SceneRepresentation::camera() const {
  return _private->active_camera;
}

const SceneRepresentation::IntegratorData& SceneRepresentation::integrator_data() const {
  return _private->integrator_data;
}

void SceneRepresentation::set_integrator_data(const IntegratorData& integrator_data) {
  _private->integrator_data = integrator_data;
}

bool SceneRepresentation::valid() const {
  return true;
}

uint32_t SceneRepresentation::add_environment_emitter(const float3& color, uint32_t medium_index) {
  uint32_t profile_index = uint32_t(_private->data.emitter_profiles.size());

  auto& e = _private->data.emitter_profiles.emplace_back(EmitterProfile::Class::Environment);
  e.emission.spectrum_index = _private->data.add_spectrum(SpectralDistribution::rgb_luminance(color));

  constexpr uint2 kUniformEnvImageDimensions = uint2{1u, 1u};
  constexpr float4 white_color = {1.0f, 1.0f, 1.0f, 1.0f};

  std::vector<float4> uniform_image_data(1, white_color);
  uint32_t image_options = Image::BuildSamplingTable | Image::RepeatU;
  e.emission.image_index = _private->data.add_image(uniform_image_data.data(), kUniformEnvImageDimensions, image_options, {}, {1.0f, 1.0f});
  e.medium_index = medium_index;
  return profile_index;
}

uint32_t SceneRepresentation::add_directional_emitter(const float3& direction, const float3& color, float angular_diameter_degrees, uint32_t medium_index) {
  uint32_t profile_index = uint32_t(_private->data.emitter_profiles.size());

  auto& e = _private->data.emitter_profiles.emplace_back(EmitterProfile::Class::Directional);
  e.emission.spectrum_index = _private->data.add_spectrum(SpectralDistribution::rgb_luminance(color));
  e.emission.image_index = kInvalidIndex;
  e.directional.direction = normalize(direction);
  e.directional.angular_size = angular_diameter_degrees * kPi / 180.0f;
  e.medium_index = medium_index;

  return profile_index;
}

void SceneRepresentation::create_area_emitters_from_materials() {
  _private->create_area_emitters_from_materials();
}

bool SceneRepresentation::delete_emitter(uint32_t emitter_index) {
  return _private->delete_emitter(emitter_index);
}

void SceneRepresentation::add_atmosphere_emitter(const AtmosphereEmitterParameters& params) {
  _private->add_atmosphere_emitter(params);
}

void SceneRepresentation::rebuild_atmosphere_emitter(uint32_t emitter_index) {
  _private->rebuild_atmosphere_emitter(emitter_index);
}

void SceneRepresentation::set_scattering_rhi(RHIContext& rhi) {
  _private->set_scattering_rhi(rhi);
}

void SceneRepresentationImpl::set_scattering_rhi(RHIContext& rhi_context) {
  if ((rhi == &rhi_context) && scattering_gpu_ready) {
    return;
  }

  if ((rhi != nullptr) && scattering_gpu.initialized) {
    scattering::gpu_cleanup(*rhi, scattering_gpu);
  }

  rhi = &rhi_context;
  scattering_gpu = {};
  scattering_gpu_ready = false;
}

bool SceneRepresentationImpl::ensure_scattering_gpu_context() {
  if (rhi == nullptr) {
    log::error("SceneRepresentation atmosphere generation requires an RHI context. Call SceneRepresentation::set_scattering_rhi() first.");
    return false;
  }

  if (scattering_gpu_ready) {
    return true;
  }

  if ((scattering_gpu.initialized == false) && (scattering::gpu_init(*rhi, scattering_gpu) == false)) {
    log::error("Failed to initialize GPU atmosphere scattering context");
    return false;
  }

  if (scattering::gpu_precompute_optical_depth(*rhi, scattering_gpu) == false) {
    log::error("Failed to precompute GPU atmosphere optical depth");
    if (scattering_gpu.initialized) {
      scattering::gpu_cleanup(*rhi, scattering_gpu);
    }
    scattering_gpu = {};
    return false;
  }

  scattering_gpu_ready = true;
  return true;
}

void SceneRepresentationImpl::add_atmosphere_emitter(const AtmosphereEmitterParameters& params) {
  uint32_t emitter_index = data.add_atmosphere_emitter(params);
  if (ensure_scattering_gpu_context() == false) {
    return;
  }

  ETX_ASSERT(rhi != nullptr);
  data.build_atmosphere_and_sun_images(emitter_index, *rhi, scattering_gpu);
}

void SceneRepresentationImpl::rebuild_atmosphere_emitter(uint32_t emitter_index) {
  if (ensure_scattering_gpu_context() == false) {
    return;
  }

  ETX_ASSERT(rhi != nullptr);
  data.rebuild_atmosphere_emitter(emitter_index, *rhi, scattering_gpu);
}

template <class T>
inline void get_values(const std::vector<T>& a, T* ptr, uint64_t count) {
  for (uint64_t i = 0, e = a.size() < count ? a.size() : count; i < e; ++i) {
    *ptr++ = a[i];
  }
}

bool SceneRepresentation::load_from_file(const char* filename, uint32_t options, IntegratorData* out_integrator) {
  IntegratorData parsed_integrator_data = {};
  IntegratorData* integrator_data = out_integrator;
  if (integrator_data == nullptr) {
    integrator_data = &parsed_integrator_data;
  } else {
    *integrator_data = {};
  }

  char base_folder[2048] = {};
  get_file_folder(filename, base_folder, sizeof(base_folder));

  _private->cleanup();
  _private->data.json_file_name = {};
  _private->data.materials_file_name = {};
  _private->data.geometry_file_name = filename;
  _private->active_camera.lens_radius = 0.0f;
  _private->active_camera.focal_distance = 0.0f;
  _private->active_camera.lens_image = kInvalidIndex;
  _private->active_camera.medium_index = kInvalidIndex;
  _private->active_camera.up = kWorldUp;

  Camera default_camera = {};
  default_camera.lens_image = kInvalidIndex;
  default_camera.medium_index = kInvalidIndex;
  default_camera.up = kWorldUp;
  default_camera.cls = Camera::Class::Perspective;

  float3 camera_target = default_camera.position + default_camera.direction;
  bool has_target = false;
  bool has_direction = false;
  float camera_focal_len = 50.0f;
  float camera_fov = focal_length_to_fov(camera_focal_len) * 180.0f / kPi;
  bool use_focal_len = false;
  bool force_tangents = false;
  bool spectral_scene = false;

  const bool raw_model_file = (strcmp(get_file_ext(filename), ".json") != 0);

  if (raw_model_file == false) {
    std::string json_content;
    if (auto f = fopen(filename, "rb")) {
      size_t file_size = get_file_size(f);
      if (file_size > 0) {
        json_content.resize(file_size);
        size_t read_bytes = fread(json_content.data(), 1, json_content.size(), f);
        json_content.resize(read_bytes);
      }
      fclose(f);
    }

    nlohmann::json js = nlohmann::json::parse(json_content, nullptr, false);
    bool parsed = js.is_discarded() == false;
    bool has_bsdfs = parsed && js.is_object() && js.contains("bsdfs");
    bool is_tungsten = parsed && js.is_object() && has_bsdfs && (js.contains("primitives") || js.contains("renderer"));
    bool is_native = parsed && js.is_object() && (has_bsdfs == false) && (js.contains("geometry") || js.contains("materials") || js.contains("integrator"));

    if (is_tungsten && (is_native == false)) {
      if (js.contains("integrator") && js["integrator"].is_object()) {
        const auto& itg = js["integrator"];
        auto map_integrator = [](const std::string& s) {
          if (s == "bidirectional_path_tracer")
            return Integrator::Type::Bidirectional;
          if ((s == "vcm") || (s == "progressive_photon_map"))
            return Integrator::Type::VCM;
          if (s == "debug")
            return Integrator::Type::Debug;
          return Integrator::Type::PathTracing;
        };
        if (integrator_data != nullptr) {
          std::string t = itg.value("type", "");
          integrator_data->selected = map_integrator(t);
        }
        if (itg.contains("min_bounces") && itg["min_bounces"].is_number_integer()) {
          _private->data.options.min_path_length = static_cast<uint32_t>(std::max<int64_t>(0, itg["min_bounces"].get<int64_t>()));
        }
        if (itg.contains("max_bounces") && itg["max_bounces"].is_number_integer()) {
          _private->data.options.max_path_length = static_cast<uint32_t>(std::max<int64_t>(0, itg["max_bounces"].get<int64_t>()));
        }
      }

      if (js.contains("renderer") && js["renderer"].is_object()) {
        const auto& rnd = js["renderer"];
        if (rnd.contains("spp") && rnd["spp"].is_number_integer()) {
          _private->data.options.samples = static_cast<uint32_t>(std::max<int64_t>(1, rnd["spp"].get<int64_t>()));
        }
      }

      uint32_t load_result = load_from_tungsten_file(filename, _private->data, _private->ior_database, _private->scheduler, _private->active_camera);
      if ((load_result & SceneLoadSucceeded) == 0)
        return false;
      return _private->finalize_scene_loading(options, base_folder, load_result, camera_fov, use_focal_len, camera_focal_len, force_tangents, spectral_scene, false);
    }

    if (parsed == false) {
      log::error("Failed to parse JSON scene %s", filename);
      return false;
    }

    if (is_native) {
      _private->data.geometry_file_name.clear();
    }

    for (auto i = js.begin(), e = js.end(); i != e; ++i) {
      const auto& key = i.key();
      const auto& obj = i.value();
      std::string str_value = {};
      float float_value = 0.0f;
      int64_t int_value = 0;
      bool bool_value = false;
      if (json_get_int(i, "samples", int_value)) {
        _private->data.options.samples = static_cast<uint32_t>(max(int64_t(1), int_value));
      } else if (json_get_int(i, "random-termination-start", int_value)) {
        _private->data.options.random_path_termination = static_cast<uint32_t>(max(int64_t(1), int_value));
      } else if (json_get_int(i, "max-path-length", int_value)) {
        _private->data.options.max_path_length = static_cast<uint32_t>(max(int64_t(1), int_value));
      } else if (json_get_int(i, "min-path-length", int_value)) {
        _private->data.options.min_path_length = static_cast<uint32_t>(max(int64_t(1), int_value));
      } else if (json_get_string(i, "geometry", str_value)) {
        _private->data.geometry_file_name = std::string(base_folder) + str_value;
      } else if (json_get_string(i, "materials", str_value)) {
        _private->data.materials_file_name = std::string(base_folder) + str_value;
      } else if (json_get_bool(i, "spectral", bool_value)) {
        spectral_scene = bool_value;
      } else if (json_get_bool(i, "energy_compensated_specular", bool_value)) {
        (void)bool_value;
      } else if (json_get_bool(i, "multiple_importance_sampling", bool_value)) {
        _private->data.options.properties[Scene::Properties::MultipleImportanceSampling] = bool_value;
      } else if (json_get_bool(i, "blue_noise", bool_value)) {
        _private->data.options.properties[Scene::Properties::BlueNoise] = bool_value;
      } else if (json_get_string(i, "light_sampling", str_value)) {
        if (str_value == "uniform") {
          _private->data.options.light_sampling = Scene::LightSampling::Uniform;
        } else if (str_value == "from_distribution") {
          _private->data.options.light_sampling = Scene::LightSampling::FromDistribution;
        } else if (str_value == "ris_uniform") {
          _private->data.options.light_sampling = Scene::LightSampling::RIS_Uniform;
        } else if (str_value == "ris_from_distribution") {
          _private->data.options.light_sampling = Scene::LightSampling::RIS_FromDistribution;
        }
      } else if (key == "strategies" && obj.is_object()) {
        uint32_t strategy_flags = Scene::Strategy::Default;
        for (auto strat_it = obj.begin(); strat_it != obj.end(); ++strat_it) {
          const std::string& strat_key = strat_it.key();
          if (strat_it.value().is_boolean() == false) {
            continue;
          }
          bool strat_value = strat_it.value().get<bool>();
          if (strat_key == "direct_hit") {
            strategy_flags = (strategy_flags & (~Scene::Strategy::DirectHit)) | (strat_value ? Scene::Strategy::DirectHit : 0u);
          } else if (strat_key == "next_event_estimation") {
            strategy_flags = (strategy_flags & (~Scene::Strategy::ConnectToLight)) | (strat_value ? Scene::Strategy::ConnectToLight : 0u);
          } else if (strat_key == "connect_to_light") {
            strategy_flags = (strategy_flags & (~Scene::Strategy::ConnectToLight)) | (strat_value ? Scene::Strategy::ConnectToLight : 0u);
          } else if (strat_key == "connect_to_camera") {
            strategy_flags = (strategy_flags & (~Scene::Strategy::ConnectToCamera)) | (strat_value ? Scene::Strategy::ConnectToCamera : 0u);
          } else if (strat_key == "connect_vertices") {
            strategy_flags = (strategy_flags & (~Scene::Strategy::ConnectVertices)) | (strat_value ? Scene::Strategy::ConnectVertices : 0u);
          } else if (strat_key == "merge_vertices") {
            strategy_flags = (strategy_flags & (~Scene::Strategy::MergeVertices)) | (strat_value ? Scene::Strategy::MergeVertices : 0u);
          } else if (strat_key == "multiple_importance_sampling") {
            _private->data.options.properties[Scene::Properties::MultipleImportanceSampling] = strat_value;
          } else if (strat_key == "blue_noise") {
            _private->data.options.properties[Scene::Properties::BlueNoise] = strat_value;
          }
        }
        _private->data.options.strategy_flags = strategy_flags;
      } else if (json_get_bool(i, "force-tangents", bool_value)) {
        force_tangents = bool_value;
      } else if ((key == "camera") && obj.is_object()) {
        for (auto ci = obj.begin(), ce = obj.end(); ci != ce; ++ci) {
          const auto& ckey = ci.key();
          const auto& cobj = ci.value();
          if (json_get_string(ci, "class", str_value)) {
            default_camera.cls = str_value == "eq" ? Camera::Class::Equirectangular : Camera::Class::Perspective;
          } else if (json_get_float(ci, "fov", float_value)) {
            camera_fov = float_value;
          } else if (json_get_float(ci, "focal-length", float_value)) {
            camera_focal_len = float_value;
            use_focal_len = true;
          } else if (json_get_float(ci, "lens-radius", float_value)) {
            default_camera.lens_radius = float_value;
          } else if (json_get_float(ci, "focal-distance", float_value)) {
            default_camera.focal_distance = float_value;
          } else if (json_get_float(ci, "clip-near", float_value)) {
            default_camera.clip_near = float_value;
          } else if (json_get_float(ci, "clip-far", float_value)) {
            default_camera.clip_far = float_value;
          } else if (cobj.is_array()) {
            if (ckey == "origin") {
              auto values = cobj.get<std::vector<float>>();
              get_values(values, &default_camera.position.x, 3llu);
            } else if (ckey == "target") {
              auto values = cobj.get<std::vector<float>>();
              get_values(values, &camera_target.x, 3llu);
              has_target = true;
            } else if (ckey == "direction") {
              auto values = cobj.get<std::vector<float>>();
              get_values(values, &default_camera.direction.x, 3llu);
              has_direction = true;
            } else if (ckey == "up") {
              auto values = cobj.get<std::vector<float>>();
              get_values(values, &default_camera.up.x, 3llu);
            } else if (ckey == "viewport") {
              auto values = cobj.get<std::vector<uint32_t>>();
              get_values(values, &default_camera.film_size.x, 2llu);
            } else {
              log::warning("Unhandled value in camera description : %s", key.c_str());
            }
          }
        }

        if (has_direction) {
          default_camera.direction = normalize(default_camera.direction);
        } else if (has_target) {
          default_camera.direction = normalize(camera_target - default_camera.position);
        } else {
          default_camera.direction = kWorldForward;
        }
      } else if ((key == "integrator") && obj.is_object()) {
        if (integrator_data != nullptr) {
          std::string selected_id_str;
          if (obj.contains("selected") && obj["selected"].is_string()) {
            selected_id_str = obj["selected"].get<std::string>();
            integrator_data->selected = legacy_integrator_selection_to_type(selected_id_str);
          }
          if (integrator_data->selected == Integrator::Type::Invalid) {
            if (obj.contains("type") && obj["type"].is_string()) {
              integrator_data->selected = legacy_integrator_selection_to_type(obj["type"].get<std::string>());
            } else if (obj.contains("name") && obj["name"].is_string()) {
              std::string name = obj["name"].get<std::string>();
              if (name.find("Path Tracing") != std::string::npos) {
                integrator_data->selected = Integrator::Type::PathTracing;
              } else if (name.find("Bidirectional") != std::string::npos) {
                integrator_data->selected = Integrator::Type::Bidirectional;
              } else if (name.find("Distilled") != std::string::npos) {
                integrator_data->selected = Integrator::Type::Bidirectional;
              } else if (name.find("VCM") != std::string::npos) {
                integrator_data->selected = Integrator::Type::VCM;
              } else if (name.find("Debug") != std::string::npos) {
                integrator_data->selected = Integrator::Type::Debug;
              }
            }
          }

          if (obj.contains("settings") && obj["settings"].is_object()) {
            const auto& settings_obj = obj["settings"];
            for (auto it = settings_obj.begin(); it != settings_obj.end(); ++it) {
              const std::string& type_id = it.key();
              const auto& options_array = it.value();

              Integrator::Type type = integrator_id_to_type(type_id.c_str());
              if (type == Integrator::Type::Invalid)
                continue;

              if (options_array.is_array()) {
                Options options;
                if (options.deserialize_from_json(options_array)) {
                  integrator_data->settings[type] = std::move(options);
                }
              }
            }
          }

          if ((integrator_data->selected != Integrator::Type::Invalid) && obj.contains("options") && obj["options"].is_array()) {
            Options options;
            if (options.deserialize_from_json(obj["options"])) {
              integrator_data->settings[integrator_data->selected] = std::move(options);
            }
          }
        }
      } else {
        log::warning("Unhandled value in scene description : %s", key.c_str());
      }
    }
    _private->data.json_file_name = filename;
  }

  _private->integrator_data = *integrator_data;

  uint32_t load_result = SceneLoadFailed;

  const char* materials_file_name = _private->data.materials_file_name.c_str();
  if (_private->data.geometry_file_name.empty()) {
    if ((materials_file_name == nullptr) || (materials_file_name[0] == 0)) {
      log::error("Scene %s does not provide geometry or materials", filename);
      return false;
    }

    char materials_base_dir[2048] = {};
    get_file_folder(materials_file_name, materials_base_dir, sizeof(materials_base_dir));
    SceneSerialization loader;
    if (loader.parse_materials_file(materials_file_name, materials_base_dir, _private->data, _private->ior_database, _private->scheduler) == false) {
      log::error("Failed to load materials from %s", materials_file_name);
      return false;
    }

    load_result = _private->data.triangles.empty() ? SceneLoadFailed : SceneLoadSucceeded;
  } else {
    const char* geometry_file_name = _private->data.geometry_file_name.c_str();
    auto ext = get_file_ext(geometry_file_name);
    if (strcmp(ext, ".etx") == 0) {
      SceneSerialization loader;
      if (loader.load_from_file(geometry_file_name, _private->data, materials_file_name, _private->ior_database, _private->scheduler) == false) {
        log::error("Failed to load ETX file from %s", geometry_file_name);
        return false;
      }
      load_result = SceneLoadSucceeded;
    } else if (strcmp(ext, ".obj") == 0) {
      load_result = load_from_obj_file(geometry_file_name, materials_file_name, _private->data, _private->ior_database, _private->scheduler);
    } else if (strcmp(ext, ".gltf") == 0) {
      load_result = load_from_gltf_file(geometry_file_name, false, _private->data, _private->scheduler, _private->active_camera);
    } else if (strcmp(ext, ".glb") == 0) {
      load_result = load_from_gltf_file(geometry_file_name, true, _private->data, _private->scheduler, _private->active_camera);
    }
  }

  if ((load_result & SceneLoadSucceeded) == 0) {
    return false;
  }

  const bool setup_camera = (options & SceneRepresentation::SetupCamera) != 0u;
  const bool create_default_camera_entry = ((raw_model_file && setup_camera) && _private->data.cameras.empty() && ((load_result & SceneLoadCameraInfo) == 0));

  if ((raw_model_file && setup_camera) && (scene_has_environment_emitter(_private->data) == false)) {
    add_default_raw_model_lighting(_private->data);
  }

  if (has_target || has_direction || default_camera.film_size.x > 0 || default_camera.lens_radius > 0.0f) {
    if (use_focal_len) {
      camera_fov = focal_length_to_fov(camera_focal_len) * 180.0f / kPi;
    }

    if (default_camera.film_size.x * default_camera.film_size.y == 0) {
      default_camera.film_size = {1280, 720};
    }

    auto& entry = _private->data.cameras.emplace_back();
    entry.id = "default";
    entry.active = _private->data.cameras.size() == 1;

    build_camera(entry.cam, default_camera.position, default_camera.direction, default_camera.up, default_camera.film_size, camera_fov);

    entry.cam.cls = default_camera.cls;
    entry.cam.lens_radius = default_camera.lens_radius;
    entry.cam.focal_distance = default_camera.focal_distance;
    entry.cam.clip_near = default_camera.clip_near;
    entry.cam.clip_far = default_camera.clip_far;
  }

  return _private->finalize_scene_loading(options, base_folder, load_result, camera_fov, use_focal_len, camera_focal_len, force_tangents, spectral_scene, create_default_camera_entry);
}

void SceneRepresentationImpl::update_medium_bounds() {
  if (data.triangles.empty() || data.vertices.pos.empty()) {
    return;
  }

  std::unordered_map<uint32_t, std::pair<float3, float3>> medium_bounds_map;

  for (const auto& tri : data.triangles) {
    if (tri.material_index >= data.materials.size()) {
      continue;
    }

    const auto& material = data.materials[tri.material_index];
    const float3& v0 = data.vertices.pos[tri.i[0]];
    const float3& v1 = data.vertices.pos[tri.i[1]];
    const float3& v2 = data.vertices.pos[tri.i[2]];

    float3 tri_min = min(min(v0, v1), v2);
    float3 tri_max = max(max(v0, v1), v2);

    if (material.int_medium != kInvalidIndex) {
      auto& bounds = medium_bounds_map[material.int_medium];
      if (bounds.first.x == kMaxFloat) {
        bounds.first = tri_min;
        bounds.second = tri_max;
      } else {
        bounds.first = min(bounds.first, tri_min);
        bounds.second = max(bounds.second, tri_max);
      }
    }

    if (material.ext_medium != kInvalidIndex) {
      auto& bounds = medium_bounds_map[material.ext_medium];
      if (bounds.first.x == kMaxFloat) {
        bounds.first = tri_min;
        bounds.second = tri_max;
      } else {
        bounds.first = min(bounds.first, tri_min);
        bounds.second = max(bounds.second, tri_max);
      }
    }
  }

  for (const auto& [medium_index, bounds_pair] : medium_bounds_map) {
    if (medium_index < data.mediums.array_size()) {
      Medium& medium = data.mediums.get(medium_index);
      medium.bounds = {bounds_pair.first, 0.0f, bounds_pair.second, 0.0f};
    }
  }
}

void SceneRepresentationImpl::set_mesh_material_impl(uint32_t mesh_index, uint32_t material_index) {
  if (mesh_index >= data.meshes.size())
    return;

  const Mesh& mesh = data.meshes[mesh_index];
  for (uint32_t i = 0; i < mesh.triangle_count; ++i) {
    uint32_t triangle_index = mesh.triangle_offset + i;
    if (triangle_index < data.triangles.size()) {
      data.triangles[triangle_index].material_index = material_index;
    }
  }
}

std::string SceneRepresentation::save_to_file(const char* filename, Integrator::Type selected_type, Integrator* integrator_array[], size_t integrator_count) {
  auto save_start = std::chrono::high_resolution_clock::now();

  auto impl = _private;

  std::string base_file = {};
  if ((filename != nullptr) && (filename[0] != 0)) {
    base_file = filename;
  } else if (impl->data.json_file_name.empty() == false) {
    base_file = impl->data.json_file_name;
  } else if (impl->data.geometry_file_name.empty() == false) {
    base_file = impl->data.geometry_file_name;
  }

  if (base_file.empty()) {
    log::error("Unable to determine base file for saving scene");
    return {};
  }

  std::filesystem::path base_path = std::filesystem::path(base_file).lexically_normal();
  std::filesystem::path base_dir = base_path.has_parent_path() ? base_path.parent_path() : std::filesystem::current_path();

  auto strip_extension = [](std::string& name, const char* ext) {
    size_t ext_length = std::strlen(ext);
    if ((name.size() >= ext_length) && (name.compare(name.size() - ext_length, ext_length, ext) == 0)) {
      name.resize(name.size() - ext_length);
      return true;
    }
    return false;
  };

  std::string base_name = base_path.filename().string();
  bool keep_stripping = true;
  while (keep_stripping) {
    keep_stripping = false;
    if (strip_extension(base_name, ".json")) {
      keep_stripping = true;
    }
    if (strip_extension(base_name, ".etx")) {
      keep_stripping = true;
    }
    if (strip_extension(base_name, ".obj")) {
      keep_stripping = true;
    }
    if (strip_extension(base_name, ".gltf")) {
      keep_stripping = true;
    }
    if (strip_extension(base_name, ".glb")) {
      keep_stripping = true;
    }
  }

  if (base_name.empty()) {
    base_name = "scene";
  }

  std::filesystem::path json_path = (base_dir / (base_name + ".etx.json")).lexically_normal();
  std::filesystem::path materials_path = (base_dir / (base_name + ".etx.materials")).lexically_normal();

  auto to_relative = [](const std::filesystem::path& target, const std::filesystem::path& base_folder) {
    std::error_code ec = {};
    auto relative_path = std::filesystem::relative(target, base_folder, ec);
    if (ec.value() == 0) {
      std::string result = relative_path.generic_string();
      if (result.empty()) {
        result = target.filename().generic_string();
      }
      return result;
    }

    return target.generic_string();
  };

  std::filesystem::path geometry_path = base_dir / (base_name + ".etx");
  std::string geometry_ref = to_relative(geometry_path, json_path.parent_path());
  std::string materials_ref = to_relative(materials_path, json_path.parent_path());

  auto geometry_export_start = std::chrono::high_resolution_clock::now();
  SceneSerialization archive;
  if (archive.save_to_file(impl->data, geometry_path) == false) {
    log::error("Failed to export geometry to %s", geometry_path.string().c_str());
    return {};
  }
  auto geometry_export_end = std::chrono::high_resolution_clock::now();
  auto geometry_export_duration = std::chrono::duration_cast<std::chrono::milliseconds>(geometry_export_end - geometry_export_start);
  log::info("Geometry export: %lld ms", geometry_export_duration.count());

  nlohmann::json js = nlohmann::json::object();
  js["samples"] = impl->data.options.samples;
  js["random-termination-start"] = impl->data.options.random_path_termination;
  js["max-path-length"] = impl->data.options.max_path_length;
  js["min-path-length"] = impl->data.options.min_path_length;
  js["geometry"] = geometry_ref;
  if (materials_ref.empty() == false) {
    js["materials"] = materials_ref;
  }
  js["spectral"] = impl->data.options.properties[Scene::Properties::Spectral];
  js["multiple_importance_sampling"] = impl->data.options.properties[Scene::Properties::MultipleImportanceSampling];
  js["blue_noise"] = impl->data.options.properties[Scene::Properties::BlueNoise];

  switch (impl->data.options.light_sampling) {
    case Scene::LightSampling::Uniform:
      js["light_sampling"] = "uniform";
      break;
    case Scene::LightSampling::FromDistribution:
      js["light_sampling"] = "from_distribution";
      break;
    case Scene::LightSampling::RIS_Uniform:
      js["light_sampling"] = "ris_uniform";
      break;
    case Scene::LightSampling::RIS_FromDistribution:
      js["light_sampling"] = "ris_from_distribution";
      break;
    default:
      js["light_sampling"] = "ris_from_distribution";
      break;
  }

  nlohmann::json strategies = nlohmann::json::object();
  strategies["direct_hit"] = ((impl->data.options.strategy_flags & Scene::Strategy::DirectHit) == Scene::Strategy::DirectHit);
  strategies["connect_to_light"] = ((impl->data.options.strategy_flags & Scene::Strategy::ConnectToLight) == Scene::Strategy::ConnectToLight);
  strategies["connect_to_camera"] = ((impl->data.options.strategy_flags & Scene::Strategy::ConnectToCamera) == Scene::Strategy::ConnectToCamera);
  strategies["connect_vertices"] = ((impl->data.options.strategy_flags & Scene::Strategy::ConnectVertices) == Scene::Strategy::ConnectVertices);
  strategies["merge_vertices"] = ((impl->data.options.strategy_flags & Scene::Strategy::MergeVertices) == Scene::Strategy::MergeVertices);
  js["strategies"] = strategies;

  if (selected_type != Integrator::Type::Invalid && integrator_array != nullptr && integrator_count > 0) {
    nlohmann::json integrator_json;

    const char* selected_id = integrator_type_to_id(selected_type);
    if (selected_id != nullptr) {
      integrator_json["selected"] = selected_id;
    }

    nlohmann::json settings_json = nlohmann::json::object();

    for (size_t i = 0; i < integrator_count; ++i) {
      Integrator* integrator = integrator_array[i];
      if (integrator == nullptr)
        continue;

      Integrator::Type type = integrator_to_type(integrator);
      if (type == Integrator::Type::Invalid)
        continue;

      const char* type_id = integrator_type_to_id(type);
      if (type_id == nullptr)
        continue;

      nlohmann::json options_json;
      integrator->options().serialize_to_json(options_json);

      if (options_json.is_array() && options_json.size() > 0) {
        settings_json[type_id] = options_json;
      }
    }

    if (settings_json.empty() == false) {
      integrator_json["settings"] = settings_json;
    }

    if (integrator_json.empty() == false) {
      js["integrator"] = integrator_json;
    }
  }

  auto json_write_start = std::chrono::high_resolution_clock::now();
  json_to_file(js, json_path.string().c_str());
  auto json_write_end = std::chrono::high_resolution_clock::now();
  auto json_write_duration = std::chrono::duration_cast<std::chrono::milliseconds>(json_write_end - json_write_start);
  log::info("JSON config write: %lld ms", json_write_duration.count());

  auto sanitize_name = [](const std::string& value) {
    std::string result = value;
    for (char& ch : result) {
      if (std::isalnum(static_cast<unsigned char>(ch)) == 0) {
        ch = '_';
      }
    }
    return result;
  };

  std::vector<std::pair<std::string, uint32_t>> medium_entries;
  medium_entries.reserve(impl->data.mediums.mapping().size());
  for (const auto& entry : impl->data.mediums.mapping()) {
    medium_entries.emplace_back(entry.first, entry.second);
  }
  std::sort(medium_entries.begin(), medium_entries.end(), [](const auto& a, const auto& b) {
    return a.first < b.first;
  });

  std::unordered_map<uint32_t, std::string> medium_names;
  medium_names.reserve(medium_entries.size());
  for (const auto& entry : medium_entries) {
    medium_names[entry.second] = entry.first;
  }

  auto spectrum_rgb = [&](uint32_t index) -> float3 {
    if ((index == kInvalidIndex) || (index >= impl->data.spectrum_values.size())) {
      return {0.0f, 0.0f, 0.0f};
    }
    return impl->data.spectrum_values[index].integrated();
  };

  auto spectrum_scalar = [&](uint32_t index, float fallback) -> float {
    if ((index == kInvalidIndex) || (index >= impl->data.spectrum_values.size())) {
      return fallback;
    }
    float3 rgb = impl->data.spectrum_values[index].integrated();
    return (rgb.x + rgb.y + rgb.z) / 3.0f;
  };

  auto spectrum_by_index = [&](uint32_t index) -> const SpectralDistribution& {
    static const SpectralDistribution null_spectrum = SpectralDistribution::constant(0.0f);
    if ((index == kInvalidIndex) || (index >= impl->data.spectrum_values.size())) {
      return null_spectrum;
    }
    return impl->data.spectrum_values[index];
  };

  auto texture_path = [&](uint32_t image_index) -> std::string {
    if (image_index == kInvalidIndex) {
      return {};
    }
    std::string stored = impl->data.images.path(image_index);
    if (stored.empty() || stored.compare(0, 5, "##mem") == 0) {
      return {};
    }
    std::filesystem::path tex_path = std::filesystem::path(stored).lexically_normal();
    return to_relative(tex_path, materials_path.parent_path());
  };

  auto write_texture_line = [&](std::ostringstream& stream, const char* label, uint32_t image_index, uint32_t channel) {
    std::string path = texture_path(image_index);
    if (path.empty() == false) {
      stream << label << " " << path;
      if (channel != kInvalidIndex) {
        stream << " channel " << channel;
      }
      stream << "\n";
    }
  };

  auto write_spectrum_line = [&](std::ostringstream& stream, const char* label, uint32_t index, bool use_gamma) {
    if ((index == kInvalidIndex) || (index >= impl->data.spectrum_values.size())) {
      return;
    }
    float3 value = spectrum_rgb(index);
    if (use_gamma) {
      value = linear_to_gamma(value);
    }
    stream << label << " " << value.x << " " << value.y << " " << value.z << "\n";
  };

  std::ostringstream materials_stream;
  materials_stream.setf(std::ios::fixed, std::ios::floatfield);
  materials_stream << std::setprecision(6);

  const Camera& camera = impl->active_camera;
  const IORDatabase& database = impl->ior_database;

  if (camera.film_size.x > 0u) {
    float3 target = camera.position + camera.direction;
    materials_stream << "newmtl et::camera\n";
    materials_stream << "class " << ((camera.cls == Camera::Class::Equirectangular) ? "eq" : "perspective") << "\n";
    materials_stream << "viewport " << camera.film_size.x << " " << camera.film_size.y << "\n";
    materials_stream << "origin " << camera.position.x << " " << camera.position.y << " " << camera.position.z << "\n";
    materials_stream << "target " << target.x << " " << target.y << " " << target.z << "\n";
    materials_stream << "up " << camera.up.x << " " << camera.up.y << " " << camera.up.z << "\n";
    materials_stream << "fov " << get_camera_fov(camera) << "\n";
    float fov_from_focal = focal_length_to_fov(get_camera_focal_length(camera)) * 180.0f / kPi;
    if (std::fabs(fov_from_focal - get_camera_fov(camera)) > 0.01f) {
      materials_stream << "focal-length " << get_camera_focal_length(camera) << "\n";
    }
    if (camera.lens_radius > 0.0f) {
      materials_stream << "lens-radius " << camera.lens_radius << "\n";
    }
    if (camera.focal_distance > 0.0f) {
      materials_stream << "focal-distance " << camera.focal_distance << "\n";
    }
    if (camera.clip_near != 0.1f) {
      materials_stream << "clip-near " << camera.clip_near << "\n";
    }
    if (camera.clip_far != 1000.0f) {
      materials_stream << "clip-far " << camera.clip_far << "\n";
    }
    bool camera_medium_valid = (camera.medium_index != kInvalidIndex) && (medium_names.count(camera.medium_index) > 0);
    if (camera_medium_valid) {
      materials_stream << "ext_medium " << medium_names[camera.medium_index] << "\n";
    }
    std::string camera_id = {};
    for (const auto& stored : impl->data.cameras) {
      if (stored.active) {
        camera_id = stored.id;
        break;
      }
    }
    if (camera_id.empty() && (impl->data.cameras.empty() == false)) {
      camera_id = impl->data.cameras.front().id;
    }
    if (camera_id.empty() == false) {
      materials_stream << "id " << camera_id << "\n";
      materials_stream << "active 1\n";
    }
    materials_stream << "\n";
  }

  std::vector<uint32_t> atmosphere_emitter_indices;

  for (uint32_t i = 0; i < impl->data.emitter_profiles.size(); ++i) {
    const auto& profile = impl->data.emitter_profiles[i];
    if ((profile.meta & EmitterProfile::Meta::Atmosphere) && (profile.cls == EmitterProfile::Class::Environment)) {
      atmosphere_emitter_indices.push_back(i);
    }
  }

  for (uint32_t emitter_index : atmosphere_emitter_indices) {
    const auto& env_profile = impl->data.emitter_profiles[emitter_index];
    float3 env_color = spectrum_rgb(env_profile.emission.spectrum_index);
    const auto& scattering = env_profile.atmosphere.scattering;
    materials_stream << "newmtl et::atmosphere\n";
    materials_stream << "anisotropy " << scattering.anisotropy << "\n";
    materials_stream << "altitude " << scattering.altitude << "\n";
    materials_stream << "rayleigh " << scattering.rayleigh_scale << "\n";
    materials_stream << "mie " << scattering.mie_scale << "\n";
    materials_stream << "ozone " << scattering.ozone_scale << "\n";
    if (scattering.primary_scattering == 0u) {
      materials_stream << "primary-scattering 0\n";
    }
    if (scattering.secondary_scattering == 0u) {
      materials_stream << "secondary-scattering 0\n";
    }
    materials_stream << "quality " << env_profile.atmosphere.quality << "\n";
    materials_stream << "color " << env_color.x << " " << env_color.y << " " << env_color.z << "\n";
    materials_stream << "\n";
  }

  for (uint32_t i = 0; i < impl->data.emitter_profiles.size(); ++i) {
    const auto& profile = impl->data.emitter_profiles[i];
    if (profile.cls != EmitterProfile::Class::Environment) {
      continue;
    }
    if (profile.meta & EmitterProfile::Meta::Atmosphere) {
      continue;
    }

    materials_stream << "newmtl et::env\n";
    std::string env_path = texture_path(profile.emission.image_index);
    if (env_path.empty() == false) {
      materials_stream << "image " << env_path << "\n";
    }
    float3 env_color = spectrum_rgb(profile.emission.spectrum_index);
    materials_stream << "color " << env_color.x << " " << env_color.y << " " << env_color.z << "\n";
    float env_rotation_offset = 0.0f;
    float env_scale_u = 1.0f;
    if (profile.emission.image_index != kInvalidIndex) {
      const Image& env_image = impl->data.images.get(profile.emission.image_index);
      env_rotation_offset = env_image.offset.x;
      env_scale_u = env_image.scale.x;
    }
    if (std::fabs(env_rotation_offset) >= kEpsilon) {
      materials_stream << "rotation " << (-env_rotation_offset * 360.0f) << "\n";
    }
    if (std::fabs(env_scale_u - 1.0f) >= kEpsilon) {
      materials_stream << "scale " << env_scale_u << "\n";
    }
    bool env_medium_valid = (profile.medium_index != kInvalidIndex) && (medium_names.count(profile.medium_index) > 0);
    if (env_medium_valid) {
      materials_stream << "ext_medium " << medium_names[profile.medium_index] << "\n";
    }
    materials_stream << "\n";
  }

  for (uint32_t i = 0; i < impl->data.emitter_profiles.size(); ++i) {
    const auto& profile = impl->data.emitter_profiles[i];
    if (profile.cls != EmitterProfile::Class::Directional) {
      continue;
    }

    materials_stream << "newmtl et::dir\n";
    float3 dir_color = spectrum_rgb(profile.emission.spectrum_index);
    materials_stream << "color " << dir_color.x << " " << dir_color.y << " " << dir_color.z << "\n";
    materials_stream << "direction " << profile.directional.direction.x << " " << profile.directional.direction.y << " " << profile.directional.direction.z << "\n";
    if (profile.directional.angular_size >= kEpsilon) {
      materials_stream << "angular_diameter " << (profile.directional.angular_size * 180.0f / kPi) << "\n";
    }
    const bool references_atmosphere = (profile.reference_emitter_index != kInvalidIndex) && (profile.reference_emitter_index < impl->data.emitter_profiles.size()) &&
                                       (impl->data.emitter_profiles[profile.reference_emitter_index].cls == EmitterProfile::Class::Environment) &&
                                       ((impl->data.emitter_profiles[profile.reference_emitter_index].meta & EmitterProfile::Meta::Atmosphere) != 0u);
    if (references_atmosphere) {
      materials_stream << "use_as_sun 1\n";
    }
    std::string dir_path = texture_path(profile.emission.image_index);
    if (dir_path.empty() == false) {
      materials_stream << "image " << dir_path << "\n";
    }
    bool dir_medium_valid = (profile.medium_index != kInvalidIndex) && (medium_names.count(profile.medium_index) > 0);
    if (dir_medium_valid) {
      materials_stream << "ext_medium " << medium_names[profile.medium_index] << "\n";
    }
    materials_stream << "\n";
  }

  for (uint64_t medium_index = 0; medium_index < medium_entries.size(); ++medium_index) {
    uint32_t pool_index = medium_entries[medium_index].second;
    const Medium& medium = impl->data.mediums.get(pool_index);
    materials_stream << "newmtl et::medium\n";
    materials_stream << "id " << medium_entries[medium_index].first << "\n";
    float3 absorption = impl->data.spectrum_values[medium.absorption_index].integrated();
    if ((std::fabs(absorption.x) >= kEpsilon) || (std::fabs(absorption.y) >= kEpsilon) || (std::fabs(absorption.z) >= kEpsilon)) {
      materials_stream << "absorption " << absorption.x << " " << absorption.y << " " << absorption.z << "\n";
    }
    float3 scattering = impl->data.spectrum_values[medium.scattering_index].integrated();
    if ((std::fabs(scattering.x) >= kEpsilon) || (std::fabs(scattering.y) >= kEpsilon) || (std::fabs(scattering.z) >= kEpsilon)) {
      materials_stream << "scattering " << scattering.x << " " << scattering.y << " " << scattering.z << "\n";
    }
    if (std::fabs(medium.phase_function_g) >= kEpsilon) {
      materials_stream << "anisotropy " << medium.phase_function_g << "\n";
    }
    if (medium.enable_explicit_connections == false) {
      materials_stream << "enclosed 1\n";
    }
    if (medium.grid_type_enum() == DensityGrid::Type::NoiseFunction) {
      materials_stream << "noise type " << static_cast<uint32_t>(medium.noise_type_enum()) << " scale " << medium.grid.noise_scale << " octaves " << medium.grid.noise_octaves
                       << " lacunarity " << medium.grid.noise_lacunarity << " persistence " << medium.grid.noise_persistence << " seed " << medium.grid.noise_seed << " power "
                       << medium.grid.noise_power << " sharpness " << medium.grid.noise_sharpness << " offset " << medium.grid.noise_offset.x << " " << medium.grid.noise_offset.y
                       << " " << medium.grid.noise_offset.z << " border_fade " << medium.grid.noise_enable_border_fade << " border_fade_distance "
                       << medium.grid.noise_border_fade_distance << "\n";
    }
    materials_stream << "\n";
  }

  auto is_internal_name = [](const std::string& name) {
    return name.compare(0, 4, "et::") == 0 || name.compare(0, 5, "etx::") == 0;
  };

  std::vector<std::pair<std::string, uint32_t>> material_entries;
  material_entries.reserve(impl->data.material_mapping.size());
  for (const auto& entry : impl->data.material_mapping) {
    material_entries.emplace_back(entry.first, entry.second);
  }
  std::sort(material_entries.begin(), material_entries.end(), [](const auto& a, const auto& b) {
    return a.first < b.first;
  });

  for (const auto& entry : material_entries) {
    const std::string& name = entry.first;
    if (is_internal_name(name)) {
      continue;
    }
    uint32_t index = entry.second;
    if (index >= impl->data.materials.size()) {
      log::warning("Material index %u out of bounds for material %s", index, name.c_str());
      continue;
    }
    const Material& material = impl->data.materials[index];

    materials_stream << "newmtl " << name << "\n";
    materials_stream << "material class " << material_class_to_string(material.cls) << "\n";

    write_spectrum_line(materials_stream, "Kd", material.scattering.spectrum_index, true);
    if ((material.cls == MaterialClass::Dielectric) || (material.cls == MaterialClass::Translucent) ||
        (material.transmission.value.x > kEpsilon)) {
      write_spectrum_line(materials_stream, "Kt", material.scattering.spectrum_index, true);
    }
    write_spectrum_line(materials_stream, "Ks", material.reflectance.spectrum_index, true);

    float rough_u = material.roughness.value.x;
    float rough_v = material.roughness.value.y;
    if ((rough_u >= kEpsilon) || (rough_v >= kEpsilon)) {
      float value_u = std::sqrt(max(0.0f, rough_u));
      float value_v = std::sqrt(max(0.0f, rough_v));
      if (std::fabs(value_u - value_v) < kEpsilon) {
        materials_stream << "Pr " << value_u << "\n";
      } else {
        materials_stream << "Pr " << value_u << " " << value_v << "\n";
      }
    }

    if (material.metalness.value.x >= kEpsilon) {
      materials_stream << "metalness " << material.metalness.value.x << "\n";
    }
    if (material.transmission.value.x >= kEpsilon) {
      materials_stream << "transmission " << material.transmission.value.x << "\n";
    }

    write_texture_line(materials_stream, "map_Kd", material.scattering.image_index, kInvalidIndex);
    write_texture_line(materials_stream, "map_Ks", material.reflectance.image_index, kInvalidIndex);
    write_texture_line(materials_stream, "map_Kt", material.scattering.image_index, kInvalidIndex);
    write_texture_line(materials_stream, "map_Pr", material.roughness.image_index, material.roughness.channel);
    write_texture_line(materials_stream, "map_Ml", material.metalness.image_index, material.metalness.channel);
    write_texture_line(materials_stream, "map_Tm", material.transmission.image_index, material.transmission.channel);

    if ((material.normal_image_index != kInvalidIndex) || (std::fabs(material.normal_scale - 1.0f) >= kEpsilon)) {
      std::string normal_path = texture_path(material.normal_image_index);
      materials_stream << "normalmap";
      if (normal_path.empty() == false) {
        materials_stream << " image " << normal_path;
      }
      materials_stream << " scale " << material.normal_scale << "\n";
    }

    int matched_int_index = -1;
    if (material.int_ior.cls != SpectralDistribution::Invalid) {
      matched_int_index = database.find_matching_index(spectrum_by_index(material.int_ior.eta_index), spectrum_by_index(material.int_ior.k_index), material.int_ior.cls);
    }
    if ((matched_int_index >= 0) && (matched_int_index < static_cast<int>(database.definitions.size()))) {
      const IORDefinition& def = database.definitions[static_cast<size_t>(matched_int_index)];
      materials_stream << "int_ior " << def.name << "\n";
    } else if ((material.int_ior.eta_index != kInvalidIndex) && (material.int_ior.cls != SpectralDistribution::Invalid)) {
      float eta_value = spectrum_scalar(material.int_ior.eta_index, 1.0f);
      if (material.int_ior.cls == SpectralDistribution::Dielectric) {
        materials_stream << "int_ior " << eta_value << "\n";
      } else if (material.int_ior.cls == SpectralDistribution::Conductor) {
        float k_value = spectrum_scalar(material.int_ior.k_index, 0.0f);
        materials_stream << "int_ior " << eta_value << " " << k_value << "\n";
      }
    }

    int matched_ext_index = -1;
    if (material.ext_ior.cls != SpectralDistribution::Invalid) {
      matched_ext_index = database.find_matching_index(spectrum_by_index(material.ext_ior.eta_index), spectrum_by_index(material.ext_ior.k_index), material.ext_ior.cls);
    }
    if ((matched_ext_index >= 0) && (matched_ext_index < static_cast<int>(database.definitions.size()))) {
      const IORDefinition& def = database.definitions[static_cast<size_t>(matched_ext_index)];
      materials_stream << "ext_ior " << def.name << "\n";
    } else {
      float ext_eta_value = spectrum_scalar(material.ext_ior.eta_index, 1.0f);
      if ((material.ext_ior.eta_index != kInvalidIndex) && (material.ext_ior.cls != SpectralDistribution::Invalid) &&
          (material.ext_ior.cls != SpectralDistribution::Dielectric || std::fabs(ext_eta_value - 1.0f) >= kEpsilon)) {
        if (material.ext_ior.cls == SpectralDistribution::Dielectric) {
          materials_stream << "ext_ior " << ext_eta_value << "\n";
        } else if (material.ext_ior.cls == SpectralDistribution::Conductor) {
          float ext_k_value = spectrum_scalar(material.ext_ior.k_index, 0.0f);
          materials_stream << "ext_ior " << ext_eta_value << " " << ext_k_value << "\n";
        }
      } else {
        materials_stream << "ext_ior 1.0\n";
      }
    }

    if (medium_names.count(material.int_medium) > 0u) {
      materials_stream << "int_medium " << medium_names[material.int_medium] << "\n";
    }
    if (medium_names.count(material.ext_medium) > 0u) {
      materials_stream << "ext_medium " << medium_names[material.ext_medium] << "\n";
    }

    if (material.two_sided != 0u) {
      materials_stream << "two_sided 1\n";
    }
    if (std::fabs(material.opacity - 1.0f) >= kEpsilon) {
      materials_stream << "opacity " << material.opacity << "\n";
    }

    bool has_emission_texture = (material.emission.image_index != kInvalidIndex);
    bool has_emission_spectrum = (material.emission.spectrum_index != kInvalidIndex) && (material.emission.spectrum_index < impl->data.spectrum_values.size());
    if (has_emission_texture || has_emission_spectrum) {
      materials_stream << "emitter";
      if (has_emission_texture) {
        std::string emission_path = texture_path(material.emission.image_index);
        if (emission_path.empty() == false) {
          materials_stream << " image " << emission_path;
        }
      }
      if (has_emission_spectrum) {
        float3 emission_value = spectrum_rgb(material.emission.spectrum_index);
        materials_stream << " color " << emission_value.x << " " << emission_value.y << " " << emission_value.z;
      }
      if (material.two_sided != 0u) {
        materials_stream << " twosided";
      }
      if (material.emission_collimation >= kEpsilon) {
        materials_stream << " collimated " << material.emission_collimation;
      }
      materials_stream << "\n";
    }

    if (material.subsurface_cls != SubsurfaceMaterial::Disabled) {
      materials_stream << "subsurface";
      if (material.subsurface_path == SubsurfaceMaterial::RefractedPath) {
        materials_stream << " path refracted";
      }
      float3 subsurface_color = spectrum_rgb(material.subsurface.spectrum_index);
      materials_stream << " distances " << subsurface_color.x << " " << subsurface_color.y << " " << subsurface_color.z;
      materials_stream << "\n";
    }

    if ((material.thinfilm.thinkness_image != kInvalidIndex) || (std::fabs(material.thinfilm.min_thickness) >= kEpsilon) ||
        (std::fabs(material.thinfilm.max_thickness) >= kEpsilon)) {
      materials_stream << "thinfilm";
      std::string thinfilm_path = texture_path(material.thinfilm.thinkness_image);
      if (thinfilm_path.empty() == false) {
        materials_stream << " image " << thinfilm_path;
      }
      materials_stream << " range " << material.thinfilm.min_thickness << " " << material.thinfilm.max_thickness;
      materials_stream << " weight " << clamp(material.thinfilm.weight, 0.0f, 1.0f);
      int matched_thinfilm_index = -1;
      if (material.thinfilm.ior.cls != SpectralDistribution::Invalid) {
        matched_thinfilm_index =
          database.find_matching_index(spectrum_by_index(material.thinfilm.ior.eta_index), spectrum_by_index(material.thinfilm.ior.k_index), material.thinfilm.ior.cls);
      }
      if ((matched_thinfilm_index >= 0) && (matched_thinfilm_index < static_cast<int>(database.definitions.size()))) {
        const IORDefinition& def = database.definitions[static_cast<size_t>(matched_thinfilm_index)];
        materials_stream << " ior " << def.name << "\n";
      } else {
        float thinfilm_eta = spectrum_scalar(material.thinfilm.ior.eta_index, 1.0f);
        materials_stream << " ior " << thinfilm_eta << "\n";
      }
    }

    materials_stream << "\n";
  }

  std::string materials_string = materials_stream.str();

  auto materials_write_start = std::chrono::high_resolution_clock::now();
  FILE* materials_file = fopen(materials_path.string().c_str(), "wb");
  if (materials_file == nullptr) {
    log::error("Failed to open materials file for writing: %s", materials_path.string().c_str());
    return {};
  }
  fwrite(materials_string.data(), 1, materials_string.size(), materials_file);
  fflush(materials_file);
  fclose(materials_file);
  auto materials_write_end = std::chrono::high_resolution_clock::now();
  auto materials_write_duration = std::chrono::duration_cast<std::chrono::milliseconds>(materials_write_end - materials_write_start);
  log::info("Materials file write: %lld ms (%zu bytes)", materials_write_duration.count(), materials_string.size());

  impl->data.json_file_name = json_path.generic_string();
  impl->data.materials_file_name = materials_path.generic_string();

  auto save_end = std::chrono::high_resolution_clock::now();
  auto save_duration = std::chrono::duration_cast<std::chrono::milliseconds>(save_end - save_start);
  log::info("Scene save total: %lld ms", save_duration.count());

  return json_path.generic_string();
}

void SceneRepresentationImpl::generate_pixel_sampler_image() {
  std::vector<float4> sampler_image;
  Film::generate_filter_image(Film::PixelFilterBlackmanHarris, sampler_image);
  uint32_t image_options = Image::BuildSamplingTable | Image::UniformSamplingTable;
  uint32_t image_index = data.images.add_from_data(sampler_image.data(), {Film::PixelFilterSize, Film::PixelFilterSize}, image_options, {}, {1.0f, 1.0f});
  data.pixel_filter = {image_index, 1.5f};
}

void SceneRepresentationImpl::setup_atmosphere_references() {
  for (auto& profile : data.emitter_profiles) {
    if (profile.cls != EmitterProfile::Class::Directional) {
      continue;
    }

    const bool invalid_reference = (profile.reference_emitter_index == kInvalidIndex) || (profile.reference_emitter_index >= data.emitter_profiles.size());
    if (invalid_reference) {
      profile.reference_emitter_index = kInvalidIndex;
      continue;
    }

    const auto& referenced = data.emitter_profiles[profile.reference_emitter_index];
    if ((referenced.cls != EmitterProfile::Class::Environment) || ((referenced.meta & EmitterProfile::Meta::Atmosphere) == 0u)) {
      profile.reference_emitter_index = kInvalidIndex;
    }
  }
}

bool SceneRepresentationImpl::finalize_scene_loading(uint32_t options, const char* base_folder, uint32_t load_result, float camera_fov, bool use_focal_len, float camera_focal_len,
  bool force_tangents, bool spectral_scene, bool create_default_camera_entry) {
  auto& camera = active_camera;
  bool needs_camera_positioning = false;
  data.options.properties[Scene::Properties::Spectral] = spectral_scene;

  if (options & SceneRepresentation::SetupCamera) {
    if (data.cameras.empty()) {
      if ((load_result & SceneLoadCameraInfo) == 0) {
        if (use_focal_len) {
          camera_fov = focal_length_to_fov(camera_focal_len) * 180.0f / kPi;
        }
        if ((camera.film_size.x == 0u) || (camera.film_size.y == 0u)) {
          camera.film_size = kDefaultModelCameraFilmSize;
        }
        if (length(camera.direction) <= kEpsilon) {
          camera.direction = kWorldForward;
        }
        build_camera(camera, camera.position, camera.direction, camera.up, camera.film_size, camera_fov);
        needs_camera_positioning = true;
      }
    } else {
      auto it = std::find_if(data.cameras.begin(), data.cameras.end(), [](const auto& e) {
        return e.active;
      });
      const auto& selected = (it != data.cameras.end()) ? *it : data.cameras.front();
      camera = selected.cam;
    }
  }

  bool has_emissive_materials = false;
  for (const auto& material : data.materials) {
    if ((material.emission.spectrum_index != kInvalidIndex) && (material.emission.spectrum_index < data.spectrum_values.size()) &&
        (data.spectrum_values[material.emission.spectrum_index].luminance() > 0.0f)) {
      has_emissive_materials = true;
      break;
    }
  }

  validate_materials();
  validate_mediums();

  generate_pixel_sampler_image();

  const bool energy_compensation_ready = (rhi != nullptr) ? ensure_energy_compensation_interfaces(data, scheduler, *rhi) : ensure_energy_compensation_interfaces(data, scheduler);
  if (energy_compensation_ready == false) {
    return false;
  }

  data.images.load_images(scheduler);

  {
    TimeMeasure m = {};
    log::warning("Validating normals and tangents...");
    bool has_invalid_tangents = false;
    std::vector<bool> referenced_vertices;
    validate_normals(referenced_vertices, has_invalid_tangents);
    log::warning("Normals validated: %.2f sec", m.lap());

    if (has_invalid_tangents || force_tangents) {
      build_tangents();
      log::warning("Tangents built: %.2f sec", m.lap());
    } else {
      log::warning("Tangents are valid, skipping rebuild");
    }

    validate_tangents(referenced_vertices, has_invalid_tangents || force_tangents);
    log::warning("Tangents validated: %.2f sec", m.lap());
  }

  setup_atmosphere_references();

  // Create area emitters from materials with emission
  create_area_emitters_from_materials();

  // Rebuild atmospheres now that references are set up
  for (uint32_t i = 0; i < data.emitter_profiles.size(); ++i) {
    const auto& profile = data.emitter_profiles[i];
    if ((profile.cls == EmitterProfile::Class::Environment) && (profile.meta & EmitterProfile::Meta::Atmosphere) != 0) {
      rebuild_atmosphere_emitter(i);
    }
  }

  update_medium_bounds();

  if (needs_camera_positioning) {
    constexpr float3 kDefaultViewDirection = {1.0f, 1.0f, 1.0f};
    float3 position = {};
    float3 target = {};
    compute_camera_position_to_fit_scene(data, camera, kDefaultViewDirection, position, target);
    const float3 direction = normalize(target - position);
    build_camera(camera, position, direction, kWorldUp, camera.film_size, camera_fov);
    if (create_default_camera_entry && (data.cameras.empty())) {
      auto& entry = data.cameras.emplace_back();
      entry.id = "default";
      entry.active = true;
      entry.cam = camera;
    }
  }

  return true;
}

void SceneRepresentationImpl::create_area_emitters_from_materials() {
  // Remove existing area emitter profiles, keeping only directional and environment emitters
  auto new_profiles_end = std::remove_if(data.emitter_profiles.begin(), data.emitter_profiles.end(), [](const EmitterProfile& profile) {
    return profile.cls == EmitterProfile::Class::Area;
  });
  data.emitter_profiles.erase(new_profiles_end, data.emitter_profiles.end());

  // Clear triangle emitter references (now point to profiles, not instances)
  for (Triangle& tri : data.triangles) {
    tri.emitter_index = kInvalidIndex;
  }

  // Create area emitter profiles for emissive materials
  std::unordered_map<uint32_t, uint32_t> material_to_profile;

  for (size_t tri_index = 0; tri_index < data.triangles.size(); ++tri_index) {
    const Triangle& tri = data.triangles[tri_index];
    if (tri.material_index >= data.materials.size())
      continue;

    const Material& mtl = data.materials[tri.material_index];
    if (mtl.emission.spectrum_index == kInvalidIndex)
      continue;

    float spectrum_weight = data.spectrum_values[mtl.emission.spectrum_index].luminance();
    if (spectrum_weight <= kEpsilon)
      continue;

    // Get or create emitter profile for this material
    uint32_t profile_index = kInvalidIndex;
    auto mat_it = material_to_profile.find(tri.material_index);
    if (mat_it != material_to_profile.end()) {
      profile_index = mat_it->second;
    } else {
      profile_index = static_cast<uint32_t>(data.emitter_profiles.size());
      material_to_profile[tri.material_index] = profile_index;

      EmitterProfile& profile = data.emitter_profiles.emplace_back(EmitterProfile::Class::Area);
      profile.emission = mtl.emission;
      profile.medium_index = kInvalidIndex;
    }

    // Mark triangle as referencing this emitter profile
    data.triangles[tri_index].emitter_index = profile_index;
  }
}

bool SceneRepresentationImpl::delete_emitter(uint32_t emitter_index) {
  if (emitter_index >= data.emitter_profiles.size()) {
    return false;
  }

  const auto& profile = data.emitter_profiles[emitter_index];

  // Don't allow deleting area emitters - they are managed by materials
  if (profile.cls == EmitterProfile::Class::Area) {
    return false;
  }

  // Remove the emitter profile
  data.emitter_profiles.erase(data.emitter_profiles.begin() + emitter_index);

  for (auto& current_profile : data.emitter_profiles) {
    if ((current_profile.cls != EmitterProfile::Class::Directional) || (current_profile.reference_emitter_index == kInvalidIndex)) {
      continue;
    }

    if (current_profile.reference_emitter_index == emitter_index) {
      current_profile.reference_emitter_index = kInvalidIndex;
    } else if (current_profile.reference_emitter_index > emitter_index) {
      current_profile.reference_emitter_index -= 1u;
    }
  }

  // Drop stale atmosphere/sun references after profile indices changed.
  setup_atmosphere_references();

  // Rebuild area emitters from materials (this will update triangle references)
  create_area_emitters_from_materials();

  for (uint32_t i = 0; i < data.emitter_profiles.size(); ++i) {
    const auto& current_profile = data.emitter_profiles[i];
    if ((current_profile.cls == EmitterProfile::Class::Environment) && ((current_profile.meta & EmitterProfile::Meta::Atmosphere) != 0u)) {
      rebuild_atmosphere_emitter(i);
    }
  }

  return true;
}

}  // namespace etx
