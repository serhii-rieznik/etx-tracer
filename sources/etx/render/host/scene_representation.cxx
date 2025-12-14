#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/core/json.hxx>

#include <etx/render/shared/base.hxx>
#include <etx/render/shared/math.hxx>
#include <etx/render/shared/scene.hxx>
#include <etx/render/shared/scattering.hxx>
#include <etx/render/shared/ior_database.hxx>
#include <etx/render/shared/vertex_utils.hxx>
#include <etx/render/shared/sampler.hxx>

#include <etx/render/host/scene_representation.hxx>
#include <etx/render/host/image_pool.hxx>
#include <etx/render/host/medium_pool.hxx>
#include <etx/render/host/distribution_builder.hxx>
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
    "principled",
    "void",
    "undefined",
  };
  static_assert(sizeof(names) / sizeof(names[0]) == uint32_t(Material::Class::Count) + 1);
  *str = cls < Material::Class::Count ? names[uint32_t(cls)] : "undefined";
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
  Scene scene;
  SceneData data;
  Camera active_camera;
  std::mutex mt;

  const IORDatabase& ior_database;

  std::filesystem::path locate_spectrum_file(const char* identifier, std::initializer_list<const char*> fallback_folders) const {
    if ((identifier == nullptr) || (identifier[0] == 0))
      return {};

    std::filesystem::path requested(identifier);
    if (requested.has_extension() == false)
      requested.replace_extension(".spd");

    std::error_code ec;
    if (requested.is_absolute()) {
      if (std::filesystem::exists(requested, ec))
        return requested;
      return {};
    }

    std::filesystem::path data_root = std::filesystem::path(env().data_folder()) / "spectrum";

    std::filesystem::path combined = data_root / requested;
    if (std::filesystem::exists(combined, ec))
      return combined;

    for (const char* folder : fallback_folders) {
      std::filesystem::path candidate = data_root / folder / requested.filename();
      if (std::filesystem::exists(candidate, ec))
        return candidate;
    }

    return {};
  }

  bool load_ior_from_identifier(const char* identifier, SpectralDistribution& eta, SpectralDistribution& k, SpectralDistribution::Class& cls) const {
    if ((identifier == nullptr) || (identifier[0] == 0))
      return false;

    if (const IORDefinition* def = ior_database.find_by_name(identifier)) {
      cls = def->cls;
      eta = def->eta;
      k = def->k;
      return true;
    }

    std::filesystem::path candidate = locate_spectrum_file(identifier, {"conductor", "dielectric"});
    if (candidate.empty())
      return false;

    cls = RefractiveIndex::load_from_file(candidate.string().c_str(), eta, k);
    return cls != SpectralDistribution::Class::Invalid;
  }

  bool load_illuminant_from_identifier(const char* identifier, SpectralDistribution& spd) const {
    if ((identifier == nullptr) || (identifier[0] == 0))
      return false;

    if (const IORDefinition* def = ior_database.find_by_name(identifier, SpectralDistribution::Class::Illuminant)) {
      spd = def->eta;
      return true;
    }

    std::filesystem::path candidate = locate_spectrum_file(identifier, {"emission"});
    if (candidate.empty())
      return false;

    auto cls = SpectralDistribution::load_from_file(candidate.string().c_str(), spd, nullptr, false);
    return cls != SpectralDistribution::Class::Invalid;
  }

  SceneRepresentationImpl(TaskScheduler& s, const IORDatabase& db)
    : scheduler(s)
    , data(s)
    , ior_database(db) {
    data.images.init(1024u);
    data.mediums.init(1024u);
    scattering::init(scheduler, data.scattering_spectrums, data.extinction_data);
    build_camera(active_camera, {5.0f, 5.0f, 5.0f}, normalize(float3{0.0f, 0.0f, 0.0f} - float3{5.0f, 5.0f, 5.0f}), kWorldUp, {1280u, 720u}, 26.99f);
  }

  ~SceneRepresentationImpl() {
    cleanup();
    data.images.cleanup();
    data.mediums.cleanup();
  }

  void init_default_values() {
    scene.black_spectrum = data.add_spectrum(SpectralDistribution::rgb_reflectance({0.0f, 0.0f, 0.0f}));
    scene.white_spectrum = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
    scene.rayleigh_spectrum = data.add_spectrum(data.scattering_spectrums.rayleigh);
    scene.mie_spectrum = data.add_spectrum(data.scattering_spectrums.mie);
    scene.ozone_spectrum = data.add_spectrum(data.scattering_spectrums.ozone);
    scene.default_dielectric_eta = data.add_spectrum(SpectralDistribution::constant(kDefaultDielectricEta));
    scene.default_conductor_eta = data.add_spectrum(SpectralDistribution::constant(0.0f));
    scene.default_conductor_k = data.add_spectrum(SpectralDistribution::constant(kDefaultConductorK));

    scene.properties[Scene::Properties::Spectral] = false;
    scene.properties[Scene::Properties::MultipleImportanceSampling] = true;
    scene.properties[Scene::Properties::BlueNoise] = true;

    scene.subsurface_scatter_material = data.add_material("etx::subsurface-scatter");
    data.materials[scene.subsurface_scatter_material].reflectance = {.spectrum_index = scene.black_spectrum};
    data.materials[scene.subsurface_scatter_material].scattering = {.spectrum_index = scene.white_spectrum};
    data.materials[scene.subsurface_scatter_material].cls = Material::Class::Translucent;

    scene.subsurface_exit_material = data.add_material("etx::subsurface-exit");
    data.materials[scene.subsurface_exit_material].reflectance = {.spectrum_index = scene.white_spectrum};
    data.materials[scene.subsurface_exit_material].scattering = {.spectrum_index = scene.white_spectrum};
    data.materials[scene.subsurface_exit_material].cls = Material::Class::Diffuse;

    scene.missing_material = data.add_material("etx::missing");
    data.materials[scene.missing_material].reflectance = {.spectrum_index = scene.white_spectrum};
    data.materials[scene.missing_material].scattering = {.spectrum_index = scene.white_spectrum};
    data.materials[scene.missing_material].cls = Material::Class::Diffuse;
  }

  void cleanup() {
    data.clear(scheduler);

    scene = {};

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
        if (mtl.emission.spectrum_index == kInvalidIndex) {
          std::unique_lock lock(mt);
          mtl.emission.spectrum_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
        }
        if ((mtl.roughness.value.x > 0.0f) || (mtl.roughness.value.y > 0.0f)) {
          mtl.roughness.value.x = max(kEpsilon, mtl.roughness.value.x);
          mtl.roughness.value.y = max(kEpsilon, mtl.roughness.value.y);
        }
        if (mtl.int_ior.eta_index == kInvalidIndex) {
          if (mtl.cls == Material::Class::Conductor) {
            mtl.int_ior.eta_index = scene.default_conductor_eta;
          } else {
            mtl.int_ior.eta_index = scene.default_dielectric_eta;
          }
        }
        if (mtl.int_ior.k_index == kInvalidIndex) {
          if (mtl.cls == Material::Class::Conductor) {
            mtl.int_ior.k_index = scene.default_conductor_k;
          } else {
            std::unique_lock lock(mt);
            mtl.int_ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
          }
        }
        if (mtl.thinfilm.ior.k_index == kInvalidIndex) {
          std::unique_lock lock(mt);
          mtl.thinfilm.ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
        }
        if (mtl.thinfilm.ior.eta_index == kInvalidIndex) {
          std::unique_lock lock(mt);
          mtl.thinfilm.ior.eta_index = data.add_spectrum(SpectralDistribution::constant(1.0f));
        }
      }
    });
  }

  void validate_mediums() {
    // Clamp medium densities to prevent extremely small mean free paths
    for (uint32_t i = 0; i < data.mediums.array_size(); ++i) {
      const Medium& medium = data.mediums.get(i);
      clamp_medium_density(scene, medium);
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
    static std::map<uint32_t, uint32_t> a = {};

    const uint64_t normal_count = data.vertices.nrm.size();
    if (data.vertices.tan.size() != normal_count)
      data.vertices.tan.resize(normal_count);
    if (data.vertices.btn.size() != normal_count)
      data.vertices.btn.resize(normal_count);

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

    SMikkTSpaceInterface contextInterface = {};
    contextInterface.m_getNumFaces = [](const SMikkTSpaceContext* pContext) -> int {
      const auto& data = reinterpret_cast<SceneRepresentationImpl*>(pContext->m_pUserData)->data;
      return static_cast<int>(data.triangles.size());
    };
    contextInterface.m_getNumVerticesOfFace = [](const SMikkTSpaceContext* pContext, const int iFace) -> int {
      return 3;
    };
    contextInterface.m_getPosition = [](const SMikkTSpaceContext* pContext, float fvPosOut[], const int iFace, const int iVert) {
      const auto& data = reinterpret_cast<SceneRepresentationImpl*>(pContext->m_pUserData)->data;
      const auto& tri = data.triangles[iFace];
      const auto& vertex = data.vertices.pos[tri.i[iVert]];
      fvPosOut[0] = vertex.x;
      fvPosOut[1] = vertex.y;
      fvPosOut[2] = vertex.z;
    };
    contextInterface.m_getNormal = [](const SMikkTSpaceContext* pContext, float fvNormOut[], const int iFace, const int iVert) {
      const auto& data = reinterpret_cast<SceneRepresentationImpl*>(pContext->m_pUserData)->data;
      const auto& tri = data.triangles[iFace];
      const auto& vertex = data.vertices.nrm[tri.i[iVert]];
      fvNormOut[0] = vertex.x;
      fvNormOut[1] = vertex.y;
      fvNormOut[2] = vertex.z;
    };
    contextInterface.m_getTexCoord = [](const SMikkTSpaceContext* pContext, float fvTexcOut[], const int iFace, const int iVert) {
      const auto& data = reinterpret_cast<SceneRepresentationImpl*>(pContext->m_pUserData)->data;
      const auto& tri = data.triangles[iFace];
      const auto& vertex = data.vertices.tex[tri.i[iVert]];
      fvTexcOut[0] = vertex.x;
      fvTexcOut[1] = vertex.y;
    };
    contextInterface.m_setTSpaceBasic = [](const SMikkTSpaceContext* pContext, const float fvTangent[], const float fSign, const int iFace, const int iVert) {
      auto& data = reinterpret_cast<SceneRepresentationImpl*>(pContext->m_pUserData)->data;
      const auto& tri = data.triangles[iFace];
      auto& nrm = data.vertices.nrm[tri.i[iVert]];
      auto& tan = data.vertices.tan[tri.i[iVert]];
      auto& btn = data.vertices.btn[tri.i[iVert]];
      if (is_valid_vector(tan) == false) {
        tan = normalize(float3{fvTangent[0], fvTangent[1], fvTangent[2]});
        btn = normalize(cross(tan, nrm) * fSign);
      }
    };

    SMikkTSpaceContext context = {};
    context.m_pUserData = this;
    context.m_pInterface = &contextInterface;

    genTangSpaceDefault(&context);
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

  void commit(bool spectral) {
    log::warning("Instancing area emitters...");

    log::warning("Building pixel sampler...");
    std::vector<float4> sampler_image;
    Film::generate_filter_image(Film::PixelFilterBlackmanHarris, sampler_image);
    uint32_t image_options = Image::BuildSamplingTable | Image::UniformSamplingTable;
    uint32_t image = data.images.add_from_data(sampler_image.data(), {Film::PixelFilterSize, Film::PixelFilterSize}, image_options, {}, {1.0f, 1.0f});
    scene.pixel_sampler = {image, 1.5f};

    float3 bbox_min = data.triangles.empty() ? float3{-1.0f, -1.0f, -1.0f} : float3{kMaxFloat, kMaxFloat, kMaxFloat};
    float3 bbox_max = data.triangles.empty() ? float3{+1.0f, +1.0f, +1.0f} : float3{-kMaxFloat, -kMaxFloat, -kMaxFloat};
    for (const auto& tri : data.triangles) {
      bbox_min = min(bbox_min, data.vertices.pos[tri.i[0]]);
      bbox_min = min(bbox_min, data.vertices.pos[tri.i[1]]);
      bbox_min = min(bbox_min, data.vertices.pos[tri.i[2]]);
      bbox_max = max(bbox_max, data.vertices.pos[tri.i[0]]);
      bbox_max = max(bbox_max, data.vertices.pos[tri.i[1]]);
      bbox_max = max(bbox_max, data.vertices.pos[tri.i[2]]);
    }

    scene.bounding_box_min = bbox_min;
    scene.bounding_box_max = bbox_max;
    scene.bounding_sphere_center = 0.5f * (bbox_min + bbox_max);
    scene.bounding_sphere_radius = length(bbox_max - scene.bounding_sphere_center);

    scene.vertices.pos = {data.vertices.pos.data(), data.vertices.pos.size()};
    scene.vertices.nrm = {data.vertices.nrm.data(), data.vertices.nrm.size()};
    scene.vertices.tan = {data.vertices.tan.data(), data.vertices.tan.size()};
    scene.vertices.btn = {data.vertices.btn.data(), data.vertices.btn.size()};
    scene.vertices.tex = {data.vertices.tex.data(), data.vertices.tex.size()};
    scene.triangles = {data.triangles.data(), data.triangles.size()};
    scene.materials = {data.materials.data(), data.materials.size()};
    scene.meshes = {data.meshes.data(), data.meshes.size()};
    scene.spectrums = {data.spectrum_values.data(), data.spectrum_values.size()};
    scene.images = {data.images.as_array(), data.images.array_size()};
    scene.mediums = {data.mediums.as_array(), data.mediums.array_size()};

    rebuild_area_emitters();

    scene.properties[Scene::Properties::Committed] = true;
    scene.properties[Scene::Properties::Spectral] = spectral;
  }

  struct TriangleEmitterData {
    uint32_t triangle_index;
    uint32_t material_index;
    float tri_area;
    float additional_weight;
    SpectralImage emission;
  };

  TriangleEmitterData compute_triangle_emitter_data(uint32_t triangle_index);
  void populate_area_emitters();
  void rebuild_area_emitters();
  void update_medium_bounds();
  void set_mesh_material(uint32_t mesh_index, uint32_t material_index);

  void set_mesh_material_impl(uint32_t mesh_index, uint32_t material_index);
  void add_atmosphere_emitter(const SceneRepresentation::AtmosphereEmitterParameters& params);
  void rebuild_atmosphere_emitter(uint32_t emitter_index);

  bool finalize_scene_loading(uint32_t options, const char* base_folder, uint32_t load_result, float camera_fov, bool use_focal_len, float camera_focal_len, bool force_tangents,
    bool spectral_scene);
};

void build_camera(Camera& camera, const float3& position, const float3& direction, const float3& up, const uint2& viewport, const float fov) {
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

ETX_PIMPL_IMPLEMENT(SceneRepresentation, Impl);

SceneRepresentation::SceneRepresentation(TaskScheduler& s, const IORDatabase& db) {
  ETX_PIMPL_INIT(SceneRepresentation, s, db);
}

SceneRepresentation::~SceneRepresentation() {
  ETX_PIMPL_CLEANUP(SceneRepresentation);
}

Scene& SceneRepresentation::mutable_scene() {
  return _private->scene;
}

Camera& SceneRepresentation::mutable_camera() {
  return _private->active_camera;
}

const Scene& SceneRepresentation::scene() const {
  return _private->scene;
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

uint32_t SceneRepresentation::add_material(const char* name) {
  uint32_t index = _private->data.add_material(name);
  auto& mat = _private->data.materials[index];
  mat.cls = Material::Class::Diffuse;
  mat.reflectance.spectrum_index = _private->data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
  mat.scattering.spectrum_index = _private->data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
  mat.emission.spectrum_index = _private->data.add_spectrum(SpectralDistribution::constant(0.0f));
  mat.subsurface.spectrum_index = _private->data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 0.2f, 0.04f}));
  mat.int_ior.eta_index = _private->scene.default_dielectric_eta;
  mat.int_ior.k_index = _private->scene.default_conductor_k;
  mat.ext_ior.eta_index = _private->scene.default_dielectric_eta;
  mat.ext_ior.k_index = _private->scene.default_conductor_k;
  _private->scene.materials = {_private->data.materials.data(), _private->data.materials.size()};
  _private->scene.spectrums = {_private->data.spectrum_values.data(), _private->data.spectrum_values.size()};
  return index;
}

std::string SceneRepresentation::rename_material(uint32_t index, const char* name) {
  auto result = rename_entry(_private->data.material_mapping, index, name, "material-");
  _private->scene.materials = {_private->data.materials.data(), _private->data.materials.size()};
  return result;
}

uint32_t SceneRepresentation::add_medium(const char* name) {
  SpectralDistribution absorption_spectrum = SpectralDistribution::constant(0.0f);
  SpectralDistribution scattering_spectrum = SpectralDistribution::constant(1.0f);
  uint32_t absorption_index = _private->data.add_spectrum(absorption_spectrum);
  uint32_t scattering_index = _private->data.add_spectrum(scattering_spectrum);
  std::string id = name && name[0] ? name : ("medium-" + std::to_string(_private->data.mediums.array_size()));
  uint32_t handle = _private->data.mediums.add(Medium::Class::Homogeneous, id, nullptr, absorption_index, scattering_index, 0.0f, true);
  _private->scene.mediums = {_private->data.mediums.as_array(), _private->data.mediums.array_size()};
  _private->scene.spectrums = {_private->data.spectrum_values.data(), _private->data.spectrum_values.size()};
  return handle;
}

std::string SceneRepresentation::rename_medium(uint32_t index, const char* name) {
  auto result = _private->data.mediums.rename(index, (name != nullptr) ? name : "");
  _private->scene.mediums = {_private->data.mediums.as_array(), _private->data.mediums.array_size()};
  _private->scene.spectrums = {_private->data.spectrum_values.data(), _private->data.spectrum_values.size()};
  return result;
}

void SceneRepresentation::update_medium_bounds() {
  _private->update_medium_bounds();
  _private->scene.mediums = {_private->data.mediums.as_array(), _private->data.mediums.array_size()};
  _private->scene.spectrums = {_private->data.spectrum_values.data(), _private->data.spectrum_values.size()};
}

std::string SceneRepresentation::rename_mesh(uint32_t index, const char* name) {
  auto result = rename_entry(_private->data.mesh_mapping, index, name, "mesh-");
  _private->scene.meshes = {_private->data.meshes.data(), _private->data.meshes.size()};
  return result;
}

void SceneRepresentation::rebuild_area_emitters() {
  _private->rebuild_area_emitters();
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

bool SceneRepresentation::valid() const {
  return _private->scene.committed();
}

uint32_t SceneRepresentation::add_environment_emitter(const float3& color, uint32_t medium_index) {
  auto& instance = _private->data.emitter_instances.emplace_back(EmitterProfile::Class::Environment);
  instance.profile = uint32_t(_private->data.emitter_profiles.size());

  auto& e = _private->data.emitter_profiles.emplace_back(EmitterProfile::Class::Environment);
  e.emission.spectrum_index = _private->data.add_spectrum(SpectralDistribution::rgb_luminance(color));

  constexpr uint2 kUniformEnvImageDimensions = uint2{1u, 1u};
  constexpr float4 white_color = {1.0f, 1.0f, 1.0f, 1.0f};

  std::vector<float4> uniform_image_data(1, white_color);
  uint32_t image_options = Image::BuildSamplingTable | Image::RepeatU;
  e.emission.image_index = _private->data.add_image(uniform_image_data.data(), kUniformEnvImageDimensions, image_options, {}, {1.0f, 1.0f});
  e.medium_index = medium_index;
  return uint32_t(_private->data.emitter_instances.size() - 1);
}

uint32_t SceneRepresentation::add_directional_emitter(const float3& direction, const float3& color, float angular_diameter_degrees, uint32_t medium_index) {
  auto& instance = _private->data.emitter_instances.emplace_back(EmitterProfile::Class::Directional);
  instance.profile = uint32_t(_private->data.emitter_profiles.size());

  auto& e = _private->data.emitter_profiles.emplace_back(EmitterProfile::Class::Directional);
  e.emission.spectrum_index = _private->data.add_spectrum(SpectralDistribution::rgb_luminance(color));
  e.emission.image_index = kInvalidIndex;
  e.directional.direction = normalize(direction);
  e.directional.angular_size = angular_diameter_degrees * kPi / 180.0f;
  e.medium_index = medium_index;

  return uint32_t(_private->data.emitter_instances.size() - 1);
}

void SceneRepresentation::add_atmosphere_emitter(const AtmosphereEmitterParameters& params) {
  _private->add_atmosphere_emitter(params);
}

void SceneRepresentation::rebuild_atmosphere_emitter(uint32_t emitter_index) {
  _private->rebuild_atmosphere_emitter(emitter_index);
}

void SceneRepresentationImpl::add_atmosphere_emitter(const SceneRepresentation::AtmosphereEmitterParameters& params) {
  data.add_atmosphere_emitter(params, scene, scheduler);
  build_emitters_distribution(data, scene);
}

void SceneRepresentationImpl::rebuild_atmosphere_emitter(uint32_t emitter_index) {
  data.rebuild_atmosphere_emitter(emitter_index, scene, scheduler);
}

template <class T>
inline void get_values(const std::vector<T>& a, T* ptr, uint64_t count) {
  for (uint64_t i = 0, e = a.size() < count ? a.size() : count; i < e; ++i) {
    *ptr++ = a[i];
  }
}

bool SceneRepresentation::load_from_file(const char* filename, uint32_t options, IntegratorData* out_integrator) {
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

  Camera json_camera = {};
  json_camera.lens_image = kInvalidIndex;
  json_camera.medium_index = kInvalidIndex;
  json_camera.up = kWorldUp;
  json_camera.cls = Camera::Class::Perspective;

  float3 camera_target = json_camera.position + json_camera.direction;
  bool has_target = false;
  bool has_direction = false;
  float camera_focal_len = 50.0f;
  float camera_fov = focal_length_to_fov(camera_focal_len) * 180.0f / kPi;
  bool use_focal_len = false;
  bool force_tangents = false;
  bool spectral_scene = false;

  if (strcmp(get_file_ext(filename), ".json") == 0) {
    std::string json_content;
    if (auto f = fopen(filename, "rb")) {
      if (_fseeki64(f, 0, SEEK_END) == 0) {
        long long size = _ftelli64(f);
        if ((size > 0) && (_fseeki64(f, 0, SEEK_SET) == 0)) {
          json_content.resize(static_cast<size_t>(size));
          size_t read_bytes = fread(json_content.data(), 1, json_content.size(), f);
          json_content.resize(read_bytes);
        }
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
        if (out_integrator != nullptr) {
          std::string t = itg.value("type", "");
          out_integrator->selected = map_integrator(t);
        }
        if (itg.contains("min_bounces") && itg["min_bounces"].is_number_integer()) {
          _private->scene.min_path_length = static_cast<uint32_t>(std::max<int64_t>(0, itg["min_bounces"].get<int64_t>()));
        }
        if (itg.contains("max_bounces") && itg["max_bounces"].is_number_integer()) {
          _private->scene.max_path_length = static_cast<uint32_t>(std::max<int64_t>(0, itg["max_bounces"].get<int64_t>()));
        }
      }

      if (js.contains("renderer") && js["renderer"].is_object()) {
        const auto& rnd = js["renderer"];
        if (rnd.contains("spp") && rnd["spp"].is_number_integer()) {
          _private->scene.samples = static_cast<uint32_t>(std::max<int64_t>(1, rnd["spp"].get<int64_t>()));
        }
      }

      uint32_t load_result = load_from_tungsten_file(filename, _private->data, _private->scene, _private->ior_database, _private->scheduler, _private->active_camera);
      if ((load_result & SceneLoadSucceeded) == 0)
        return false;
      return _private->finalize_scene_loading(options, base_folder, load_result, camera_fov, use_focal_len, camera_focal_len, force_tangents, spectral_scene);
    }

    if (parsed == false) {
      log::error("Failed to parse JSON scene %s", filename);
      return false;
    }

    for (auto i = js.begin(), e = js.end(); i != e; ++i) {
      const auto& key = i.key();
      const auto& obj = i.value();
      std::string str_value = {};
      float float_value = 0.0f;
      int64_t int_value = 0;
      bool bool_value = false;
      if (json_get_int(i, "samples", int_value)) {
        _private->scene.samples = static_cast<uint32_t>(std::max(int64_t(1), int_value));
      } else if (json_get_int(i, "random-termination-start", int_value)) {
        _private->scene.random_path_termination = static_cast<uint32_t>(std::max(int64_t(1), int_value));
      } else if (json_get_int(i, "max-path-length", int_value)) {
        _private->scene.max_path_length = static_cast<uint32_t>(std::max(int64_t(1), int_value));
      } else if (json_get_int(i, "min-path-length", int_value)) {
        _private->scene.min_path_length = static_cast<uint32_t>(std::max(int64_t(1), int_value));
      } else if (json_get_string(i, "geometry", str_value)) {
        _private->data.geometry_file_name = std::string(base_folder) + str_value;
      } else if (json_get_string(i, "materials", str_value)) {
        _private->data.materials_file_name = std::string(base_folder) + str_value;
      } else if (json_get_bool(i, "spectral", bool_value)) {
        spectral_scene = bool_value;
      } else if (json_get_bool(i, "multiple_importance_sampling", bool_value)) {
        _private->scene.properties[Scene::Properties::MultipleImportanceSampling] = bool_value;
      } else if (json_get_bool(i, "blue_noise", bool_value)) {
        _private->scene.properties[Scene::Properties::BlueNoise] = bool_value;
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
            _private->scene.properties[Scene::Properties::MultipleImportanceSampling] = strat_value;
          } else if (strat_key == "blue_noise") {
            _private->scene.properties[Scene::Properties::BlueNoise] = strat_value;
          }
        }
        _private->scene.strategy_flags = strategy_flags;
      } else if (json_get_bool(i, "force-tangents", bool_value)) {
        force_tangents = bool_value;
      } else if ((key == "camera") && obj.is_object()) {
        for (auto ci = obj.begin(), ce = obj.end(); ci != ce; ++ci) {
          const auto& ckey = ci.key();
          const auto& cobj = ci.value();
          if (json_get_string(ci, "class", str_value)) {
            json_camera.cls = str_value == "eq" ? Camera::Class::Equirectangular : Camera::Class::Perspective;
          } else if (json_get_float(ci, "fov", float_value)) {
            camera_fov = float_value;
          } else if (json_get_float(ci, "focal-length", float_value)) {
            camera_focal_len = float_value;
            use_focal_len = true;
          } else if (json_get_float(ci, "lens-radius", float_value)) {
            json_camera.lens_radius = float_value;
          } else if (json_get_float(ci, "focal-distance", float_value)) {
            json_camera.focal_distance = float_value;
          } else if (json_get_float(ci, "clip-near", float_value)) {
            json_camera.clip_near = float_value;
          } else if (json_get_float(ci, "clip-far", float_value)) {
            json_camera.clip_far = float_value;
          } else if (cobj.is_array()) {
            if (ckey == "origin") {
              auto values = cobj.get<std::vector<float>>();
              get_values(values, &json_camera.position.x, 3llu);
            } else if (ckey == "target") {
              auto values = cobj.get<std::vector<float>>();
              get_values(values, &camera_target.x, 3llu);
              has_target = true;
            } else if (ckey == "direction") {
              auto values = cobj.get<std::vector<float>>();
              get_values(values, &json_camera.direction.x, 3llu);
              has_direction = true;
            } else if (ckey == "up") {
              auto values = cobj.get<std::vector<float>>();
              get_values(values, &json_camera.up.x, 3llu);
            } else if (ckey == "viewport") {
              auto values = cobj.get<std::vector<uint32_t>>();
              get_values(values, &json_camera.film_size.x, 2llu);
            } else {
              log::warning("Unhandled value in camera description : %s", key.c_str());
            }
          }
        }

        if (has_direction) {
          json_camera.direction = normalize(json_camera.direction);
        } else if (has_target) {
          json_camera.direction = normalize(camera_target - json_camera.position);
        } else {
          json_camera.direction = kWorldForward;
        }
      } else if ((key == "integrator") && obj.is_object()) {
        if (out_integrator != nullptr) {
          std::string selected_id_str;
          if (obj.contains("selected") && obj["selected"].is_string()) {
            selected_id_str = obj["selected"].get<std::string>();
            out_integrator->selected = integrator_id_to_type(selected_id_str.c_str());
          }
          if (out_integrator->selected == Integrator::Type::Invalid) {
            if (obj.contains("type") && obj["type"].is_string()) {
              out_integrator->selected = integrator_id_to_type(obj["type"].get<std::string>().c_str());
            } else if (obj.contains("name") && obj["name"].is_string()) {
              std::string name = obj["name"].get<std::string>();
              if (name.find("Path Tracing") != std::string::npos) {
                out_integrator->selected = Integrator::Type::PathTracing;
              } else if (name.find("Bidirectional") != std::string::npos) {
                out_integrator->selected = Integrator::Type::Bidirectional;
              } else if (name.find("VCM") != std::string::npos) {
                out_integrator->selected = Integrator::Type::VCM;
              } else if (name.find("Debug") != std::string::npos) {
                out_integrator->selected = Integrator::Type::Debug;
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
                  out_integrator->settings[type] = std::move(options);
                }
              }
            }
          }

          if (out_integrator->selected != Integrator::Type::Invalid && obj.contains("options") && obj["options"].is_array()) {
            Options options;
            if (options.deserialize_from_json(obj["options"])) {
              out_integrator->settings[out_integrator->selected] = std::move(options);
            }
          }
        }
      } else {
        log::warning("Unhandled value in scene description : %s", key.c_str());
      }
    }
    _private->data.json_file_name = filename;
  }

  uint32_t load_result = SceneLoadFailed;

  auto ext = get_file_ext(_private->data.geometry_file_name.c_str());
  if (strcmp(ext, ".etx") == 0) {
    SceneSerialization loader;
    if (!loader.load_from_file(_private->data.geometry_file_name.c_str(), _private->data, _private->data.materials_file_name.c_str(), _private->scene, _private->ior_database,
          _private->scheduler)) {
      log::error("Failed to load ETX file from %s", _private->data.geometry_file_name.c_str());
      return false;
    }
    load_result = SceneLoadSucceeded;
  } else if (strcmp(ext, ".obj") == 0) {
    load_result = load_from_obj_file(_private->data.geometry_file_name.c_str(), _private->data.materials_file_name.c_str(), _private->data, _private->scene, _private->ior_database,
      _private->scheduler);
  } else if (strcmp(ext, ".gltf") == 0) {
    load_result = load_from_gltf_file(_private->data.geometry_file_name.c_str(), false, _private->data, _private->scene, _private->scheduler, _private->active_camera);
  } else if (strcmp(ext, ".glb") == 0) {
    load_result = load_from_gltf_file(_private->data.geometry_file_name.c_str(), true, _private->data, _private->scene, _private->scheduler, _private->active_camera);
  }

  if ((load_result & SceneLoadSucceeded) == 0) {
    return false;
  }

  if (has_target || has_direction || json_camera.film_size.x > 0 || json_camera.lens_radius > 0.0f) {
    if (use_focal_len) {
      camera_fov = focal_length_to_fov(camera_focal_len) * 180.0f / kPi;
    }

    if (json_camera.film_size.x * json_camera.film_size.y == 0) {
      json_camera.film_size = {1280, 720};
    }

    auto& entry = _private->data.cameras.emplace_back();
    entry.id = "json_camera";
    entry.active = _private->data.cameras.size() == 1;

    build_camera(entry.cam, json_camera.position, json_camera.direction, json_camera.up, json_camera.film_size, camera_fov);

    entry.cam.cls = json_camera.cls;
    entry.cam.lens_radius = json_camera.lens_radius;
    entry.cam.focal_distance = json_camera.focal_distance;
    entry.cam.clip_near = json_camera.clip_near;
    entry.cam.clip_far = json_camera.clip_far;
  }

  return _private->finalize_scene_loading(options, base_folder, load_result, camera_fov, use_focal_len, camera_focal_len, force_tangents, spectral_scene);
}

SceneRepresentationImpl::TriangleEmitterData SceneRepresentationImpl::compute_triangle_emitter_data(uint32_t triangle_index) {
  TriangleEmitterData result = {};
  result.triangle_index = kInvalidIndex;

  if (triangle_index >= data.triangles.size())
    return result;

  const auto& tri = data.triangles[triangle_index];
  uint32_t material_index = tri.material_index;
  if (material_index >= data.materials.size())
    return result;

  const auto& mtl = data.materials[material_index];
  if (mtl.emission.spectrum_index == kInvalidIndex)
    return result;

  if (mtl.emission.spectrum_index >= data.spectrum_values.size())
    return result;

  float texture_emission = 1.0f;
  if (mtl.emission.image_index != kInvalidIndex) {
    const auto& img = data.images.get(mtl.emission.image_index);
    constexpr float kBCScale = 2.0f;

    const float2& tex0 = data.vertices.tex[tri.i[0]];
    const float2& tex1 = data.vertices.tex[tri.i[1]];
    const float2& tex2 = data.vertices.tex[tri.i[2]];

    auto min_uv = min(tex0, min(tex1, tex2));
    auto max_uv = max(tex0, max(tex1, tex2));

    float uv_area = fabsf((max_uv.x - min_uv.x) * (max_uv.y - min_uv.y));
    uint32_t sample_count = max(16u, uint32_t(uv_area * img.fsize.x * img.fsize.y * kBCScale));

    Sampler smp(triangle_index, material_index);

    float sum = 0.0f;
    for (uint32_t i = 0; i < sample_count; ++i) {
      float2 rnd = smp.next_2d();
      auto b = random_barycentric(rnd);
      auto uv = tex0 * b.x + tex1 * b.y + tex2 * b.z;
      float4 val = img.evaluate(uv, nullptr);
      sum += luminance(to_float3(val)) * val.w;
    }

    texture_emission = sum / float(sample_count);
  }

  float tri_area = triangle_area(tri);
  float spectrum_weight = data.spectrum_values[mtl.emission.spectrum_index].luminance();
  float additional_weight = (mtl.two_sided ? 2.0f : 1.0f) * (tri_area * kPi) * texture_emission;
  if ((additional_weight <= 0.0f) || (spectrum_weight <= 0.0f))
    return result;

  result.triangle_index = triangle_index;
  result.material_index = material_index;
  result.tri_area = tri_area;
  result.additional_weight = additional_weight;
  result.emission = mtl.emission;

  return result;
}

void SceneRepresentationImpl::populate_area_emitters() {
  std::unordered_map<uint32_t, uint32_t> profile_remap;
  std::vector<EmitterProfile> preserved_profiles;
  preserved_profiles.reserve(data.emitter_profiles.size());
  for (uint32_t i = 0; i < data.emitter_profiles.size(); ++i) {
    const auto& profile = data.emitter_profiles[i];
    if (profile.cls != EmitterProfile::Class::Area) {
      uint32_t new_index = static_cast<uint32_t>(preserved_profiles.size());
      profile_remap[i] = new_index;
      preserved_profiles.emplace_back(profile);
    }
  }

  std::vector<Emitter> preserved_instances;
  preserved_instances.reserve(data.emitter_instances.size());
  for (const auto& emitter : data.emitter_instances) {
    if (emitter.cls != EmitterProfile::Class::Area) {
      auto remap = profile_remap.find(emitter.profile);
      ETX_CRITICAL(remap != profile_remap.end());
      auto copy = emitter;
      copy.profile = remap->second;
      preserved_instances.emplace_back(copy);
    }
  }

  data.emitter_profiles = std::move(preserved_profiles);
  data.emitter_instances = std::move(preserved_instances);

  data.triangle_to_emitter.resize(data.triangles.size(), kInvalidIndex);
  std::fill(data.triangle_to_emitter.begin(), data.triangle_to_emitter.end(), kInvalidIndex);

  for (auto it = data.material_to_emitter_profile.begin(); it != data.material_to_emitter_profile.end();) {
    uint32_t profile_index = it->second;
    auto remap_it = profile_remap.find(profile_index);
    if (remap_it == profile_remap.end()) {
      it = data.material_to_emitter_profile.erase(it);
    } else {
      it->second = remap_it->second;
      ++it;
    }
  }

  data.images.load_images(scheduler);

  std::mutex emitter_mutex;
  scheduler.execute(data.triangles.size(), [this, &emitter_mutex](uint32_t begin, uint32_t end, uint32_t) {
    std::vector<TriangleEmitterData> local_emitters;
    std::unordered_map<uint32_t, std::pair<uint32_t, SpectralImage>> local_material_profiles;

    for (uint32_t tri_index = begin; tri_index < end; ++tri_index) {
      auto emitter_data = compute_triangle_emitter_data(tri_index);
      if (emitter_data.triangle_index == kInvalidIndex)
        continue;

      uint32_t local_profile_index = kInvalidIndex;
      auto profile_it = local_material_profiles.find(emitter_data.material_index);
      if (profile_it != local_material_profiles.end()) {
        local_profile_index = profile_it->second.first;
      } else {
        local_profile_index = static_cast<uint32_t>(local_material_profiles.size());
        local_material_profiles[emitter_data.material_index] = {local_profile_index, emitter_data.emission};
      }

      local_emitters.push_back(emitter_data);
    }

    {
      std::lock_guard<std::mutex> lock(emitter_mutex);
      std::unordered_map<uint32_t, uint32_t> local_to_global_profile;

      for (const auto& [material_index, profile_data] : local_material_profiles) {
        uint32_t global_profile_index = kInvalidIndex;
        auto mapping_it = data.material_to_emitter_profile.find(material_index);
        if (mapping_it != data.material_to_emitter_profile.end()) {
          global_profile_index = mapping_it->second;
        } else {
          global_profile_index = static_cast<uint32_t>(data.emitter_profiles.size());
          data.material_to_emitter_profile[material_index] = global_profile_index;
          data.emitter_profiles.emplace_back(EmitterProfile::Class::Area);
          data.emitter_profiles[global_profile_index].emission = profile_data.second;
        }
        local_to_global_profile[profile_data.first] = global_profile_index;
      }

      for (const auto& emitter_data : local_emitters) {
        auto profile_it = local_material_profiles.find(emitter_data.material_index);
        ETX_ASSERT(profile_it != local_material_profiles.end());
        uint32_t local_profile_index = profile_it->second.first;
        uint32_t global_profile_index = local_to_global_profile[local_profile_index];

        uint32_t global_emitter_index = static_cast<uint32_t>(data.emitter_instances.size());
        auto& emitter = data.emitter_instances.emplace_back(EmitterProfile::Class::Area);
        emitter.profile = global_profile_index;
        emitter.triangle_index = emitter_data.triangle_index;
        emitter.triangle_area = emitter_data.tri_area;
        emitter.additional_weight = emitter_data.additional_weight;
        data.triangle_to_emitter[emitter_data.triangle_index] = global_emitter_index;
      }
    }
  });
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

void SceneRepresentationImpl::rebuild_area_emitters() {
  populate_area_emitters();
  scene.triangle_to_emitter = {data.triangle_to_emitter.data(), data.triangle_to_emitter.size()};
  scene.emitter_profiles = {data.emitter_profiles.data(), data.emitter_profiles.size()};
  scene.emitter_instances = {data.emitter_instances.data(), data.emitter_instances.size()};
  scene.spectrums = {data.spectrum_values.data(), data.spectrum_values.size()};
  scene.images = {data.images.as_array(), data.images.array_size()};
  build_emitters_distribution(data, scene);
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

  const Scene& scene_data = impl->scene;

  auto geometry_export_start = std::chrono::high_resolution_clock::now();
  SceneSerialization archive;
  if (!archive.save_to_file(impl->data, geometry_path)) {
    log::error("Failed to export geometry to %s", geometry_path.string().c_str());
    return {};
  }
  auto geometry_export_end = std::chrono::high_resolution_clock::now();
  auto geometry_export_duration = std::chrono::duration_cast<std::chrono::milliseconds>(geometry_export_end - geometry_export_start);
  log::info("Geometry export: %lld ms", geometry_export_duration.count());

  nlohmann::json js = nlohmann::json::object();
  js["samples"] = scene_data.samples;
  js["random-termination-start"] = scene_data.random_path_termination;
  js["max-path-length"] = scene_data.max_path_length;
  js["min-path-length"] = scene_data.min_path_length;
  js["geometry"] = geometry_ref;
  if (materials_ref.empty() == false) {
    js["materials"] = materials_ref;
  }
  js["spectral"] = scene_data.properties[Scene::Properties::Spectral];
  js["multiple_importance_sampling"] = scene_data.properties[Scene::Properties::MultipleImportanceSampling];
  js["blue_noise"] = scene_data.properties[Scene::Properties::BlueNoise];

  nlohmann::json strategies = nlohmann::json::object();
  strategies["direct_hit"] = ((scene_data.strategy_flags & Scene::Strategy::DirectHit) == Scene::Strategy::DirectHit);
  strategies["connect_to_light"] = ((scene_data.strategy_flags & Scene::Strategy::ConnectToLight) == Scene::Strategy::ConnectToLight);
  strategies["connect_to_camera"] = ((scene_data.strategy_flags & Scene::Strategy::ConnectToCamera) == Scene::Strategy::ConnectToCamera);
  strategies["connect_vertices"] = ((scene_data.strategy_flags & Scene::Strategy::ConnectVertices) == Scene::Strategy::ConnectVertices);
  strategies["merge_vertices"] = ((scene_data.strategy_flags & Scene::Strategy::MergeVertices) == Scene::Strategy::MergeVertices);
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
    static const SpectralDistribution null_spectrum = SpectralDistribution::null();
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
    if (stored.empty()) {
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

  const EmitterProfile* environment_profile = nullptr;
  const EmitterProfile* directional_profile = nullptr;
  for (const auto& profile : impl->data.emitter_profiles) {
    if ((profile.cls == EmitterProfile::Class::Environment) && (environment_profile == nullptr)) {
      environment_profile = &profile;
    } else if ((profile.cls == EmitterProfile::Class::Directional) && (directional_profile == nullptr)) {
      directional_profile = &profile;
    }
  }

  if (environment_profile != nullptr) {
    materials_stream << "newmtl et::env\n";
    std::string env_path = texture_path(environment_profile->emission.image_index);
    if (env_path.empty() == false) {
      materials_stream << "image " << env_path << "\n";
    }
    float3 env_color = spectrum_rgb(environment_profile->emission.spectrum_index);
    materials_stream << "color " << env_color.x << " " << env_color.y << " " << env_color.z << "\n";
    float env_rotation_offset = 0.0f;
    float env_scale_u = 1.0f;
    if (environment_profile->emission.image_index != kInvalidIndex) {
      const Image& env_image = impl->data.images.get(environment_profile->emission.image_index);
      env_rotation_offset = env_image.offset.x;
      env_scale_u = env_image.scale.x;
    }
    if (std::fabs(env_rotation_offset) >= kEpsilon) {
      materials_stream << "rotation " << (-env_rotation_offset * 360.0f) << "\n";
    }
    if (std::fabs(env_scale_u - 1.0f) >= kEpsilon) {
      materials_stream << "scale " << env_scale_u << "\n";
    }
    bool env_medium_valid = (environment_profile->medium_index != kInvalidIndex) && (medium_names.count(environment_profile->medium_index) > 0);
    if (env_medium_valid) {
      materials_stream << "ext_medium " << medium_names[environment_profile->medium_index] << "\n";
    }
    materials_stream << "\n";
  }

  if (directional_profile != nullptr) {
    materials_stream << "newmtl et::dir\n";
    float3 dir_color = spectrum_rgb(directional_profile->emission.spectrum_index);
    materials_stream << "color " << dir_color.x << " " << dir_color.y << " " << dir_color.z << "\n";
    materials_stream << "direction " << directional_profile->directional.direction.x << " " << directional_profile->directional.direction.y << " "
                     << directional_profile->directional.direction.z << "\n";
    if (directional_profile->directional.angular_size >= kEpsilon) {
      materials_stream << "angular_diameter " << (directional_profile->directional.angular_size * 180.0f / kPi) << "\n";
    }
    std::string dir_path = texture_path(directional_profile->emission.image_index);
    if (dir_path.empty() == false) {
      materials_stream << "image " << dir_path << "\n";
    }
    bool dir_medium_valid = (directional_profile->medium_index != kInvalidIndex) && (medium_names.count(directional_profile->medium_index) > 0);
    if (dir_medium_valid) {
      materials_stream << "ext_medium " << medium_names[directional_profile->medium_index] << "\n";
    }
    materials_stream << "\n";
  }

  for (uint64_t medium_index = 0; medium_index < medium_entries.size(); ++medium_index) {
    uint32_t pool_index = medium_entries[medium_index].second;
    const Medium& medium = impl->data.mediums.get(pool_index);
    materials_stream << "newmtl et::medium\n";
    materials_stream << "id " << medium_entries[medium_index].first << "\n";
    float3 absorption = medium_absorption_rgb(impl->scene, medium);
    if ((std::fabs(absorption.x) >= kEpsilon) || (std::fabs(absorption.y) >= kEpsilon) || (std::fabs(absorption.z) >= kEpsilon)) {
      materials_stream << "absorption " << absorption.x << " " << absorption.y << " " << absorption.z << "\n";
    }
    float3 scattering = medium_scattering_rgb(impl->scene, medium);
    if ((std::fabs(scattering.x) >= kEpsilon) || (std::fabs(scattering.y) >= kEpsilon) || (std::fabs(scattering.z) >= kEpsilon)) {
      materials_stream << "scattering " << scattering.x << " " << scattering.y << " " << scattering.z << "\n";
    }
    if (std::fabs(medium.phase_function_g) >= kEpsilon) {
      materials_stream << "anisotropy " << medium.phase_function_g << "\n";
    }
    if (medium.enable_explicit_connections == false) {
      materials_stream << "enclosed 1\n";
    }
    if (medium.grid.type == DensityGrid::Type::NoiseFunction) {
      materials_stream << "noise type " << static_cast<uint32_t>(medium.grid.noise_type) << " scale " << medium.grid.noise.scale << " octaves " << medium.grid.noise.octaves
                       << " lacunarity " << medium.grid.noise.lacunarity << " persistence " << medium.grid.noise.persistence << " seed " << medium.grid.noise.seed << " power "
                       << medium.grid.noise.power << " sharpness " << medium.grid.noise.sharpness << " offset " << medium.grid.noise.offset.x << " " << medium.grid.noise.offset.y
                       << " " << medium.grid.noise.offset.z << " border_fade " << medium.grid.noise.enable_border_fade << " border_fade_distance "
                       << medium.grid.noise.border_fade_distance << "\n";
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
    if ((material.cls == Material::Class::Dielectric) || (material.cls == Material::Class::Translucent) || (material.transmission.value.x > kEpsilon)) {
      write_spectrum_line(materials_stream, "Kt", material.scattering.spectrum_index, true);
    }
    write_spectrum_line(materials_stream, "Ks", material.reflectance.spectrum_index, true);

    float rough_u = material.roughness.value.x;
    float rough_v = material.roughness.value.y;
    if ((rough_u >= kEpsilon) || (rough_v >= kEpsilon)) {
      float value_u = std::sqrt(std::max(0.0f, rough_u));
      float value_v = std::sqrt(std::max(0.0f, rough_v));
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
    if (material.int_ior.cls != SpectralDistribution::Class::Invalid) {
      matched_int_index = database.find_matching_index(spectrum_by_index(material.int_ior.eta_index), spectrum_by_index(material.int_ior.k_index), material.int_ior.cls);
    }
    if ((matched_int_index >= 0) && (matched_int_index < static_cast<int>(database.definitions.size()))) {
      const IORDefinition& def = database.definitions[static_cast<size_t>(matched_int_index)];
      materials_stream << "int_ior " << def.name << "\n";
    } else if ((material.int_ior.eta_index != kInvalidIndex) && (material.int_ior.cls != SpectralDistribution::Class::Invalid)) {
      float eta_value = spectrum_scalar(material.int_ior.eta_index, 1.0f);
      if (material.int_ior.cls == SpectralDistribution::Class::Dielectric) {
        materials_stream << "int_ior " << eta_value << "\n";
      } else if (material.int_ior.cls == SpectralDistribution::Class::Conductor) {
        float k_value = spectrum_scalar(material.int_ior.k_index, 0.0f);
        materials_stream << "int_ior " << eta_value << " " << k_value << "\n";
      }
    }

    int matched_ext_index = -1;
    if (material.ext_ior.cls != SpectralDistribution::Class::Invalid) {
      matched_ext_index = database.find_matching_index(spectrum_by_index(material.ext_ior.eta_index), spectrum_by_index(material.ext_ior.k_index), material.ext_ior.cls);
    }
    if ((matched_ext_index >= 0) && (matched_ext_index < static_cast<int>(database.definitions.size()))) {
      const IORDefinition& def = database.definitions[static_cast<size_t>(matched_ext_index)];
      materials_stream << "ext_ior " << def.name << "\n";
    } else {
      float ext_eta_value = spectrum_scalar(material.ext_ior.eta_index, 1.0f);
      if ((material.ext_ior.eta_index != kInvalidIndex) && (material.ext_ior.cls != SpectralDistribution::Class::Invalid) &&
          (material.ext_ior.cls != SpectralDistribution::Class::Dielectric || std::fabs(ext_eta_value - 1.0f) >= kEpsilon)) {
        if (material.ext_ior.cls == SpectralDistribution::Class::Dielectric) {
          materials_stream << "ext_ior " << ext_eta_value << "\n";
        } else if (material.ext_ior.cls == SpectralDistribution::Class::Conductor) {
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
    if (material.diffuse_variation != 0u) {
      materials_stream << "diffuse " << material.diffuse_variation << "\n";
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

    if (material.subsurface.cls != SubsurfaceMaterial::Class::Disabled) {
      materials_stream << "subsurface";
      if (material.subsurface.cls == SubsurfaceMaterial::Class::ChristensenBurley) {
        materials_stream << " class approximate";
      }
      if (material.subsurface.path == SubsurfaceMaterial::Path::Refracted) {
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
      int matched_thinfilm_index = -1;
      if (material.thinfilm.ior.cls != SpectralDistribution::Class::Invalid) {
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

bool SceneRepresentationImpl::finalize_scene_loading(uint32_t options, const char* base_folder, uint32_t load_result, float camera_fov, bool use_focal_len, float camera_focal_len,
  bool force_tangents, bool spectral_scene) {
  auto& camera = active_camera;
  bool needs_camera_positioning = false;

  if (options & SceneRepresentation::SetupCamera) {
    if (data.cameras.empty()) {
      if ((load_result & SceneLoadCameraInfo) == 0) {
        if (use_focal_len) {
          camera_fov = focal_length_to_fov(camera_focal_len) * 180.0f / kPi;
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

  if (data.emitter_profiles.empty() && !has_emissive_materials) {
    add_atmosphere_emitter({});
    data.images.load_images(scheduler);
  }

  validate_materials();
  validate_mediums();

  {
    TimeMeasure m = {};
    log::warning("Validating normals and tangents...");
    bool has_invalid_tangents = false;
    std::vector<bool> referenced_vertices;
    validate_normals(referenced_vertices, has_invalid_tangents);
    log::warning("Normals validated: %.2f sec", m.lap());
    build_tangents();
    log::warning("Tangents built: %.2f sec", m.lap());
    validate_tangents(referenced_vertices, has_invalid_tangents || force_tangents);
    log::warning("Tangents validated: %.2f sec", m.lap());
  }

  update_medium_bounds();
  commit(spectral_scene);

  if (needs_camera_positioning) {
    constexpr float3 kDefaultViewDirection = {1.0f, 1.0f, 1.0f};
    float3 position = {};
    float3 target = {};
    compute_camera_position_to_fit_scene(scene, camera, kDefaultViewDirection, position, target);
    const float3 direction = normalize(target - position);
    build_camera(camera, position, direction, kWorldUp, camera.film_size, camera_fov);
  }

  return true;
}

}  // namespace etx
