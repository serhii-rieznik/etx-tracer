#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/core/json.hxx>

#include <etx/render/shared/base.hxx>
#include <etx/render/shared/math.hxx>
#include <etx/render/shared/scene.hxx>
#include <etx/render/shared/scattering.hxx>
#include <etx/render/shared/ior_database.hxx>

#include <etx/render/host/scene_representation.hxx>
#include <etx/render/host/image_pool.hxx>
#include <etx/render/host/medium_pool.hxx>
#include <etx/render/host/distribution_builder.hxx>
#include <etx/render/host/gltf_accessor.hxx>
#include <etx/render/host/scene_data.hxx>

#include <filesystem>
#include <sstream>
#include <iomanip>
#include <system_error>
#include <cmath>
#include <vector>
#include <string>
#include <set>
#include <algorithm>
#include <cstring>
#include <initializer_list>

#include <mikktspace.h>
#include <tiny_obj_loader.hxx>

namespace etx {

static scattering::ScatteringSpectrums shared_scattering_spectrums;

inline Material::Class material_string_to_class(const char* s) {
  if (strcmp(s, "diffuse") == 0)
    return Material::Class::Diffuse;
  if (strcmp(s, "translucent") == 0)
    return Material::Class::Translucent;
  else if (strcmp(s, "plastic") == 0)
    return Material::Class::Plastic;
  else if (strcmp(s, "conductor") == 0)
    return Material::Class::Conductor;
  else if (strcmp(s, "dielectric") == 0)
    return Material::Class::Dielectric;
  else if (strcmp(s, "thinfilm") == 0)
    return Material::Class::Thinfilm;
  else if (strcmp(s, "mirror") == 0)
    return Material::Class::Mirror;
  else if (strcmp(s, "boundary") == 0)
    return Material::Class::Boundary;
  else if (strcmp(s, "velvet") == 0)
    return Material::Class::Velvet;
  else if (strcmp(s, "principled") == 0)
    return Material::Class::Principled;
  else if (strcmp(s, "void") == 0)
    return Material::Class::Void;
  else {
    log::error("Undefined BSDF: `%s`", s);
    return Material::Class::Diffuse;
  }
}

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
  TaskScheduler& scheduler;
  Scene scene;
  SceneData data;
  SceneLoaderContext context;
  Camera active_camera;

  const IORDatabase& ior_database;

  char data_buffer[2048] = {};

  bool get_param(const tinyobj::material_t& m, const char* param) {
    memset(data_buffer, 0, sizeof(data_buffer));
    for (const auto& p : m.unknown_parameter) {
      if (_stricmp(p.first.c_str(), param) == 0) {
        memcpy(data_buffer, p.second.c_str(), p.second.size());
        data_buffer[p.second.size()] = 0;
        return true;
      }
    }
    return false;
  }

  bool get_file(const char* base_dir, const std::string& base) {
    memset(data_buffer, 0, sizeof(data_buffer));
    if (base.empty() == false) {
      snprintf(data_buffer, sizeof(data_buffer), "%s/%s", base_dir, base.c_str());
      return true;
    }
    return false;
  }

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
    , context(s)
    , ior_database(db) {
    context.images.init(1024u);
    context.mediums.init(1024u);
    scattering::init(scheduler, shared_scattering_spectrums, data.atmosphere_extinction);
    build_camera(active_camera, {5.0f, 5.0f, 5.0f}, {0.0f, 0.0f, 0.0f}, kWorldUp, {1280u, 720u}, 26.99f);
  }

  ~SceneRepresentationImpl() {
    cleanup();
    context.images.free_image(data.atmosphere_extinction);
    context.images.cleanup();
    context.mediums.cleanup();
  }

  void init_default_values() {
    scene.black_spectrum = data.add_spectrum(SpectralDistribution::rgb_reflectance({0.0f, 0.0f, 0.0f}));
    scene.white_spectrum = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
    scene.rayleigh_spectrum = data.add_spectrum(shared_scattering_spectrums.rayleigh);
    scene.mie_spectrum = data.add_spectrum(shared_scattering_spectrums.mie);
    scene.ozone_spectrum = data.add_spectrum(shared_scattering_spectrums.ozone);
    scene.default_dielectric_eta = data.add_spectrum(SpectralDistribution::constant(1.5f));
    scene.default_conductor_eta = data.add_spectrum(SpectralDistribution::constant(0.0f));
    scene.default_conductor_k = data.add_spectrum(SpectralDistribution::constant(1000000.0f));

    scene.subsurface_scatter_material = data.add_material("etx::subsurface-scatter");
    data.materials[scene.subsurface_scatter_material].reflectance = {.spectrum_index = scene.black_spectrum};
    data.materials[scene.subsurface_scatter_material].scattering = {.spectrum_index = scene.white_spectrum};
    data.materials[scene.subsurface_scatter_material].cls = Material::Class::Translucent;

    scene.subsurface_exit_material = data.add_material("etx::subsurface-exit");
    data.materials[scene.subsurface_exit_material].reflectance = {.spectrum_index = scene.white_spectrum};
    data.materials[scene.subsurface_exit_material].scattering = {.spectrum_index = scene.white_spectrum};
    data.materials[scene.subsurface_exit_material].cls = Material::Class::Diffuse;
  }

  void cleanup() {
    context.images.remove_all();
    context.mediums.remove_all();

    free(scene.emitters_distribution.values.a);
    scene.emitters_distribution = {};

    data = {};
    scene = {};

    active_camera = {};
    active_camera.lens_image = kInvalidIndex;
    active_camera.medium_index = kInvalidIndex;
    active_camera.up = kWorldUp;

    scattering::init(scheduler, shared_scattering_spectrums, data.atmosphere_extinction);
    build_camera(active_camera, {5.0f, 5.0f, 5.0f}, {0.0f, 0.0f, 0.0f}, kWorldUp, {1280u, 720u}, 26.99f);

    init_default_values();
  }

  float triangle_area(const Triangle& t) {
    return 0.5f * length(cross(data.vertices[t.i[1]].pos - data.vertices[t.i[0]].pos, data.vertices[t.i[2]].pos - data.vertices[t.i[0]].pos));
  }

  bool validate_triangle(Triangle& t) {
    t.geo_n = cross(data.vertices[t.i[1]].pos - data.vertices[t.i[0]].pos, data.vertices[t.i[2]].pos - data.vertices[t.i[0]].pos);
    float l = length(t.geo_n);
    if (l == 0.0f)
      return false;

    t.geo_n /= l;
    return true;
  }

  void validate_materials() {
    for (auto& mtl : data.materials) {
      if (mtl.reflectance.spectrum_index == kInvalidIndex) {
        mtl.reflectance.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
      }
      if (mtl.scattering.spectrum_index == kInvalidIndex) {
        mtl.scattering.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
      }
      if (mtl.subsurface.spectrum_index == kInvalidIndex) {
        mtl.subsurface.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 0.2f, 0.04f}));
      }
      if (mtl.emission.spectrum_index == kInvalidIndex) {
        mtl.emission.spectrum_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
      }
      if ((mtl.roughness.value.x > 0.0f) || (mtl.roughness.value.y > 0.0f)) {
        constexpr float kEpsilon = 1e-6f;
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
          mtl.int_ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
        }
      }
      if (mtl.thinfilm.ior.k_index == kInvalidIndex) {
        mtl.thinfilm.ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
      }
      if (mtl.thinfilm.ior.eta_index == kInvalidIndex) {
        mtl.thinfilm.ior.eta_index = data.add_spectrum(SpectralDistribution::constant(1.0f));
      }
    }
  }

  void validate_normals(std::vector<bool>& referenced_vertices) {
    std::set<uint32_t> reset_normals;

    referenced_vertices.resize(data.vertices.size());
    for (const auto& tri : data.triangles) {
      const float tri_area = triangle_area(tri);
      for (uint32_t i = 0; i < 3; ++i) {
        uint32_t index = tri.i[i];
        ETX_CRITICAL(is_valid_vector(tri.geo_n));
        referenced_vertices[index] = true;

        if (is_valid_vector(data.vertices[index].nrm))
          continue;

        if (reset_normals.count(index) == 0) {
          data.vertices[index].nrm = tri.geo_n * tri_area;
          reset_normals.insert(index);
        } else {
          data.vertices[index].nrm += tri.geo_n * tri_area;
        }
      }
    }

    if (reset_normals.empty())
      return;

    for (auto i : reset_normals) {
      ETX_ASSERT(is_valid_vector(data.vertices[i].nrm));
      data.vertices[i].nrm = normalize(data.vertices[i].nrm);
      ETX_ASSERT(is_valid_vector(data.vertices[i].nrm));
    }
  }

  void build_tangents() {
    static std::map<uint32_t, uint32_t> a = {};

    float2 min_uv = {kMaxFloat, kMaxFloat};
    float2 max_uv = {-kMaxFloat, -kMaxFloat};
    for (const auto& v : data.vertices) {
      min_uv = min(min_uv, v.tex);
      max_uv = max(max_uv, v.tex);
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
      const auto& vertex = data.vertices[tri.i[iVert]];
      fvPosOut[0] = vertex.pos.x;
      fvPosOut[1] = vertex.pos.y;
      fvPosOut[2] = vertex.pos.z;
    };
    contextInterface.m_getNormal = [](const SMikkTSpaceContext* pContext, float fvNormOut[], const int iFace, const int iVert) {
      const auto& data = reinterpret_cast<SceneRepresentationImpl*>(pContext->m_pUserData)->data;
      const auto& tri = data.triangles[iFace];
      const auto& vertex = data.vertices[tri.i[iVert]];
      fvNormOut[0] = vertex.nrm.x;
      fvNormOut[1] = vertex.nrm.y;
      fvNormOut[2] = vertex.nrm.z;
    };
    contextInterface.m_getTexCoord = [](const SMikkTSpaceContext* pContext, float fvTexcOut[], const int iFace, const int iVert) {
      const auto& data = reinterpret_cast<SceneRepresentationImpl*>(pContext->m_pUserData)->data;
      const auto& tri = data.triangles[iFace];
      const auto& vertex = data.vertices[tri.i[iVert]];
      fvTexcOut[0] = vertex.tex.x;
      fvTexcOut[1] = vertex.tex.y;
    };
    contextInterface.m_setTSpaceBasic = [](const SMikkTSpaceContext* pContext, const float fvTangent[], const float fSign, const int iFace, const int iVert) {
      auto& data = reinterpret_cast<SceneRepresentationImpl*>(pContext->m_pUserData)->data;
      const auto& tri = data.triangles[iFace];
      auto& vertex = data.vertices[tri.i[iVert]];
      if (is_valid_vector(vertex.tan) == false) {
        vertex.tan = normalize(float3{fvTangent[0], fvTangent[1], fvTangent[2]});
        vertex.btn = normalize(cross(vertex.tan, vertex.nrm) * fSign);
      }
    };

    SMikkTSpaceContext context = {};
    context.m_pUserData = this;
    context.m_pInterface = &contextInterface;

    genTangSpaceDefault(&context);
  }

  void validate_tangents(std::vector<bool>& referenced_vertices, bool force) {
    for (uint64_t vertex_index = 0, e = data.vertices.size(); vertex_index < e; ++vertex_index) {
      auto& v = data.vertices[vertex_index];
      if ((force == false) && is_valid_vector(v.tan) && is_valid_vector(v.btn)) {
        continue;
      }

      if (force || referenced_vertices[vertex_index]) {
        ETX_ASSERT(is_valid_vector(v.nrm));
        auto [t, b] = orthonormal_basis(v.nrm);
        v.tan = t;
        v.btn = b;
      }
    }

    for (uint64_t vertex_index = 0, e = data.vertices.size(); vertex_index < e; ++vertex_index) {
      orthogonalize(data.vertices[vertex_index]);
    }
  }

  void commit(bool spectral) {
    log::warning("Instancing area emitters...");

    log::warning("Building pixel sampler...");
    std::vector<float4> sampler_image;
    Film::generate_filter_image(Film::PixelFilterBlackmanHarris, sampler_image);
    uint32_t image_options = Image::BuildSamplingTable | Image::UniformSamplingTable;
    uint32_t image = context.images.add_from_data(sampler_image.data(), {Film::PixelFilterSize, Film::PixelFilterSize}, image_options, {}, {1.0f, 1.0f});
    scene.pixel_sampler = {image, 1.5f};

    float3 bbox_min = data.triangles.empty() ? float3{-1.0f, -1.0f, -1.0f} : float3{kMaxFloat, kMaxFloat, kMaxFloat};
    float3 bbox_max = data.triangles.empty() ? float3{+1.0f, +1.0f, +1.0f} : float3{-kMaxFloat, -kMaxFloat, -kMaxFloat};
    for (const auto& tri : data.triangles) {
      bbox_min = min(bbox_min, data.vertices[tri.i[0]].pos);
      bbox_min = min(bbox_min, data.vertices[tri.i[1]].pos);
      bbox_min = min(bbox_min, data.vertices[tri.i[2]].pos);
      bbox_max = max(bbox_max, data.vertices[tri.i[0]].pos);
      bbox_max = max(bbox_max, data.vertices[tri.i[1]].pos);
      bbox_max = max(bbox_max, data.vertices[tri.i[2]].pos);
    }

    scene.bounding_sphere_center = 0.5f * (bbox_min + bbox_max);
    scene.bounding_sphere_radius = length(bbox_max - scene.bounding_sphere_center);
    scene.vertices = {data.vertices.data(), data.vertices.size()};
    scene.triangles = {data.triangles.data(), data.triangles.size()};
    scene.materials = {data.materials.data(), data.materials.size()};
    scene.spectrums = {data.spectrum_values.data(), data.spectrum_values.size()};
    scene.images = {context.images.as_array(), context.images.array_size()};
    scene.mediums = {context.mediums.as_array(), context.mediums.array_size()};

    rebuild_area_emitters();

    scene.flags = Scene::Committed | (spectral ? Scene::Spectral : 0u);
  }

  float2 make_float2(const float values[]) {
    return {values[0], values[1]};
  }

  float3 make_float3(const float values[]) {
    return {values[0], values[1], values[2]};
  }

  std::vector<const char*> split_params(char* data) {
    std::vector<const char*> params;
    const char* begin = data;
    char* token = data;
    while (*token != 0) {
      if (*token == 0x20) {
        *token++ = 0;
        params.emplace_back(begin);
        begin = token;
      } else {
        ++token;
      }
    }
    params.emplace_back(begin);
    return params;
  }

  uint32_t load_reflectance_spectrum(SceneData& data, char* values) {
    auto params = split_params(values);

    if (params.size() == 1) {
      uint32_t index = data.find_spectrum(params[0]);
      if (index != kInvalidIndex)
        return index;
    }

    if (params.size() == 3) {
      float3 value = gamma_to_linear({
        static_cast<float>(atof(params[0])),
        static_cast<float>(atof(params[1])),
        static_cast<float>(atof(params[2])),
      });
      return data.add_spectrum(SpectralDistribution::rgb_reflectance(value));
    }

    return 0u;
  }

  SpectralDistribution load_illuminant_spectrum(SceneData& data, char* values) {
    auto params = split_params(values);

    if (params.size() == 1) {
      float value = 0.0f;
      if (sscanf(params[0], "%f", &value) == 1) {
        return SpectralDistribution::rgb_luminance({value, value, value});
      }

      auto i = data.find_spectrum(params[0]);
      if (i != kInvalidIndex)
        return data.spectrum_values[i];
    }

    if (params.size() == 3) {
      float3 value = {
        static_cast<float>(atof(params[0])),
        static_cast<float>(atof(params[1])),
        static_cast<float>(atof(params[2])),
      };
      return SpectralDistribution::rgb_luminance(value);
    }

    SpectralDistribution emitter_spectrum = SpectralDistribution::rgb_luminance({1.0f, 1.0f, 1.0f});

    float scale = 1.0f;
    for (uint64_t i = 0, count = params.size(); i < count; ++i) {
      if ((strcmp(params[i], "blackbody") == 0) && (i + 1 < count)) {
        float temperature = static_cast<float>(atof(params[i + 1]));
        emitter_spectrum = SpectralDistribution::from_black_body(temperature, 1.0f);
        i += 1;
      } else if ((strcmp(params[i], "nblackbody") == 0) && (i + 1 < count)) {
        float temperature = static_cast<float>(atof(params[i + 1]));
        emitter_spectrum = SpectralDistribution::from_normalized_black_body(temperature, 1.0f);
        i += 1;
      } else if ((strcmp(params[i], "scale") == 0) && (i + 1 < count)) {
        scale = static_cast<float>(atof(params[i + 1]));
        i += 1;
      }
    }

    emitter_spectrum.scale(scale);
    return emitter_spectrum;
  }

  enum : uint32_t {
    LoadFailed = 0u,
    LoadSucceeded = 1u << 0u,
    LoadCameraInfo = 1u << 1u,
  };

  void add_area_emitters_for_triangle(uint32_t triangle_index);
  void populate_area_emitters();
  void rebuild_area_emitters();

  uint32_t load_from_obj(const char* file_name, const char* mtl_file);

  uint32_t load_from_gltf(const char* file_name, bool binary);
  void load_gltf_materials(const tinygltf::Model&);
  void load_gltf_node(const tinygltf::Model& model, const tinygltf::Node&, const float4x4& transform);
  void load_gltf_mesh(const tinygltf::Node& node, const tinygltf::Model& model, const tinygltf::Mesh&, const float4x4& transform);
  void load_gltf_camera(const tinygltf::Node& node, const tinygltf::Model& model, const tinygltf::Camera&, const float4x4& transform);
  float4x4 build_gltf_node_transform(const tinygltf::Node& node);

  void parse_material(const char* base_dir, const tinyobj::material_t& material);
  void parse_camera(const char* base_dir, const tinyobj::material_t& material);
  void parse_medium(const char* base_dir, const tinyobj::material_t& material);
  void parse_directional_light(const char* base_dir, const tinyobj::material_t& material);
  void parse_env_light(const char* base_dir, const tinyobj::material_t& material);
  void parse_atmosphere_light(const char* base_dir, const tinyobj::material_t& material);
  void parse_spectrum(const char* base_dir, const tinyobj::material_t& material);

  uint32_t load_reflectance_spectrum(char* data);
  SpectralDistribution load_illuminant_spectrum(char* data);

  void parse_obj_materials(const char* base_dir, const std::vector<tinyobj::material_t>& obj_materials);
};

void build_camera(Camera& camera, const float3& origin, const float3& target, const float3& up, const uint2& viewport, const float fov) {
  float4x4 view = look_at(origin, target, up);
  float4x4 proj = perspective(fov * kPi / 180.0f, viewport.x, viewport.y, camera.clip_near, camera.clip_far);
  float4x4 inv_view = inverse(view);

  camera.target = target;
  camera.position = {inv_view.col[3].x, inv_view.col[3].y, inv_view.col[3].z};
  camera.side = {view.col[0].x, view.col[1].x, view.col[2].x};
  camera.up = {view.col[0].y, view.col[1].y, view.col[2].y};
  camera.direction = {-view.col[0].z, -view.col[1].z, -view.col[2].z};
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
  return _private->context.mediums.mapping();
}

uint32_t SceneRepresentation::add_medium(const char* name) {
  SpectralDistribution null_spectrum = SpectralDistribution::null();
  uint32_t handle = _private->context.add_medium(_private->scene, _private->data, Medium::Class::Homogeneous, name, nullptr, null_spectrum, null_spectrum, 0.0f, true);
  _private->scene.mediums = {_private->context.mediums.as_array(), _private->context.mediums.array_size()};
  return handle;
}

void SceneRepresentation::rebuild_area_emitters() {
  _private->rebuild_area_emitters();
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

template <class T>
inline void get_values(const std::vector<T>& a, T* ptr, uint64_t count) {
  for (uint64_t i = 0, e = a.size() < count ? a.size() : count; i < e; ++i) {
    *ptr++ = a[i];
  }
}

uint32_t load_reflectance_spectrum(SceneData& data, char* values);
SpectralDistribution load_illuminant_spectrum(SceneData& data, char* values);

bool SceneRepresentation::load_from_file(const char* filename, uint32_t options) {
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

  auto& camera = _private->active_camera;
  float3 camera_target = camera.position + camera.direction;
  float camera_fov = get_camera_fov(camera);
  float camera_focal_len = fov_to_focal_length(camera_fov);
  bool use_focal_len = false;
  bool force_tangents = false;
  bool spectral_scene = false;

  if (strcmp(get_file_ext(filename), ".json") == 0) {
    auto js = json_from_file(filename);
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
      } else if (json_get_bool(i, "force-tangents", bool_value)) {
        force_tangents = bool_value;
      } else if ((key == "camera") && obj.is_object()) {
        for (auto ci = obj.begin(), ce = obj.end(); ci != ce; ++ci) {
          const auto& ckey = ci.key();
          const auto& cobj = ci.value();
          if (json_get_string(ci, "class", str_value)) {
            camera.cls = str_value == "eq" ? Camera::Class::Equirectangular : Camera::Class::Perspective;
          } else if (json_get_float(ci, "fov", float_value)) {
            camera_fov = float_value;
          } else if (json_get_float(ci, "focal-length", float_value)) {
            camera_focal_len = float_value;
            use_focal_len = true;
          } else if (json_get_float(ci, "lens-radius", float_value)) {
            _private->active_camera.lens_radius = float_value;
          } else if (json_get_float(ci, "focal-distance", float_value)) {
            _private->active_camera.focal_distance = float_value;
          } else if (json_get_float(ci, "clip-near", float_value)) {
            _private->active_camera.clip_near = float_value;
          } else if (json_get_float(ci, "clip-far", float_value)) {
            _private->active_camera.clip_far = float_value;
          } else if (cobj.is_array()) {
            if (ckey == "origin") {
              auto values = cobj.get<std::vector<float>>();
              get_values(values, &camera.position.x, 3llu);
            } else if (ckey == "target") {
              auto values = cobj.get<std::vector<float>>();
              get_values(values, &camera_target.x, 3llu);
            } else if (ckey == "up") {
              auto values = cobj.get<std::vector<float>>();
              get_values(values, &camera.up.x, 3llu);
            } else if (ckey == "viewport") {
              auto values = cobj.get<std::vector<uint32_t>>();
              get_values(values, &camera.film_size.x, 2llu);
            } else {
              log::warning("Unhandled value in camera description : %s", key.c_str());
            }
          }
        }
      } else {
        log::warning("Unhandled value in scene description : %s", key.c_str());
      }
    }
    _private->data.json_file_name = filename;
  }

  if (camera.film_size.x * camera.film_size.y == 0) {
    camera.film_size = {1280, 720};
  }

  uint32_t load_result = SceneRepresentationImpl::LoadFailed;

  auto ext = get_file_ext(_private->data.geometry_file_name.c_str());
  if (strcmp(ext, ".obj") == 0) {
    load_result = _private->load_from_obj(_private->data.geometry_file_name.c_str(), _private->data.materials_file_name.c_str());
  } else if (strcmp(ext, ".gltf") == 0) {
    load_result = _private->load_from_gltf(_private->data.geometry_file_name.c_str(), false);
  } else if (strcmp(ext, ".glb") == 0) {
    load_result = _private->load_from_gltf(_private->data.geometry_file_name.c_str(), true);
  }

  if ((load_result & SceneRepresentationImpl::LoadSucceeded) == 0) {
    return false;
  }

  if (options & SetupCamera) {
    if (_private->data.cameras.empty()) {
      if ((load_result & SceneRepresentationImpl::LoadCameraInfo) == 0) {
        if (use_focal_len) {
          camera_fov = focal_length_to_fov(camera_focal_len) * 180.0f / kPi;
        }
        build_camera(camera, camera.position, camera_target, camera.up, camera.film_size, camera_fov);
      }
    } else {
      auto it = std::find_if(_private->data.cameras.begin(), _private->data.cameras.end(), [](const auto& e) {
        return e.active;
      });
      const auto& selected = (it != _private->data.cameras.end()) ? *it : _private->data.cameras.front();
      _private->active_camera = selected.cam;
    }
  }

  if (_private->data.emitter_profiles.empty()) {
    tinyobj::material_t mtl = {};
    mtl.unknown_parameter.emplace_back("direction", "0.0 2.0 1.0");
    mtl.unknown_parameter.emplace_back("quality", ETX_DEBUG ? "0.0625" : "0.125");
    mtl.unknown_parameter.emplace_back("angular_diameter", "0.5422");
    mtl.unknown_parameter.emplace_back("anisotropy", "0.825");
    mtl.unknown_parameter.emplace_back("altitude", "1000.0");
    mtl.unknown_parameter.emplace_back("scale", "1.0");
    mtl.unknown_parameter.emplace_back("sky_scale", "1.0");
    mtl.unknown_parameter.emplace_back("sun_scale", "1.0");
    mtl.unknown_parameter.emplace_back("rayleigh", "1.0");
    mtl.unknown_parameter.emplace_back("mie", "1.0");
    mtl.unknown_parameter.emplace_back("ozone", "1.0");
    _private->parse_atmosphere_light(base_folder, mtl);
    _private->context.images.load_images();
  }

  _private->validate_materials();

  {
    TimeMeasure m = {};
    log::warning("Validating normals and tangents...");
    std::vector<bool> referenced_vertices;
    _private->validate_normals(referenced_vertices);
    _private->build_tangents();
    _private->validate_tangents(referenced_vertices, force_tangents);
    log::warning("Normals and tangents validated in %.2f sec", m.measure());
  }

  _private->commit(spectral_scene);

  return true;
}

void SceneRepresentationImpl::add_area_emitters_for_triangle(uint32_t triangle_index) {
  if (triangle_index >= data.triangles.size())
    return;

  const auto& tri = data.triangles[triangle_index];
  uint32_t material_index = tri.material_index;
  if (material_index >= data.materials.size())
    return;

  const auto& mtl = data.materials[material_index];
  if (mtl.emission.spectrum_index == kInvalidIndex)
    return;
  if (mtl.emission.spectrum_index >= data.spectrum_values.size())
    return;

  float texture_emission = 1.0f;
  if (mtl.emission.image_index != kInvalidIndex) {
    const auto& img = context.images.get(mtl.emission.image_index);
    constexpr float kBCScale = 4.0f;
    auto min_uv = min(data.vertices[tri.i[0]].tex, min(data.vertices[tri.i[1]].tex, data.vertices[tri.i[2]].tex));
    auto max_uv = max(data.vertices[tri.i[0]].tex, max(data.vertices[tri.i[1]].tex, data.vertices[tri.i[2]].tex));
    float u_size = kBCScale * max(1.0f, ceil((max_uv.x - min_uv.x) * img.fsize.x));
    float du = 1.0f / u_size;
    float v_size = kBCScale * max(1.0f, ceil((max_uv.y - min_uv.y) * img.fsize.y));
    float dv = 1.0f / v_size;
    for (float v = 0.0f; v < 1.0f; v += dv) {
      for (float u = 0.0f; u < 1.0f; u += dv) {
        float2 uv = lerp_uv({data.vertices.data(), data.vertices.size()}, tri, random_barycentric({u, v}));
        float4 val = img.evaluate(uv, nullptr);
        texture_emission += luminance(to_float3(val)) * du * dv * val.w;
      }
    }
  }

  float tri_area = triangle_area(tri);
  float spectrum_weight = data.spectrum_values[mtl.emission.spectrum_index].luminance();
  float additional_weight = (mtl.two_sided ? 2.0f : 1.0f) * (tri_area * kPi) * texture_emission;
  if ((additional_weight <= 0.0f) || (spectrum_weight <= 0.0f))
    return;

  uint32_t profile_index = kInvalidIndex;
  auto mapping_it = data.material_to_emitter_profile.find(material_index);
  if (mapping_it != data.material_to_emitter_profile.end()) {
    profile_index = mapping_it->second;
  } else {
    profile_index = static_cast<uint32_t>(data.emitter_profiles.size());
    data.material_to_emitter_profile[material_index] = profile_index;
    data.emitter_profiles.emplace_back(EmitterProfile::Class::Area);
  }

  ETX_ASSERT(profile_index < data.emitter_profiles.size());
  auto& profile = data.emitter_profiles[profile_index];
  profile.emission = mtl.emission;

  uint32_t emitter_index = static_cast<uint32_t>(data.emitter_instances.size());
  auto& emitter = data.emitter_instances.emplace_back(EmitterProfile::Class::Area);
  emitter.profile = profile_index;
  emitter.triangle_index = triangle_index;
  emitter.triangle_area = tri_area;
  emitter.additional_weight = additional_weight;
  emitter.spectrum_weight = spectrum_weight;

  if (triangle_index < data.triangle_to_emitter.size()) {
    data.triangle_to_emitter[triangle_index] = emitter_index;
  }
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

  context.images.load_images();

  for (uint32_t tri_index = 0; tri_index < data.triangles.size(); ++tri_index) {
    add_area_emitters_for_triangle(tri_index);
  }
}

void SceneRepresentationImpl::rebuild_area_emitters() {
  populate_area_emitters();
  scene.triangle_to_emitter = {data.triangle_to_emitter.data(), data.triangle_to_emitter.size()};
  scene.emitter_profiles = {data.emitter_profiles.data(), data.emitter_profiles.size()};
  scene.emitter_instances = {data.emitter_instances.data(), data.emitter_instances.size()};
  build_emitters_distribution(scene);
}

uint32_t SceneRepresentationImpl::load_from_obj(const char* file_name, const char* mtl_file) {
  auto& triangles = data.triangles;
  auto& triangle_to_emitter = data.triangle_to_emitter;
  auto& vertices = data.vertices;
  auto& material_mapping = data.material_mapping;

  tinyobj::attrib_t obj_attrib;
  std::vector<tinyobj::shape_t> obj_shapes;
  std::vector<tinyobj::material_t> obj_materials;
  std::string warnings;
  std::string errors;

  char base_dir[2048] = {};
  get_file_folder(file_name, base_dir, sizeof(base_dir));

  if (tinyobj::LoadObj(&obj_attrib, &obj_shapes, &obj_materials, &warnings, &errors, file_name, base_dir, mtl_file) == false) {
    log::error("Failed to load OBJ from file: `%s`\n%s", file_name, errors.c_str());
    return LoadFailed;
  }

  if (warnings.empty() == false) {
    log::warning("Loaded OBJ from file: `%s`\n%s", file_name, warnings.c_str());
  }

  parse_obj_materials(base_dir, obj_materials);

  uint64_t total_triangles = 0;
  for (const auto& shape : obj_shapes) {
    total_triangles += shape.mesh.num_face_vertices.size();
  }

  triangles.reserve(total_triangles);
  triangle_to_emitter.reserve(total_triangles);
  vertices.reserve(total_triangles * 3);

  for (const auto& shape : obj_shapes) {
    uint64_t index_offset = 0;
    float3 shape_bbox_min = {kMaxFloat, kMaxFloat, kMaxFloat};
    float3 shape_bbox_max = {-kMaxFloat, -kMaxFloat, -kMaxFloat};

    for (uint64_t face = 0, face_e = shape.mesh.num_face_vertices.size(); face < face_e; ++face) {
      int material_id = shape.mesh.material_ids[face];
      if (material_id == -1) {
        continue;
      }
      const auto& source_material = obj_materials[material_id];

      uint64_t face_size = shape.mesh.num_face_vertices[face];
      ETX_ASSERT(face_size == 3);

      triangle_to_emitter.emplace_back(kInvalidIndex);
      auto& tri = triangles.emplace_back();
      tri.material_index = material_mapping.at(source_material.name);

      for (uint64_t vertex_index = 0; vertex_index < face_size; ++vertex_index) {
        const auto& index = shape.mesh.indices[index_offset + vertex_index];
        tri.i[vertex_index] = static_cast<uint32_t>(vertices.size());
        auto& vertex = vertices.emplace_back();
        vertex.pos = make_float3(obj_attrib.vertices.data() + (3 * index.vertex_index));
        if (index.normal_index >= 0) {
          vertex.nrm = make_float3(obj_attrib.normals.data() + (3 * index.normal_index));
        }
        if (index.texcoord_index >= 0) {
          vertex.tex = make_float2(obj_attrib.texcoords.data() + (2 * index.texcoord_index));
        }
      }
      index_offset += face_size;

      if (validate_triangle(tri) == false) {
        triangles.pop_back();
      }

      // TODO : deal with bounds!
      shape_bbox_max = max(shape_bbox_max, vertices[tri.i[0]].pos);
      shape_bbox_max = max(shape_bbox_max, vertices[tri.i[1]].pos);
      shape_bbox_max = max(shape_bbox_max, vertices[tri.i[2]].pos);
      shape_bbox_min = min(shape_bbox_min, vertices[tri.i[0]].pos);
      shape_bbox_min = min(shape_bbox_min, vertices[tri.i[1]].pos);
      shape_bbox_min = min(shape_bbox_min, vertices[tri.i[2]].pos);

      auto& mtl = data.materials[tri.material_index];
      if (mtl.int_medium != kInvalidIndex) {
        context.mediums.get(mtl.int_medium).bounds = {shape_bbox_min, 0.0f, shape_bbox_max, 0.0f};
      }
    }
  }

  return true;
}

void SceneRepresentationImpl::parse_camera(const char* base_dir, const tinyobj::material_t& material) {
  auto& entry = data.cameras.emplace_back();

  if (get_param(material, "class")) {
    entry.cam.cls = (strcmp(data_buffer, "eq") == 0) ? Camera::Class::Equirectangular : Camera::Class::Perspective;
  }

  if (get_param(material, "viewport")) {
    uint32_t val[2] = {};
    if (sscanf(data_buffer, "%u %u", val + 0, val + 1) == 2) {
      entry.cam.film_size = {val[0], val[1]};
    }
  }

  if (get_param(material, "origin")) {
    float val[3] = {};
    if (sscanf(data_buffer, "%f %f %f", val + 0, val + 1, val + 2) == 3) {
      entry.cam.position = {val[0], val[1], val[2]};
    }
  }

  float3 target = entry.cam.position + kWorldForward;
  if (get_param(material, "target")) {
    float val[3] = {};
    if (sscanf(data_buffer, "%f %f %f", val + 0, val + 1, val + 2) == 3) {
      target = {val[0], val[1], val[2]};
    }
  }

  entry.cam.up = kWorldUp;
  if (get_param(material, "up")) {
    float val[3] = {};
    if (sscanf(data_buffer, "%f %f %f", val + 0, val + 1, val + 2) == 3) {
      entry.cam.up = {val[0], val[1], val[2]};
    }
  }

  float fov = 50.0f;
  if (get_param(material, "fov")) {
    float val = 0.0f;
    if (sscanf(data_buffer, "%f", &val) == 1) {
      fov = val;
    }
  }

  if (get_param(material, "focal-length")) {
    float val = 0.0f;
    if (sscanf(data_buffer, "%f", &val) == 1) {
      fov = focal_length_to_fov(val) * 180.0f / kPi;
    }
  }

  if (get_param(material, "lens-radius")) {
    float val = 0.0f;
    if (sscanf(data_buffer, "%f", &val) == 1) {
      entry.cam.lens_radius = val;
    }
  }

  if (get_param(material, "focal-distance")) {
    float val = 0.0f;
    if (sscanf(data_buffer, "%f", &val) == 1) {
      entry.cam.focal_distance = val;
    }
  }

  if (get_param(material, "clip-near")) {
    float val = 0.0f;
    if (sscanf(data_buffer, "%f", &val) == 1) {
      entry.cam.clip_near = val;
    }
  }

  if (get_param(material, "clip-far")) {
    float val = 0.0f;
    if (sscanf(data_buffer, "%f", &val) == 1) {
      entry.cam.clip_far = val;
    }
  }

  if (get_param(material, "shape")) {
    char tmp_buffer[2048] = {};
    snprintf(tmp_buffer, sizeof(tmp_buffer), "%s/%s", base_dir, data_buffer);
    entry.cam.lens_image = context.add_image(tmp_buffer, Image::BuildSamplingTable | Image::UniformSamplingTable, {}, {1.0f, 1.0f});
  }

  if (get_param(material, "ext_medium")) {
    auto m = context.mediums.find(data_buffer);
    if (m == kInvalidIndex) {
      log::warning("Medium %s was not declared, but used in material %s as external medium\n", data_buffer, material.name.c_str());
    }
    entry.cam.medium_index = m;
  }

  if (get_param(material, "id")) {
    entry.id = data_buffer;
  }

  int active_flag = 0;
  if (get_param(material, "active")) {
    if (sscanf(data_buffer, "%d", &active_flag) == 1) {
      entry.active = (active_flag != 0);
    }
  }

  build_camera(entry.cam, entry.cam.position, target, entry.cam.up, entry.cam.film_size, fov);
}

void SceneRepresentationImpl::parse_medium(const char* base_dir, const tinyobj::material_t& material) {
  if (get_param(material, "id") == false) {
    log::warning("Medium does not have identifier - skipped");
    return;
  }

  std::string name = data_buffer;

  float anisotropy = 0.0f;
  if (get_param(material, "g")) {
    float val = {};
    if (sscanf(data_buffer, "%f", &val) == 1) {
      anisotropy = val;
    }
  }

  if (get_param(material, "anisotropy")) {
    float val = {};
    if (sscanf(data_buffer, "%f", &val) == 1) {
      anisotropy = val;
    }
  }

  SpectralDistribution s_a = SpectralDistribution::null();
  if (get_param(material, "absorption")) {
    float val[3] = {};
    int params_read = sscanf(data_buffer, "%f %f %f", val + 0, val + 1, val + 2);
    if (params_read == 3) {
      s_a = SpectralDistribution::rgb_reflectance({val[0], val[1], val[2]});
    } else if (params_read == 1) {
      s_a = SpectralDistribution::rgb_reflectance({val[0], val[0], val[0]});
    }
  }

  if (get_param(material, "absorbtion")) {
    log::warning("absorBtion used in medium: %s", name.c_str());
    float val[3] = {};
    int params_read = sscanf(data_buffer, "%f %f %f", val + 0, val + 1, val + 2);
    if (params_read == 3) {
      s_a = SpectralDistribution::rgb_reflectance({val[0], val[1], val[2]});
    } else if (params_read == 1) {
      s_a = SpectralDistribution::rgb_reflectance({val[0], val[0], val[0]});
    }
  }

  SpectralDistribution s_t = SpectralDistribution::null();
  if (get_param(material, "scattering")) {
    float val[3] = {};
    int params_read = sscanf(data_buffer, "%f %f %f", val + 0, val + 1, val + 2);
    if (params_read == 3) {
      s_t = SpectralDistribution::rgb_reflectance({val[0], val[1], val[2]});
    } else if (params_read == 1) {
      s_t = SpectralDistribution::rgb_reflectance({val[0], val[0], val[0]});
    }
  }

  if (get_param(material, "rayleigh")) {
    s_t = shared_scattering_spectrums.rayleigh;

    float scale = 1.0f;
    auto params = split_params(data_buffer);
    for (uint64_t i = 0, e = params.size(); i < e; ++i) {
      if ((strcmp(params[i], "scale") == 0) && (i + 1 < e)) {
        scale = static_cast<float>(atof(params[i + 1]));
      }
    }
    s_t.scale(scale / s_t.maximum_spectral_power());
  }

  if (get_param(material, "mie")) {
    s_t = shared_scattering_spectrums.mie;

    float scale = 1.0f;
    auto params = split_params(data_buffer);
    for (uint64_t i = 0, e = params.size(); i < e; ++i) {
      if ((strcmp(params[i], "scale") == 0) && (i + 1 < e)) {
        scale = static_cast<float>(atof(params[i + 1]));
      }
    }
    s_t.scale(scale / s_t.maximum_spectral_power());
  }

  if (get_param(material, "parametric")) {
    float3 color = {1.0f, 1.0f, 1.0f};
    float3 scattering_distances = {0.25f, 0.25f, 0.25f};

    float scale = 1.0f;
    auto params = split_params(data_buffer);
    for (uint64_t i = 0, e = params.size(); i < e; ++i) {
      if ((strcmp(params[i], "color") == 0) && (i + 3 < e)) {
        color = {
          static_cast<float>(atof(params[i + 1])),
          static_cast<float>(atof(params[i + 2])),
          static_cast<float>(atof(params[i + 3])),
        };
        i += 3;
      }
      if ((strcmp(params[i], "distance") == 0) && (i + 1 < e)) {
        float value = static_cast<float>(atof(params[i + 1]));
        scattering_distances = {value, value, value};
        i += 1;
      }
      if ((strcmp(params[i], "distances") == 0) && (i + 1 < e)) {
        scattering_distances = {
          static_cast<float>(atof(params[i + 1])),
          static_cast<float>(atof(params[i + 2])),
          static_cast<float>(atof(params[i + 3])),
        };
        i += 3;
      }
      if ((strcmp(params[i], "scale") == 0) && (i + 1 < e)) {
        scale = static_cast<float>(atof(params[i + 1]));
        i += 1;
      }
    }

    float3 albedo = {};
    float3 extinction = {};
    float3 scattering = {};
    subsurface::remap(color, scale * scattering_distances, albedo, extinction, scattering);

    float3 absorption = max({}, extinction - scattering);
    ETX_VALIDATE(absorption);

    s_t = SpectralDistribution::rgb_reflectance(scattering);
    s_a = SpectralDistribution::rgb_reflectance(absorption);
  }

  bool explicit_connections = true;
  if (get_param(material, "enclosed")) {
    explicit_connections = false;
  }

  Medium::Class cls = Medium::Class::Homogeneous;

  char tmp_buffer[2048] = {};

  if (get_param(material, "volume")) {
    if (strlen(data_buffer) > 0) {
      snprintf(tmp_buffer, sizeof(tmp_buffer), "%s%s", base_dir, data_buffer);
      cls = Medium::Class::Heterogeneous;
    }
  }

  context.add_medium(scene, data, cls, name.c_str(), tmp_buffer, s_a, s_t, anisotropy, explicit_connections);
}

void SceneRepresentationImpl::parse_directional_light(const char* base_dir, const tinyobj::material_t& material) {
  auto& instance = data.emitter_instances.emplace_back(EmitterProfile::Class::Directional);
  instance.profile = uint32_t(data.emitter_profiles.size());

  auto& e = data.emitter_profiles.emplace_back(EmitterProfile::Class::Directional);
  if (get_param(material, "color")) {
    e.emission.spectrum_index = data.add_spectrum(load_illuminant_spectrum(data_buffer));
  } else {
    e.emission.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_luminance({1.0f, 1.0f, 1.0f}));
  }

  e.direction = float3{1.0f, 1.0f, 1.0f};
  if (get_param(material, "direction")) {
    float value[3] = {};
    if (sscanf(data_buffer, "%f %f %f", value + 0, value + 1, value + 2) == 3) {
      e.direction = {value[0], value[1], value[2]};
    }
  }
  e.direction = normalize(e.direction);

  if (get_param(material, "image")) {
    char tmp_buffer[2048] = {};
    snprintf(tmp_buffer, sizeof(tmp_buffer), "%s/%s", base_dir, data_buffer);
    e.emission.image_index = context.add_image(tmp_buffer, Image::Regular, {}, {1.0f, 1.0f});
  }

  if (get_param(material, "angular_diameter")) {
    float val = {};
    if (sscanf(data_buffer, "%f", &val) == 1) {
      e.angular_size = val * kPi / 180.0f;
    }
  }
}

void SceneRepresentationImpl::parse_env_light(const char* base_dir, const tinyobj::material_t& material) {
  auto& instance = data.emitter_instances.emplace_back(EmitterProfile::Class::Environment);
  instance.profile = uint32_t(data.emitter_profiles.size());

  auto& e = data.emitter_profiles.emplace_back(EmitterProfile::Class::Environment);

  char tmp_buffer[2048] = {};
  if (get_param(material, "image")) {
    snprintf(tmp_buffer, sizeof(tmp_buffer), "%s/%s", base_dir, data_buffer);
  }

  float rotation = 0.0f;
  if (get_param(material, "rotation")) {
    // minus to align with Blender
    rotation = -static_cast<float>(atof(data_buffer)) / 360.0f;
  }

  float u_scale = 1.0f;
  if (get_param(material, "scale")) {
    float val = {};
    if (sscanf(data_buffer, "%f", &val) == 1) {
      u_scale = val;
    }
  }
  e.emission.image_index = context.add_image(tmp_buffer, Image::BuildSamplingTable | Image::RepeatU, {rotation, 0.0f}, {u_scale, 1.0f});

  if (get_param(material, "color")) {
    e.emission.spectrum_index = data.add_spectrum(load_illuminant_spectrum(data_buffer));
  } else {
    e.emission.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_luminance({1.0f, 1.0f, 1.0f}));
  }
}

void SceneRepresentationImpl::parse_atmosphere_light(const char* base_dir, const tinyobj::material_t& material) {
  float quality = 1.0f;
  if (get_param(material, "quality")) {
    float val = {};
    if (sscanf(data_buffer, "%f", &val) == 1) {
      quality = val;
    }
  }

  float scale = 1.0f;
  if (get_param(material, "scale")) {
    float val = {};
    if (sscanf(data_buffer, "%f", &val) == 1) {
      scale = val;
    }
  }
  float sun_scale = 1.0f;
  if (get_param(material, "sun_scale")) {
    float val = {};
    if (sscanf(data_buffer, "%f", &val) == 1) {
      sun_scale = val;
    }
  }
  float sky_scale = 1.0f;
  if (get_param(material, "sky_scale")) {
    float val = {};
    if (sscanf(data_buffer, "%f", &val) == 1) {
      sky_scale = val;
    }
  }

  float3 direction = normalize(float3{1.0f, 1.0f, 1.0f});
  float angular_size = 0.5422f * (kPi / 180.0f);

  scattering::Parameters scattering_parameters = {};

  if (get_param(material, "direction")) {
    float value[3] = {};
    if (sscanf(data_buffer, "%f %f %f", value + 0, value + 1, value + 2) == 3) {
      direction = normalize(float3{value[0], value[1], value[2]});
    }
  }
  if (get_param(material, "angular_diameter")) {
    float val = {};
    if (sscanf(data_buffer, "%f", &val) == 1) {
      angular_size = val * (kPi / 180.0f);
    }
  }
  if (get_param(material, "anisotropy")) {
    float val = {};
    if (sscanf(data_buffer, "%f", &val) == 1) {
      scattering_parameters.anisotropy = val;
    }
  }
  if (get_param(material, "altitude")) {
    float val = {};
    if (sscanf(data_buffer, "%f", &val) == 1) {
      scattering_parameters.altitude = val;
    }
  }
  if (get_param(material, "rayleigh")) {
    float val = {};
    if (sscanf(data_buffer, "%f", &val) == 1) {
      scattering_parameters.rayleigh_scale = val;
    }
  }
  if (get_param(material, "mie")) {
    float val = {};
    if (sscanf(data_buffer, "%f", &val) == 1) {
      scattering_parameters.mie_scale = val;
    }
  }
  if (get_param(material, "ozone")) {
    float val = {};
    if (sscanf(data_buffer, "%f", &val) == 1) {
      scattering_parameters.ozone_scale = val;
    }
  }

  float radiance_scale = scale * (kDoublePi * (1.0f - cosf(0.5f * angular_size)));
  auto sun_spectrum = SpectralDistribution::from_black_body(5900.0f, radiance_scale);

  constexpr uint2 kSunImageDimensions = uint2{128u, 128u};
  constexpr uint32_t kSkyImageBaseDimensions = 1024u;

  uint2 sky_image_dimensions = uint2{2u * kSkyImageBaseDimensions, kSkyImageBaseDimensions};
  sky_image_dimensions.x = max(64u, uint32_t(sky_image_dimensions.x * quality));
  sky_image_dimensions.y = max(64u, uint32_t(sky_image_dimensions.y * quality));

  {
    auto& instance = data.emitter_instances.emplace_back(EmitterProfile::Class::Directional);
    instance.profile = uint32_t(data.emitter_profiles.size());

    auto& d = data.emitter_profiles.emplace_back(EmitterProfile::Class::Directional);
    d.emission.spectrum_index = data.add_spectrum(sun_spectrum);
    data.spectrum_values[d.emission.spectrum_index].scale(sun_scale);
    d.angular_size = angular_size;
    d.direction = direction;

    if (angular_size > 0.0f) {
      d.emission.image_index = context.add_image(nullptr, kSunImageDimensions, Image::Delay, {}, {1.0f, 1.0f});
      auto& img = context.images.get(d.emission.image_index);
      scattering::generate_sun_image(scattering_parameters, kSunImageDimensions, direction, angular_size, img.pixels.f32.a, shared_scattering_spectrums, scheduler);
    }
  }

  {
    auto& instance = data.emitter_instances.emplace_back(EmitterProfile::Class::Environment);
    instance.profile = uint32_t(data.emitter_profiles.size());

    auto& e = data.emitter_profiles.emplace_back(EmitterProfile::Class::Environment);
    e.emission.spectrum_index = data.add_spectrum(sun_spectrum);
    data.spectrum_values[e.emission.spectrum_index].scale(sky_scale);
    uint32_t image_options = Image::BuildSamplingTable | Image::Delay;
    e.emission.image_index = context.add_image(nullptr, sky_image_dimensions, image_options, {}, {1.0f, 1.0f});
    e.direction = direction;

    auto& img = context.images.get(e.emission.image_index);
    scattering::generate_sky_image(scattering_parameters, sky_image_dimensions, direction, data.atmosphere_extinction, img.pixels.f32.a, shared_scattering_spectrums, scheduler);
  }
}

void SceneRepresentationImpl::parse_spectrum(const char* base_dir, const tinyobj::material_t& material) {
  if (get_param(material, "id") == false) {
    log::warning("Spectrum does not have identifier - skipped");
    return;
  }
  std::string name = data_buffer;

  bool initialized = false;
  bool illuminant = false;

  float scale = 1.0f;

  if (get_param(material, "scale")) {
    scale = static_cast<float>(atof(data_buffer));
  }

  if (get_param(material, "illuminant")) {
    illuminant = true;
  }

  if (get_param(material, "rgb")) {
    auto params = split_params(data_buffer);

    if (params.size() < 3) {
      log::warning("Spectrum `%s` uses RGB but did not provide all values - skipped", name.c_str());
      return;
    }

    float3 value = {
      static_cast<float>(atof(params[0])),
      static_cast<float>(atof(params[1])),
      static_cast<float>(atof(params[2])),
    };
    value = gamma_to_linear(value);

    data.add_spectrum(name.c_str(), illuminant ? SpectralDistribution::rgb_luminance(value) : SpectralDistribution::rgb_reflectance(value));
    initialized = true;
  } else if (get_param(material, "blackbody")) {
    auto params = split_params(data_buffer);
    if (params.size() < 1) {
      log::warning("Spectrum `%s` uses blackbody but did not provide temperature value - skipped", name.c_str());
      return;
    }

    float t = static_cast<float>(atof(params[0]));
    data.add_spectrum(name.c_str(), SpectralDistribution::from_black_body(t, scale));
    initialized = true;
  } else if (get_param(material, "nblackbody")) {
    auto params = split_params(data_buffer);
    if (params.size() < 1) {
      log::warning("Spectrum `%s` uses nblackbody but did not provide temperature value - skipped", name.c_str());
      return;
    }

    float scale = 1.0f;
    for (uint64_t i = 0, e = params.size(); i < e; ++i) {
      if ((i + 1 < e) && (strcmp(params[i], "scale") == 0)) {
        scale = static_cast<float>(atof(params[i + 1]));
        ++i;
      }
    }

    float t = static_cast<float>(atof(params[0]));
    data.add_spectrum(name.c_str(), SpectralDistribution::from_normalized_black_body(t, scale));
    initialized = true;
  }

  bool have_samples = get_param(material, "samples");

  if ((have_samples == false) && (initialized == false)) {
    log::warning("Spectrum `%s` does not have samples or RBG or (n)blackbody - skipped", name.c_str());
    return;
  } else if (initialized && have_samples) {
    log::warning("Spectrum `%s` uses both RGB or (n)blackbody and samples set - samples will be used", name.c_str());
  } else if (initialized == false) {
    auto params = split_params(data_buffer);
    if (params.size() % 2) {
      log::warning("Spectrum `%s` have uneven number samples - skipped", name.c_str());
      return;
    }

    std::vector<float2> samples;
    samples.reserve(params.size() / 2 + 1);

    for (uint64_t i = 0, e = params.size(); i < e; i += 2) {
      float2& smp = samples.emplace_back();
      smp.x = static_cast<float>(atof(params[i + 0]));
      smp.y = static_cast<float>(atof(params[i + 1]));
    }

    if (samples.empty() == false) {
      log::warning("Spectrum `%s` sample set is empty - skipped", name.c_str());
    }

    auto spectrum = SpectralDistribution::from_samples(samples.data(), samples.size());

    if (get_param(material, "normalize")) {
      float3 xyz = spectrum.integrate_to_xyz();
      bool normalize_rgb = strcmp(data_buffer, "luminance") != 0;
      float3 rgb = spectrum::xyz_to_rgb(xyz);
      float lum = normalize_rgb ? fmaxf(fmaxf(0.0f, rgb.x), fmaxf(rgb.y, rgb.z)) : xyz.y;
      if (lum > kEpsilon) {
        spectrum.scale(1.0f / lum);
      }
    }
    data.add_spectrum(name.c_str(), spectrum);
  }

  uint32_t i = data.find_spectrum(name.c_str());
  if (i != kInvalidIndex) {
    data.spectrum_values[i].scale(scale);
  }
}

uint32_t SceneRepresentationImpl::load_reflectance_spectrum(char* adata) {
  auto params = split_params(adata);

  if (params.size() == 1) {
    auto i = data.find_spectrum(params[0]);
    if (i != kInvalidIndex)
      return i;
  }

  if (params.size() == 3) {
    float3 value = gamma_to_linear({
      static_cast<float>(atof(params[0])),
      static_cast<float>(atof(params[1])),
      static_cast<float>(atof(params[2])),
    });
    return data.add_spectrum(SpectralDistribution::rgb_reflectance(value));
  }

  return 0;
}

SpectralDistribution SceneRepresentationImpl::load_illuminant_spectrum(char* adata) {
  auto params = split_params(adata);

  if (params.size() == 1) {
    float value = 0.0f;
    if (sscanf(params[0], "%f", &value) == 1) {
      return SpectralDistribution::rgb_luminance({value, value, value});
    }

    auto i = data.find_spectrum(params[0]);
    if (i != kInvalidIndex)
      return data.spectrum_values[i];
  }

  if (params.size() == 3) {
    float3 value = {
      static_cast<float>(atof(params[0])),
      static_cast<float>(atof(params[1])),
      static_cast<float>(atof(params[2])),
    };
    return SpectralDistribution::rgb_luminance(value);
  }

  SpectralDistribution emitter_spectrum = SpectralDistribution::rgb_luminance({1.0f, 1.0f, 1.0f});

  float scale = 1.0f;
  for (uint64_t i = 0, e = params.size(); i < e; ++i) {
    if ((strcmp(params[i], "blackbody") == 0) && (i + 1 < e)) {
      float t = static_cast<float>(atof(params[i + 1]));
      emitter_spectrum = SpectralDistribution::from_black_body(t, 1.0f);
      i += 1;
    } else if ((strcmp(params[i], "nblackbody") == 0) && (i + 1 < e)) {
      float t = static_cast<float>(atof(params[i + 1]));
      emitter_spectrum = SpectralDistribution::from_normalized_black_body(t, 1.0f);
      i += 1;
    } else if ((strcmp(params[i], "scale") == 0) && (i + 1 < e)) {
      scale = static_cast<float>(atof(params[i + 1]));
      i += 1;
    }
  }
  emitter_spectrum.scale(scale);

  return emitter_spectrum;
}

void SceneRepresentationImpl::parse_material(const char* base_dir, const tinyobj::material_t& material) {
  auto& material_mapping = data.material_mapping;

  uint32_t material_index = kInvalidIndex;

  if (material_mapping.count(material.name) == 0) {
    material_index = data.add_material(material.name.c_str());
  } else {
    material_index = material_mapping.at(material.name);
  }

  auto& mtl = data.materials[material_index];

  mtl.cls = Material::Class::Diffuse;
  mtl.emission = {};
  mtl.emission_collimation = 0.0f;

  if (get_param(material, "base")) {
    auto i = material_mapping.find(data_buffer);
    if (i != material_mapping.end()) {
      mtl = data.materials[i->second];
    }
  }

  if (get_param(material, "Kd")) {
    mtl.scattering.spectrum_index = load_reflectance_spectrum(data_buffer);
  }

  if (get_param(material, "Ks")) {
    mtl.reflectance.spectrum_index = load_reflectance_spectrum(data_buffer);
  }

  if (get_param(material, "Kt")) {
    mtl.scattering.spectrum_index = load_reflectance_spectrum(data_buffer);
  }

  if (get_param(material, "two_sided")) {
    int val = 0;
    if (sscanf(data_buffer, "%d", &val) == 1) {
      mtl.two_sided = (val != 0) ? 1u : 0u;
    } else {
      // accept tokens like "true"/"false"
      mtl.two_sided = ((strcmp(data_buffer, "true") == 0) || (strcmp(data_buffer, "on") == 0)) ? 1u : 0u;
    }
  }

  if (get_param(material, "opacity")) {
    float val = 1.0f;
    if (sscanf(data_buffer, "%f", &val) == 1) {
      mtl.opacity = clamp(val, 0.0f, 1.0f);
    }
  }

  if (get_param(material, "Pr")) {
    float4 value = {};
    if (sscanf(data_buffer, "%f %f", &value.x, &value.y) == 2) {
      mtl.roughness.value = sqr(value);
    } else if (sscanf(data_buffer, "%f", &value.x) == 1) {
      mtl.roughness = {sqr(value.x), sqr(value.x)};
    }
  }

  // Principled extras: metalness / transmission / roughness texture maps
  if (get_param(material, "metalness")) {
    float val = 0.0f;
    if (sscanf(data_buffer, "%f", &val) == 1) {
      mtl.metalness.value = {val, val, val, val};
    }
  }

  if (get_param(material, "transmission")) {
    float val = 0.0f;
    if (sscanf(data_buffer, "%f", &val) == 1) {
      mtl.transmission.value = {val, val, val, val};
    }
  }

  if (get_param(material, "map_Pr")) {
    auto params = split_params(data_buffer);
    const char* path = (params.empty() == false) ? params[0] : nullptr;
    int channel = 0;
    for (uint64_t i = 0, e = params.size(); i < e; ++i) {
      if ((strcmp(params[i], "channel") == 0) && (i + 1 < e)) {
        channel = std::max(0, atoi(params[i + 1]));
        ++i;
      }
    }
    if (path && get_file(base_dir, path)) {
      mtl.roughness.image_index = context.add_image(data_buffer, Image::RepeatU | Image::RepeatV, {}, {1.0f, 1.0f});
      mtl.roughness.channel = static_cast<uint32_t>(channel);
    }
  }

  if (get_param(material, "map_Ml")) {
    auto params = split_params(data_buffer);
    const char* path = (params.empty() == false) ? params[0] : nullptr;
    int channel = 0;
    for (uint64_t i = 0, e = params.size(); i < e; ++i) {
      if ((strcmp(params[i], "channel") == 0) && (i + 1 < e)) {
        channel = std::max(0, atoi(params[i + 1]));
        ++i;
      }
    }
    if (path && get_file(base_dir, path)) {
      mtl.metalness.image_index = context.add_image(data_buffer, Image::RepeatU | Image::RepeatV, {}, {1.0f, 1.0f});
      mtl.metalness.channel = static_cast<uint32_t>(channel);
    }
  }

  if (get_param(material, "map_Tm")) {
    auto params = split_params(data_buffer);
    const char* path = (params.empty() == false) ? params[0] : nullptr;
    int channel = 0;
    for (uint64_t i = 0, e = params.size(); i < e; ++i) {
      if ((strcmp(params[i], "channel") == 0) && (i + 1 < e)) {
        channel = std::max(0, atoi(params[i + 1]));
        ++i;
      }
    }
    if (path && get_file(base_dir, path)) {
      mtl.transmission.image_index = context.add_image(data_buffer, Image::RepeatU | Image::RepeatV, {}, {1.0f, 1.0f});
      mtl.transmission.channel = static_cast<uint32_t>(channel);
    }
  }

  if (get_file(base_dir, material.diffuse_texname)) {
    mtl.scattering.image_index = context.add_image(data_buffer, Image::RepeatU | Image::RepeatV, {}, {1.0f, 1.0f});
  }

  if (get_file(base_dir, material.specular_texname)) {
    mtl.reflectance.image_index = context.add_image(data_buffer, Image::RepeatU | Image::RepeatV, {}, {1.0f, 1.0f});
  }

  if (get_file(base_dir, material.transmittance_texname)) {
    mtl.scattering.image_index = context.add_image(data_buffer, Image::RepeatU | Image::RepeatV, {}, {1.0f, 1.0f});
  }

  if (get_param(material, "material")) {
    auto params = split_params(data_buffer);
    for (uint64_t i = 0, e = params.size(); i < e; ++i) {
      if ((strcmp(params[i], "class") == 0) && (i + 1 < e)) {
        mtl.cls = material_string_to_class(params[i + 1]);
        i += 1;
      }
    }
  }

  if (get_param(material, "diffuse")) {
    uint32_t var = 0;
    if (sscanf(data_buffer, "%u", &var) == 1) {
      mtl.diffuse_variation = var;
    }
  }

  auto load_ior = [this](RefractiveIndex& target, const char* data_buffer) {
    float2 values = {};
    int values_read = sscanf(data_buffer, "%f %f", &values.x, &values.y);
    target.cls = SpectralDistribution::Class::Dielectric;
    if (values_read == 1) {
      target.eta_index = data.add_spectrum(SpectralDistribution::constant(values.x));
      target.k_index = kInvalidIndex;
    } else if (values_read == 2) {
      target.cls = SpectralDistribution::Class::Conductor;
      target.eta_index = data.add_spectrum(SpectralDistribution::constant(values.x));
      target.k_index = data.add_spectrum(SpectralDistribution::constant(values.y));
    } else {
      SpectralDistribution eta_spd = {};
      SpectralDistribution k_spd = {};
      SpectralDistribution::Class cls = SpectralDistribution::Class::Invalid;
      if (load_ior_from_identifier(data_buffer, eta_spd, k_spd, cls) == false) {
        std::filesystem::path fallback = locate_spectrum_file(data_buffer, {});
        if (fallback.empty() == false) {
          cls = RefractiveIndex::load_from_file(fallback.string().c_str(), eta_spd, k_spd);
        }
      }

      if (cls == SpectralDistribution::Class::Invalid) {
        log::warning("Unable to load IOR spectrum `%s`, falling back to 1.5 dielectric", data_buffer);
        cls = SpectralDistribution::Class::Dielectric;
        eta_spd = SpectralDistribution::constant(1.5f);
        k_spd = SpectralDistribution::constant(0.0f);
      }

      target.cls = cls;
      target.eta_index = data.add_spectrum(eta_spd);
      if (cls == SpectralDistribution::Class::Conductor) {
        target.k_index = data.add_spectrum(k_spd);
      } else {
        target.k_index = k_spd.empty() ? data.add_spectrum(SpectralDistribution::constant(0.0f)) : data.add_spectrum(k_spd);
      }
    }
  };

  if (get_param(material, "int_ior")) {
    load_ior(mtl.int_ior, data_buffer);
  } else {
    mtl.int_ior.cls = SpectralDistribution::Class::Dielectric;
    mtl.int_ior.eta_index = data.add_spectrum(SpectralDistribution::constant(1.5f));
    mtl.int_ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
  }

  if (get_param(material, "ext_ior")) {
    load_ior(mtl.ext_ior, data_buffer);
  } else {
    mtl.ext_ior.cls = SpectralDistribution::Class::Dielectric;
    mtl.ext_ior.eta_index = data.add_spectrum(SpectralDistribution::constant(1.0f));
    mtl.ext_ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
  }

  if (get_param(material, "int_medium")) {
    auto m = context.mediums.find(data_buffer);
    if (m == kInvalidIndex) {
      log::warning("Medium %s was not declared, but used in material %s as internal medium", data_buffer, material.name.c_str());
    }
    mtl.int_medium = m;
  }

  if (get_param(material, "ext_medium")) {
    auto m = context.mediums.find(data_buffer);
    if (m == kInvalidIndex) {
      log::warning("Medium %s was not declared, but used in material %s as external medium\n", data_buffer, material.name.c_str());
    }
    mtl.ext_medium = m;
  }

  if (get_param(material, "normalmap")) {
    auto params = split_params(data_buffer);
    for (uint64_t i = 0, e = params.size(); i < e; ++i) {
      if ((strcmp(params[i], "image") == 0) && (i + 1 < e)) {
        char buffer[1024] = {};
        snprintf(buffer, sizeof(buffer), "%s/%s", base_dir, params[i + 1]);
        mtl.normal_image_index = context.add_image(buffer, Image::RepeatU | Image::RepeatV | Image::SkipSRGBConversion, {}, {1.0f, 1.0f});
        i += 1;
      }
      if ((strcmp(params[i], "scale") == 0) && (i + 1 < e)) {
        mtl.normal_scale = static_cast<float>(atof(params[i + 1]));
        i += 1;
      }
    }
  }

  if (get_param(material, "thinfilm")) {
    auto params = split_params(data_buffer);

    for (uint64_t i = 0, e = params.size(); i < e; ++i) {
      if ((strcmp(params[i], "image") == 0) && (i + 1 < e)) {
        char buffer[1024] = {};
        snprintf(buffer, sizeof(buffer), "%s/%s", base_dir, params[i + 1]);
        mtl.thinfilm.thinkness_image = context.add_image(buffer, Image::RepeatU | Image::RepeatV, {}, {1.0f, 1.0f});
        i += 1;
      }

      if ((strcmp(params[i], "range") == 0) && (i + 2 < e)) {
        mtl.thinfilm.min_thickness = static_cast<float>(atof(params[i + 1]));
        mtl.thinfilm.max_thickness = static_cast<float>(atof(params[i + 2]));
        i += 2;
      }

      if ((strcmp(params[i], "ior") == 0) && (i + 1 < e)) {
        float value = 0.0f;
        if (sscanf(params[i + 1], "%f", &value) == 1) {
          mtl.thinfilm.ior.cls = SpectralDistribution::Class::Dielectric;
          mtl.thinfilm.ior.eta_index = data.add_spectrum(SpectralDistribution::constant(value));
          mtl.thinfilm.ior.k_index = kInvalidIndex;
        } else {
          SpectralDistribution eta_spd = {};
          SpectralDistribution k_spd = {};
          SpectralDistribution::Class cls = SpectralDistribution::Class::Invalid;
          if (load_ior_from_identifier(params[i + 1], eta_spd, k_spd, cls) == false) {
            std::filesystem::path fallback = locate_spectrum_file(params[i + 1], {});
            if (fallback.empty() == false) {
              cls = RefractiveIndex::load_from_file(fallback.string().c_str(), eta_spd, k_spd);
            }
          }

          if (cls == SpectralDistribution::Class::Invalid) {
            log::warning("Unable to load thinfilm IOR `%s`, using dielectric 1.5", params[i + 1]);
            cls = SpectralDistribution::Class::Dielectric;
            eta_spd = SpectralDistribution::constant(1.5f);
            k_spd = SpectralDistribution::constant(0.0f);
          }

          mtl.thinfilm.ior.cls = cls;
          mtl.thinfilm.ior.eta_index = data.add_spectrum(eta_spd);
          if (cls == SpectralDistribution::Class::Conductor) {
            mtl.thinfilm.ior.k_index = data.add_spectrum(k_spd);
          } else {
            mtl.thinfilm.ior.k_index = k_spd.empty() ? kInvalidIndex : data.add_spectrum(k_spd);
          }
        }
      }
    }
  }

  if (get_param(material, "subsurface")) {
    mtl.subsurface.cls = SubsurfaceMaterial::Class::RandomWalk;

    float subsurface_scale = 1.0f;
    float3 scattering_distances = {1.0f, 0.2f, 0.04f};

    auto params = split_params(data_buffer);
    for (uint64_t i = 0, e = params.size(); i < e; ++i) {
      if ((strcmp(params[i], "path") == 0) && (i + 1 < e)) {
        bool is_refraction = (strcmp(params[i + 1], "refracted") == 0) || (strcmp(params[i + 1], "refraction") == 0) || (strcmp(params[i + 1], "refract") == 0);
        mtl.subsurface.path = is_refraction ? SubsurfaceMaterial::Path::Refracted : SubsurfaceMaterial::Path::Diffuse;
      }

      if ((strcmp(params[i], "distances") == 0) && (i + 3 < e)) {
        scattering_distances.x = static_cast<float>(atof(params[i + 1]));
        scattering_distances.y = static_cast<float>(atof(params[i + 2]));
        scattering_distances.z = static_cast<float>(atof(params[i + 3]));
        i += 3;
      }

      if ((strcmp(params[i], "scale") == 0) && (i + 1 < e)) {
        subsurface_scale = static_cast<float>(atof(params[i + 1]));
        i += 1;
      }

      if ((strcmp(params[i], "class") == 0) && (i + 1 < e)) {
        if (strcmp(params[i + 1], "approximate") == 0) {
          mtl.subsurface.cls = SubsurfaceMaterial::Class::ChristensenBurley;
        }
        i += 1;
      }
    }

    mtl.subsurface.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance(scattering_distances));
    data.spectrum_values[mtl.subsurface.spectrum_index].scale(subsurface_scale);
  }

  SpectralDistribution emission_spd = SpectralDistribution::null();
  float pending_scale = 1.0f;
  bool is_emitter = false;

  bool emission_spd_defined = false;
  float collimation = mtl.emission_collimation;

  if (get_param(material, "Ke")) {
    is_emitter = true;
    emission_spd = load_illuminant_spectrum(data_buffer);
    emission_spd_defined = true;
    if (get_file(base_dir, material.emissive_texname)) {
      mtl.emission.image_index = context.add_image(data_buffer, Image::RepeatU | Image::RepeatV | Image::BuildSamplingTable, {}, {1.0f, 1.0f});
    }
  }

  if (get_param(material, "emitter")) {
    is_emitter = true;
    auto params = split_params(data_buffer);
    for (uint64_t i = 0, end = params.size(); i < end; ++i) {
      if ((strcmp(params[i], "image") == 0) && (i + 1 < end) && get_file(base_dir, params[i + 1])) {
        mtl.emission.image_index = context.add_image(data_buffer, Image::RepeatU | Image::RepeatV | Image::BuildSamplingTable, {}, {1.0f, 1.0f});
      } else if (strcmp(params[i], "twosided") == 0) {
        mtl.two_sided = 1u;
      } else if ((strcmp(params[i], "collimated") == 0) && (i + 1 < end)) {
        collimation = static_cast<float>(atof(params[i + 1]));
        i += 1;
      } else if ((strcmp(params[i], "color") == 0) && (i + 3 < end)) {
        float3 value = {
          static_cast<float>(atof(params[i + 1])),
          static_cast<float>(atof(params[i + 2])),
          static_cast<float>(atof(params[i + 3])),
        };
        emission_spd = SpectralDistribution::rgb_luminance(value);
        emission_spd_defined = true;
        i += 3;
      } else if ((strcmp(params[i], "blackbody") == 0) && (i + 1 < end)) {
        emission_spd = SpectralDistribution::from_black_body(static_cast<float>(atof(params[i + 1])), 1.0f);
        emission_spd_defined = true;
        i += 1;
      } else if ((strcmp(params[i], "nblackbody") == 0) && (i + 1 < end)) {
        emission_spd = SpectralDistribution::from_normalized_black_body(static_cast<float>(atof(params[i + 1])), 1.0f);
        emission_spd_defined = true;
        i += 1;
      } else if ((strcmp(params[i], "scale") == 0) && (i + 1 < end)) {
        pending_scale *= static_cast<float>(atof(params[i + 1]));
        i += 1;
      } else if ((strcmp(params[i], "spectrum") == 0) && (i + 1 < end)) {
        char buffer[2048] = {};
        snprintf(buffer, sizeof(buffer), "%s/%s", base_dir, data_buffer);
        auto cls = SpectralDistribution::load_from_file(buffer, emission_spd, nullptr, false);
        if (cls != SpectralDistribution::Class::Illuminant) {
          log::warning("Spectrum %s is not illuminant", buffer);
        }
        emission_spd_defined = true;
      }
    }
    collimation = saturate(collimation);
  }

  if (is_emitter) {
    emission_spd.scale(pending_scale);
    mtl.emission_collimation = collimation;
    if (emission_spd_defined && (emission_spd.luminance() > 0.0f)) {
      mtl.emission.spectrum_index = data.add_spectrum(emission_spd);
    } else if (emission_spd_defined == false && mtl.emission.spectrum_index != kInvalidIndex) {
      // keep existing spectrum (e.g. inherited from base)
    } else {
      mtl.emission.spectrum_index = kInvalidIndex;
    }
    if (mtl.emission.spectrum_index == kInvalidIndex) {
      mtl.emission.image_index = kInvalidIndex;
    }
  } else if (mtl.emission.spectrum_index == kInvalidIndex) {
    mtl.emission.image_index = kInvalidIndex;
    mtl.emission_collimation = 0.0f;
  }
}

void SceneRepresentationImpl::parse_obj_materials(const char* base_dir, const std::vector<tinyobj::material_t>& obj_materials) {
  for (const auto& material : obj_materials) {
    if (material.name == "et::camera") {
      parse_camera(base_dir, material);
    } else if (material.name == "et::medium") {
      parse_medium(base_dir, material);
    } else if (material.name == "et::dir") {
      parse_directional_light(base_dir, material);
    } else if (material.name == "et::env") {
      parse_env_light(base_dir, material);
    } else if (material.name == "et::atmosphere") {
      parse_atmosphere_light(base_dir, material);
    } else if (material.name == "et::spectrum") {
      parse_spectrum(base_dir, material);
    } else {
      parse_material(base_dir, material);
    }
  }

  context.images.load_images();
}

float4x4 SceneRepresentationImpl::build_gltf_node_transform(const tinygltf::Node& node) {
  float4x4 transform = {};
  if (node.matrix.size() == 16) {
    for (uint32_t i = 0; i < 16u; ++i) {
      transform.val[i] = float(node.matrix[i]);
    }
  } else {
    float3 translation = {0.0f, 0.0f, 0.0f};
    if (node.translation.size() == 3) {
      translation = {float(node.translation[0]), float(node.translation[1]), float(node.translation[2])};
    }
    float4 rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    if (node.rotation.size() == 4) {
      rotation = {float(node.rotation[0]), float(node.rotation[1]), float(node.rotation[2]), float(node.rotation[3])};
    }
    float3 scale = {1.0f, 1.0f, 1.0f};
    if (node.scale.size() == 3) {
      scale = {float(node.scale[0]), float(node.scale[1]), float(node.scale[2])};
    }
    transform = transform_matrix(translation, rotation, scale);
  }
  return transform;
}

void SceneRepresentationImpl::load_gltf_camera(const tinygltf::Node& node, const tinygltf::Model& model, const tinygltf::Camera& pcam, const float4x4& transform) {
  if (pcam.type != "perspective") {
    log::warning("Loading non-perspective not yet supported");
    return;
  }

  const auto& cam = pcam.perspective;

  auto position = to_float3(transform * float4{0.0f, 0.0f, 0.0f, 1.0f});
  auto direction = to_float3(transform.col[2]);
  auto up = to_float3(transform.col[1]);
  build_camera(active_camera, position, position - direction, up, active_camera.film_size, float(cam.yfov) * 180.0f / kPi);
}

void SceneRepresentationImpl::load_gltf_node(const tinygltf::Model& model, const tinygltf::Node& node, const float4x4& parent_transform) {
  auto current_transform = parent_transform * build_gltf_node_transform(node);

  if ((node.mesh >= 0) && (node.mesh < model.meshes.size())) {
    load_gltf_mesh(node, model, model.meshes.at(node.mesh), current_transform);
  }

  if ((node.camera >= 0) && (node.camera < model.cameras.size())) {
    load_gltf_camera(node, model, model.cameras.at(node.camera), current_transform);
  }

  for (const auto& child : node.children) {
    load_gltf_node(model, model.nodes[child], current_transform);
  }
}

uint32_t SceneRepresentationImpl::load_from_gltf(const char* file_name, bool binary) {
  tinygltf::TinyGLTF loader;
  tinygltf::Model model;
  std::string errors;
  std::string warnings;

  bool load_result = false;

  auto& gltf_image_mapping = data.gltf_image_mapping;
  gltf_image_mapping.clear();
  auto image_loader = [](tinygltf::Image* image, const int image_index, std::string* errors, std::string* warnings,  //
                        int width, int height, const unsigned char* data, int data_size, void* user_pointer) -> bool {
    auto self = reinterpret_cast<SceneRepresentationImpl*>(user_pointer);

    if (((width == 0) || (height == 0)) && (data != nullptr)) {
      bool can_write = std::filesystem::exists("./tmp") || std::filesystem::create_directory("./tmp");

      if (can_write) {
        char buffer[2048] = {};
        uint32_t hash = fnv1a32(data, data_size, kFnv1a32Begin);
        snprintf(buffer, sizeof(buffer), "./tmp/img-%x.png", hash);
        if (auto fout = fopen(buffer, "wb")) {
          if (fwrite(data, 1, data_size, fout) == data_size) {
            self->data.gltf_image_mapping[image_index] = self->context.add_image(buffer, Image::RepeatU | Image::RepeatV, {}, {1.0f, 1.0f});
          }
          fclose(fout);
        }
      }
    }

    return true;
  };

  loader.SetImageLoader(image_loader, this);

  if (binary) {
    load_result = loader.LoadBinaryFromFile(&model, &errors, &warnings, file_name);
  } else {
    load_result = loader.LoadASCIIFromFile(&model, &errors, &warnings, file_name);
  }

  if (warnings.empty() == false) {
    log::warning("GLTF warning(s): %s", warnings.c_str());
  }

  if (errors.empty() == false) {
    log::error("GLTF error(s): %s", errors.c_str());
  }

  if (load_result == false) {
    log::error("Failed to load GLTF from %s:\n%s", file_name, errors.c_str());
    return LoadFailed;
  }

  load_gltf_materials(model);

  bool camera_loaded = false;
  for (const auto& scene : model.scenes) {
    for (int32_t node_index : scene.nodes) {
      if ((node_index < 0) || (node_index >= model.nodes.size()))
        continue;

      const float4x4 identity = build_gltf_node_transform({});
      const auto& node = model.nodes[node_index];
      load_gltf_node(model, node, identity);
    }
  }

  return LoadSucceeded | (camera_loaded ? LoadCameraInfo : 0u);
}

void SceneRepresentationImpl::load_gltf_materials(const tinygltf::Model& model) {
  for (auto& material : model.materials) {
    std::string material_name = material.name;
    uint32_t index = 1;
    while (data.has_material(material_name.c_str())) {
      char buffer[1024] = {};
      snprintf(buffer, sizeof(buffer), "%s-%04u", material.name.c_str(), index);
      material_name = buffer;
      ++index;
    }

    const auto& pbr = material.pbrMetallicRoughness;

    uint32_t material_index = data.add_material(material_name.c_str());

    auto& mtl = data.materials[material_index];
    mtl.cls = Material::Class::Principled;
    mtl.roughness.value = {float(pbr.roughnessFactor), float(pbr.roughnessFactor)};
    mtl.metalness.value = {float(pbr.metallicFactor), float(pbr.metallicFactor)};
    mtl.ext_ior.cls = SpectralDistribution::Class::Dielectric;
    mtl.ext_ior.eta_index = data.add_spectrum(SpectralDistribution::constant(1.0f));
    mtl.ext_ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
    mtl.int_ior.cls = SpectralDistribution::Class::Conductor;
    mtl.int_ior.eta_index = data.add_spectrum(SpectralDistribution::constant(1.5f));
    mtl.int_ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
    mtl.subsurface.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 0.2f, 0.04f}));
    mtl.emission = {};
    mtl.emission_collimation = 0.0f;

    float3 rgb = {1.0f, 1.0f, 1.0f};
    const auto& base_color = material.pbrMetallicRoughness.baseColorFactor;
    if (base_color.size() >= 3) {
      rgb = {float(base_color[0]), float(base_color[1]), float(base_color[2])};
    }
    mtl.scattering.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance(rgb));

    if ((pbr.baseColorTexture.index != -1) && (data.gltf_image_mapping.count(pbr.baseColorTexture.index) > 0)) {
      mtl.scattering.image_index = data.gltf_image_mapping.at(pbr.baseColorTexture.index);
      mtl.reflectance.image_index = mtl.scattering.image_index;
    }

    if ((pbr.metallicRoughnessTexture.index != -1) && (data.gltf_image_mapping.count(pbr.metallicRoughnessTexture.index) > 0)) {
      auto image_index = data.gltf_image_mapping.at(pbr.metallicRoughnessTexture.index);
      mtl.roughness.image_index = image_index;
      mtl.roughness.channel = 1u;
      mtl.metalness.image_index = image_index;
      mtl.metalness.channel = 2u;
    }

    if ((material.normalTexture.index != -1) && (data.gltf_image_mapping.count(material.normalTexture.index) > 0)) {
      mtl.normal_image_index = data.gltf_image_mapping.at(material.normalTexture.index);
      mtl.normal_scale = 1.0f;
      context.add_image_options(mtl.normal_image_index, Image::SkipSRGBConversion);
    }

    if (material.emissiveFactor.size() >= 3) {
      float3 emission = {float(material.emissiveFactor[0]), float(material.emissiveFactor[1]), float(material.emissiveFactor[2])};
      if (dot(emission, emission) >= kEpsilon) {
        auto spd = SpectralDistribution::rgb_luminance(emission);

        for (const auto& ext : material.extensions) {
          if (ext.first == "KHR_materials_emissive_strength") {
            if (ext.second.IsObject() && ext.second.Has("emissiveStrength")) {
              const auto& value = ext.second.Get("emissiveStrength");
              if (value.IsNumber()) {
                float scale = float(value.GetNumberAsDouble());
                spd.scale(scale);
              }
            }
          }
        }

        mtl.emission.spectrum_index = data.add_spectrum(spd);
        mtl.emission_collimation = 0.0f;
        if ((material.emissiveTexture.index != -1) && (data.gltf_image_mapping.count(material.emissiveTexture.index) > 0)) {
          mtl.emission.image_index = data.gltf_image_mapping.at(material.emissiveTexture.index);
        }
      } else {
        mtl.emission.spectrum_index = kInvalidIndex;
        mtl.emission.image_index = kInvalidIndex;
      }

      for (const auto& ext : material.extensions) {
        if (ext.first == "KHR_materials_transmission") {
          if (ext.second.IsObject() && ext.second.Has("transmissionFactor")) {
            const auto& value = ext.second.Get("transmissionFactor");
            if (value.IsNumber()) {
              float transmission = float(value.GetNumberAsDouble());
              mtl.transmission.value = {transmission, transmission, transmission, transmission};
            }
          }
        }
      }
    }

    mtl.reflectance.spectrum_index = data.add_spectrum(SpectralDistribution::constant(1.0f));
  }

  if (data.materials.empty()) {
    uint32_t material_index = data.add_material("default");
    auto& mtl = data.materials[material_index];
    mtl.cls = Material::Class::Diffuse;
    mtl.scattering.spectrum_index = data.add_spectrum(SpectralDistribution::constant(1.0f));
    mtl.reflectance.spectrum_index = data.add_spectrum(SpectralDistribution::constant(1.0f));
  }

  context.images.load_images();
}

void SceneRepresentationImpl::load_gltf_mesh(const tinygltf::Node& node, const tinygltf::Model& model, const tinygltf::Mesh& mesh, const float4x4& transform) {
  auto& triangles = data.triangles;
  auto& triangle_to_emitter = data.triangle_to_emitter;
  auto& vertices = data.vertices;

  if (mesh.primitives.empty())
    return;

  for (const auto& primitive : mesh.primitives) {
    bool has_positions = primitive.attributes.count("POSITION") > 0;
    bool has_normals = primitive.attributes.count("NORMAL") > 0;
    bool has_tex_coords = primitive.attributes.count("TEXCOORD_0") > 0;
    bool has_tangents = primitive.attributes.count("TANGENT") > 0;

    if (has_positions == false)
      continue;

    uint32_t material_index = (primitive.material >= 0) && (primitive.material < static_cast<int>(model.materials.size())) ? static_cast<uint32_t>(primitive.material) : 0;

    const tinygltf::Accessor& pos_accessor = model.accessors[primitive.attributes.find("POSITION")->second];
    const tinygltf::BufferView& pos_buffer_view = model.bufferViews[pos_accessor.bufferView];
    const tinygltf::Buffer& pos_buffer = model.buffers[pos_buffer_view.buffer];

    const tinygltf::Accessor* nrm_accessor = nullptr;
    const tinygltf::BufferView* nrm_buffer_view = nullptr;
    const tinygltf::Buffer* nrm_buffer = nullptr;
    if (has_normals) {
      nrm_accessor = model.accessors.data() + primitive.attributes.find("NORMAL")->second;
      nrm_buffer_view = model.bufferViews.data() + nrm_accessor->bufferView;
      nrm_buffer = model.buffers.data() + nrm_buffer_view->buffer;
    }

    const tinygltf::Accessor* tex_accessor = nullptr;
    const tinygltf::BufferView* tex_buffer_view = nullptr;
    const tinygltf::Buffer* tex_buffer = nullptr;
    if (has_tex_coords) {
      tex_accessor = model.accessors.data() + primitive.attributes.find("TEXCOORD_0")->second;
      tex_buffer_view = model.bufferViews.data() + tex_accessor->bufferView;
      tex_buffer = model.buffers.data() + tex_buffer_view->buffer;
    }

    const tinygltf::Accessor* tan_accessor = nullptr;
    const tinygltf::BufferView* tan_buffer_view = nullptr;
    const tinygltf::Buffer* tan_buffer = nullptr;
    if (has_tangents) {
      tan_accessor = model.accessors.data() + primitive.attributes.find("TANGENT")->second;
      tan_buffer_view = model.bufferViews.data() + tan_accessor->bufferView;
      tan_buffer = model.buffers.data() + tan_buffer_view->buffer;
    }

    bool has_indices = (primitive.indices >= 0) && (primitive.indices < model.accessors.size());

    const tinygltf::Accessor* idx_accessor = nullptr;
    const tinygltf::BufferView* idx_buffer_view = nullptr;
    const tinygltf::Buffer* idx_buffer = nullptr;
    if (has_indices) {
      idx_accessor = model.accessors.data() + primitive.indices;
      idx_buffer_view = model.bufferViews.data() + idx_accessor->bufferView;
      idx_buffer = model.buffers.data() + idx_buffer_view->buffer;
    }

    ETX_ASSERT(idx_accessor->count % 3 == 0);
    uint32_t triangle_count = static_cast<uint32_t>(has_indices ? idx_accessor->count : pos_accessor.count) / 3u;

    uint32_t linear_index = 0;
    for (uint32_t tri_index = 0; tri_index < triangle_count; ++tri_index) {
      triangle_to_emitter.emplace_back(kInvalidIndex);

      uint32_t base_index = static_cast<uint32_t>(vertices.size());
      Triangle& tri = triangles.emplace_back();
      tri.i[0] = base_index + 0;
      tri.i[1] = base_index + 1;
      tri.i[2] = base_index + 2;
      tri.material_index = material_index;

      for (uint32_t j = 0; j < 3; ++j, ++linear_index) {
        auto index = has_indices ? gltf_read_buffer_as_uint(*idx_buffer, *idx_accessor, *idx_buffer_view, 3u * tri_index + j) : linear_index;

        auto& v0 = vertices.emplace_back();

        auto p = gltf_read_buffer<float3>(pos_buffer, pos_accessor, pos_buffer_view, index);
        auto tp = transform * float4{p.x, p.y, p.z, 1.0f};
        v0.pos = {tp.x, tp.y, tp.z};

        if (has_normals) {
          auto n = gltf_read_buffer<float3>(*nrm_buffer, *nrm_accessor, *nrm_buffer_view, index);
          auto tn = transform * float4{n.x, n.y, n.z, 0.0f};
          v0.nrm = {tn.x, tn.y, tn.z};
        }

        if (has_tex_coords) {
          v0.tex = gltf_read_buffer<float2>(*tex_buffer, *tex_accessor, *tex_buffer_view, index);
        }

        if (has_tangents) {
          auto t = gltf_read_buffer<float4>(*tan_buffer, *tan_accessor, *tan_buffer_view, index);
          auto tt = transform * float4{t.x, t.y, t.z, 0.0f};
          v0.tan = normalize(float3{tt.x, tt.y, tt.z});
          v0.btn = cross(v0.nrm, v0.tan) * t.w;
        }
      }

      if (validate_triangle(tri) == false) {
        triangles.pop_back();
        continue;
      }

      if (has_normals == false) {
        vertices[vertices.size() - 1u].nrm = tri.geo_n;
        vertices[vertices.size() - 2u].nrm = tri.geo_n;
        vertices[vertices.size() - 3u].nrm = tri.geo_n;
      }
    }
  }
}

void build_emitters_distribution(Scene& scene) {
  for (uint32_t i = 0; i < scene.emitter_profiles.count; ++i) {
    auto& emitter = scene.emitter_profiles[i];
    if (emitter.is_distant()) {
      emitter.equivalent_disk_size = 2.0f * std::tan(emitter.angular_size / 2.0f);
      emitter.angular_size_cosine = std::cos(emitter.angular_size / 2.0f);
      float additional_weight = kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius;
      for (uint32_t j = 0; j < scene.emitter_instances.count; ++j) {
        if (scene.emitter_instances[j].profile == i) {
          scene.emitter_instances[j].additional_weight = additional_weight;
        }
      }
    }
  }

  log::warning("Building emitters distribution for %llu emitters...", scene.emitter_instances.count);

  scene.environment_emitters.count = 0;

  DistributionBuilder emitters_distribution(scene.emitters_distribution, static_cast<uint32_t>(scene.emitter_instances.count));
  for (uint32_t i = 0; i < scene.emitter_instances.count; ++i) {
    auto& emitter = scene.emitter_instances[i];

    float spectrum_weight = 0.0f;

    const auto& profile = scene.emitter_profiles[emitter.profile];
    if (profile.emission.spectrum_index != kInvalidIndex) {
      spectrum_weight = scene.spectrums[profile.emission.spectrum_index].luminance();
    }
    emitter.spectrum_weight = spectrum_weight;

    float total_weight = emitter.spectrum_weight * emitter.additional_weight;
    emitters_distribution.add(total_weight);
    if (emitter.is_local()) {
      scene.triangle_to_emitter[emitter.triangle_index] = i;
    } else if (emitter.is_distant() && (total_weight > 0.0f)) {
      scene.environment_emitters.emitters[scene.environment_emitters.count++] = i;
    }
  }
  emitters_distribution.finalize();
}

std::string SceneRepresentation::save_to_file(const char* filename) {
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

  std::filesystem::path geometry_path = impl->data.geometry_file_name.empty() ? base_path : std::filesystem::path(impl->data.geometry_file_name).lexically_normal();
  std::string geometry_ref = to_relative(geometry_path, json_path.parent_path());
  std::string materials_ref = to_relative(materials_path, json_path.parent_path());

  const Scene& scene_data = impl->scene;

  nlohmann::json js = nlohmann::json::object();
  js["samples"] = scene_data.samples;
  js["random-termination-start"] = scene_data.random_path_termination;
  js["max-path-length"] = scene_data.max_path_length;
  js["min-path-length"] = scene_data.min_path_length;
  js["geometry"] = geometry_ref;
  if (materials_ref.empty() == false) {
    js["materials"] = materials_ref;
  }
  js["spectral"] = ((scene_data.flags & Scene::Spectral) == Scene::Spectral);

  json_to_file(js, json_path.string().c_str());

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
  medium_entries.reserve(impl->context.mediums.mapping().size());
  for (const auto& entry : impl->context.mediums.mapping()) {
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
    std::string stored = impl->context.images.path(image_index);
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
    materials_stream << "newmtl et::camera\n";
    materials_stream << "class " << ((camera.cls == Camera::Class::Equirectangular) ? "eq" : "perspective") << "\n";
    materials_stream << "viewport " << camera.film_size.x << " " << camera.film_size.y << "\n";
    materials_stream << "origin " << camera.position.x << " " << camera.position.y << " " << camera.position.z << "\n";
    materials_stream << "target " << camera.target.x << " " << camera.target.y << " " << camera.target.z << "\n";
    materials_stream << "up " << camera.up.x << " " << camera.up.y << " " << camera.up.z << "\n";
    materials_stream << "fov " << get_camera_fov(camera) << "\n";
    materials_stream << "focal-length " << get_camera_focal_length(camera) << "\n";
    materials_stream << "lens-radius " << camera.lens_radius << "\n";
    materials_stream << "focal-distance " << camera.focal_distance << "\n";
    materials_stream << "clip-near " << camera.clip_near << "\n";
    materials_stream << "clip-far " << camera.clip_far << "\n";
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
      const Image& env_image = impl->context.images.get(environment_profile->emission.image_index);
      env_rotation_offset = env_image.offset.x;
      env_scale_u = env_image.scale.x;
    }
    if (std::fabs(env_rotation_offset) >= kEpsilon) {
      materials_stream << "rotation " << (-env_rotation_offset * 360.0f) << "\n";
    }
    if (std::fabs(env_scale_u - 1.0f) >= kEpsilon) {
      materials_stream << "scale " << env_scale_u << "\n";
    }
    materials_stream << "\n";
  }

  if (directional_profile != nullptr) {
    materials_stream << "newmtl et::dir\n";
    float3 dir_color = spectrum_rgb(directional_profile->emission.spectrum_index);
    materials_stream << "color " << dir_color.x << " " << dir_color.y << " " << dir_color.z << "\n";
    materials_stream << "direction " << directional_profile->direction.x << " " << directional_profile->direction.y << " " << directional_profile->direction.z << "\n";
    if (directional_profile->angular_size >= kEpsilon) {
      materials_stream << "angular_diameter " << (directional_profile->angular_size * 180.0f / kPi) << "\n";
    }
    std::string dir_path = texture_path(directional_profile->emission.image_index);
    if (dir_path.empty() == false) {
      materials_stream << "image " << dir_path << "\n";
    }
    materials_stream << "\n";
  }

  for (uint64_t medium_index = 0; medium_index < medium_entries.size(); ++medium_index) {
    uint32_t pool_index = medium_entries[medium_index].second;
    const Medium& medium = impl->context.mediums.get(pool_index);
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
    materials_stream << "\n";
  }

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
    if ((name.compare(0, 4, "et::") == 0) || (name.compare(0, 5, "et::") == 0)) {
      continue;
    }
    uint32_t index = entry.second;
    if (index >= impl->data.materials.size()) {
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
  FILE* materials_file = fopen(materials_path.string().c_str(), "wb");
  if (materials_file == nullptr) {
    log::error("Failed to open materials file for writing: %s", materials_path.string().c_str());
    return {};
  }
  fwrite(materials_string.data(), 1, materials_string.size(), materials_file);
  fflush(materials_file);
  fclose(materials_file);

  impl->data.json_file_name = json_path.generic_string();
  impl->data.materials_file_name = materials_path.generic_string();

  return json_path.generic_string();
}

}  // namespace etx
