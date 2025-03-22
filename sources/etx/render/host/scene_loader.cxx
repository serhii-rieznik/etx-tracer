#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/core/json.hxx>

#include <etx/render/shared/base.hxx>
#include <etx/render/shared/scene.hxx>
#include <etx/render/shared/scattering.hxx>

#include <etx/render/host/scene_loader.hxx>
#include <etx/render/host/image_pool.hxx>
#include <etx/render/host/medium_pool.hxx>
#include <etx/render/host/distribution_builder.hxx>
#include <etx/render/host/gltf_accessor.hxx>

#include <vector>
#include <string>
#include <set>

#include <mikktspace.h>
#include <tiny_obj_loader.hxx>

namespace etx {

static Spectrums shared_spectrums;

namespace {

inline void init_spectrums(TaskScheduler& scheduler, Image& extinction) {
  using SPD = SpectralDistribution;
  scattering::init(scheduler, &shared_spectrums, extinction);
  {
    static const float2 chrome_samples_eta[] = {{0.354f, 1.84f}, {0.368f, 1.87f}, {0.381f, 1.92f}, {0.397f, 2.00f}, {0.413f, 2.08f}, {0.431f, 2.19f}, {0.451f, 2.33f},
      {0.471f, 2.51f}, {0.496f, 2.75f}, {0.521f, 2.94f}, {0.549f, 3.18f}, {0.582f, 3.22f}, {0.617f, 3.17f}, {0.659f, 3.09f}, {0.704f, 3.05f}, {0.756f, 3.08f}, {0.821f, 3.20f},
      {0.892f, 3.30f}};

    static const float2 chrome_samples_k[] = {{0.354f, 2.64f}, {0.368f, 2.69f}, {0.381f, 2.74f}, {0.397f, 2.83f}, {0.413f, 2.93f}, {0.431f, 3.04f}, {0.451f, 3.14f},
      {0.471f, 3.24f}, {0.496f, 3.30f}, {0.521f, 3.33f}, {0.549f, 3.33f}, {0.582f, 3.30f}, {0.617f, 3.30f}, {0.659f, 3.34f}, {0.704f, 3.39f}, {0.756f, 3.42f}, {0.821f, 3.48f},
      {0.892f, 3.52f}};

    static const float2 plastic_samples_eta[] = {{40.0000f, 1.519f}, {41.6667f, 1.519f}, {43.4783f, 1.519f}, {45.4545f, 1.520f}, {47.6190f, 1.521f}, {50.0000f, 1.521f},
      {52.6316f, 1.521f}, {55.5556f, 1.521f}, {58.8235f, 1.521f}, {62.5000f, 1.521f}, {66.6667f, 1.521f}, {71.4286f, 1.521f}, {76.9231f, 1.520f}, {83.3333f, 1.520f},
      {90.9091f, 1.520f}};

    shared_spectrums.thinfilm.eta = SPD::constant(1.5f);
    shared_spectrums.thinfilm.k = SPD::null();

    shared_spectrums.conductor.eta = SPD::from_samples(chrome_samples_eta, uint32_t(std::size(chrome_samples_eta)));
    shared_spectrums.conductor.k = SPD::from_samples(chrome_samples_k, uint32_t(std::size(chrome_samples_k)));
    shared_spectrums.dielectric.eta = SPD::from_samples(plastic_samples_eta, uint32_t(std::size(plastic_samples_eta)));
    shared_spectrums.dielectric.k = SPD::null();
  }
}

inline auto to_float2(const float values[]) -> float2 {
  return {values[0], values[1]};
};

inline auto to_float3(const float values[]) -> float3 {
  return {values[0], values[1], values[2]};
};

inline std::vector<const char*> split_params(char* data) {
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

}  // namespace

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
  struct SpectrumID {
    SpectralDistribution spectrum;
    std::string id;
  };

  TaskScheduler& scheduler;
  std::vector<Vertex> vertices;
  std::vector<Triangle> triangles;
  std::vector<uint32_t> triangle_to_material;
  std::vector<uint32_t> triangle_to_emitter;
  std::vector<Material> materials;
  std::vector<Emitter> emitters;
  std::vector<SpectrumID> scene_spectrums;
  std::vector<SpectralDistribution> scene_spectrum_array;

  std::string json_file_name;
  std::string geometry_file_name;
  std::string mtl_file_name;

  Image extinction;
  ImagePool images;
  MediumPool mediums;

  SceneRepresentation::MaterialMapping material_mapping;
  std::unordered_map<uint32_t, uint32_t> gltf_image_mapping;

  Scene scene;
  Camera camera;

  bool loaded = false;

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

  uint32_t add_spectrum(const char* id, const SpectralDistribution& spd) {
    auto& spectrum = scene_spectrums.emplace_back(SpectrumID{
      .spectrum = spd,
      .id = id,
    });
    return uint32_t(scene_spectrums.size() - 1u);
  }

  uint32_t add_spectrum(const char* id) {
    return add_spectrum(id, {});
  }

  uint32_t add_spectrum() {
    char buffer[64] = {};
    snprintf(buffer, sizeof(buffer), "##spectrum_%llu", scene_spectrums.size());
    return add_spectrum(buffer);
  }

  uint32_t add_spectrum(const SpectralDistribution& spd) {
    uint32_t i = add_spectrum();
    scene_spectrums[i].spectrum = spd;
    return i;
  }

  uint32_t find_spectrum(const char* id) const {
    auto i = std::find_if(scene_spectrums.begin(), scene_spectrums.end(), [id](const SpectrumID& value) {
      return value.id == id;
    });

    if (i == scene_spectrums.end())
      return kInvalidIndex;

    return uint32_t(std::distance(scene_spectrums.begin(), i));
  }

  uint32_t add_image(const char* path, uint32_t options, const float2& offset) {
    std::string id = path ? path : ("image-" + std::to_string(images.array_size()));
    return images.add_from_file(id, options | Image::Delay, offset);
  }

  uint32_t add_image(const Image& img) {
    return images.add_copy(img);
  }

  uint32_t add_image(const char* path, uint32_t options) {
    return add_image(path, options, {});
  }

  uint32_t add_image(const float4* data, const uint2& dim, uint32_t options) {
    return images.add_from_data(data, dim, options, {});
  }

  void add_image_options(uint32_t index, uint32_t options) {
    images.add_options(index, options);
  }

  bool has_material(const char* name) const {
    return material_mapping.count(name) > 0;
  }

  uint32_t add_material(const char* name) {
    std::string id = (name != nullptr) && (name[0] != 0) ? name : ("material-" + std::to_string(materials.size()));
    auto i = material_mapping.find(id);
    if (i != material_mapping.end()) {
      return i->second;
    }
    uint32_t index = static_cast<uint32_t>(materials.size());
    materials.emplace_back();
    material_mapping[id] = index;
    return index;
  }

  uint32_t add_medium(Medium::Class cls, const char* name, const char* volume_file, const SpectralDistribution& s_a, const SpectralDistribution& s_t, float g,
    bool explicit_connections) {
    std::string id = name ? name : ("medium-" + std::to_string(mediums.array_size()));
    return mediums.add(cls, id, volume_file, s_a, s_t, g, explicit_connections);
  }

  SceneRepresentationImpl(TaskScheduler& s)
    : scheduler(s)
    , images(s) {
    images.init(1024u);
    mediums.init(1024u);
    init_spectrums(s, extinction);
    build_camera(camera, {5.0f, 5.0f, 5.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1280u, 720u}, 26.99f);
  }

  ~SceneRepresentationImpl() {
    cleanup();
    images.free_image(extinction);
    images.cleanup();
    mediums.cleanup();
  }

  void cleanup() {
    vertices.clear();
    triangles.clear();
    materials.clear();
    emitters.clear();
    material_mapping.clear();
    triangle_to_material.clear();
    triangle_to_emitter.clear();
    scene_spectrums.clear();

    images.remove_all();
    mediums.remove_all();
    materials.reserve(1024);  // TODO : fix, images when reallocated are destroyed releasing memory

    free(scene.emitters_distribution.values.a);
    scene.emitters_distribution = {};

    scene = {};
    camera.lens_image = kInvalidIndex;

    loaded = false;
  }

  float triangle_area(const Triangle& t) {
    return 0.5f * length(cross(vertices[t.i[1]].pos - vertices[t.i[0]].pos, vertices[t.i[2]].pos - vertices[t.i[0]].pos));
  }

  bool validate_triangle(Triangle& t) {
    t.geo_n = cross(vertices[t.i[1]].pos - vertices[t.i[0]].pos, vertices[t.i[2]].pos - vertices[t.i[0]].pos);
    float l = length(t.geo_n);
    if (l == 0.0f)
      return false;

    t.geo_n /= l;
    return true;
  }

  void validate_materials() {
    for (auto& mtl : materials) {
      if ((mtl.roughness.value.x > 0.0f) || (mtl.roughness.value.y > 0.0f)) {
        mtl.roughness.value.x = max(kEpsilon, mtl.roughness.value.x);
        mtl.roughness.value.y = max(kEpsilon, mtl.roughness.value.y);
      }

      if (mtl.int_ior.eta.empty() && mtl.int_ior.k.empty()) {
        if (mtl.cls == Material::Class::Conductor) {
          mtl.int_ior = shared_spectrums.conductor;
        } else {
          mtl.int_ior = shared_spectrums.dielectric;
        }
      }
    }
  }

  void validate_normals(std::vector<bool>& referenced_vertices) {
    std::set<uint32_t> reset_normals;

    referenced_vertices.resize(vertices.size());
    for (const auto& tri : triangles) {
      const float tri_area = triangle_area(tri);
      for (uint32_t i = 0; i < 3; ++i) {
        uint32_t index = tri.i[i];
        ETX_CRITICAL(is_valid_vector(tri.geo_n));
        referenced_vertices[index] = true;

        if (is_valid_vector(vertices[index].nrm))
          continue;

        if (reset_normals.count(index) == 0) {
          vertices[index].nrm = tri.geo_n * tri_area;
          reset_normals.insert(index);
        } else {
          vertices[index].nrm += tri.geo_n * tri_area;
        }
      }
    }

    if (reset_normals.empty())
      return;

    for (auto i : reset_normals) {
      ETX_ASSERT(is_valid_vector(vertices[i].nrm));
      vertices[i].nrm = normalize(vertices[i].nrm);
      ETX_ASSERT(is_valid_vector(vertices[i].nrm));
    }
  }

  void build_tangents() {
    static std::map<uint32_t, uint32_t> a = {};

    SMikkTSpaceInterface contextInterface = {};
    contextInterface.m_getNumFaces = [](const SMikkTSpaceContext* pContext) -> int {
      auto data = reinterpret_cast<SceneRepresentationImpl*>(pContext->m_pUserData);
      return static_cast<int>(data->triangles.size());
    };
    contextInterface.m_getNumVerticesOfFace = [](const SMikkTSpaceContext* pContext, const int iFace) -> int {
      return 3;
    };
    contextInterface.m_getPosition = [](const SMikkTSpaceContext* pContext, float fvPosOut[], const int iFace, const int iVert) {
      auto data = reinterpret_cast<SceneRepresentationImpl*>(pContext->m_pUserData);
      const auto& tri = data->triangles[iFace];
      const auto& vertex = data->vertices[tri.i[iVert]];
      fvPosOut[0] = vertex.pos.x;
      fvPosOut[1] = vertex.pos.y;
      fvPosOut[2] = vertex.pos.z;
    };
    contextInterface.m_getNormal = [](const SMikkTSpaceContext* pContext, float fvNormOut[], const int iFace, const int iVert) {
      auto data = reinterpret_cast<SceneRepresentationImpl*>(pContext->m_pUserData);
      const auto& tri = data->triangles[iFace];
      const auto& vertex = data->vertices[tri.i[iVert]];
      fvNormOut[0] = vertex.nrm.x;
      fvNormOut[1] = vertex.nrm.y;
      fvNormOut[2] = vertex.nrm.z;
    };
    contextInterface.m_getTexCoord = [](const SMikkTSpaceContext* pContext, float fvTexcOut[], const int iFace, const int iVert) {
      auto data = reinterpret_cast<SceneRepresentationImpl*>(pContext->m_pUserData);
      const auto& tri = data->triangles[iFace];
      const auto& vertex = data->vertices[tri.i[iVert]];
      fvTexcOut[0] = vertex.tex.x;
      fvTexcOut[1] = vertex.tex.y;
    };
    contextInterface.m_setTSpaceBasic = [](const SMikkTSpaceContext* pContext, const float fvTangent[], const float fSign, const int iFace, const int iVert) {
      auto data = reinterpret_cast<SceneRepresentationImpl*>(pContext->m_pUserData);
      const auto& tri = data->triangles[iFace];
      auto& vertex = data->vertices[tri.i[iVert]];
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

  void validate_tangents(std::vector<bool>& referenced_vertices) {
    for (uint64_t vertex_index = 0, e = vertices.size(); vertex_index < e; ++vertex_index) {
      auto& v = vertices[vertex_index];
      if (is_valid_vector(v.tan) && is_valid_vector(v.btn)) {
        continue;
      }

      if (referenced_vertices[vertex_index]) {
        ETX_ASSERT(is_valid_vector(v.nrm));
        auto [t, b] = orthonormal_basis(v.nrm);
        v.tan = t;
        v.btn = b;
      }
    }
  }

  void commit() {
    log::info("Building pixel sampler...");

    std::vector<float4> sampler_image;
    Film::generate_filter_image(Film::PixelFilterBlackmanHarris, sampler_image);
    uint32_t image = images.add_from_data(sampler_image.data(), {Film::PixelFilterSize, Film::PixelFilterSize}, Image::BuildSamplingTable | Image::UniformSamplingTable, {});
    scene.pixel_sampler = {image, 1.5f};

    scene_spectrum_array.resize(scene_spectrums.size());
    scene_spectrum_array.clear();

    for (const auto& spd : scene_spectrums) {
      scene_spectrum_array.emplace_back(spd.spectrum);
    }

    float3 bbox_min = triangles.empty() ? float3{-1.0f, -1.0f, -1.0f} : float3{kMaxFloat, kMaxFloat, kMaxFloat};
    float3 bbox_max = triangles.empty() ? float3{+1.0f, +1.0f, +1.0f} : float3{-kMaxFloat, -kMaxFloat, -kMaxFloat};
    for (const auto& tri : triangles) {
      bbox_min = min(bbox_min, vertices[tri.i[0]].pos);
      bbox_min = min(bbox_min, vertices[tri.i[1]].pos);
      bbox_min = min(bbox_min, vertices[tri.i[2]].pos);
      bbox_max = max(bbox_max, vertices[tri.i[0]].pos);
      bbox_max = max(bbox_max, vertices[tri.i[1]].pos);
      bbox_max = max(bbox_max, vertices[tri.i[2]].pos);
    }
    scene.bounding_sphere_center = 0.5f * (bbox_min + bbox_max);
    scene.bounding_sphere_radius = length(bbox_max - scene.bounding_sphere_center);
    scene.vertices = {vertices.data(), vertices.size()};
    scene.triangles = {triangles.data(), triangles.size()};
    scene.triangle_to_material = {triangle_to_material.data(), triangle_to_material.size()};
    scene.triangle_to_emitter = {triangle_to_emitter.data(), triangle_to_emitter.size()};
    scene.materials = {materials.data(), materials.size()};
    scene.emitters = {emitters.data(), emitters.size()};
    scene.images = {images.as_array(), images.array_size()};
    scene.mediums = {mediums.as_array(), mediums.array_size()};
    scene.spectrums = {scene_spectrum_array.data(), scene_spectrum_array.size()};
    scene.environment_emitters.count = 0;

    log::info("Building emitters distribution for %llu emitters...", scene.emitters.count);
    build_emitters_distribution(scene);

    loaded = true;
  }

  enum : uint32_t {
    LoadFailed = 0u,
    LoadSucceeded = 1u << 0u,
  };

  uint32_t load_from_obj(const char* file_name, const char* mtl_file);

  uint32_t load_from_gltf(const char* file_name, bool binary);
  float4x4 build_gltf_node_transform(const tinygltf::Node& node);
  void load_gltf_node(const tinygltf::Model& model, const tinygltf::Node&, const float4x4& transform);
  void load_gltf_mesh(const tinygltf::Node& node, const tinygltf::Model& model, const tinygltf::Mesh&, const float4x4& transform);

  void parse_material(const char* base_dir, const tinyobj::material_t& material);
  void parse_camera(const char* base_dir, const tinyobj::material_t& material);
  void parse_medium(const char* base_dir, const tinyobj::material_t& material);
  void parse_directional_light(const char* base_dir, const tinyobj::material_t& material);
  void parse_env_light(const char* base_dir, const tinyobj::material_t& material);
  void parse_atmosphere_light(const char* base_dir, const tinyobj::material_t& material);
  void parse_spectrum(const char* base_dir, const tinyobj::material_t& material);

  uint32_t load_reflectance_spectrum(char* data);
  uint32_t load_illuminant_spectrum(char* data);

  void parse_obj_materials(const char* base_dir, const std::vector<tinyobj::material_t>& obj_materials);
};

void build_camera(Camera& camera, const float3& origin, const float3& target, const float3& up, const uint2& viewport, const float fov) {
  float4x4 view = look_at(origin, target, up);
  float4x4 proj = perspective(fov * kPi / 180.0f, viewport.x, viewport.y, 1.0f, 1024.0f);
  float4x4 inv_view = inverse(view);

  camera.target = target;
  camera.position = {inv_view.col[3].x, inv_view.col[3].y, inv_view.col[3].z};
  camera.side = {view.col[0].x, view.col[1].x, view.col[2].x};
  camera.up = {view.col[0].y, view.col[1].y, view.col[2].y};
  camera.direction = {-view.col[0].z, -view.col[1].z, -view.col[2].z};
  camera.tan_half_fov = 1.0f / std::abs(proj.col[1].y);
  camera.aspect = proj.col[1].y / proj.col[0].x;
  camera.view_proj = proj * view;

  float plane_w = 2.0f * camera.tan_half_fov * camera.aspect;
  float plane_h = 2.0f * camera.tan_half_fov;
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

SceneRepresentation::SceneRepresentation(TaskScheduler& s) {
  ETX_PIMPL_INIT(SceneRepresentation, s);
}

SceneRepresentation::~SceneRepresentation() {
  ETX_PIMPL_CLEANUP(SceneRepresentation);
}

Scene& SceneRepresentation::mutable_scene() {
  return _private->scene;
}

Scene* SceneRepresentation::mutable_scene_pointer() {
  return &_private->scene;
}

const Scene& SceneRepresentation::scene() const {
  return _private->scene;
}

const SceneRepresentation::MaterialMapping& SceneRepresentation::material_mapping() const {
  return _private->material_mapping;
}

const SceneRepresentation::MediumMapping& SceneRepresentation::medium_mapping() const {
  return _private->mediums.mapping();
}

Camera& SceneRepresentation::camera() {
  return _private->camera;
}

const Camera& SceneRepresentation::camera() const {
  return _private->camera;
}

bool SceneRepresentation::valid() const {
  return _private->loaded;
}

template <class T>
inline void get_values(const std::vector<T>& a, T* ptr, uint64_t count) {
  for (uint64_t i = 0, e = a.size() < count ? a.size() : count; i < e; ++i) {
    *ptr++ = a[i];
  }
}

bool SceneRepresentation::load_from_file(const char* filename, uint32_t options) {
  auto s = SpectralDistribution::from_black_body(5800.0f, 1.0f);
  auto q = s.integrated();
  q /= fmaxf(q.x, fmaxf(q.y, q.z));
  log::info("%.5f, %.5f, %.5f", q.x, q.y, q.z);

  char base_folder[2048] = {};
  get_file_folder(filename, base_folder, sizeof(base_folder));

  _private->cleanup();
  _private->json_file_name = {};
  _private->mtl_file_name = {};
  _private->geometry_file_name = filename;
  _private->camera.lens_radius = 0.0f;
  _private->camera.focal_distance = 0.0f;
  _private->camera.lens_image = kInvalidIndex;
  _private->camera.medium_index = kInvalidIndex;
  _private->camera.up = {0.0f, 1.0f, 0.0f};

  _private->add_spectrum(SpectralDistribution::null());
  _private->add_spectrum(SpectralDistribution::rgb_luminance({1.0f, 1.0f, 1.0f}));

  auto& camera = _private->camera;
  float3 camera_target = camera.position + camera.direction;
  float camera_fov = get_camera_fov(camera);
  float camera_focal_len = fov_to_focal_length(camera_fov);
  bool use_focal_len = false;

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
      } else if (json_get_string(i, "geometry", str_value)) {
        _private->geometry_file_name = std::string(base_folder) + str_value;
      } else if (json_get_string(i, "materials", str_value)) {
        _private->mtl_file_name = std::string(base_folder) + str_value;
      } else if (json_get_bool(i, "spectral", bool_value)) {
        _private->scene.spectral = bool_value;
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
            _private->camera.lens_radius = float_value;
          } else if (json_get_float(ci, "focal-distance", float_value)) {
            _private->camera.focal_distance = float_value;
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
    _private->json_file_name = filename;
  }

  uint32_t load_result = SceneRepresentationImpl::LoadFailed;

  auto ext = get_file_ext(_private->geometry_file_name.c_str());
  if (strcmp(ext, ".obj") == 0) {
    load_result = _private->load_from_obj(_private->geometry_file_name.c_str(), _private->mtl_file_name.c_str());
  } else if (strcmp(ext, ".gltf") == 0) {
    load_result = _private->load_from_gltf(_private->geometry_file_name.c_str(), false);
  } else if (strcmp(ext, ".glb") == 0) {
    load_result = _private->load_from_gltf(_private->geometry_file_name.c_str(), true);
  }

  if ((load_result & SceneRepresentationImpl::LoadSucceeded) == 0) {
    return false;
  }

  if (options & SetupCamera) {
    if (camera.film_size.x * camera.film_size.y == 0) {
      camera.film_size = {1280, 720};
    }
    if (use_focal_len) {
      camera_fov = focal_length_to_fov(camera_focal_len) * 180.0f / kPi;
    }
    build_camera(camera, camera.position, camera_target, camera.up, camera.film_size, camera_fov);
  }

  if (_private->emitters.empty()) {
    tinyobj::material_t mtl;
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
    _private->images.load_images();
  }

  _private->validate_materials();

  {
    TimeMeasure m = {};
    log::warning("Validating normals and tangents...");
    std::vector<bool> referenced_vertices;
    _private->validate_normals(referenced_vertices);
    _private->build_tangents();
    _private->validate_tangents(referenced_vertices);
    log::warning("Tangents calculated in %.2f sec\n", m.measure());
  }

  _private->commit();

  return true;
}

uint32_t SceneRepresentationImpl::load_from_obj(const char* file_name, const char* mtl_file) {
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
  triangle_to_material.reserve(total_triangles);
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

      uint32_t material_index = material_mapping.at(source_material.name);

      triangle_to_emitter.emplace_back(kInvalidIndex);
      triangle_to_material.emplace_back(material_index);
      auto& tri = triangles.emplace_back();
      auto& mtl = materials[material_index];

      for (uint64_t vertex_index = 0; vertex_index < face_size; ++vertex_index) {
        const auto& index = shape.mesh.indices[index_offset + vertex_index];
        tri.i[vertex_index] = static_cast<uint32_t>(vertices.size());
        auto& vertex = vertices.emplace_back();
        vertex.pos = to_float3(obj_attrib.vertices.data() + (3 * index.vertex_index));
        if (index.normal_index >= 0) {
          vertex.nrm = to_float3(obj_attrib.normals.data() + (3 * index.normal_index));
        }
        if (index.texcoord_index >= 0) {
          vertex.tex = to_float2(obj_attrib.texcoords.data() + (2 * index.texcoord_index));
        }
      }
      index_offset += face_size;

      if (validate_triangle(tri) == false) {
        triangles.pop_back();
      }

      if (get_param(source_material, "Ke")) {
        Emitter e = {};
        e.cls = Emitter::Class::Area;
        e.emission.spectrum_index = load_illuminant_spectrum(data_buffer);

        uint32_t emissive_image_index = kInvalidIndex;
        if (get_file(base_dir, source_material.emissive_texname)) {
          emissive_image_index = add_image(data_buffer, Image::RepeatU | Image::RepeatV | Image::BuildSamplingTable);
          images.load_images();
        }

        float texture_emission = 1.0f;

        if (emissive_image_index != kInvalidIndex) {
          const auto& img = images.get(emissive_image_index);

          constexpr float kBCScale = 4.0f;

          auto min_uv = min(vertices[tri.i[0]].tex, min(vertices[tri.i[1]].tex, vertices[tri.i[2]].tex));
          auto max_uv = max(vertices[tri.i[0]].tex, max(vertices[tri.i[1]].tex, vertices[tri.i[2]].tex));

          float u_size = kBCScale * max(1.0f, ceil((max_uv.x - min_uv.x) * img.fsize.x));
          float du = 1.0f / u_size;

          float v_size = kBCScale * max(1.0f, ceil((max_uv.y - min_uv.y) * img.fsize.y));
          float dv = 1.0f / v_size;

          for (float v = 0.0f; v < 1.0f; v += dv) {
            for (float u = 0.0f; u < 1.0f; u += dv) {
              float2 uv = lerp_uv({vertices.data(), vertices.size()}, tri, random_barycentric({u, v}));
              float4 val = img.evaluate(uv, nullptr);
              texture_emission += luminance(to_float3(val)) * du * dv * val.w;
            }
          }
        }

        if (get_param(obj_materials[material_id], "emitter")) {
          auto params = split_params(data_buffer);

          for (uint64_t i = 0, end = params.size(); i < end; ++i) {
            if (strcmp(params[i], "twosided") == 0) {
              e.emission_direction = Emitter::Direction::TwoSided;
            } else if (strcmp(params[i], "omni") == 0) {
              e.emission_direction = Emitter::Direction::Omni;
            } else if ((strcmp(params[i], "collimated") == 0) && (i + 1 < end)) {
              e.collimation = static_cast<float>(atof(params[i + 1]));
              i += 1;
            } else if ((strcmp(params[i], "blackbody") == 0) && (i + 1 < end)) {
              e.emission.spectrum_index = add_spectrum(SpectralDistribution::from_black_body(static_cast<float>(atof(params[i + 1])), 1.0f));
              i += 1;
            } else if ((strcmp(params[i], "nblackbody") == 0) && (i + 1 < end)) {
              e.emission.spectrum_index = add_spectrum(SpectralDistribution::from_normalized_black_body(static_cast<float>(atof(params[i + 1])), 1.0f));
              i += 1;
            } else if ((strcmp(params[i], "scale") == 0) && (i + 1 < end)) {
              if (e.emission.spectrum_index != kInvalidIndex) {
                scene_spectrums[e.emission.spectrum_index].spectrum.scale(static_cast<float>(atof(params[i + 1])));
              }
              i += 1;
            } else if ((strcmp(params[i], "spectrum") == 0) && (i + 1 < end)) {
              char buffer[2048] = {};
              snprintf(buffer, sizeof(buffer), "%s/%s", base_dir, data_buffer);
              SpectralDistribution emission_spectrum;
              auto cls = SpectralDistribution::load_from_file(buffer, emission_spectrum, nullptr, false);
              if (cls == SpectralDistribution::Class::Illuminant) {
                e.emission.spectrum_index = add_spectrum(emission_spectrum);
              } else {
                log::warning("Spectrum %s is not illuminant", buffer);
              }
            }
          }
        }

        float power_scale = 1.0f;
        switch (e.emission_direction) {
          case Emitter::Direction::TwoSided: {
            power_scale = 2.0f;
            break;
          }
          case Emitter::Direction::Omni: {
            power_scale = 4.0f * kPi;
            break;
          }
          default:
            break;
        }

        if (e.emission.spectrum_index != kInvalidIndex) {
          auto lum = scene_spectrums[e.emission.spectrum_index].spectrum.luminance();

          e.medium_index = mtl.ext_medium;
          e.triangle_index = static_cast<uint32_t>(triangles.size() - 1llu);
          e.triangle_area = triangle_area(tri);
          e.weight = power_scale * (e.triangle_area * kPi) * (lum * texture_emission);
          e.emission.image_index = emissive_image_index;

          if (e.weight > 0.0f) {
            emitters.emplace_back(e);
          }
        }
      }

      // TODO : deal with bounds!
      shape_bbox_max = max(shape_bbox_max, vertices[tri.i[0]].pos);
      shape_bbox_max = max(shape_bbox_max, vertices[tri.i[1]].pos);
      shape_bbox_max = max(shape_bbox_max, vertices[tri.i[2]].pos);
      shape_bbox_min = min(shape_bbox_min, vertices[tri.i[0]].pos);
      shape_bbox_min = min(shape_bbox_min, vertices[tri.i[1]].pos);
      shape_bbox_min = min(shape_bbox_min, vertices[tri.i[2]].pos);

      if (mtl.int_medium != kInvalidIndex) {
        mediums.get(mtl.int_medium).bounds = {shape_bbox_min, shape_bbox_max};
      }
    }
  }

  return true;
}

void SceneRepresentationImpl::parse_camera(const char* base_dir, const tinyobj::material_t& material) {
  if (get_param(material, "shape")) {
    char tmp_buffer[2048] = {};
    snprintf(tmp_buffer, sizeof(tmp_buffer), "%s/%s", base_dir, data_buffer);
    camera.lens_image = add_image(tmp_buffer, Image::BuildSamplingTable | Image::UniformSamplingTable);
  }
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
    s_t = shared_spectrums.rayleigh;

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
    s_t = shared_spectrums.mie;

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
    }

    float3 albedo = {};
    float3 extinction = {};
    float3 scattering = {};
    subsurface::remap(color, scattering_distances, albedo, extinction, scattering);

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

  uint32_t medium_index = add_medium(cls, name.c_str(), tmp_buffer, s_a, s_t, anisotropy, explicit_connections);

  if (name == "camera") {
    camera.medium_index = medium_index;
  }
}

void SceneRepresentationImpl::parse_directional_light(const char* base_dir, const tinyobj::material_t& material) {
  auto& e = emitters.emplace_back(Emitter::Class::Directional);

  if (get_param(material, "color")) {
    e.emission.spectrum_index = load_illuminant_spectrum(data_buffer);
  } else {
    e.emission.spectrum_index = 1u;
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
    e.emission.image_index = add_image(tmp_buffer, Image::Regular);
  }

  if (get_param(material, "angular_diameter")) {
    float val = {};
    if (sscanf(data_buffer, "%f", &val) == 1) {
      e.angular_size = val * kPi / 180.0f;
    }
  }
}

void SceneRepresentationImpl::parse_env_light(const char* base_dir, const tinyobj::material_t& material) {
  auto& e = emitters.emplace_back(Emitter::Class::Environment);

  char tmp_buffer[2048] = {};
  if (get_param(material, "image")) {
    snprintf(tmp_buffer, sizeof(tmp_buffer), "%s/%s", base_dir, data_buffer);
  }

  float rotation = 0.0f;
  if (get_param(material, "rotation")) {
    rotation = static_cast<float>(atof(data_buffer)) / 360.0f;
  }
  e.emission.image_index = add_image(tmp_buffer, Image::BuildSamplingTable | Image::RepeatU, {rotation, 0.0f});

  if (get_param(material, "color")) {
    e.emission.spectrum_index = load_illuminant_spectrum(data_buffer);
  } else {
    e.emission.spectrum_index = 1u;
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
    auto& d = emitters.emplace_back(Emitter::Class::Directional);
    d.emission.spectrum_index = add_spectrum(sun_spectrum);
    scene_spectrums[d.emission.spectrum_index].spectrum.scale(sun_scale);
    d.angular_size = angular_size;
    d.direction = direction;

    if (angular_size > 0.0f) {
      d.emission.image_index = add_image(nullptr, kSunImageDimensions, Image::Delay);
      auto& img = images.get(d.emission.image_index);
      scattering::generate_sun_image(scattering_parameters, kSunImageDimensions, direction, angular_size, img.pixels.f32.a, scheduler);
    }
  }

  {
    auto& e = emitters.emplace_back(Emitter::Class::Environment);
    e.emission.spectrum_index = add_spectrum(sun_spectrum);
    scene_spectrums[e.emission.spectrum_index].spectrum.scale(sky_scale);
    e.emission.image_index = add_image(nullptr, sky_image_dimensions, Image::BuildSamplingTable | Image::Delay);
    e.direction = direction;

    auto& img = images.get(e.emission.image_index);
    scattering::generate_sky_image(scattering_parameters, sky_image_dimensions, direction, extinction, img.pixels.f32.a, scheduler);
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

    add_spectrum(name.c_str(), illuminant ? SpectralDistribution::rgb_luminance(value) : SpectralDistribution::rgb_reflectance(value));
    initialized = true;
  } else if (get_param(material, "blackbody")) {
    auto params = split_params(data_buffer);
    if (params.size() < 1) {
      log::warning("Spectrum `%s` uses blackbody but did not provide temperature value - skipped", name.c_str());
      return;
    }

    float t = static_cast<float>(atof(params[0]));
    add_spectrum(name.c_str(), SpectralDistribution::from_black_body(t, scale));
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
    add_spectrum(name.c_str(), SpectralDistribution::from_normalized_black_body(t, scale));
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
    add_spectrum(name.c_str(), spectrum);
  }

  uint32_t i = find_spectrum(name.c_str());
  if (i != kInvalidIndex) {
    scene_spectrums[i].spectrum.scale(scale);
  }
}

uint32_t SceneRepresentationImpl::load_reflectance_spectrum(char* data) {
  auto params = split_params(data);

  if (params.size() == 1) {
    auto i = find_spectrum(params[0]);
    if (i != kInvalidIndex)
      return i;
  }

  if (params.size() == 3) {
    float3 value = gamma_to_linear({
      static_cast<float>(atof(params[0])),
      static_cast<float>(atof(params[1])),
      static_cast<float>(atof(params[2])),
    });
    char buffer[128] = {};
    snprintf(buffer, sizeof(buffer), "spectrum%.4f%.4f%.4f", value.x, value.y, value.z);
    return add_spectrum(buffer, SpectralDistribution::rgb_reflectance(value));
  }

  return 0;
}

uint32_t SceneRepresentationImpl::load_illuminant_spectrum(char* data) {
  auto params = split_params(data);

  if (params.size() == 1) {
    float value = 0.0f;
    if (sscanf(params[0], "%f", &value) == 1) {
      return add_spectrum(params[0], SpectralDistribution::rgb_luminance({value, value, value}));
    }

    auto i = find_spectrum(params[0]);
    if (i != kInvalidIndex)
      return i;
  }

  if (params.size() == 3) {
    float3 value = {
      static_cast<float>(atof(params[0])),
      static_cast<float>(atof(params[1])),
      static_cast<float>(atof(params[2])),
    };
    char buffer[128] = {};
    snprintf(buffer, sizeof(buffer), "spectrum%.4f%.4f%.4f", value.x, value.y, value.z);
    return add_spectrum(buffer, SpectralDistribution::rgb_luminance(value));
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

  return add_spectrum(data, emitter_spectrum);
}

void SceneRepresentationImpl::parse_material(const char* base_dir, const tinyobj::material_t& material) {
  uint32_t material_index = kInvalidIndex;

  if (material_mapping.count(material.name) == 0) {
    material_index = add_material(material.name.c_str());
  } else {
    material_index = material_mapping.at(material.name);
  }

  auto& mtl = materials[material_index];

  mtl.cls = Material::Class::Diffuse;
  mtl.reflectance = {add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}))};
  mtl.transmittance = {add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}))};
  mtl.subsurface.scattering_distance_spectrum = add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 0.2f, 0.04f}));

  if (get_param(material, "base")) {
    auto i = material_mapping.find(data_buffer);
    if (i != material_mapping.end()) {
      mtl = materials[i->second];
    }
  }

  if (get_param(material, "Kd")) {
    mtl.transmittance.spectrum_index = load_reflectance_spectrum(data_buffer);
  }

  if (get_param(material, "Ks")) {
    mtl.reflectance.spectrum_index = load_reflectance_spectrum(data_buffer);
  }

  if (get_param(material, "Kt")) {
    mtl.transmittance.spectrum_index = load_reflectance_spectrum(data_buffer);
  }

  if (get_param(material, "Pr")) {
    float4 value = {};
    if (sscanf(data_buffer, "%f %f", &value.x, &value.y) == 2) {
      mtl.roughness.value = sqr(value);
    } else if (sscanf(data_buffer, "%f", &value.x) == 1) {
      mtl.roughness = {sqr(value.x), sqr(value.x)};
    }
  }

  if (get_file(base_dir, material.diffuse_texname)) {
    mtl.transmittance.image_index = add_image(data_buffer, Image::RepeatU | Image::RepeatV);
  }

  if (get_file(base_dir, material.specular_texname)) {
    mtl.reflectance.image_index = add_image(data_buffer, Image::RepeatU | Image::RepeatV);
  }

  if (get_file(base_dir, material.transmittance_texname)) {
    mtl.transmittance.image_index = add_image(data_buffer, Image::RepeatU | Image::RepeatV);
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

  if (get_param(material, "int_ior")) {
    float2 values = {};
    int values_read = sscanf(data_buffer, "%f %f", &values.x, &values.y);
    if (values_read == 1) {
      // interpret as eta
      mtl.int_ior.eta = SpectralDistribution::constant(values.x);
      mtl.int_ior.k = SpectralDistribution::null();
    } else if (values_read == 2) {
      // interpret as eta / k
      mtl.int_ior.eta = SpectralDistribution::constant(values.x);
      mtl.int_ior.k = SpectralDistribution::constant(values.y);
    } else {
      char buffer[1024] = {};
      snprintf(buffer, sizeof(buffer), "%sspectrum/%s.spd", env().data_folder(), data_buffer);
      mtl.int_ior = RefractiveIndex::load_from_file(buffer);
    }
  }

  if (get_param(material, "ext_ior")) {
    float2 values = {};
    if (sscanf(data_buffer, "%f %f", &values.x, &values.y) == 2) {
      // interpret as eta/k
      mtl.ext_ior.eta = SpectralDistribution::constant(values.x);
      mtl.ext_ior.k = SpectralDistribution::constant(values.y);
    } else {
      char buffer[256] = {};
      snprintf(buffer, sizeof(buffer), "%sspectrum/%s.spd", env().data_folder(), data_buffer);
      mtl.ext_ior = RefractiveIndex::load_from_file(buffer);
    }
  } else {
    mtl.ext_ior.eta = SpectralDistribution::constant(1.0f);
    mtl.ext_ior.k = SpectralDistribution::null();
  }

  if (get_param(material, "int_medium")) {
    auto m = mediums.find(data_buffer);
    if (m == kInvalidIndex) {
      log::warning("Medium %s was not declared, but used in material %s as internal medium", data_buffer, material.name.c_str());
    }
    mtl.int_medium = m;
  }

  if (get_param(material, "ext_medium")) {
    auto m = mediums.find(data_buffer);
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
        mtl.normal_image_index = add_image(buffer, Image::RepeatU | Image::RepeatV | Image::SkipSRGBConversion);
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
        mtl.thinfilm.thinkness_image = add_image(buffer, Image::RepeatU | Image::RepeatV);
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
          mtl.thinfilm.ior.eta = SpectralDistribution::constant(value);
        } else {
          char buffer[256] = {};
          snprintf(buffer, sizeof(buffer), "%sspectrum/%s.spd", env().data_folder(), params[i + 1]);
          mtl.thinfilm.ior = RefractiveIndex::load_from_file(buffer);
        }
      }
    }
  }

  if (get_param(material, "subsurface")) {
    mtl.subsurface.cls = SubsurfaceMaterial::Class::RandomWalk;

    auto params = split_params(data_buffer);
    for (uint64_t i = 0, e = params.size(); i < e; ++i) {
      if ((strcmp(params[i], "path") == 0) && (i + 1 < e)) {
        bool is_refraction = (strcmp(params[i + 1], "refracted") == 0) || (strcmp(params[i + 1], "refraction") == 0);
        mtl.subsurface.path = is_refraction ? SubsurfaceMaterial::Path::Refracted : SubsurfaceMaterial::Path::Diffuse;
      }

      if ((strcmp(params[i], "distances") == 0) && (i + 3 < e)) {
        float dr = static_cast<float>(atof(params[i + 1]));
        float dg = static_cast<float>(atof(params[i + 2]));
        float db = static_cast<float>(atof(params[i + 3]));
        mtl.subsurface.scattering_distance_spectrum = add_spectrum(SpectralDistribution::rgb_reflectance({dr, dg, db}));
        i += 3;
      }

      if ((strcmp(params[i], "scale") == 0) && (i + 1 < e)) {
        mtl.subsurface.scale = static_cast<float>(atof(params[i + 1]));
        i += 1;
      }

      if ((strcmp(params[i], "class") == 0) && (i + 1 < e)) {
        if (strcmp(params[i + 1], "approximate") == 0) {
          mtl.subsurface.cls = SubsurfaceMaterial::Class::ChristensenBurley;
        }
        i += 1;
      }
    }
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

  images.load_images();
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

void SceneRepresentationImpl::load_gltf_node(const tinygltf::Model& model, const tinygltf::Node& node, const float4x4& transform) {
  if ((node.mesh >= 0) && (node.mesh < model.meshes.size())) {
    load_gltf_mesh(node, model, model.meshes.at(node.mesh), transform);
  }

  for (const auto& child : node.children) {
    auto child_transform = transform * build_gltf_node_transform(model.nodes[child]);
    load_gltf_node(model, model.nodes[child], child_transform);
  }
}

uint32_t SceneRepresentationImpl::load_from_gltf(const char* file_name, bool binary) {
  tinygltf::TinyGLTF loader;
  tinygltf::Model model;
  std::string errors;
  std::string warnings;

  bool load_result = false;

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
            self->gltf_image_mapping[image_index] = self->add_image(buffer, Image::RepeatU | Image::RepeatV, {});
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

  for (auto& material : model.materials) {
    std::string material_name = material.name;
    uint32_t index = 1;
    while (has_material(material_name.c_str())) {
      char buffer[1024] = {};
      snprintf(buffer, sizeof(buffer), "%s-%04u", material.name.c_str(), index);
      material_name = buffer;
      ++index;
    }

    const auto& pbr = material.pbrMetallicRoughness;

    uint32_t material_index = add_material(material_name.c_str());

    auto& mtl = materials[material_index];
    mtl.cls = Material::Class::Principled;
    mtl.roughness.value = {float(pbr.roughnessFactor), float(pbr.roughnessFactor)};
    mtl.metalness.value = {float(pbr.metallicFactor), float(pbr.metallicFactor)};
    mtl.ext_ior.cls = SpectralDistribution::Class::Dielectric;
    mtl.ext_ior.eta = SpectralDistribution::constant(1.0f);
    mtl.ext_ior.k = SpectralDistribution::null();
    mtl.int_ior.cls = SpectralDistribution::Class::Conductor;
    mtl.int_ior.eta = SpectralDistribution::constant(1.5f);
    mtl.int_ior.k = SpectralDistribution::null();

    float3 rgb = {1.0f, 1.0f, 1.0f};
    const auto& base_color = material.pbrMetallicRoughness.baseColorFactor;
    if (base_color.size() >= 3) {
      rgb = {float(base_color[0]), float(base_color[1]), float(base_color[2])};
    }
    mtl.transmittance.spectrum_index = add_spectrum(SpectralDistribution::rgb_reflectance(rgb));

    if ((pbr.baseColorTexture.index != -1) && (gltf_image_mapping.count(pbr.baseColorTexture.index) > 0)) {
      mtl.transmittance.image_index = gltf_image_mapping.at(pbr.baseColorTexture.index);
      mtl.reflectance.image_index = mtl.transmittance.image_index;
    }

    if ((pbr.metallicRoughnessTexture.index != -1) && (gltf_image_mapping.count(pbr.metallicRoughnessTexture.index) > 0)) {
      auto image_index = gltf_image_mapping.at(pbr.metallicRoughnessTexture.index);
      mtl.roughness.image_index = image_index;
      mtl.roughness.channel = 1u;
      mtl.metalness.image_index = image_index;
      mtl.metalness.channel = 2u;
    }

    if ((material.normalTexture.index != -1) && gltf_image_mapping.count(material.normalTexture.index) > 0) {
      mtl.normal_image_index = gltf_image_mapping.at(material.normalTexture.index);
      mtl.normal_scale = 1.0f;
      add_image_options(mtl.normal_image_index, Image::SkipSRGBConversion);
    }

    mtl.reflectance.spectrum_index = add_spectrum(SpectralDistribution::constant(1.0f));
  }

  if (materials.empty()) {
    uint32_t material_index = add_material("default");
    auto& mtl = materials[material_index];
    mtl.cls = Material::Class::Diffuse;
    mtl.transmittance.spectrum_index = add_spectrum(SpectralDistribution::constant(1.0f));
    mtl.reflectance.spectrum_index = add_spectrum(SpectralDistribution::constant(1.0f));
  }

  for (const auto& scene : model.scenes) {
    for (int32_t node_index : scene.nodes) {
      if ((node_index < 0) || (node_index >= model.nodes.size()))
        continue;

      const float4x4 identity = build_gltf_node_transform({});
      const auto& node = model.nodes[node_index];
      load_gltf_node(model, node, identity);
    }
  }

  return LoadSucceeded;
}

void SceneRepresentationImpl::load_gltf_mesh(const tinygltf::Node& node, const tinygltf::Model& model, const tinygltf::Mesh& mesh, const float4x4& transform) {
  for (const auto& primitive : mesh.primitives) {
    bool has_positions = primitive.attributes.count("POSITION") > 0;
    bool has_normals = primitive.attributes.count("NORMAL") > 0;
    bool has_tex_coords = primitive.attributes.count("TEXCOORD_0") > 0;
    bool has_tangents = primitive.attributes.count("TANGENT") > 0;

    if (has_positions == false)
      continue;

    bool valid_material = (primitive.material >= 0) && (primitive.material < model.materials.size());
    uint32_t material_index = valid_material ? static_cast<uint32_t>(primitive.material) : 0;

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
      triangle_to_material.emplace_back(material_index);
      triangle_to_emitter.emplace_back(kInvalidIndex);

      uint32_t base_index = static_cast<uint32_t>(vertices.size());
      Triangle& tri = triangles.emplace_back();
      tri.i[0] = base_index + 0;
      tri.i[1] = base_index + 1;
      tri.i[2] = base_index + 2;

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
        triangle_to_material.pop_back();
      } else if (has_normals == false) {
        vertices[vertices.size() - 1u].nrm = tri.geo_n;
        vertices[vertices.size() - 2u].nrm = tri.geo_n;
        vertices[vertices.size() - 3u].nrm = tri.geo_n;
      }
    }
  }
}

void build_emitters_distribution(Scene& scene) {
  DistributionBuilder emitters_distribution(scene.emitters_distribution, static_cast<uint32_t>(scene.emitters.count));
  scene.environment_emitters.count = 0;
  for (uint32_t i = 0; i < scene.emitters.count; ++i) {
    auto& emitter = scene.emitters[i];
    if (emitter.is_distant()) {
      uint32_t spectrum_index = emitter.emission.spectrum_index;
      ETX_CRITICAL(spectrum_index != kInvalidIndex);
      emitter.weight = scene.spectrums[spectrum_index].luminance() * kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius;
    }
    emitter.equivalent_disk_size = 2.0f * std::tan(emitter.angular_size / 2.0f);
    emitter.angular_size_cosine = std::cos(emitter.angular_size / 2.0f);
    emitters_distribution.add(emitter.weight);

    if (emitter.is_local()) {
      scene.triangle_to_emitter[emitter.triangle_index] = i;
    } else if (emitter.is_distant() && (emitter.weight > 0.0f)) {
      scene.environment_emitters.emitters[scene.environment_emitters.count++] = i;
    }
  }
  emitters_distribution.finalize();
}

namespace spectrum {

Pointer<Spectrums> shared() {
  return &shared_spectrums;
}

}  // namespace spectrum

}  // namespace etx
