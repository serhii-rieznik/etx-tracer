#include <etx/core/core.hxx>
#include <etx/log/log.hxx>
#include <etx/core/environment.hxx>

#include <etx/render/shared/base.hxx>
#include <etx/render/shared/scene.hxx>

#include <etx/render/host/scene_loader.hxx>
#include <etx/render/host/image_pool.hxx>
#include <etx/render/host/medium_pool.hxx>
#include <etx/render/host/distribution_builder.hxx>

#include <vector>
#include <unordered_map>
#include <string>
#include <set>

#include <mikktspace.h>
#include <tiny_obj_loader.hxx>
#include <jansson.h>

namespace etx {

Spectrums* spectrums() {
  static Spectrums _spectrums;
  static auto invoke_once = []() {
    using SPD = SpectralDistribution;
    rgb::init_spectrums(_spectrums);
    {
      static float w[2] = {spectrum::kShortestWavelength, spectrum::kLongestWavelength};
      static float eta[2] = {1.5f, 1.5f};
      static float k[2] = {0.0f, 0.0f};

      static const float2 chrome_samples_eta[] = {{0.354f, 1.84f}, {0.368f, 1.87f}, {0.381f, 1.92f}, {0.397f, 2.00f}, {0.413f, 2.08f}, {0.431f, 2.19f}, {0.451f, 2.33f},
        {0.471f, 2.51f}, {0.496f, 2.75f}, {0.521f, 2.94f}, {0.549f, 3.18f}, {0.582f, 3.22f}, {0.617f, 3.17f}, {0.659f, 3.09f}, {0.704f, 3.05f}, {0.756f, 3.08f}, {0.821f, 3.20f},
        {0.892f, 3.30f}};

      static const float2 chrome_samples_k[] = {{0.354f, 2.64f}, {0.368f, 2.69f}, {0.381f, 2.74f}, {0.397f, 2.83f}, {0.413f, 2.93f}, {0.431f, 3.04f}, {0.451f, 3.14f},
        {0.471f, 3.24f}, {0.496f, 3.30f}, {0.521f, 3.33f}, {0.549f, 3.33f}, {0.582f, 3.30f}, {0.617f, 3.30f}, {0.659f, 3.34f}, {0.704f, 3.39f}, {0.756f, 3.42f}, {0.821f, 3.48f},
        {0.892f, 3.52f}};

      static const float2 plastic_samples_eta[] = {{40.0000f, 1.519f}, {41.6667f, 1.519f}, {43.4783f, 1.519f}, {45.4545f, 1.520f}, {47.6190f, 1.521f}, {50.0000f, 1.521f},
        {52.6316f, 1.521f}, {55.5556f, 1.521f}, {58.8235f, 1.521f}, {62.5000f, 1.521f}, {66.6667f, 1.521f}, {71.4286f, 1.521f}, {76.9231f, 1.520f}, {83.3333f, 1.520f},
        {90.9091f, 1.520f}};

      _spectrums.thinfilm.eta = SPD::from_samples(w, eta, 2, SPD::Class::Reflectance, &_spectrums);
      _spectrums.thinfilm.k = SPD::from_samples(w, k, 2, SPD::Class::Reflectance, &_spectrums);

      _spectrums.conductor.eta = SPD::from_samples(chrome_samples_eta, std::size(chrome_samples_eta), SPD::Class::Reflectance, &_spectrums);
      _spectrums.conductor.k = SPD::from_samples(chrome_samples_eta, std::size(chrome_samples_k), SPD::Class::Reflectance, &_spectrums);

      _spectrums.dielectric.eta = SPD::from_samples(plastic_samples_eta, std::size(plastic_samples_eta), SPD::Class::Reflectance, &_spectrums);
      _spectrums.dielectric.k = SPD::from_constant(0.0f);
    }
    return true;
  }();

  return &_spectrums;
}

inline bool value_is_correct(float t) {
  return !std::isnan(t) && !std::isinf(t);
}

inline bool value_is_correct(const float3& v) {
  return value_is_correct(v.x) && value_is_correct(v.y) && value_is_correct(v.z);
}

inline bool is_valid_vector(const float3& v) {
  return value_is_correct(v) && (dot(v, v) > 0.0f);
}

struct SceneRepresentationImpl {
  TaskScheduler& scheduler;
  std::vector<Vertex> vertices;
  std::vector<Triangle> triangles;
  std::vector<Material> materials;
  std::vector<Emitter> emitters;

  ImagePool images;
  MediumPool mediums;

  std::unordered_map<std::string, uint32_t> material_mapping;
  uint32_t camera_medium_index = kInvalidIndex;
  uint32_t camera_lens_shape_image_index = kInvalidIndex;

  Scene scene;
  bool loaded = false;

  uint32_t add_image(const char* path, uint32_t options) {
    std::string id = path ? path : ("image-" + std::to_string(images.array_size()));
    return images.add_from_file(path, options);
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

  uint32_t add_medium(Medium::Class cls, const char* name, const char* volume_file, const SpectralDistribution& s_a, const SpectralDistribution& s_t, float g) {
    std::string id = name ? name : ("medium-" + std::to_string(mediums.array_size()));
    return mediums.add(cls, id, volume_file, s_a, s_t, g, spectrums());
  }

  SceneRepresentationImpl(TaskScheduler& s)
    : scheduler(s)
    , images(s) {
    images.init(1024u);
    mediums.init(1024u);
    scene.camera = build_camera({5.0f, 5.0f, 5.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1280u, 720u}, 26.99f, 0.0f, 1.0f);
  }

  ~SceneRepresentationImpl() {
    cleanup();
    images.cleanup();
    mediums.cleanup();
  }

  void cleanup() {
    vertices.clear();
    triangles.clear();
    materials.clear();
    emitters.clear();
    material_mapping.clear();
    camera_medium_index = kInvalidIndex;
    camera_lens_shape_image_index = kInvalidIndex;

    images.remove_all();
    mediums.remove_all();
    materials.reserve(1024);  // TODO : fix, images when reallocated are destroyed releasing memory

    free(scene.emitters_distribution.values);
    scene.emitters_distribution = {};

    auto camera = scene.camera;
    scene = {};
    scene.camera = camera;

    loaded = false;
  }

  bool calculate_area(Triangle& t) {
    t.geo_n = cross(vertices[t.i[1]].pos - vertices[t.i[0]].pos, vertices[t.i[2]].pos - vertices[t.i[0]].pos);
    t.area = 0.5f * length(t.geo_n);
    t.geo_n *= 0.5f / t.area;
    return t.area > 0.0f;
  }

  void validate_materials() {
    for (auto& mtl : materials) {
      if ((mtl.roughness.x > 0.0f) || (mtl.roughness.y > 0.0f)) {
        mtl.roughness.x = max(kEpsilon, mtl.roughness.x);
        mtl.roughness.y = max(kEpsilon, mtl.roughness.y);
      }

      if (mtl.int_ior.eta.empty() && mtl.int_ior.k.empty()) {
        if (mtl.cls == Material::Class::Conductor) {
          mtl.int_ior = spectrums()->conductor;
        } else {
          mtl.int_ior = spectrums()->dielectric;
        }
      }
    }
  }

  void validate_normals(std::vector<bool>& referenced_vertices) {
    std::set<uint32_t> reset_normals;

    referenced_vertices.resize(vertices.size());
    for (const auto& tri : triangles) {
      for (uint32_t i = 0; i < 3; ++i) {
        uint32_t index = tri.i[i];
        ETX_CRITICAL(is_valid_vector(tri.geo_n));
        referenced_vertices[index] = true;
        if (is_valid_vector(vertices[index].nrm) == false) {
          if (reset_normals.count(index) == 0) {
            vertices[index].nrm = tri.geo_n * tri.area;
            reset_normals.insert(index);
          } else {
            vertices[index].nrm += tri.geo_n * tri.area;
          }
        }
      }
    }

    if (reset_normals.empty() == false) {
      for (auto i : reset_normals) {
        ETX_ASSERT(is_valid_vector(vertices[i].nrm));
        vertices[i].nrm = normalize(vertices[i].nrm);
        ETX_ASSERT(is_valid_vector(vertices[i].nrm));
      }
    }
  }

  void build_tangents() {
    SMikkTSpaceInterface contextInterface = {};
    contextInterface.m_getNumFaces = [](const SMikkTSpaceContext* pContext) -> int {
      auto data = reinterpret_cast<Scene*>(pContext->m_pUserData);
      return static_cast<int>(data->triangles.count);
    };
    contextInterface.m_getNumVerticesOfFace = [](const SMikkTSpaceContext* pContext, const int iFace) -> int {
      return 3;
    };
    contextInterface.m_getPosition = [](const SMikkTSpaceContext* pContext, float fvPosOut[], const int iFace, const int iVert) {
      auto data = reinterpret_cast<Scene*>(pContext->m_pUserData);
      const auto& tri = data->triangles[iFace];
      const auto& vertex = data->vertices[tri.i[iVert]];
      fvPosOut[0] = vertex.pos.x;
      fvPosOut[1] = vertex.pos.y;
      fvPosOut[2] = vertex.pos.z;
    };
    contextInterface.m_getNormal = [](const SMikkTSpaceContext* pContext, float fvNormOut[], const int iFace, const int iVert) {
      auto data = reinterpret_cast<Scene*>(pContext->m_pUserData);
      const auto& tri = data->triangles[iFace];
      const auto& vertex = data->vertices[tri.i[iVert]];
      fvNormOut[0] = vertex.nrm.x;
      fvNormOut[1] = vertex.nrm.y;
      fvNormOut[2] = vertex.nrm.z;
    };
    contextInterface.m_getTexCoord = [](const SMikkTSpaceContext* pContext, float fvTexcOut[], const int iFace, const int iVert) {
      auto data = reinterpret_cast<Scene*>(pContext->m_pUserData);
      const auto& tri = data->triangles[iFace];
      const auto& vertex = data->vertices[tri.i[iVert]];
      fvTexcOut[0] = vertex.tex.x;
      fvTexcOut[1] = vertex.tex.y;
    };
    contextInterface.m_setTSpaceBasic = [](const SMikkTSpaceContext* pContext, const float fvTangent[], const float fSign, const int iFace, const int iVert) {
      auto data = reinterpret_cast<Scene*>(pContext->m_pUserData);
      const auto& tri = data->triangles[iFace];
      auto& vertex = data->vertices[tri.i[iVert]];
      vertex.tan.x = fvTangent[0];
      vertex.tan.y = fvTangent[1];
      vertex.tan.z = fvTangent[2];
      vertex.btn = normalize(cross(vertex.tan, vertex.nrm) * fSign);
    };

    SMikkTSpaceContext context = {};
    context.m_pUserData = &scene;
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
    if (triangles.empty()) {
      scene.bounding_sphere_center = {};
      scene.bounding_sphere_radius = kPlanetRadius + kAtmosphereRadius;
    } else {
      float3 bbox_min = {kMaxFloat, kMaxFloat, kMaxFloat};
      float3 bbox_max = {-kMaxFloat, -kMaxFloat, -kMaxFloat};
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
    }
    scene.camera_medium_index = camera_medium_index;
    scene.camera_lens_shape_image_index = camera_lens_shape_image_index;

    for (auto& emitter : emitters) {
      if (emitter.is_distant()) {
        ETX_ASSERT(emitter.weight == 0.0f);
        emitter.weight = kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius;
      }
    }

    scene.vertices = {vertices.data(), vertices.size()};
    scene.triangles = {triangles.data(), triangles.size()};
    scene.materials = {materials.data(), materials.size()};
    scene.emitters = {emitters.data(), emitters.size()};
    scene.images = {images.as_array(), images.array_size()};
    scene.mediums = {mediums.as_array(), mediums.array_size()};
    scene.spectrums = spectrums();
    scene.environment_emitters.count = 0;

    log::info("Building emitters distribution for %llu emitters...\n", scene.emitters.count);
    DistributionBuilder emitters_distribution(scene.emitters_distribution, static_cast<uint32_t>(scene.emitters.count));
    for (uint32_t i = 0; i < scene.emitters.count; ++i) {
      auto& emitter = emitters[i];
      emitter.equivalent_disk_size = 2.0f * std::tan(emitter.angular_size / 2.0f);
      emitter.angular_size_cosine = std::cos(emitter.angular_size / 2.0f);
      emitters_distribution.add(emitter.weight);
      if (emitter.is_local()) {
        triangles[emitter.triangle_index].emitter_index = i;
      } else if ((emitter.cls == Emitter::Class::Environment) || (emitter.cls == Emitter::Class::Directional)) {
        scene.environment_emitters.emitters[scene.environment_emitters.count++] = i;
      }
    }
    emitters_distribution.finalize();

    build_gpu_data();

    loaded = true;
  }

  void build_gpu_data() {
    /*/
    _gpu.image_buffers.resize(images.count);
    std::vector<ImageData> images_array(images.count);
    for (uint64_t i = 0, e = _images.size(); i < e; ++i) {
      const auto& img = _images[i];
      const auto& dim = img.dimensions();
      const auto& y_dist = img.y_distribution();

      uint64_t buffer_size = align_up(sizeof(float4) * dim.x * dim.y, 16llu);             // pixels
      buffer_size += align_up(y_dist.size() * sizeof(Distribution1DData::Entry), 16llu);  // y-values
      buffer_size += align_up(y_dist.size() * sizeof(Distribution1DData), 16llu);         // x-distributions
      for (uint64_t j = 0; j < y_dist.size(); ++j) {
        buffer_size += align_up(img.x_distribution(j).size() * sizeof(Distribution1DData::Entry), 16llu);  // x-values
      }

      _gpu.image_buffers[i] = gpu_device.create_buffer(buffer_size, nullptr, optix::Buffer::Suballocations);
      images_array[i] = img.data();

      if (images_array[i].pixels != nullptr) {
        auto pixels = gpu_device.upload_gpu_data(_gpu.image_buffers[i], nullptr, images_array[i].pixels, sizeof(float4) * dim.x * dim.y);
        images_array[i].pixels = reinterpret_cast<float4*>(pixels);
      } else {
        images_array[i].pixels = nullptr;
      }

      if (y_dist.size() > 0) {
        auto y_values = gpu_device.upload_gpu_data(_gpu.image_buffers[i], nullptr, y_dist.values(), y_dist.size() * sizeof(Distribution1DData::Entry));
        images_array[i].y_distribution.values = reinterpret_cast<Distribution1DData::Entry*>(y_values);

        std::vector<Distribution1DData> x_dists(y_dist.size());
        for (uint64_t j = 0; j < y_dist.size(); ++j) {
          const auto& x_dist = img.x_distribution(j);
          x_dists[j].size = x_dist.size();
          x_dists[j].total_weight = x_dist.total_weight();

          auto x_values = gpu_device.upload_gpu_data(_gpu.image_buffers[i], nullptr, x_dist.values(), x_dist.size() * sizeof(Distribution1DData::Entry));
          x_dists[j].values = reinterpret_cast<Distribution1DData::Entry*>(x_values);
        }

        auto x_distributions = gpu_device.upload_gpu_data(_gpu.image_buffers[i], nullptr, x_dists.data(), y_dist.size() * sizeof(Distribution1DData));
        images_array[i].x_distributions = reinterpret_cast<Distribution1DData*>(x_distributions);
      }
    }

    if (mediums.count > 0) {
      std::vector<MediumData> gpu_medium_data;
      gpu_medium_data.resize(mediums.count);

      uint64_t medium_buffer_size = align_up(sizeof(MediumData) * mediums.count, 16llu);
      for (uint64_t i = 0; i < mediums.count; ++i) {
        gpu_medium_data[i] = mediums[i];
        medium_buffer_size += align_up(mediums[i].density.count * sizeof(float), 16llu);
      }
      _gpu.mediums = gpu_device.create_buffer(medium_buffer_size, nullptr, optix::Buffer::Suballocations);

      for (uint64_t i = 0; i < mediums.count; ++i) {
        if (gpu_medium_data[i].density.count > 0) {
          auto ptr = gpu_device.upload_gpu_data(_gpu.mediums, nullptr, gpu_medium_data[i].density.a, sizeof(float) * gpu_medium_data[i].density.count);
          gpu_medium_data[i].density = make_array_view<float>(ptr, gpu_medium_data[i].density.count);  //
        }
      }

      auto ptr = gpu_device.upload_gpu_data(_gpu.mediums, nullptr, gpu_medium_data.data(), gpu_medium_data.size() * sizeof(MediumData));
      _gpu.data.mediums = make_array_view<MediumData>(ptr, mediums.count);
    }

    uint64_t emitter_buffer_size = align_up(emitters.count * sizeof(Emitter), 16llu) + align_up(emitters_distribution.size * sizeof(Distribution1DData::Entry), 16llu);
    _gpu.emitters = gpu_device.create_buffer(emitter_buffer_size, nullptr, optix::Buffer::Suballocations);
    auto emitters_ptr = gpu_device.upload_gpu_data(_gpu.emitters, nullptr, emitters.a, emitters.count * sizeof(Emitter));
    auto emitters_dist_ptr = gpu_device.upload_gpu_data(_gpu.emitters, nullptr, emitters_distribution.values, emitters_distribution.size * sizeof(Distribution1DData::Entry));

    _gpu.vertices = gpu_device.create_buffer(vertices.count * sizeof(Vertex), vertices.a, 0u);
    _gpu.triangles = gpu_device.create_buffer(triangles.count * sizeof(Triangle), triangles.a, 0u);
    _gpu.materials = gpu_device.create_buffer(materials.count * sizeof(Material), materials.a, 0u);
    _gpu.images = gpu_device.create_buffer(images.count * sizeof(ImageData), images_array.data(), 0u);

    _gpu.spectrums = gpu_device.create_buffer(sizeof(Spectrums), spectrums, 0);

    _gpu.acceleration_structure = gpu_device.build_acceleration_structure(  //
      _gpu.vertices, static_cast<uint32_t>(vertices.count),                 //
      _gpu.triangles, static_cast<uint32_t>(triangles.count));

    _gpu.data.vertices = make_array_view<Vertex>(gpu_device.get_buffer_device_pointer(_gpu.vertices), vertices.count);
    _gpu.data.triangles = make_array_view<Triangle>(gpu_device.get_buffer_device_pointer(_gpu.triangles), triangles.count);
    _gpu.data.materials = make_array_view<Material>(gpu_device.get_buffer_device_pointer(_gpu.materials), materials.count);
    _gpu.data.emitters = make_array_view<Emitter>(emitters_ptr, emitters.count);
    _gpu.data.images = make_array_view<ImageData>(gpu_device.get_buffer_device_pointer(_gpu.images), images.count);
    _gpu.data.emitters_distribution.size = emitters_distribution.size;
    _gpu.data.emitters_distribution.total_weight = emitters_distribution.total_weight;
    _gpu.data.emitters_distribution.values = reinterpret_cast<Distribution1DData::Entry*>(emitters_dist_ptr);
    _gpu.data.spectrums = reinterpret_cast<Spectrums*>(gpu_device.get_buffer_device_pointer(_gpu.spectrums));
    _gpu.data.environment_emitters = environment_emitters;
    _gpu.data.bounding_sphere_center = bounding_sphere_center;
    _gpu.data.bounding_sphere_radius = bounding_sphere_radius;
    _gpu.data.acceleration_structure = reinterpret_cast<uint64_t>(gpu_device.get_acceleration_structure_device_pointer(_gpu.acceleration_structure));
    _gpu.data.camera_medium_index = camera_medium_index;
    _gpu.data.camera_lens_shape_image_index = camera_lens_shape_image_index;
    // */
  }

  enum : uint32_t {
    LoadFailed = 0u,
    LoadSucceeded = 1u << 0u,
    HaveTangents = 1u << 1u,
  };

  uint32_t load_from_obj(const char* file_name, const char* mtl_file);
  void parse_obj_materials(const char* base_dir, const std::vector<tinyobj::material_t>& obj_materials);
};

Camera build_camera(const float3& origin, const float3& target, const float3& up, const uint2& viewport, float fov, float lens_radius, float focal_distance) {
  Camera result = {};
  result.focal_distance = focal_distance;
  result.lens_radius = lens_radius;
  update_camera(result, origin, target, up, viewport, fov);
  return result;
}

void update_camera(Camera& camera, const float3& origin, const float3& target, const float3& up, const uint2& viewport, float fov) {
  float4x4 view = look_at(origin, target, up);
  float4x4 proj = perspective(fov * kPi / 180.0f, viewport.x, viewport.y, 1.0f, 1024.0f);
  float4x4 inv_view = inverse(view);

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
  camera.image_size = viewport;
  camera.image_plane = float(camera.image_size.x) / (2.0f * camera.tan_half_fov);
}

float get_camera_fov(const Camera& camera) {
  return 2.0f * atanf(camera.tan_half_fov) * 180.0f / kPi;
}

ETX_PIMPL_IMPLEMENT(SceneRepresentation, Impl);

SceneRepresentation::SceneRepresentation(TaskScheduler& s) {
  ETX_PIMPL_INIT(SceneRepresentation, s);
}

SceneRepresentation::~SceneRepresentation() {
  ETX_PIMPL_CLEANUP(SceneRepresentation);
}

const Scene& SceneRepresentation::scene() const {
  return _private->scene;
}

Camera& SceneRepresentation::camera() {
  return _private->scene.camera;
}

SceneRepresentation::operator bool() const {
  return _private->loaded;
}

uint2 json_to_u2(json_t* a) {
  uint2 result = {};
  int i = 0;
  json_t* val = {};
  json_array_foreach(a, i, val) {
    if ((i >= 2) || (json_is_number(val) == false))
      break;

    switch (i) {
      case 0:
        result.x = static_cast<uint32_t>(json_number_value(val));
        break;
      case 1:
        result.y = static_cast<uint32_t>(json_number_value(val));
        break;
      default:
        break;
    }
  }
  return result;
}

float3 json_to_f3(json_t* a) {
  float3 result = {};
  int i = 0;
  json_t* val = {};
  json_array_foreach(a, i, val) {
    if ((i >= 3) || (json_is_number(val) == false))
      break;

    switch (i) {
      case 0:
        result.x = static_cast<float>(json_number_value(val));
        break;
      case 1:
        result.y = static_cast<float>(json_number_value(val));
        break;
      case 2:
        result.z = static_cast<float>(json_number_value(val));
        break;
      default:
        break;
    }
  }
  return result;
}

bool SceneRepresentation::load_from_file(const char* filename, uint32_t options) {
  _private->cleanup();

  uint32_t load_result = SceneRepresentationImpl::LoadFailed;

  std::string file_to_load = filename;
  std::string material_file = {};

  char base_folder[2048] = {};
  get_file_folder(filename, base_folder, sizeof(base_folder));

  auto& cam = _private->scene.camera;
  Camera::Class camera_cls = Camera::Class::Perspective;
  float3 camera_pos = cam.position;
  float3 camera_up = {0.0f, 1.0f, 0.0f};
  float3 camera_view = cam.position + cam.direction;
  uint2 viewport = cam.image_size;
  float camera_fov = get_camera_fov(cam);

  if (strcmp(get_file_ext(filename), ".json") == 0) {
    json_error_t err = {};
    auto js = json_load_file(filename, 0, &err);
    if (js == nullptr) {
      log::error("Failed to parse json file: %s\n%d / %d : %s", filename, err.line, err.column, err.text);
      return false;
    }

    if (json_is_object(js) == false) {
      log::error("Invalid scene description file: %s", filename);
      json_decref(js);
      return false;
    }

    const char* key = {};
    json_t* js_value = {};
    json_object_foreach(js, key, js_value) {
      if (strcmp(key, "geometry") == 0) {
        if (json_is_string(js_value) == false) {
          log::error("`geometry` in scene description should be a string (file name)");
          json_decref(js);
          return false;
        }
        file_to_load = std::string(base_folder) + json_string_value(js_value);
      } else if (strcmp(key, "materials") == 0) {
        if (json_is_string(js_value) == false) {
          log::error("`materials` in scene description should be a string (file name)");
          json_decref(js);
          return false;
        }
        material_file = json_string_value(js_value);
      } else if (strcmp(key, "camera") == 0) {
        if (json_is_object(js_value) == false) {
          log::error("`camera` in scene description should be an object");
          continue;
        }
        const char* cam_key = {};
        json_t* cam_value = {};
        json_object_foreach(js_value, cam_key, cam_value) {
          if ((strcmp(cam_key, "class") == 0) && json_is_string(cam_value)) {
            if (strcmp(json_string_value(cam_value), "eq") == 0) {
              camera_cls = Camera::Class::Equirectangular;
            }
          } else if ((strcmp(cam_key, "origin") == 0) && json_is_array(cam_value)) {
            camera_pos = json_to_f3(cam_value);
          } else if ((strcmp(cam_key, "target") == 0) && json_is_array(cam_value)) {
            camera_view = json_to_f3(cam_value);
          } else if ((strcmp(cam_key, "up") == 0) && json_is_array(cam_value)) {
            camera_up = json_to_f3(cam_value);
          } else if ((strcmp(cam_key, "viewport") == 0) && json_is_array(cam_value)) {
            viewport = json_to_u2(cam_value);
          } else if ((strcmp(cam_key, "fov") == 0) && json_is_number(cam_value)) {
            camera_fov = static_cast<float>(json_number_value(cam_value));
          } else if ((strcmp(cam_key, "lens-radius") == 0) && json_is_number(cam_value)) {
            cam.lens_radius = static_cast<float>(json_number_value(cam_value));
          } else if ((strcmp(cam_key, "focal-distance") == 0) && json_is_number(cam_value)) {
            cam.focal_distance = static_cast<float>(json_number_value(cam_value));
          }
        }
      }
    }
    json_decref(js);
  }

  auto ext = get_file_ext(file_to_load.c_str());
  if (strcmp(ext, ".obj") == 0) {
    load_result = _private->load_from_obj(file_to_load.c_str(), material_file.c_str());
  }

  if ((load_result & SceneRepresentationImpl::LoadSucceeded) == 0) {
    return false;
  }

  if (options & SetupCamera) {
    if (viewport.x * viewport.y == 0) {
      viewport = {1280, 720};
    }
    cam.cls = camera_cls;
    update_camera(cam, camera_pos, camera_view, camera_up, viewport, camera_fov);
  }

  if (_private->emitters.empty()) {
    printf("No emitters found, adding default environment image...\n");
    auto& sky = _private->emitters.emplace_back(Emitter::Class::Environment);
    sky.emission.spectrum = SpectralDistribution::from_constant(1.0f);
    sky.emission.image_index = _private->add_image(env().file_in_data("assets/hdri/environment.exr"), Image::RepeatU | Image::BuildSamplingTable);
  }

  _private->validate_materials();

  std::vector<bool> referenced_vertices;
  _private->validate_normals(referenced_vertices);

  if ((load_result & SceneRepresentationImpl::HaveTangents) == 0) {
    TimeMeasure m = {};
    log::warning("Calculating tangents...");
    _private->build_tangents();
    log::warning("Tangents calculated in %.2f sec\n", m.measure());
  }
  _private->validate_tangents(referenced_vertices);
  _private->commit();

  return true;
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

inline auto get_param(const tinyobj::material_t& m, const char* param, char buffer[]) -> bool {
  for (const auto& p : m.unknown_parameter) {
    if (_stricmp(p.first.c_str(), param) == 0) {
      if (buffer != nullptr) {
        memcpy(buffer, p.second.c_str(), p.second.size());
        buffer[p.second.size()] = 0;
      }
      return true;
    }
  }
  return false;
}

inline Material::Class material_string_to_class(const char* s) {
  if (strcmp(s, "diffuse") == 0)
    return Material::Class::Diffuse;
  else if (strcmp(s, "msdiffuse") == 0)
    return Material::Class::MultiscatteringDiffuse;
  else if (strcmp(s, "plastic") == 0)
    return Material::Class::Plastic;
  else if (strcmp(s, "conductor") == 0)
    return Material::Class::Conductor;
  else if (strcmp(s, "msconductor") == 0)
    return Material::Class::MultiscatteringConductor;
  else if (strcmp(s, "dielectric") == 0)
    return Material::Class::Dielectric;
  else if (strcmp(s, "msdielectric") == 0)
    return Material::Class::MultiscatteringDielectric;
  else if (strcmp(s, "thinfilm") == 0)
    return Material::Class::Thinfilm;
  else if (strcmp(s, "translucent") == 0)
    return Material::Class::Translucent;
  else if (strcmp(s, "mirror") == 0)
    return Material::Class::Mirror;
  else if (strcmp(s, "boundary") == 0)
    return Material::Class::Boundary;
  else if (strcmp(s, "generic") == 0)
    return Material::Class::Generic;
  else if (strcmp(s, "coating") == 0)
    return Material::Class::Coating;
  else if (strcmp(s, "mixture") == 0)
    return Material::Class::Mixture;
  else
    return Material::Class::Undefined;
}

inline bool get_file(const char* base_dir, const std::string& base, char buffer[]) {
  if (base.empty()) {
    return false;
  }
  sprintf(buffer, "%s/%s", base_dir, base.c_str());
  return true;
};

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
  vertices.reserve(total_triangles * 3);

  for (const auto& shape : obj_shapes) {
    uint64_t index_offset = 0;
    float3 shape_bbox_min = {FLT_MAX, FLT_MAX, FLT_MAX};
    float3 shape_bbox_max = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

    for (uint64_t face = 0, face_e = shape.mesh.num_face_vertices.size(); face < face_e; ++face) {
      int material_id = shape.mesh.material_ids[face];
      if (material_id == -1) {
        continue;
      }
      const auto& source_material = obj_materials[material_id];

      uint64_t face_size = shape.mesh.num_face_vertices[face];
      ETX_ASSERT(face_size == 3);

      auto& tri = triangles.emplace_back();
      tri.material_index = material_mapping[source_material.name];
      auto& mtl = materials[tri.material_index];

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

      if (calculate_area(tri) == false) {
        triangles.pop_back();
      }

      bool emissive_material = (source_material.emission[0] > 0.0f) || (source_material.emission[1] > 0.0f) || (source_material.emission[2] > 0.0f);

      if (emissive_material) {
        char data_buffer[2048] = {};
        uint32_t emissive_image_index = kInvalidIndex;
        if (get_file(base_dir, source_material.emissive_texname, data_buffer)) {
          emissive_image_index = add_image(data_buffer, Image::RepeatU | Image::RepeatV | Image::BuildSamplingTable);
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
              float2 uv = lerp_uv({vertices.data(), vertices.size()}, tri, random_barycentric(u, v));
              float4 val = img.evaluate(uv);
              texture_emission += luminance(to_float3(val)) * du * dv * val.w;
            }
          }
        }

        auto& e = emitters.emplace_back(Emitter::Class::Area);
        e.emission.spectrum = rgb::make_illuminant_spd(to_float3(source_material.emission), spectrums());

        if (get_param(obj_materials[material_id], "emitter", data_buffer)) {
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
              e.emission.spectrum = SpectralDistribution::from_black_body(static_cast<float>(atof(params[i + 1])), spectrums());
              i += 1;
            } else if ((strcmp(params[i], "nblackbody") == 0) && (i + 1 < end)) {
              float t = static_cast<float>(atof(params[i + 1]));
              float w = spectrum::black_body_radiation_maximum_wavelength(t);
              float r = spectrum::black_body_radiation(w, t);
              e.emission.spectrum = SpectralDistribution::from_black_body(t, spectrums()) / r;
              i += 1;
            } else if ((strcmp(params[i], "scale") == 0) && (i + 1 < end)) {
              e.emission.spectrum *= static_cast<float>(atof(params[i + 1]));
              i += 1;
            } else if ((strcmp(params[i], "spectrum") == 0) && (i + 1 < end)) {
              char buffer[2048] = {};
              snprintf(buffer, sizeof(buffer), "%s/%s", base_dir, data_buffer);
              SpectralDistribution::load_from_file(buffer, e.emission.spectrum, nullptr, SpectralDistribution::Class::Illuminant, spectrums());
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
        e.medium_index = mtl.ext_medium;
        e.triangle_index = static_cast<uint32_t>(triangles.size() - 1llu);
        e.weight = power_scale * (tri.area * kPi) * (e.emission.spectrum.total_power() * texture_emission);
        e.emission.image_index = emissive_image_index;
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

void SceneRepresentationImpl::parse_obj_materials(const char* base_dir, const std::vector<tinyobj::material_t>& obj_materials) {
  for (const auto& material : obj_materials) {
    char data_buffer[1024] = {};
    char tmp_buffer[2048] = {};

    if (material.name == "et::camera") {
      if (get_param(material, "shape", data_buffer)) {
        snprintf(tmp_buffer, sizeof(tmp_buffer), "%s/%s", base_dir, data_buffer);
        camera_lens_shape_image_index = add_image(tmp_buffer, Image::BuildSamplingTable | Image::UniformSamplingTable);
      }
    } else if (material.name == "et::medium") {
      char name_buffer[2048] = {};
      if (get_param(material, "id", name_buffer) == false) {
        continue;
      }

      SpectralDistribution s_a = SpectralDistribution::from_constant(0.0f);
      if (get_param(material, "sigma_a", data_buffer)) {
        float val[3] = {};
        if (sscanf(data_buffer, "%f %f %f", val + 0, val + 1, val + 2) == 3) {
          s_a = rgb::make_reflectance_spd({val[0], val[1], val[2]}, spectrums());
        }
      }

      SpectralDistribution s_t = SpectralDistribution::from_constant(0.0f);
      if (get_param(material, "sigma_s", data_buffer)) {
        float val[3] = {};
        if (sscanf(data_buffer, "%f %f %f", val + 0, val + 1, val + 2) == 3) {
          s_t = rgb::make_reflectance_spd({val[0], val[1], val[2]}, spectrums());
        }
      }

      float g = 0.0f;
      if (get_param(material, "g", data_buffer)) {
        float val = {};
        if (sscanf(data_buffer, "%f", &val) == 1) {
          g = val;
        }
      }

      Medium::Class cls = Medium::Class::Homogeneous;

      if (get_param(material, "volume", data_buffer)) {
        if (strlen(data_buffer) > 0) {
          snprintf(tmp_buffer, sizeof(tmp_buffer), "%s%s", base_dir, data_buffer);
          cls = Medium::Class::Heterogeneous;
        }
      }

      uint32_t medium_index = add_medium(cls, name_buffer, tmp_buffer, s_a, s_t, g);

      if (strcmp(name_buffer, "camera") == 0) {
        camera_medium_index = medium_index;
      }

    } else if (material.name == "et::dir") {
      SpectralDistribution emitter_spectrum = SpectralDistribution::from_constant(1.0f);
      auto dir = float3{1.0f, 1.0f, 1.0f};

      if (get_param(material, "color", data_buffer)) {
        float value[3] = {};
        if (sscanf(data_buffer, "%f %f %f", value + 0, value + 1, value + 2) == 3) {
          emitter_spectrum = rgb::make_illuminant_spd({value[0], value[1], value[2]}, spectrums());
        } else {
          float scale = 1.0f;
          auto params = split_params(data_buffer);
          for (uint64_t i = 0, e = params.size(); i < e; ++i) {
            if ((strcmp(params[i], "blackbody") == 0) && (i + 1 < e)) {
              float t = static_cast<float>(atof(params[i + 1]));
              emitter_spectrum = SpectralDistribution::from_black_body(t, spectrums());
              i += 1;
            } else if ((strcmp(params[i], "nblackbody") == 0) && (i + 1 < e)) {
              float t = static_cast<float>(atof(params[i + 1]));
              float w = spectrum::black_body_radiation_maximum_wavelength(t);
              float r = spectrum::black_body_radiation(w, t);
              emitter_spectrum = SpectralDistribution::from_black_body(t, spectrums()) / r;
              i += 1;
            } else if ((strcmp(params[i], "scale") == 0) && (i + 1 < e)) {
              scale = static_cast<float>(atof(params[i + 1]));
              i += 1;
            }
          }
          emitter_spectrum *= scale;
        }
      }

      if (get_param(material, "direction", data_buffer)) {
        float value[3] = {};
        if (sscanf(data_buffer, "%f %f %f", value + 0, value + 1, value + 2) == 3) {
          dir = {value[0], value[1], value[2]};
        }
      }

      auto& e = emitters.emplace_back(Emitter::Class::Directional);
      e.emission.spectrum = emitter_spectrum;  // rgb::make_illuminant_spd(color, spectrums());
      e.direction = normalize(dir);

      if (get_param(material, "image", data_buffer)) {
        snprintf(tmp_buffer, sizeof(tmp_buffer), "%s/%s", base_dir, data_buffer);
        e.emission.image_index = add_image(tmp_buffer, Image::Regular);
      }

      if (get_param(material, "angular_diameter", data_buffer)) {
        float val = {};
        if (sscanf(data_buffer, "%f", &val) == 1) {
          e.angular_size = val * kPi / 180.0f;
        }
      }
    } else if (material.name == "et::env") {
      auto& e = emitters.emplace_back(Emitter::Class::Environment);

      if (get_param(material, "image", data_buffer)) {
        snprintf(tmp_buffer, sizeof(tmp_buffer), "%s/%s", base_dir, data_buffer);
      }

      e.emission.spectrum = rgb::make_illuminant_spd({1.0f, 1.0f, 1.0f}, spectrums());
      e.emission.image_index = add_image(tmp_buffer, Image::BuildSamplingTable | Image::RepeatU);

      if (get_param(material, "color", data_buffer)) {
        float color[3] = {};
        if (sscanf(data_buffer, "%f %f %f", color + 0, color + 1, color + 2) == 3) {
          e.emission.spectrum = rgb::make_illuminant_spd({color[0], color[1], color[2]}, spectrums());
        }
      }
    } else {
      if (material_mapping.count(material.name) == 0) {
        add_material(material.name.c_str());
      }
      uint32_t material_index = material_mapping[material.name];
      auto& mtl = materials[material_index];

      mtl.diffuse.spectrum = rgb::make_reflectance_spd(to_float3(material.diffuse), spectrums());
      mtl.specular.spectrum = rgb::make_reflectance_spd(to_float3(material.specular), spectrums());
      mtl.transmittance.spectrum = rgb::make_reflectance_spd(to_float3(material.transmittance), spectrums());

      mtl.roughness = {material.roughness, material.roughness};
      mtl.metalness = material.metallic;

      if (get_file(base_dir, material.diffuse_texname, data_buffer)) {
        mtl.diffuse.image_index = add_image(data_buffer, Image::RepeatU | Image::RepeatV);
      }

      if (get_file(base_dir, material.specular_texname, data_buffer)) {
        mtl.specular.image_index = add_image(data_buffer, Image::RepeatU | Image::RepeatV);
      }

      if (get_param(material, "material", data_buffer)) {
        auto params = split_params(data_buffer);
        for (uint64_t i = 0, e = params.size(); i < e; ++i) {
          if ((strcmp(params[i], "class") == 0) && (i + 1 < e)) {
            mtl.cls = material_string_to_class(params[i + 1]);
            i += 1;
          }
          if ((strcmp(params[i], "uroughness") == 0) && (i + 1 < e)) {
            float param = 0.0f;
            if (sscanf(params[i + 1], "%f", &param) == 1) {
              mtl.roughness.x = param;
            }
            i += 1;
          }
          if ((strcmp(params[i], "vroughness") == 0) && (i + 1 < e)) {
            float param = 0.0f;
            if (sscanf(params[i + 1], "%f", &param) == 1) {
              mtl.roughness.y = param;
            }
            i += 1;
          }
        }
      } else {
        mtl.cls = Material::Class::Diffuse;
      }

      if (get_param(material, "int_ior", data_buffer)) {
        float2 values = {};
        if (sscanf(data_buffer, "%f %f", &values.x, &values.y) == 2) {
          // interpret as eta/k
          mtl.int_ior.eta = SpectralDistribution::from_constant(values.x);
          mtl.int_ior.k = SpectralDistribution::from_constant(values.y);
        } else {
          char buffer[256] = {};
          snprintf(buffer, sizeof(buffer), "%sspectrum/%s.spd", env().data_folder(), data_buffer);
          SpectralDistribution::load_from_file(buffer, mtl.int_ior.eta, &mtl.int_ior.k, SpectralDistribution::Class::Reflectance, spectrums());
        }
      }

      if (get_param(material, "ext_ior", data_buffer)) {
        float2 values = {};
        if (sscanf(data_buffer, "%f %f", &values.x, &values.y) == 2) {
          // interpret as eta/k
          mtl.ext_ior.eta = SpectralDistribution::from_constant(values.x);
          mtl.ext_ior.k = SpectralDistribution::from_constant(values.y);
        } else {
          char buffer[256] = {};
          snprintf(buffer, sizeof(buffer), "%sspectrum/%s.spd", env().data_folder(), data_buffer);
          SpectralDistribution::load_from_file(buffer, mtl.ext_ior.eta, &mtl.ext_ior.k, SpectralDistribution::Class::Reflectance, spectrums());
        }
      }

      if (get_param(material, "int_medium", data_buffer)) {
        auto m = mediums.find(data_buffer);
        if (m == kInvalidIndex) {
          log::warning("Medium %s was not declared, but used in material %s as internal medium", data_buffer, material.name.c_str());
        }
        mtl.int_medium = m;
      }

      if (get_param(material, "ext_medium", data_buffer)) {
        auto m = mediums.find(data_buffer);
        if (m == kInvalidIndex) {
          log::warning("Medium %s was not declared, but used in material %s as external medium\n", data_buffer, material.name.c_str());
        }
        mtl.ext_medium = m;
      }

      if (get_param(material, "spectrum_kd", data_buffer)) {
        char buffer[1024] = {};
        snprintf(buffer, sizeof(buffer), "%s/%s", base_dir, data_buffer);
        SpectralDistribution::load_from_file(buffer, mtl.diffuse.spectrum, nullptr, SpectralDistribution::Class::Reflectance, spectrums());
      }
      if (get_param(material, "spectrum_ks", data_buffer)) {
        char buffer[1024] = {};
        snprintf(buffer, sizeof(buffer), "%s/%s", base_dir, data_buffer);
        SpectralDistribution::load_from_file(buffer, mtl.specular.spectrum, nullptr, SpectralDistribution::Class::Reflectance, spectrums());
      }
      if (get_param(material, "spectrum_kt", data_buffer)) {
        char buffer[1024] = {};
        snprintf(buffer, sizeof(buffer), "%s/%s", base_dir, data_buffer);
        SpectralDistribution::load_from_file(buffer, mtl.transmittance.spectrum, nullptr, SpectralDistribution::Class::Reflectance, spectrums());
      }

      if (get_param(material, "normalmap", data_buffer)) {
        auto params = split_params(data_buffer);
        for (uint64_t i = 0, e = params.size(); i < e; ++i) {
          if ((strcmp(params[i], "image") == 0) && (i + 1 < e)) {
            char buffer[1024] = {};
            snprintf(buffer, sizeof(buffer), "%s/%s", base_dir, params[i + 1]);
            mtl.normal_image_index = add_image(buffer, Image::RepeatU | Image::RepeatV | Image::Linear);
            i += 1;
          }
          if ((strcmp(params[i], "scale") == 0) && (i + 1 < e)) {
            mtl.normal_scale = static_cast<float>(atof(params[i + 1]));
            i += 1;
          }
        }
      }

      if (get_param(material, "thinfilm", data_buffer)) {
        auto params = split_params(data_buffer);

        for (uint64_t i = 0, e = params.size(); i < e; ++i) {
          if ((strcmp(params[i], "image") == 0) && (i + 1 < e)) {
            char buffer[1024] = {};
            snprintf(buffer, sizeof(buffer), "%s/%s", base_dir, params[i + 1]);
            mtl.thinfilm.thinkness_image = add_image(buffer, Image::RepeatU | Image::RepeatV | Image::Linear);
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
              mtl.thinfilm.ior.eta = SpectralDistribution::from_constant(value);
            } else {
              char buffer[256] = {};
              snprintf(buffer, sizeof(buffer), "%sspectrum/%s.spd", env().data_folder(), params[i + 1]);
              SpectralDistribution::load_from_file(buffer, mtl.thinfilm.ior.eta, &mtl.thinfilm.ior.k, SpectralDistribution::Class::Reflectance, spectrums());
            }
          }
        }
      }

      if (get_param(material, "mixture", data_buffer)) {
        auto params = split_params(data_buffer);
        for (uint64_t i = 0, e = params.size(); i < e; ++i) {
          if ((strcmp(params[i], "material1") == 0) && (i + 1 < e)) {
            auto ref = params[i + 1];
            mtl.mixture_0 = material_mapping.count(ref) > 0 ? material_mapping[ref] : kInvalidIndex;
            i += 1;
          }
          if ((strcmp(params[i], "material2") == 0) && (i + 1 < e)) {
            auto ref = params[i + 1];
            mtl.mixture_1 = material_mapping.count(ref) > 0 ? material_mapping[ref] : kInvalidIndex;
            i += 1;
          }
          if ((strcmp(params[i], "factor") == 0) && (i + 1 < e)) {
            float value = 0.0f;
            if (sscanf(params[i + 1], "%f", &value) == 1) {
              mtl.mixture = value;
            }
            i += 1;
          }
          if ((strcmp(params[i], "image") == 0) && (i + 1 < e)) {
            char buffer[1024] = {};
            snprintf(buffer, sizeof(buffer), "%s/%s", base_dir, params[i + 1]);
            mtl.mixture_image_index = add_image(buffer, Image::RepeatU | Image::RepeatV);
            i += 1;
          }
        }
      }
    }
  }
}

}  // namespace etx
