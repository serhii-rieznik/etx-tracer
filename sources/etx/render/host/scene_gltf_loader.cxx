#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/render/shared/base.hxx>
#include <etx/render/shared/math.hxx>
#include <etx/render/shared/scene.hxx>
#include <etx/render/shared/spectrum.hxx>
#include <etx/render/host/scene_gltf_loader.hxx>
#include <etx/render/host/gltf_accessor.hxx>
#include <etx/render/host/scene_representation.hxx>

#include <tinyexr.hxx>

#include <filesystem>
#include <set>
#include <cmath>

namespace etx {
namespace {

static constexpr float kDefaultDielectricEta = 1.5f;

struct GltfLoaderState {
  SceneData& data;
  Camera& active_camera;
  TaskScheduler& scheduler;
};

float4x4 build_gltf_node_transform(const tinygltf::Node& node) {
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

bool load_gltf_camera(const tinygltf::Node& node, const tinygltf::Model& model, const tinygltf::Camera& pcam, const float4x4& transform, GltfLoaderState& state) {
  auto& data = state.data;
  auto& active_camera = state.active_camera;

  auto& entry = data.cameras.emplace_back();

  std::string camera_id = pcam.name.empty() == false ? pcam.name : (node.name.empty() == false ? node.name : "camera");
  uint32_t index = 1;
  std::string camera_name = camera_id;
  while (std::find_if(data.cameras.begin(), data.cameras.end() - 1, [&](const auto& e) {
    return e.id == camera_name;
  }) != data.cameras.end() - 1) {
    char buffer[1024] = {};
    snprintf(buffer, sizeof(buffer), "%s-%04u", camera_id.c_str(), index);
    camera_name = buffer;
    ++index;
  }
  entry.id = camera_name;
  entry.active = (data.cameras.size() == 1);

  auto position = to_float3(transform.col[3]);
  auto forward = normalize(to_float3(transform.col[2]));
  auto direction = -forward;
  auto up = normalize(to_float3(transform.col[1]));

  uint2 film_size = active_camera.film_size;
  if (film_size.x == 0 || film_size.y == 0) {
    film_size = {1280u, 720u};
  }

  if (pcam.type == "perspective") {
    const auto& cam = pcam.perspective;
    entry.cam.cls = Camera::Class::Perspective;

    if (cam.aspectRatio > 0.0) {
      film_size.y = static_cast<uint32_t>(film_size.x / cam.aspectRatio);
    }

    if (cam.znear > 0.0) {
      entry.cam.clip_near = static_cast<float>(cam.znear);
    }
    if (cam.zfar > 0.0) {
      entry.cam.clip_far = static_cast<float>(cam.zfar);
    }

    float fov = static_cast<float>(cam.yfov) * 180.0f / kPi;
    build_camera(entry.cam, position, direction, up, film_size, fov);
  } else if (pcam.type == "orthographic") {
    const auto& cam = pcam.orthographic;
    entry.cam.cls = Camera::Class::Perspective;

    if (cam.znear > 0.0) {
      entry.cam.clip_near = static_cast<float>(cam.znear);
    }
    if (cam.zfar > 0.0 && cam.zfar > cam.znear) {
      entry.cam.clip_far = static_cast<float>(cam.zfar);
    }

    if (cam.xmag > 0.0 && cam.ymag > 0.0) {
      float aspect = static_cast<float>(cam.xmag / cam.ymag);
      film_size.y = static_cast<uint32_t>(film_size.x / aspect);
    }

    float ortho_fov = 2.0f * atanf(static_cast<float>(cam.ymag) * 0.5f) * 180.0f / kPi;
    build_camera(entry.cam, position, direction, up, film_size, ortho_fov);
  } else {
    log::warning("Unknown camera type: %s", pcam.type.c_str());
    data.cameras.pop_back();
    return false;
  }

  if (entry.active) {
    active_camera = entry.cam;
  }

  return true;
}

void load_gltf_mesh(const tinygltf::Node& node, const tinygltf::Model& model, const tinygltf::Mesh& mesh, const float4x4& transform, GltfLoaderState& state) {
  auto& data = state.data;
  auto& triangles = data.triangles;
  auto& vertices = data.vertices;

  if (mesh.primitives.empty())
    return;

  for (size_t primitive_index = 0; primitive_index < mesh.primitives.size(); ++primitive_index) {
    const auto& primitive = mesh.primitives[primitive_index];
    bool has_positions = primitive.attributes.count("POSITION") > 0;
    bool has_normals = primitive.attributes.count("NORMAL") > 0;
    bool has_tex_coords = primitive.attributes.count("TEXCOORD_0") > 0;
    bool has_tangents = primitive.attributes.count("TANGENT") > 0;

    if (has_positions == false)
      continue;

    uint32_t material_index = data.defaults.missing_material;
    if ((primitive.material >= 0) && (primitive.material < static_cast<int32_t>(model.materials.size()))) {
      auto it = data.gltf_material_mapping.find(static_cast<int32_t>(primitive.material));
      if (it != data.gltf_material_mapping.end()) {
        material_index = it->second;
      }
    }

    uint32_t triangle_start = static_cast<uint32_t>(triangles.size());
    float3 mesh_bbox_min = {kMaxFloat, kMaxFloat, kMaxFloat};
    float3 mesh_bbox_max = {-kMaxFloat, -kMaxFloat, -kMaxFloat};

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
      auto tangent_it = primitive.attributes.find("TANGENT");
      if (tangent_it != primitive.attributes.end()) {
        int tangent_accessor_idx = tangent_it->second;
        if ((tangent_accessor_idx >= 0) && (tangent_accessor_idx < static_cast<int>(model.accessors.size()))) {
          tan_accessor = model.accessors.data() + tangent_accessor_idx;
          if ((tan_accessor->bufferView >= 0) && (tan_accessor->bufferView < static_cast<int>(model.bufferViews.size()))) {
            tan_buffer_view = model.bufferViews.data() + tan_accessor->bufferView;
            if ((tan_buffer_view->buffer >= 0) && (tan_buffer_view->buffer < static_cast<int>(model.buffers.size()))) {
              tan_buffer = model.buffers.data() + tan_buffer_view->buffer;
            } else {
              log::warning("GLTF primitive %zu: Invalid buffer index %d for tangent", primitive_index, tan_buffer_view->buffer);
              has_tangents = false;
            }
          } else {
            log::warning("GLTF primitive %zu: Invalid bufferView index %d for tangent", primitive_index, tan_accessor->bufferView);
            has_tangents = false;
          }
        } else {
          log::warning("GLTF primitive %zu: Invalid accessor index %d for tangent", primitive_index, tangent_accessor_idx);
          has_tangents = false;
        }
      } else {
        log::warning("GLTF primitive %zu: TANGENT attribute not found in attributes map", primitive_index);
        has_tangents = false;
      }
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
    uint32_t expected_triangle_count = static_cast<uint32_t>(has_indices ? idx_accessor->count : pos_accessor.count) / 3u;

    uint32_t linear_index = 0;
    for (uint32_t tri_index = 0; tri_index < expected_triangle_count; ++tri_index) {
      uint32_t base_index = static_cast<uint32_t>(vertices.pos.size());
      Triangle& tri = triangles.emplace_back();
      tri.i[0] = base_index + 0;
      tri.i[1] = base_index + 1;
      tri.i[2] = base_index + 2;
      tri.material_index = material_index;

      for (uint32_t j = 0; j < 3; ++j, ++linear_index) {
        auto index = has_indices ? gltf_read_buffer_as_uint(*idx_buffer, *idx_accessor, *idx_buffer_view, 3u * tri_index + j) : linear_index;

        auto p = gltf_read_buffer<float3>(pos_buffer, pos_accessor, pos_buffer_view, index);
        auto pos = transform * float4{p.x, p.y, p.z, 1.0f};

        float3 nrm = {0.0f, 1.0f, 0.0f};
        if (has_normals) {
          auto n = gltf_read_buffer<float3>(*nrm_buffer, *nrm_accessor, *nrm_buffer_view, index);
          auto t = transform * float4{n.x, n.y, n.z, 0.0f};
          float3 transformed_nrm = float3{t.x, t.y, t.z};
          float nrm_length_sq = dot(transformed_nrm, transformed_nrm);
          if (nrm_length_sq > kEpsilon) {
            nrm = normalize(transformed_nrm);
          } else {
            nrm = {0.0f, 1.0f, 0.0f};
            static uint32_t zero_normal_count = 0;
            if (++zero_normal_count <= 10) {
              log::warning("GLTF: Zero-length normal detected at vertex %u, using default", static_cast<uint32_t>(vertices.pos.size()));
            }
          }
        }

        float2 tex = {};
        if (has_tex_coords) {
          tex = gltf_read_buffer<float2>(*tex_buffer, *tex_accessor, *tex_buffer_view, index);
        }

        float3 tan = {1.0f, 0.0f, 0.0f};
        float3 btn = {0.0f, 0.0f, 1.0f};
        bool should_compute_tangents = true;

        if (has_tangents) {
          auto gltf_tangent = gltf_read_buffer<float4>(*tan_buffer, *tan_accessor, *tan_buffer_view, index);
          auto tt = transform * float4{gltf_tangent.x, gltf_tangent.y, gltf_tangent.z, 0.0f};
          float3 transformed_tan = float3{tt.x, tt.y, tt.z};
          float tan_length_sq = dot(transformed_tan, transformed_tan);

          if (tan_length_sq > kEpsilon) {
            tan = normalize(transformed_tan);
            tan = normalize(tan - dot(tan, nrm) * nrm);
            float tan_ortho_length_sq = dot(tan, tan);
            if (tan_ortho_length_sq > kEpsilon) {
              tan = normalize(tan);
              float3 computed_btn = cross(tan, nrm) * (-gltf_tangent.w);
              float btn_length_sq = dot(computed_btn, computed_btn);
              if (btn_length_sq > kEpsilon) {
                btn = normalize(computed_btn);
                should_compute_tangents = false;
              }
            }
          }
        }

        if (should_compute_tangents) {
          tan = {0.0f, 0.0f, 0.0f};
          btn = {0.0f, 0.0f, 0.0f};
        }

        vertices.pos.emplace_back(float3{pos.x, pos.y, pos.z});
        vertices.nrm.emplace_back(float3{nrm.x, nrm.y, nrm.z});
        vertices.tan.emplace_back(float3{tan.x, tan.y, tan.z});
        vertices.btn.emplace_back(float3{btn.x, btn.y, btn.z});
        vertices.tex.emplace_back(float2{tex.x, tex.y});
      }

      if (validate_triangle(tri, vertices.pos) == false) {
        triangles.pop_back();
        vertices.pos.pop_back();
        vertices.pos.pop_back();
        vertices.pos.pop_back();
        vertices.nrm.pop_back();
        vertices.nrm.pop_back();
        vertices.nrm.pop_back();
        vertices.tan.pop_back();
        vertices.tan.pop_back();
        vertices.tan.pop_back();
        vertices.btn.pop_back();
        vertices.btn.pop_back();
        vertices.btn.pop_back();
        vertices.tex.pop_back();
        vertices.tex.pop_back();
        vertices.tex.pop_back();
        continue;
      }

      mesh_bbox_min = min(mesh_bbox_min, vertices.pos[tri.i[0]]);
      mesh_bbox_min = min(mesh_bbox_min, vertices.pos[tri.i[1]]);
      mesh_bbox_min = min(mesh_bbox_min, vertices.pos[tri.i[2]]);
      mesh_bbox_max = max(mesh_bbox_max, vertices.pos[tri.i[0]]);
      mesh_bbox_max = max(mesh_bbox_max, vertices.pos[tri.i[1]]);
      mesh_bbox_max = max(mesh_bbox_max, vertices.pos[tri.i[2]]);

      if (has_normals == false) {
        vertices.nrm[vertices.nrm.size() - 1u] = tri.geo_n;
        vertices.nrm[vertices.nrm.size() - 2u] = tri.geo_n;
        vertices.nrm[vertices.nrm.size() - 3u] = tri.geo_n;
      }
    }

    uint32_t triangle_end = static_cast<uint32_t>(triangles.size());
    uint32_t triangle_count = triangle_end - triangle_start;
    if (triangle_count > 0) {
      std::string mesh_name;
      if (node.name.empty() == false) {
        mesh_name = node.name;
        if (mesh.primitives.size() > 1) {
          mesh_name += "_" + std::to_string(primitive_index);
        }
      } else if (mesh.name.empty() == false) {
        mesh_name = mesh.name + "_" + std::to_string(primitive_index);
      } else {
        mesh_name = "mesh_" + std::to_string(primitive_index);
      }
      data.add_mesh(mesh_name.c_str(), triangle_start, triangle_count, mesh_bbox_min, mesh_bbox_max);
    }
  }
}

bool load_gltf_node(const tinygltf::Model& model, const tinygltf::Node& node, const float4x4& parent_transform, GltfLoaderState& state) {
  auto current_transform = parent_transform * build_gltf_node_transform(node);

  bool camera_found = false;

  if ((node.mesh >= 0) && (node.mesh < model.meshes.size())) {
    load_gltf_mesh(node, model, model.meshes.at(node.mesh), current_transform, state);
  }

  if ((node.camera >= 0) && (node.camera < model.cameras.size())) {
    camera_found = load_gltf_camera(node, model, model.cameras.at(node.camera), current_transform, state);
  }

  for (const auto& child : node.children) {
    if (load_gltf_node(model, model.nodes[child], current_transform, state)) {
      camera_found = true;
    }
  }

  return camera_found;
}

void load_gltf_materials(const tinygltf::Model& model, GltfLoaderState& state) {
  auto& data = state.data;

  for (int32_t gltf_material_index = 0; gltf_material_index < static_cast<int32_t>(model.materials.size()); ++gltf_material_index) {
    auto& material = model.materials[gltf_material_index];
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
    data.gltf_material_mapping[gltf_material_index] = material_index;

    auto& mtl = data.materials[material_index];

    bool is_unlit = false;
    for (const auto& ext : material.extensions) {
      if (ext.first == "KHR_materials_unlit") {
        is_unlit = true;
        break;
      }
    }

    mtl.cls = is_unlit ? Material::Class::Diffuse : Material::Class::Principled;
    mtl.ext_ior.cls = SpectralDistribution::Class::Dielectric;
    mtl.ext_ior.eta_index = data.add_spectrum(SpectralDistribution::constant(1.0f));
    mtl.ext_ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
    mtl.int_ior.cls = SpectralDistribution::Class::Conductor;
    mtl.int_ior.eta_index = data.add_spectrum(SpectralDistribution::constant(kDefaultDielectricEta));
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

    if (is_unlit == false) {
      bool has_metallic_roughness_texture = (pbr.metallicRoughnessTexture.index != -1) && (data.gltf_image_mapping.count(pbr.metallicRoughnessTexture.index) > 0);
      if (has_metallic_roughness_texture) {
        auto image_index = data.gltf_image_mapping.at(pbr.metallicRoughnessTexture.index);
        mtl.roughness.image_index = image_index;
        mtl.roughness.channel = 1u;
        mtl.roughness.value = {1.0f, 1.0f};
        mtl.metalness.image_index = image_index;
        mtl.metalness.channel = 2u;
        mtl.metalness.value = {1.0f, 1.0f};
      } else {
        float roughness = float(pbr.roughnessFactor);
        float metalness = float(pbr.metallicFactor);
        mtl.roughness.value = {roughness, roughness};
        mtl.metalness.value = {metalness, metalness};
        mtl.roughness.image_index = kInvalidIndex;
        mtl.metalness.image_index = kInvalidIndex;
      }
    }

    if ((pbr.baseColorTexture.index != -1) && (data.gltf_image_mapping.count(pbr.baseColorTexture.index) > 0)) {
      mtl.scattering.image_index = data.gltf_image_mapping.at(pbr.baseColorTexture.index);
      mtl.reflectance.image_index = mtl.scattering.image_index;
    }

    if ((material.normalTexture.index != -1) && (data.gltf_image_mapping.count(material.normalTexture.index) > 0)) {
      mtl.normal_image_index = data.gltf_image_mapping.at(material.normalTexture.index);
      mtl.normal_scale = 1.0f;
      data.add_image_options(mtl.normal_image_index, Image::SkipSRGBConversion);
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
          bool has_transmission_texture = false;
          if (ext.second.IsObject() && ext.second.Has("transmissionTexture")) {
            const auto& tex_obj = ext.second.Get("transmissionTexture");
            if (tex_obj.IsObject() && tex_obj.Has("index")) {
              const auto& tex_index = tex_obj.Get("index");
              if (tex_index.IsNumber()) {
                int32_t tex_idx = tex_index.GetNumberAsInt();
                if ((tex_idx >= 0) && (data.gltf_image_mapping.count(tex_idx) > 0)) {
                  mtl.transmission.image_index = data.gltf_image_mapping.at(tex_idx);
                  mtl.transmission.channel = 0u;
                  mtl.transmission.value = {1.0f, 1.0f, 1.0f, 1.0f};
                  has_transmission_texture = true;
                }
              }
            }
          }
          if ((has_transmission_texture == false) && ext.second.IsObject() && ext.second.Has("transmissionFactor")) {
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

  data.images.load_images(state.scheduler);
}

}  // namespace

uint32_t load_from_gltf_file(const char* file_name, bool binary, SceneData& data, TaskScheduler& scheduler, Camera& active_camera) {
  GltfLoaderState state{data, active_camera, scheduler};

  tinygltf::TinyGLTF loader;
  tinygltf::Model model;
  std::string errors;
  std::string warnings;

  bool load_result = false;

  auto& gltf_image_mapping = data.gltf_image_mapping;
  auto& gltf_material_mapping = data.gltf_material_mapping;
  gltf_image_mapping.clear();
  gltf_material_mapping.clear();
  auto image_loader = [](tinygltf::Image* image, const int image_index, std::string* errors, std::string* warnings, int width, int height, const unsigned char* data_ptr,
                        int data_size, void* user_pointer) -> bool {
    (void)errors;
    (void)warnings;
    auto self = reinterpret_cast<GltfLoaderState*>(user_pointer);

    if (((width == 0) || (height == 0)) && (data_ptr != nullptr)) {
      uint32_t hash = etx_hash32(data_ptr, data_size);
      char file_name[64] = {};
      snprintf(file_name, sizeof(file_name), "img-%x.png", hash);

      char buffer[2048] = {};
      env().file_in_tmp(file_name, buffer, sizeof(buffer));
      if (auto fout = fopen(buffer, "wb")) {
        if (fwrite(data_ptr, 1, data_size, fout) == data_size) {
          self->data.gltf_image_mapping[image_index] = self->data.add_image(buffer, Image::RepeatU | Image::RepeatV, {}, {1.0f, 1.0f});
        }
        fclose(fout);
      }
    }

    return true;
  };

  loader.SetImageLoader(image_loader, &state);

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
    return SceneLoadFailed;
  }

  constexpr auto kDataBufferSize = 2048llu;
  static char base_dir[kDataBufferSize] = {};
  get_base_directory(file_name, base_dir, sizeof(base_dir));

  for (size_t image_index = 0; image_index < model.images.size(); ++image_index) {
    if (gltf_image_mapping.count(static_cast<int>(image_index)) > 0) {
      continue;
    }

    const auto& gltf_image = model.images[image_index];
    if (gltf_image.uri.empty()) {
      continue;
    }

    if (gltf_image.uri.compare(0, 5, "data:") == 0) {
      continue;
    }

    std::filesystem::path image_path(gltf_image.uri);
    if (image_path.is_absolute() == false) {
      std::filesystem::path base_path(base_dir);
      image_path = base_path / gltf_image.uri;
    }

    std::string image_path_str = image_path.lexically_normal().string();
    gltf_image_mapping[static_cast<int>(image_index)] = data.add_image(image_path_str.c_str(), Image::RepeatU | Image::RepeatV, {}, {1.0f, 1.0f});
  }

  load_gltf_materials(model, state);

  const std::string ext_name = "EXT_lights_image_based";

  if (model.extensionsUsed.end() != std::find(model.extensionsUsed.begin(), model.extensionsUsed.end(), ext_name)) {
    log::info("GLTF: extension %s is used by model", ext_name.c_str());
  }
  if (model.extensionsRequired.end() != std::find(model.extensionsRequired.begin(), model.extensionsRequired.end(), ext_name)) {
    log::info("GLTF: extension %s is required by model", ext_name.c_str());
  }

  auto& gltf_image_mapping_ref = state.data.gltf_image_mapping;

  auto parse_sh_data = [](const tinygltf::Value& value, float3 sh_coeffs[9]) -> bool {
    const auto* array = value.IsArray() ? &value : nullptr;
    if (array == nullptr)
      return false;

    if (array->ArrayLen() != 9)
      return false;

    for (size_t i = 0; i < 9; ++i) {
      auto c = array->Get(static_cast<int>(i));
      if (c.IsArray() == false)
        return false;

      if (c.ArrayLen() < 3)
        return false;

      sh_coeffs[i] = {float(c.Get(0).GetNumberAsDouble()), float(c.Get(1).GetNumberAsDouble()), float(c.Get(2).GetNumberAsDouble())};
    }

    return true;
  };

  auto quaternion_to_yaw_rotation_offset = [](const float4& q) -> float {
    const float yaw = std::atan2(2.0f * (q.w * q.y + q.x * q.z), 1.0f - 2.0f * (q.y * q.y + q.z * q.z));
    return yaw;
  };

  auto parse_spherical_harmonics = [&](const tinygltf::Value& value, float3 sh_coeffs[9]) -> bool {
    if (!value.IsObject() || !value.Has("coefficients")) {
      return false;
    }

    const auto& coeffs_value = value.Get("coefficients");
    return parse_sh_data(coeffs_value, sh_coeffs);
  };

  for (const auto& ext : model.extensions) {
    if (ext.first == "EXT_lights_image_based") {
      const auto& lights = ext.second;
      if (lights.IsObject() == false)
        continue;

      if (lights.Has("reflectionProbes")) {
        const auto& probes = lights.Get("reflectionProbes");
        if (probes.IsArray()) {
          for (size_t light_idx = 0; light_idx < probes.ArrayLen(); ++light_idx) {
            float3 sh_coeffs[9] = {};
            const auto& light_obj = probes.Get(static_cast<int>(light_idx));
            if (parse_spherical_harmonics(light_obj, sh_coeffs) == false)
              continue;

            float intensity = 1.0f;
            if (light_obj.Has("intensity")) {
              const auto& intensity_val = light_obj.Get("intensity");
              if (intensity_val.IsNumber()) {
                intensity = float(intensity_val.GetNumberAsDouble());
              }
            }

            for (size_t i = 0; i < 9; ++i) {
              sh_coeffs[i] = sh_coeffs[i] * intensity;
            }

            float4 rotation_quat = {0.0f, 0.0f, 0.0f, 1.0f};
            if (light_obj.Has("rotation")) {
              const auto& rot_array = light_obj.Get("rotation");
              if (rot_array.IsArray() && rot_array.ArrayLen() >= 4) {
                rotation_quat = {float(rot_array.Get(0).GetNumberAsDouble()), float(rot_array.Get(1).GetNumberAsDouble()), float(rot_array.Get(2).GetNumberAsDouble()),
                  float(rot_array.Get(3).GetNumberAsDouble())};
              }
            }

            float rotation_offset = quaternion_to_yaw_rotation_offset(rotation_quat);

            constexpr uint2 env_image_dimensions = {512u, 256u};
            uint32_t image_options = Image::BuildSamplingTable | Image::RepeatU;
            uint32_t image_index = data.images.add_from_spherical_harmonics(scheduler, sh_coeffs, env_image_dimensions, image_options, {rotation_offset, 0.0f}, {1.0f, 1.0f});

            auto& e = data.emitter_profiles.emplace_back(EmitterProfile::Class::Environment);

            e.emission.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
            e.emission.image_index = image_index;
            e.medium_index = kInvalidIndex;

            log::info("Created environment emitter from EXT_lights_image_based extension (light index %zu)", light_idx);

            if (light_obj.Has("specularImages")) {
              const auto& spec_images_value = light_obj.Get("specularImages");
              if (spec_images_value.IsArray() && spec_images_value.ArrayLen() > 0) {
                const auto& mip0_faces = spec_images_value.Get(0);
                if (mip0_faces.IsArray() && mip0_faces.ArrayLen() >= 6) {
                  int32_t cube_face_indices[6] = {};
                  bool all_faces_valid = true;
                  for (size_t face_idx = 0; face_idx < 6 && face_idx < mip0_faces.ArrayLen(); ++face_idx) {
                    const auto& face_ref = mip0_faces.Get(int(face_idx));
                    if (face_ref.IsInt() || face_ref.IsNumber()) {
                      cube_face_indices[face_idx] = face_ref.GetNumberAsInt();
                      if ((cube_face_indices[face_idx] < 0) || (gltf_image_mapping_ref.count(cube_face_indices[face_idx]) == 0)) {
                        all_faces_valid = false;
                        break;
                      }
                    } else {
                      all_faces_valid = false;
                      break;
                    }
                  }

                  if (all_faces_valid) {
                    uint32_t cube_size = 256u;
                    if (light_obj.Has("specularImageSize")) {
                      const auto& size_val = light_obj.Get("specularImageSize");
                      if (size_val.IsInt() || size_val.IsNumber()) {
                        cube_size = static_cast<uint32_t>(size_val.GetNumberAsInt());
                      }
                    }

                    uint32_t cube_face_images[6] = {};
                    for (size_t i = 0; i < 6; ++i) {
                      cube_face_images[i] = gltf_image_mapping_ref.at(cube_face_indices[i]);
                    }
                    constexpr uint2 equirect_dimensions = {1024u, 512u};
                    uint32_t specular_image_options = Image::BuildSamplingTable | Image::RepeatU;
                    uint32_t specular_image_index =
                      data.images.add_from_cubemap(state.scheduler, cube_face_images, equirect_dimensions, specular_image_options, {rotation_offset, 0.0f}, {1.0f, 1.0f});

                    const auto& equirect_image = data.images.get(specular_image_index);
                    char exr_filename[512] = {};
                    char exr_name[64] = {};
                    snprintf(exr_name, sizeof(exr_name), "specular_env_%zu.exr", light_idx);
                    env().file_in_tmp(exr_name, exr_filename, sizeof(exr_filename));
                    const char* error = nullptr;
                    if (SaveEXR(reinterpret_cast<const float*>(equirect_image.pixels.f32.a), equirect_dimensions.x, equirect_dimensions.y, 4, false, exr_filename, &error) !=
                        TINYEXR_SUCCESS) {
                      log::warning("Failed to save specular environment map to %s: %s", exr_filename, error ? error : "unknown error");
                    } else {
                      log::info("Saved specular environment map to %s", exr_filename);
                    }

                    auto& spec_e = data.emitter_profiles.emplace_back(EmitterProfile::Class::Environment);
                    spec_e.emission.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
                    spec_e.emission.image_index = specular_image_index;
                    spec_e.medium_index = kInvalidIndex;

                    log::info("Created specular environment emitter from EXT_lights_image_based extension (light index %zu)", light_idx);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  std::set<size_t> referenced_light_indices;
  for (const auto& scene_ref : model.scenes) {
    auto scene_ext_it = scene_ref.extensions.find("EXT_lights_image_based");
    if (scene_ext_it != scene_ref.extensions.end()) {
      const auto& scene_ext_value = scene_ext_it->second;
      if (scene_ext_value.IsObject() && scene_ext_value.Has("light")) {
        const auto& light_ref = scene_ext_value.Get("light");
        if (light_ref.IsInt() || light_ref.IsNumber()) {
          int32_t light_index = light_ref.GetNumberAsInt();
          if (light_index >= 0) {
            referenced_light_indices.insert(static_cast<size_t>(light_index));
          }
        }
      }
    }
  }

  bool camera_loaded = false;
  for (const auto& scene_ref : model.scenes) {
    for (int32_t node_index : scene_ref.nodes) {
      if ((node_index < 0) || (node_index >= model.nodes.size()))
        continue;

      const float4x4 identity = build_gltf_node_transform({});
      const auto& node = model.nodes[node_index];
      if (load_gltf_node(model, node, identity, state)) {
        camera_loaded = true;
      }
    }
  }

  (void)referenced_light_indices;
  (void)scheduler;
  return SceneLoadSucceeded | (camera_loaded ? SceneLoadCameraInfo : 0u);
}

}  // namespace etx
