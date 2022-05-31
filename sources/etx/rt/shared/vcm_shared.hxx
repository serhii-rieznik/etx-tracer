#pragma once

#include <etx/render/shared/scene.hxx>

namespace etx {

struct VCMOptions {
  uint32_t options : 8;
  uint32_t radius_decay : 24;

  enum : uint32_t {
    ConnectToCamera = 1u << 0u,
    DirectHit = 1u << 1u,
    ConnectToLight = 1u << 2u,
    ConnectVertices = 1u << 3u,
    MergeVertices = 1u << 4u,
    EnableMis = 1u << 5u,
    OnlyConnection = 1u << 6u,

    Default = ConnectToCamera | DirectHit | ConnectToLight | ConnectVertices | MergeVertices | EnableMis,
    BDPT = ConnectToCamera | DirectHit | ConnectToLight | ConnectVertices | OnlyConnection | EnableMis,
  };

  void set_option(uint32_t option, bool enabled) {
    options = (options & (~option)) | (enabled ? option : 0u);
  }

#if !ETX_NVCC_COMPILER
  VCMOptions() {
    options = Default;
    radius_decay = 256u;
  }
#endif

  ETX_GPU_CODE bool connect_to_camera() const {
    return options & ConnectToCamera;
  }
  ETX_GPU_CODE bool direct_hit() const {
    return options & DirectHit;
  }
  ETX_GPU_CODE bool connect_to_light() const {
    return options & ConnectToLight;
  }
  ETX_GPU_CODE bool connect_vertices() const {
    return options & ConnectVertices;
  }
  ETX_GPU_CODE bool merge_vertices() const {
    return (options & MergeVertices) && ((options & OnlyConnection) == 0);
  }
  ETX_GPU_CODE bool enable_mis() const {
    return options & EnableMis;
  }
  ETX_GPU_CODE bool only_connection() const {
    return options & OnlyConnection;
  }
};

enum class VCMState : uint32_t {
  Stopped,
  GatheringLightVertices,
  GatheringCameraVertices,
};

struct ETX_ALIGNED VCMIteration {
  float current_radius = 0.0f;
  float vm_weight = {};
  float vc_weight = {};
  float vm_normalization = {};
};

struct ETX_ALIGNED GPUCameraLaunchParams {
  ArrayView<float4> camera_image ETX_EMPTY_INIT;
};

struct ETX_ALIGNED VCMPathState {
  enum : uint32_t {
    Active = 0u,
    ConnectingVertices = 1u,
    Completed = 2u,
  };

  SpectralResponse throughput = {};
  Ray ray = {};
  float d_vcm = 0.0f;
  float d_vc = 0.0f;
  float d_vm = 0.0f;
  uint32_t state = Active;
};

struct ETX_ALIGNED VCMLightVertex {
  VCMLightVertex() = default;

  ETX_GPU_CODE VCMLightVertex(const VCMPathState& s, const float3& p, const float3& b, uint32_t tri, uint32_t len, uint32_t index)
    : throughput(s.throughput)
    , w_i(s.ray.d)
    , d_vcm(s.d_vcm)
    , bc(b)
    , d_vc(s.d_vc)
    , pos(p)
    , d_vm(s.d_vm)
    , triangle_index(tri)
    , path_length(len)
    , path_index(index) {
  }

  SpectralResponse throughput = {};

  float3 w_i = {};
  float d_vcm = 0.0f;

  float3 bc = {};
  float d_vc = 0.0f;

  float3 pos = {};
  float d_vm = 0.0f;
  uint32_t triangle_index = kInvalidIndex;
  uint32_t path_length = 0;
  uint32_t path_index = 0;

  ETX_GPU_CODE Vertex vertex(const Scene& s) const {
    return lerp_vertex(s.vertices, s.triangles[triangle_index], bc);
  }

  ETX_GPU_CODE float3 position(const Scene& s) const {
    return pos;  // lerp_pos(s.vertices, s.triangles[triangle_index], bc);
  }

  ETX_GPU_CODE float3 normal(const Scene& s) const {
    return lerp_normal(s.vertices, s.triangles[triangle_index], bc);
  }
};

ETX_GPU_CODE bool vcm_next_ray(const Scene& scene, SpectralQuery spect, const PathSource path_source, const Intersection& i, const uint64_t path_length, uint32_t rr_start,
  Sampler& smp, VCMPathState& state, uint32_t& state_medium, float& state_eta, const VCMIteration& it) {
  const auto& tri = scene.triangles[i.triangle_index];
  const auto& mat = scene.materials[tri.material_index];
  auto bsdf_data = BSDFData{spect, state_medium, path_source, i, state.ray.d, {}};
  auto bsdf_sample = bsdf::sample(bsdf_data, mat, scene, smp);
  bsdf_data.w_o = bsdf_sample.w_o;
  if (bsdf_sample.valid() == false) {
    return false;
  }

  state.throughput *= bsdf_sample.weight;
  ETX_VALIDATE(state.throughput);

  if (bsdf_sample.properties & BSDFSample::MediumChanged) {
    state_medium = bsdf_sample.medium_index;
  }

  if (path_source == PathSource::Light) {
    state.throughput *= fix_shading_normal(tri.geo_n, bsdf_data.nrm, bsdf_data.w_i, bsdf_data.w_o);
  }

  if (state.throughput.is_zero()) {
    return false;
  }

  if ((path_length >= rr_start) && (apply_rr(state_eta, smp.next(), state.throughput) == false)) {
    return false;
  }

  float cos_theta_bsdf = fabsf(dot(i.nrm, bsdf_sample.w_o));

  if (bsdf_sample.is_delta()) {
    state.d_vc *= cos_theta_bsdf;
    ETX_VALIDATE(state.d_vc);

    state.d_vm *= cos_theta_bsdf;
    ETX_VALIDATE(state.d_vm);

    state.d_vcm = 0.0f;
  } else {
    auto rev_sample_pdf = bsdf::pdf(bsdf_data.swap_directions(), mat, scene, smp);
    ETX_VALIDATE(rev_sample_pdf);

    state.d_vc = (cos_theta_bsdf / bsdf_sample.pdf) * (state.d_vc * rev_sample_pdf + state.d_vcm + it.vm_weight);
    ETX_VALIDATE(state.d_vc);

    state.d_vm = (cos_theta_bsdf / bsdf_sample.pdf) * (state.d_vm * rev_sample_pdf + state.d_vcm * it.vc_weight + 1.0f);
    ETX_VALIDATE(state.d_vm);

    state.d_vcm = 1.0f / bsdf_sample.pdf;
    ETX_VALIDATE(state.d_vcm);
  }
  state.ray.d = bsdf_sample.w_o;
  state.ray.o = shading_pos(scene.vertices, tri, i.barycentric, bsdf_sample.w_o);
  state_eta *= bsdf_sample.eta;
  return true;
}

ETX_GPU_CODE SpectralResponse vcm_get_radiance(const Scene& scene, SpectralQuery spect, const Emitter& emitter, const Intersection& intersection, const VCMPathState& state,
  const uint64_t path_length, bool mis) {
  float pdf_emitter_area = 0.0f;
  float pdf_emitter_dir = 0.0f;
  float pdf_emitter_dir_out = 0.0f;

  SpectralResponse radiance = {};

  if (emitter.is_local()) {
    radiance = emitter_get_radiance(emitter, spect, intersection.tex, state.ray.o, intersection.pos,  //
      pdf_emitter_area, pdf_emitter_dir, pdf_emitter_dir_out, scene, (path_length == 1));             //
  } else {
    radiance = emitter_get_radiance(emitter, spect, state.ray.d,       //
      pdf_emitter_area, pdf_emitter_dir, pdf_emitter_dir_out, scene);  //
  }

  if (pdf_emitter_dir <= kEpsilon) {
    return {spect.wavelength, 0.0f};
  }

  float emitter_sample_pdf = emitter_discrete_pdf(emitter, scene.emitters_distribution);
  pdf_emitter_area *= emitter_sample_pdf;
  pdf_emitter_dir_out *= emitter_sample_pdf;
  float w_camera = state.d_vcm * pdf_emitter_area + state.d_vc * pdf_emitter_dir_out;
  float weight = (mis && (path_length > 1)) ? (1.0f / (1.0f + w_camera)) : 1.0f;
  return weight * (state.throughput * radiance);
}

ETX_GPU_CODE bool vcm_connect_to_light_vertex(const Scene& scene, const SpectralQuery& spect, const VCMPathState& state, const Intersection& intersection,
  const VCMLightVertex& light_vertex, float vm_weight, uint32_t state_medium, float3& target_position, SpectralResponse& value, Sampler& smp) {
  auto light_v = light_vertex.vertex(scene);
  target_position = light_v.pos;

  auto w_o = light_v.pos - intersection.pos;
  float distance_squared = dot(w_o, w_o);

  if (distance_squared == 0.0f) {
    return false;
  }

  w_o /= sqrtf(distance_squared);

  float w_dot_c = dot(intersection.nrm, w_o);
  if (w_dot_c < 0.0f) {
    return false;
  }

  float w_dot_l = -dot(light_v.nrm, w_o);
  if (w_dot_l < 0.0f) {
    return false;
  }

  const auto& tri = scene.triangles[intersection.triangle_index];
  const auto& mat = scene.materials[tri.material_index];
  auto camera_data = BSDFData{spect, state_medium, PathSource::Camera, intersection, intersection.w_i, w_o};
  auto camera_bsdf = bsdf::evaluate(camera_data, mat, scene, smp);
  if (camera_bsdf.valid() == false) {
    return false;
  }

  auto camera_area_pdf = camera_bsdf.pdf * w_dot_l / distance_squared;
  ETX_VALIDATE(camera_area_pdf);

  auto camera_rev_pdf = bsdf::pdf(camera_data.swap_directions(), mat, scene, smp);
  ETX_VALIDATE(camera_rev_pdf);
  if (camera_rev_pdf <= kEpsilon) {
    return false;
  }

  const auto& light_tri = scene.triangles[light_vertex.triangle_index];
  const auto& light_mat = scene.materials[light_tri.material_index];
  auto light_data = BSDFData{spect, state_medium, PathSource::Light, light_v, light_vertex.w_i, -w_o};
  auto light_bsdf = bsdf::evaluate(light_data, light_mat, scene, smp);
  if (light_bsdf.valid() == false) {
    return false;
  }

  auto light_area_pdf = light_bsdf.pdf * w_dot_c / distance_squared;
  ETX_VALIDATE(light_area_pdf);

  auto light_rev_pdf = bsdf::pdf(light_data.swap_directions(), light_mat, scene, smp);
  ETX_VALIDATE(light_rev_pdf);
  if (light_rev_pdf <= kEpsilon) {
    return false;
  }

  float w_light = camera_area_pdf * (vm_weight + light_vertex.d_vcm + light_vertex.d_vc * light_rev_pdf);
  ETX_VALIDATE(w_light);

  float w_camera = light_area_pdf * (vm_weight + state.d_vcm + state.d_vc * camera_rev_pdf);
  ETX_VALIDATE(w_camera);

  float weight = 1.0f / (1.0f + w_light + w_camera);
  ETX_VALIDATE(weight);

  light_bsdf.bsdf *= fix_shading_normal(light_tri.geo_n, light_data.nrm, light_data.w_i, light_data.w_o);
  value = (camera_bsdf.bsdf * state.throughput) * (light_bsdf.bsdf * light_vertex.throughput) * (weight / distance_squared);
  return true;
}

struct ETX_ALIGNED VCMSpatialGridData {
  ArrayView<uint32_t> indices ETX_EMPTY_INIT;
  ArrayView<uint32_t> cell_ends ETX_EMPTY_INIT;
  BoundingBox bounding_box ETX_EMPTY_INIT;
  uint32_t hash_table_mask ETX_EMPTY_INIT;
  float cell_size ETX_EMPTY_INIT;
  float radius_squared ETX_EMPTY_INIT;
  float pad ETX_EMPTY_INIT;

  ETX_GPU_CODE uint32_t cell_index(int32_t x, int32_t y, int32_t z) const {
    return ((x * 73856093u) ^ (y * 19349663) ^ (z * 83492791)) & hash_table_mask;
  }

  ETX_GPU_CODE uint32_t position_to_index(const float3& pos) const {
    auto m = floor((pos - bounding_box.p_min) / cell_size);
    return cell_index(static_cast<int32_t>(m.x), static_cast<int32_t>(m.y), static_cast<int32_t>(m.z));
  }

  ETX_GPU_CODE float3 gather(const Scene& scene, SpectralQuery spect, const VCMPathState& state, const VCMLightVertex* samples, const Intersection& intersection,
    uint32_t state_medium, uint32_t path_length, uint32_t max_path_length, float vc_weight, Sampler& smp) const {
    if ((indices.count == 0) || (bounding_box.contains(intersection.pos) == false)) {
      return {};
    }

    float3 m = (intersection.pos - bounding_box.p_min) / cell_size;
    float3 mf = floor(m);
    float3 md = m - mf;

    int32_t acx = static_cast<int32_t>(mf.x);
    int32_t acy = static_cast<int32_t>(mf.y);
    int32_t acz = static_cast<int32_t>(mf.z);

    int32_t bcx = acx + ((md.x < 0.5f) ? -1 : +1);
    int32_t bcy = acy + ((md.y < 0.5f) ? -1 : +1);
    int32_t bcz = acz + ((md.z < 0.5f) ? -1 : +1);

    uint32_t cell_indices[8] = {
      cell_index(acx, acy, acz),
      cell_index(bcx, acy, acz),
      cell_index(acx, bcy, acz),
      cell_index(bcx, bcy, acz),
      cell_index(acx, acy, bcz),
      cell_index(bcx, acy, bcz),
      cell_index(acx, bcy, bcz),
      cell_index(bcx, bcy, bcz),
    };

    float3 merged = {};
    for (uint32_t i = 0; i < 8; ++i) {
      uint32_t index = cell_indices[i];
      uint32_t range_begin = (index == 0) ? 0 : cell_ends[index - 1llu];

      for (uint32_t j = range_begin, range_end = cell_ends[index]; j < range_end; ++j) {
        auto& light_vertex = samples[indices[j]];

        auto d = light_vertex.position(scene) - intersection.pos;
        float distance_squared = dot(d, d);
        if ((distance_squared > radius_squared) || (light_vertex.path_length + path_length > max_path_length)) {
          continue;
        }

        if (dot(intersection.nrm, light_vertex.normal(scene)) <= kEpsilon) {
          continue;
        }

        const auto& tri = scene.triangles[intersection.triangle_index];
        const auto& mat = scene.materials[tri.material_index];
        auto camera_data = BSDFData{spect, state_medium, PathSource::Camera, intersection, state.ray.d, -light_vertex.w_i};
        auto camera_bsdf = bsdf::evaluate(camera_data, mat, scene, smp);
        if (camera_bsdf.valid() == false) {
          continue;
        }

        auto camera_rev_pdf = bsdf::pdf(camera_data.swap_directions(), mat, scene, smp);

        float w_light = light_vertex.d_vcm * vc_weight + light_vertex.d_vm * camera_bsdf.pdf;
        float w_camera = state.d_vcm * vc_weight + state.d_vm * camera_rev_pdf;
        float weight = 1.0f / (1.0f + w_light + w_camera);

        if constexpr (spectrum::kSpectralRendering) {
          auto c_value = ((camera_bsdf.func * state.throughput / spectrum::sample_pdf()).to_xyz());
          c_value = max(c_value, float3{0.0f, 0.0f, 0.0f});
          ETX_VALIDATE(c_value);
          auto l_value = ((light_vertex.throughput / spectrum::sample_pdf()).to_xyz());
          l_value = max(l_value, float3{0.0f, 0.0f, 0.0f});
          ETX_VALIDATE(l_value);
          merged += (c_value * l_value) * weight;
          ETX_VALIDATE(merged);
        } else {
          // mul as RGB
          auto value = (light_vertex.throughput * camera_bsdf.func * state.throughput) * (weight / sqr(spectrum::sample_pdf()));
          ETX_VALIDATE(value);
          merged += value.to_xyz();
          ETX_VALIDATE(merged);
        }
      }
    }

    return merged;
  }
};

struct VCMSpatialGrid {
  VCMSpatialGridData data;

  void construct(const Scene& scene, const VCMLightVertex* samples, uint64_t sample_count, float radius);

 private:
  std::vector<uint32_t> _indices;
  std::vector<uint32_t> _cell_ends;
};

}  // namespace etx
