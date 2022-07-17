#pragma once

#include <etx/render/shared/scene.hxx>

namespace etx {

struct Options;

struct ETX_ALIGNED VCMOptions {
  uint32_t options ETX_EMPTY_INIT;
  uint32_t max_samples ETX_EMPTY_INIT;
  uint32_t max_depth ETX_EMPTY_INIT;
  uint32_t rr_start ETX_EMPTY_INIT;
  uint32_t radius_decay ETX_EMPTY_INIT;
  float initial_radius ETX_EMPTY_INIT;

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

  static VCMOptions default_values();

  void store(Options&);
  void load(const Options&);
};

enum class VCMState : uint32_t {
  Stopped,
  GatheringLightVertices,
  GatheringCameraVertices,
};

struct ETX_ALIGNED VCMIteration {
  uint32_t iteration ETX_EMPTY_INIT;
  uint32_t active_paths ETX_EMPTY_INIT;
  uint32_t light_vertices ETX_EMPTY_INIT;
  uint32_t pad ETX_EMPTY_INIT;
  float current_radius ETX_EMPTY_INIT;
  float vm_weight ETX_EMPTY_INIT;
  float vc_weight ETX_EMPTY_INIT;
  float vm_normalization ETX_EMPTY_INIT;
};

struct ETX_ALIGNED VCMPathState {
  SpectralResponse throughput = {};
  SpectralResponse gathered = {};
  Ray ray = {};

  float3 merged = {};
  Sampler sampler = {};

  float2 uv = {};
  SpectralQuery spect = {};
  float path_distance = 0.0f;

  float d_vcm = 0.0f;
  float d_vc = 0.0f;
  float d_vm = 0.0f;
  float eta = 1.0f;

  uint32_t path_length = 0;
  uint32_t medium_index = kInvalidIndex;
  uint32_t delta_emitter = 0;
  uint32_t index = 0u;
};

struct ETX_ALIGNED VCMLightVertex {
  VCMLightVertex() = default;

  ETX_GPU_CODE VCMLightVertex(const VCMPathState& s, const float3& p, const float3& b, uint32_t tri, uint32_t index)
    : throughput(s.throughput)
    , w_i(s.ray.d)
    , d_vcm(s.d_vcm)
    , bc(b)
    , d_vc(s.d_vc)
    , pos(p)
    , d_vm(s.d_vm)
    , triangle_index(tri)
    , path_length(s.path_length)
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
  uint32_t pad = 0;

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

struct ETX_ALIGNED VCMLightPath {
  uint32_t index ETX_EMPTY_INIT;
  uint32_t count ETX_EMPTY_INIT;
  SpectralQuery spect ETX_EMPTY_INIT;
  uint32_t pad ETX_EMPTY_INIT;
};

ETX_GPU_CODE bool vcm_next_ray(const Scene& scene, const PathSource path_source, const Intersection& i, uint32_t rr_start, VCMPathState& state, const VCMIteration& it) {
  const auto& tri = scene.triangles[i.triangle_index];
  const auto& mat = scene.materials[tri.material_index];
  auto bsdf_data = BSDFData{state.spect, state.medium_index, path_source, i, state.ray.d, {}};
  auto bsdf_sample = bsdf::sample(bsdf_data, mat, scene, state.sampler);
  bsdf_data.w_o = bsdf_sample.w_o;
  if (bsdf_sample.valid() == false) {
    return false;
  }

  state.throughput *= bsdf_sample.weight;
  ETX_VALIDATE(state.throughput);

  if (path_source == PathSource::Light) {
    state.throughput *= fix_shading_normal(tri.geo_n, bsdf_data.nrm, bsdf_data.w_i, bsdf_data.w_o);
  }

  if (state.throughput.is_zero()) {
    return false;
  }

  if ((state.path_length >= rr_start) && (apply_rr(state.eta, state.sampler.next(), state.throughput) == false)) {
    return false;
  }

  if (bsdf_sample.properties & BSDFSample::MediumChanged) {
    state.medium_index = bsdf_sample.medium_index;
  }

  float cos_theta_bsdf = fabsf(dot(i.nrm, bsdf_sample.w_o));

  if (bsdf_sample.is_delta()) {
    state.d_vc *= cos_theta_bsdf;
    ETX_VALIDATE(state.d_vc);

    state.d_vm *= cos_theta_bsdf;
    ETX_VALIDATE(state.d_vm);

    state.d_vcm = 0.0f;
  } else {
    auto rev_sample_pdf = bsdf::pdf(bsdf_data.swap_directions(), mat, scene, state.sampler);
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
  state.eta *= bsdf_sample.eta;
  return true;
}

ETX_GPU_CODE SpectralResponse vcm_get_radiance(const Scene& scene, const Emitter& emitter, const Intersection& intersection, const VCMPathState& state, const VCMOptions& options) {
  float pdf_emitter_area = 0.0f;
  float pdf_emitter_dir = 0.0f;
  float pdf_emitter_dir_out = 0.0f;

  SpectralResponse radiance = {};

  if (emitter.is_local()) {
    radiance = emitter_get_radiance(emitter, state.spect, intersection.tex, state.ray.o, intersection.pos,  //
      pdf_emitter_area, pdf_emitter_dir, pdf_emitter_dir_out, scene, (state.path_length == 1));             //
  } else {
    radiance = emitter_get_radiance(emitter, state.spect, state.ray.d,  //
      pdf_emitter_area, pdf_emitter_dir, pdf_emitter_dir_out, scene);   //
  }

  if (pdf_emitter_dir <= kEpsilon) {
    return {state.spect.wavelength, 0.0f};
  }

  float emitter_sample_pdf = emitter_discrete_pdf(emitter, scene.emitters_distribution);
  pdf_emitter_area *= emitter_sample_pdf;
  pdf_emitter_dir_out *= emitter_sample_pdf;
  float w_camera = state.d_vcm * pdf_emitter_area + state.d_vc * pdf_emitter_dir_out;
  float weight = (options.enable_mis() && (state.path_length > 1)) ? (1.0f / (1.0f + w_camera)) : 1.0f;
  return weight * (state.throughput * radiance);
}

ETX_GPU_CODE VCMPathState vcm_generate_emitter_state(uint32_t index, const Scene& scene, const VCMIteration& it) {
  auto sampler = Sampler(index, it.iteration);
  auto spect = spectrum::sample(sampler.next());
  auto emitter_sample = sample_emission(scene, spect, sampler);

  VCMPathState state = {};

  float cos_t = dot(emitter_sample.direction, emitter_sample.normal);
  state.throughput = emitter_sample.value * (cos_t / (emitter_sample.pdf_dir * emitter_sample.pdf_area * emitter_sample.pdf_sample));

  state.ray = {emitter_sample.origin, emitter_sample.direction};
  if (emitter_sample.triangle_index != kInvalidIndex) {
    state.ray.o = shading_pos(scene.vertices, scene.triangles[emitter_sample.triangle_index], emitter_sample.barycentric, state.ray.d);
  }

  state.d_vcm = emitter_sample.pdf_area / emitter_sample.pdf_dir_out;
  ETX_VALIDATE(state.d_vcm);

  state.d_vc = emitter_sample.is_delta ? 0.0f : (cos_t / (emitter_sample.pdf_dir_out * emitter_sample.pdf_sample));
  ETX_VALIDATE(state.d_vc);

  state.d_vm = state.d_vc * it.vc_weight;
  ETX_VALIDATE(state.d_vm);

  state.eta = 1.0f;
  state.spect = spect;
  state.medium_index = emitter_sample.medium_index;
  state.delta_emitter = emitter_sample.is_delta ? 1 : 0;
  state.sampler = sampler;
  state.index = index;
  return state;
}

ETX_GPU_CODE static VCMPathState vcm_generate_camera_state(const uint2& coord, const Scene& scene, const VCMIteration& it, const SpectralQuery spect) {
  VCMPathState state = {};
  state.index = coord.x + coord.y * scene.camera.image_size.x;
  state.sampler = Sampler(state.index, it.iteration);
  state.spect = (spect.wavelength == 0.0f) ? spectrum::sample(state.sampler.next()) : spect;
  state.uv = get_jittered_uv(state.sampler, coord, scene.camera.image_size);
  state.ray = generate_ray(state.sampler, scene, state.uv);
  state.throughput = {state.spect.wavelength, 1.0f};
  state.gathered = {state.spect.wavelength, 0.0f};
  state.merged = {};

  auto film_eval = film_evaluate_out(state.spect, scene.camera, state.ray);
  state.d_vcm = 1.0f / film_eval.pdf_dir;
  state.d_vc = 0.0f;
  state.d_vm = 0.0f;
  state.medium_index = scene.camera_medium_index;
  state.eta = 1.0f;
  state.path_distance = 0.0f;
  state.path_length = 1;
  return state;
}

ETX_GPU_CODE Medium::Sample vcm_try_sampling_medium(const Scene& scene, VCMPathState& state, float max_t) {
  if (state.medium_index == kInvalidIndex)
    return {};

  auto medium_sample = scene.mediums[state.medium_index].sample(state.spect, state.sampler, state.ray.o, state.ray.d, max_t);
  state.throughput *= medium_sample.weight;
  ETX_VALIDATE(state.throughput);
  return medium_sample;
}

ETX_GPU_CODE void vcm_handle_sampled_medium(const Scene& scene, const Medium::Sample& medium_sample, VCMPathState& state) {
  state.path_distance += medium_sample.t;
  state.path_length += 1;
  const auto& medium = scene.mediums[state.medium_index];
  state.ray.o = medium_sample.pos;
  state.ray.d = medium.sample_phase_function(state.spect, state.sampler, medium_sample.pos, state.ray.d);
}

ETX_GPU_CODE bool vcm_handle_boundary_bsdf(const Scene& scene, const Material& mat, const Intersection& intersection, const PathSource path_source, VCMPathState& state) {
  if (mat.cls != Material::Class::Boundary)
    return false;

  auto bsdf_sample = bsdf::sample({state.spect, state.medium_index, path_source, intersection, intersection.w_i, {}}, mat, scene, state.sampler);
  if (bsdf_sample.properties & BSDFSample::MediumChanged) {
    state.medium_index = bsdf_sample.medium_index;
  }
  state.ray.o = intersection.pos;
  state.ray.d = bsdf_sample.w_o;
  return true;
}

ETX_GPU_CODE void vcm_update_light_vcm(const Intersection& intersection, VCMPathState& state) {
  if ((state.path_length > 0) || (state.delta_emitter == 0)) {
    state.d_vcm *= sqr(state.path_distance);
  }

  float cos_to_prev = fabsf(dot(intersection.nrm, -state.ray.d));
  state.d_vcm /= cos_to_prev;
  state.d_vc /= cos_to_prev;
  state.d_vm /= cos_to_prev;
  state.path_distance = 0.0f;
}

template <class RT>
ETX_GPU_CODE static float3 vcm_connect_to_camera(const RT& rt, const Scene& scene, const Intersection& intersection, const Material& mat, const Triangle& tri,
  const VCMIteration& vcm_iteration, const VCMOptions& options, VCMPathState& state, float2& uv) {
  if ((options.connect_to_camera() == false) || (state.path_length + 1 >= options.max_depth)) {
    return {};
  }

  auto camera_sample = sample_film(state.sampler, scene, intersection.pos);
  if (camera_sample.pdf_dir <= 0.0f) {
    return {};
  }

  auto direction = camera_sample.position - intersection.pos;
  auto w_o = normalize(direction);
  auto data = BSDFData{state.spect, state.medium_index, PathSource::Light, intersection, state.ray.d, w_o};
  auto eval = bsdf::evaluate(data, mat, scene, state.sampler);
  if (eval.valid() == false) {
    return {};
  }

  float3 p0 = shading_pos(scene.vertices, tri, intersection.barycentric, w_o);
  auto tr = transmittance(state.spect, state.sampler, p0, camera_sample.position, state.medium_index, scene, rt);
  if (tr.is_zero()) {
    return {};
  }

  float reverse_pdf = bsdf::pdf(data.swap_directions(), mat, scene, state.sampler);
  float camera_pdf = camera_sample.pdf_dir_out * fabsf(dot(intersection.nrm, w_o)) / dot(direction, direction);
  ETX_VALIDATE(camera_pdf);

  float w_light = camera_pdf * (vcm_iteration.vm_weight + state.d_vcm + state.d_vc * reverse_pdf);
  ETX_VALIDATE(w_light);

  float weight = options.enable_mis() ? (1.0f / (1.0f + w_light)) : 1.0f;
  ETX_VALIDATE(weight);

  eval.bsdf *= fix_shading_normal(tri.geo_n, data.nrm, data.w_i, data.w_o);
  auto result = (tr * eval.bsdf * state.throughput * camera_sample.weight) * weight;

  uv = camera_sample.uv;
  return (result / spectrum::sample_pdf()).to_xyz();
}

ETX_GPU_CODE static void vcm_cam_handle_miss(const Scene& scene, const Intersection& intersection, const VCMOptions& options, VCMPathState& state) {
  if (options.direct_hit() == false)
    return;

  for (uint32_t ie = 0; ie < scene.environment_emitters.count; ++ie) {
    const auto& emitter = scene.emitters[scene.environment_emitters.emitters[ie]];
    state.gathered += vcm_get_radiance(scene, emitter, intersection, state, options);
  }
}

ETX_GPU_CODE static void vcm_update_camera_vcm(const Intersection& intersection, VCMPathState& state) {
  float cos_to_prev = fabsf(dot(intersection.nrm, -state.ray.d));
  state.d_vcm *= sqr(state.path_distance) / cos_to_prev;
  state.d_vc /= cos_to_prev;
  state.d_vm /= cos_to_prev;
  state.path_distance = 0.0f;
}

ETX_GPU_CODE static void vcm_handle_direct_hit(const Scene& scene, const Triangle& tri, const Intersection& intersection, const VCMOptions& options, VCMPathState& state) {
  if ((options.direct_hit() == false) || (state.path_length > options.max_depth) || (tri.emitter_index == kInvalidIndex))
    return;

  const auto& emitter = scene.emitters[tri.emitter_index];
  state.gathered += vcm_get_radiance(scene, emitter, intersection, state, options);
}

template <class RT>
ETX_GPU_CODE static void vcm_connect_to_light(const Scene& scene, const Triangle& tri, const Material& mat, const Intersection& intersection, const VCMIteration& vcm_iteration,
  const VCMOptions& options, const RT& rt, VCMPathState& state) {
  if ((options.connect_to_light() == false) || ((state.path_length + 1) > options.max_depth))
    return;

  auto emitter_sample = sample_emitter(state.spect, state.sampler, intersection.pos, scene);

  if (emitter_sample.pdf_dir > 0) {
    BSDFData connection_data = {state.spect, state.medium_index, PathSource::Camera, intersection, state.ray.d, emitter_sample.direction};
    BSDFEval connection_eval = bsdf::evaluate(connection_data, mat, scene, state.sampler);
    if (connection_eval.valid()) {
      float3 p0 = shading_pos(scene.vertices, tri, intersection.barycentric, normalize(emitter_sample.origin - intersection.pos));
      auto tr = transmittance(state.spect, state.sampler, p0, emitter_sample.origin, state.medium_index, scene, rt);
      if (tr.is_zero() == false) {
        float l_dot_n = fabsf(dot(emitter_sample.direction, intersection.nrm));
        float l_dot_e = fabsf(dot(emitter_sample.direction, emitter_sample.normal));
        float reverse_pdf = bsdf::pdf(connection_data.swap_directions(), mat, scene, state.sampler);

        float w_light = emitter_sample.is_delta ? 0.0f : (connection_eval.pdf / (emitter_sample.pdf_dir * emitter_sample.pdf_sample));

        float w_camera = (emitter_sample.pdf_dir_out * l_dot_n) / (emitter_sample.pdf_dir * l_dot_e) *  //
                         (vcm_iteration.vm_weight + state.d_vcm + state.d_vc * reverse_pdf);            //

        float weight = options.enable_mis() ? 1.0f / (1.0f + w_light + w_camera) : 1.0f;
        state.gathered += tr * state.throughput * connection_eval.bsdf * emitter_sample.value * (weight / (emitter_sample.pdf_dir * emitter_sample.pdf_sample));
        ETX_VALIDATE(state.gathered);
      }
    }
  }
}

ETX_GPU_CODE bool vcm_connect_to_light_vertex(const Scene& scene, const SpectralQuery& spect, VCMPathState& state, const Intersection& intersection,
  const VCMLightVertex& light_vertex, const VCMOptions& options, float vm_weight, uint32_t state_medium, float3& target_position, SpectralResponse& value) {
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
  auto camera_bsdf = bsdf::evaluate(camera_data, mat, scene, state.sampler);
  if (camera_bsdf.valid() == false) {
    return false;
  }

  auto camera_area_pdf = camera_bsdf.pdf * w_dot_l / distance_squared;
  ETX_VALIDATE(camera_area_pdf);

  auto camera_rev_pdf = bsdf::pdf(camera_data.swap_directions(), mat, scene, state.sampler);
  ETX_VALIDATE(camera_rev_pdf);
  if (camera_rev_pdf <= kEpsilon) {
    return false;
  }

  const auto& light_tri = scene.triangles[light_vertex.triangle_index];
  const auto& light_mat = scene.materials[light_tri.material_index];
  auto light_data = BSDFData{spect, state_medium, PathSource::Light, light_v, light_vertex.w_i, -w_o};
  auto light_bsdf = bsdf::evaluate(light_data, light_mat, scene, state.sampler);
  if (light_bsdf.valid() == false) {
    return false;
  }

  auto light_area_pdf = light_bsdf.pdf * w_dot_c / distance_squared;
  ETX_VALIDATE(light_area_pdf);

  auto light_rev_pdf = bsdf::pdf(light_data.swap_directions(), light_mat, scene, state.sampler);
  ETX_VALIDATE(light_rev_pdf);
  if (light_rev_pdf <= kEpsilon) {
    return false;
  }

  float w_light = camera_area_pdf * (vm_weight + light_vertex.d_vcm + light_vertex.d_vc * light_rev_pdf);
  ETX_VALIDATE(w_light);

  float w_camera = light_area_pdf * (vm_weight + state.d_vcm + state.d_vc * camera_rev_pdf);
  ETX_VALIDATE(w_camera);

  float weight = options.enable_mis() ? 1.0f / (1.0f + w_light + w_camera) : 1.0f;
  ETX_VALIDATE(weight);

  light_bsdf.bsdf *= fix_shading_normal(light_tri.geo_n, light_data.nrm, light_data.w_i, light_data.w_o);
  value = (camera_bsdf.bsdf * state.throughput) * (light_bsdf.bsdf * light_vertex.throughput) * (weight / distance_squared);
  return true;
}

template <class RT>
ETX_GPU_CODE static void vcm_connect_to_light_path(const Scene& scene, const Triangle& tri, const Material& mat, const Intersection& intersection,
  const VCMIteration& vcm_iteration, const VCMLightPath& light_path, const ArrayView<VCMLightVertex> light_vertices, const VCMOptions& options, const RT& rt, VCMPathState& state) {
  if (options.connect_vertices() == false)
    return;

  for (uint64_t i = 0; (i < light_path.count) && (state.path_length + i + 2 <= options.max_depth); ++i) {
    float3 target_position = {};
    SpectralResponse value = {};
    bool connected = vcm_connect_to_light_vertex(scene, state.spect, state, intersection, light_vertices[light_path.index + i],  //
      options, vcm_iteration.vm_weight, state.medium_index, target_position, value);

    if (connected) {
      float3 p0 = shading_pos(scene.vertices, tri, intersection.barycentric, normalize(target_position - intersection.pos));
      auto tr = transmittance(state.spect, state.sampler, p0, target_position, state.medium_index, scene, rt);
      if (tr.is_zero() == false) {
        state.gathered += tr * value;
        ETX_VALIDATE(state.gathered);
      }
    }
  }
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

  ETX_GPU_CODE float3 gather(const Scene& scene, VCMPathState& state, const VCMLightVertex* samples, const Intersection& intersection, const VCMOptions& options,
    float vc_weight) const {
    if ((options.merge_vertices() == false) || (indices.count == 0) || (state.path_length + 1 > options.max_depth)) {
      return {};
    }

    if (bounding_box.contains(intersection.pos) == false) {
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
        if ((distance_squared > radius_squared) || (light_vertex.path_length + state.path_length > options.max_depth)) {
          continue;
        }

        if (dot(intersection.nrm, light_vertex.normal(scene)) <= kEpsilon) {
          continue;
        }

        const auto& tri = scene.triangles[intersection.triangle_index];
        const auto& mat = scene.materials[tri.material_index];
        auto camera_data = BSDFData{state.spect, state.medium_index, PathSource::Camera, intersection, state.ray.d, -light_vertex.w_i};
        auto camera_bsdf = bsdf::evaluate(camera_data, mat, scene, state.sampler);
        if (camera_bsdf.valid() == false) {
          continue;
        }

        auto camera_rev_pdf = bsdf::pdf(camera_data.swap_directions(), mat, scene, state.sampler);

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
          auto value = light_vertex.throughput * camera_bsdf.func * state.throughput * (weight / sqr(spectrum::sample_pdf()));
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

struct ETX_ALIGNED VCMGlobal {
  Scene scene ETX_EMPTY_INIT;
  ArrayView<VCMPathState> input_state ETX_EMPTY_INIT;
  ArrayView<VCMPathState> output_state ETX_EMPTY_INIT;
  ArrayView<VCMLightPath> light_paths ETX_EMPTY_INIT;
  ArrayView<VCMLightVertex> light_vertices ETX_EMPTY_INIT;
  ArrayView<float4> camera_image ETX_EMPTY_INIT;
  ArrayView<float4> light_iteration_image ETX_EMPTY_INIT;
  ArrayView<float4> light_final_image ETX_EMPTY_INIT;
  VCMSpatialGridData spatial_grid ETX_EMPTY_INIT;
  VCMOptions options ETX_EMPTY_INIT;
  VCMIteration* iteration ETX_EMPTY_INIT;
};

enum VCMMemoryRequirements : uint64_t {
  VCMMaxOutputWidth = 3840llu,
  VCMMaxOutputHeight = 2160llu,
  VCMPixelCount = VCMMaxOutputWidth * VCMMaxOutputHeight,
  VCMPathStateSize = sizeof(VCMPathState),
  VCMGlobalSize = sizeof(VCMGlobal),
  VCMLightVertexSize = sizeof(VCMLightVertex),
  VCMLightVerticesMinSize = VCMLightVertexSize * VCMPixelCount,
  VCMPathStateBuffersSize = 2llu * VCMPathStateSize * VCMPixelCount,
  VCMImagesSize = 3llu * sizeof(float4) * VCMPixelCount,

  VCMTotalSize = VCMPathStateBuffersSize + VCMLightVerticesMinSize + VCMImagesSize,

  MB_PathStateBuffersSize = VCMPathStateBuffersSize / 1024llu / 1024llu,
  MB_ImagesSize = VCMImagesSize / 1024llu / 1024llu,
  MB_VCMTotalSize = VCMTotalSize / 1024llu / 1024llu,
};

}  // namespace etx
