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
    Default = ConnectToCamera | DirectHit | ConnectToLight | ConnectVertices | MergeVertices | EnableMis,
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
    return options & MergeVertices;
  }
  ETX_GPU_CODE bool enable_mis() const {
    return options & EnableMis;
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
  enum : uint32_t {
    DeltaEmitter = 1u << 0u,
    ContinueRay = 1u << 1u,
    RayActionSet = 1u << 2u,
  };

  Intersection intersection = {};
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

  uint32_t total_path_depth = 0;
  uint32_t medium_index = kInvalidIndex;
  uint32_t global_index = 0u;
  uint32_t flags = 0u;

  ETX_GPU_CODE bool delta_emitter() const {
    return (flags & DeltaEmitter) == DeltaEmitter;
  }

  ETX_GPU_CODE bool should_continue_ray() const {
    return (flags & ContinueRay) == ContinueRay;
  }

  ETX_GPU_CODE bool ray_action_set() const {
    return (flags & RayActionSet) == RayActionSet;
  }

  ETX_GPU_CODE void set_flags(uint32_t f, bool enable) {
    flags = enable ? (flags | f) : (flags & (~f));
  }

  ETX_GPU_CODE void clear_ray_action() {
    set_flags(RayActionSet | ContinueRay, false);
  }

  ETX_GPU_CODE void continue_ray(bool cont) {
    set_flags(ContinueRay, cont);
    set_flags(RayActionSet, true);
  }
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
    , path_length(s.total_path_depth)
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

ETX_GPU_CODE bool vcm_next_medium_ray(const VCMOptions& options, VCMPathState& state) {
  if (state.total_path_depth + 1 > options.max_depth)
    return false;

  return random_continue(state.total_path_depth, options.rr_start, state.eta, state.sampler, state.throughput);
}

ETX_GPU_CODE bool vcm_next_ray(const Scene& scene, const PathSource path_source, const VCMOptions& options, VCMPathState& state, const VCMIteration& it) {
  if (state.total_path_depth + 1 > options.max_depth)
    return false;

  const auto& tri = scene.triangles[state.intersection.triangle_index];
  const auto& mat = scene.materials[tri.material_index];
  auto bsdf_data = BSDFData{state.spect, state.medium_index, path_source, state.intersection, state.ray.d, {}};
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

  if (random_continue(state.total_path_depth, options.rr_start, state.eta, state.sampler, state.throughput) == false) {
    return false;
  }

  if (bsdf_sample.properties & BSDFSample::MediumChanged) {
    state.medium_index = bsdf_sample.medium_index;
  }

  float cos_theta_bsdf = fabsf(dot(state.intersection.nrm, bsdf_sample.w_o));

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
  state.ray.o = shading_pos(scene.vertices, tri, state.intersection.barycentric, bsdf_sample.w_o);
  state.eta *= bsdf_sample.eta;
  state.total_path_depth += 1;

  return true;
}

ETX_GPU_CODE SpectralResponse vcm_get_radiance(const Scene& scene, const Emitter& emitter, const VCMPathState& state, const VCMOptions& options) {
  float pdf_emitter_area = 0.0f;
  float pdf_emitter_dir = 0.0f;
  float pdf_emitter_dir_out = 0.0f;

  SpectralResponse radiance = {};

  if (emitter.is_local()) {
    radiance = emitter_get_radiance(emitter, state.spect, state.intersection.tex, state.ray.o, state.intersection.pos,  //
      pdf_emitter_area, pdf_emitter_dir, pdf_emitter_dir_out, scene, (state.total_path_depth == 1));                    //
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
  float weight = (options.enable_mis() && (state.total_path_depth > 1)) ? (1.0f / (1.0f + w_camera)) : 1.0f;
  return weight * (state.throughput * radiance);
}

ETX_GPU_CODE VCMPathState vcm_generate_emitter_state(uint32_t index, const Scene& scene, const VCMIteration& it) {
  VCMPathState state = {};
  state.sampler.init(index, it.iteration);
  state.spect = spectrum::sample(state.sampler.next());
  state.global_index = index;

  auto emitter_sample = sample_emission(scene, state.spect, state.sampler);

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
  state.medium_index = emitter_sample.medium_index;
  state.set_flags(VCMPathState::DeltaEmitter, emitter_sample.is_delta);
  return state;
}

ETX_GPU_CODE VCMPathState vcm_generate_camera_state(const uint2& coord, const Scene& scene, const VCMIteration& it, const SpectralQuery spect) {
  VCMPathState state = {};
  state.global_index = coord.x + coord.y * scene.camera.image_size.x;

  state.sampler.init(state.global_index, it.iteration);
  auto sampled_spectrum = spectrum::sample(state.sampler.next());
  state.spect = (spect.wavelength == 0.0f) ? sampled_spectrum : spect;

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
  state.total_path_depth = 1;
  return state;
}

ETX_GPU_CODE Medium::Sample vcm_try_sampling_medium(const Scene& scene, VCMPathState& state) {
  if (state.medium_index == kInvalidIndex)
    return {};

  float max_t = state.intersection.t;
  auto medium_sample = scene.mediums[state.medium_index].sample(state.spect, state.sampler, state.ray.o, state.ray.d, max_t);
  state.throughput *= medium_sample.weight;

  ETX_VALIDATE(state.throughput);
  return medium_sample;
}

ETX_GPU_CODE bool vcm_handle_sampled_medium(const Scene& scene, const Medium::Sample& medium_sample, const VCMOptions& options, VCMPathState& state) {
  state.path_distance += medium_sample.t;
  state.total_path_depth += 1;

  const auto& medium = scene.mediums[state.medium_index];
  state.ray.o = medium_sample.pos;
  state.ray.d = medium.sample_phase_function(state.spect, state.sampler, medium_sample.pos, state.ray.d);

  return vcm_next_medium_ray(options, state);
}

ETX_GPU_CODE bool vcm_handle_boundary_bsdf(const Scene& scene, const PathSource path_source, VCMPathState& state) {
  const auto& tri = scene.triangles[state.intersection.triangle_index];
  const auto& mat = scene.materials[tri.material_index];
  if (mat.cls != Material::Class::Boundary)
    return false;

  auto bsdf_sample = bsdf::sample({state.spect, state.medium_index, path_source, state.intersection, state.intersection.w_i, {}}, mat, scene, state.sampler);
  if (bsdf_sample.properties & BSDFSample::MediumChanged) {
    state.medium_index = bsdf_sample.medium_index;
  }

  state.path_distance += state.intersection.t;
  state.ray.o = state.intersection.pos;
  state.ray.d = bsdf_sample.w_o;
  return true;
}

ETX_GPU_CODE void vcm_update_light_vcm(VCMPathState& state) {
  if ((state.total_path_depth > 0) || (state.delta_emitter() == false)) {
    state.d_vcm *= sqr(state.path_distance + state.intersection.t);
  }

  float cos_to_prev = fabsf(dot(state.intersection.nrm, -state.ray.d));
  state.d_vcm /= cos_to_prev;
  state.d_vc /= cos_to_prev;
  state.d_vm /= cos_to_prev;
  state.path_distance = 0.0f;
}

template <class RT>
ETX_GPU_CODE float3 vcm_connect_to_camera(const RT& rt, const Scene& scene, const Material& mat, const Triangle& tri, const VCMIteration& vcm_iteration, const VCMOptions& options,
  VCMPathState& state, float2& uv) {
  if ((options.connect_to_camera() == false) || (state.total_path_depth + 1 >= options.max_depth)) {
    return {};
  }

  auto camera_sample = sample_film(state.sampler, scene, state.intersection.pos);
  if (camera_sample.pdf_dir <= 0.0f) {
    return {};
  }

  auto direction = camera_sample.position - state.intersection.pos;
  auto w_o = normalize(direction);
  auto data = BSDFData{state.spect, state.medium_index, PathSource::Light, state.intersection, state.ray.d, w_o};
  auto eval = bsdf::evaluate(data, mat, scene, state.sampler);
  if (eval.valid() == false) {
    return {};
  }

  float3 p0 = shading_pos(scene.vertices, tri, state.intersection.barycentric, w_o);
  auto tr = transmittance(state.spect, state.sampler, p0, camera_sample.position, state.medium_index, scene, rt);
  if (tr.is_zero()) {
    return {};
  }

  float reverse_pdf = bsdf::pdf(data.swap_directions(), mat, scene, state.sampler);
  float camera_pdf = camera_sample.pdf_dir_out * fabsf(dot(state.intersection.nrm, w_o)) / dot(direction, direction);
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

ETX_GPU_CODE void vcm_cam_handle_miss(const Scene& scene, const VCMOptions& options, VCMPathState& state) {
  if (options.direct_hit() == false)
    return;

  for (uint32_t ie = 0; ie < scene.environment_emitters.count; ++ie) {
    const auto& emitter = scene.emitters[scene.environment_emitters.emitters[ie]];
    state.gathered += vcm_get_radiance(scene, emitter, state, options);
  }
}

ETX_GPU_CODE void vcm_update_camera_vcm(VCMPathState& state) {
  float cos_to_prev = fabsf(dot(state.intersection.nrm, -state.ray.d));
  state.d_vcm *= sqr(state.path_distance + state.intersection.t) / cos_to_prev;
  state.d_vc /= cos_to_prev;
  state.d_vm /= cos_to_prev;
  state.path_distance = 0.0f;
}

ETX_GPU_CODE void vcm_handle_direct_hit(const Scene& scene, const VCMOptions& options, VCMPathState& state) {
  const auto& tri = scene.triangles[state.intersection.triangle_index];
  if ((options.direct_hit() == false) || (tri.emitter_index == kInvalidIndex))
    return;

  const auto& emitter = scene.emitters[tri.emitter_index];
  state.gathered += vcm_get_radiance(scene, emitter, state, options);
}

template <class RT>
ETX_GPU_CODE void vcm_connect_to_light(const Scene& scene, const VCMIteration& vcm_iteration, const VCMOptions& options, const RT& rt, VCMPathState& state) {
  if ((options.connect_to_light() == false) || (state.total_path_depth + 1 > options.max_depth))
    return;

  const auto& tri = scene.triangles[state.intersection.triangle_index];
  const auto& mat = scene.materials[tri.material_index];
  auto emitter_sample = sample_emitter(state.spect, state.sampler, state.intersection.pos, scene);

  if (emitter_sample.pdf_dir > 0) {
    BSDFData connection_data = {state.spect, state.medium_index, PathSource::Camera, state.intersection, state.ray.d, emitter_sample.direction};
    BSDFEval connection_eval = bsdf::evaluate(connection_data, mat, scene, state.sampler);
    if (connection_eval.valid()) {
      float3 p0 = shading_pos(scene.vertices, tri, state.intersection.barycentric, normalize(emitter_sample.origin - state.intersection.pos));
      auto tr = transmittance(state.spect, state.sampler, p0, emitter_sample.origin, state.medium_index, scene, rt);
      if (tr.is_zero() == false) {
        float l_dot_n = fabsf(dot(emitter_sample.direction, state.intersection.nrm));
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

ETX_GPU_CODE bool vcm_connect_to_light_vertex(const Scene& scene, const SpectralQuery& spect, VCMPathState& state, const VCMLightVertex& light_vertex, const VCMOptions& options,
  float vm_weight, uint32_t state_medium, float3& target_position, SpectralResponse& value) {
  auto light_v = light_vertex.vertex(scene);
  target_position = light_v.pos;

  auto w_o = light_v.pos - state.intersection.pos;
  float distance_squared = dot(w_o, w_o);

  if (distance_squared == 0.0f) {
    return false;
  }

  w_o /= sqrtf(distance_squared);

  float w_dot_c = dot(state.intersection.nrm, w_o);
  if (w_dot_c < 0.0f) {
    return false;
  }

  float w_dot_l = -dot(light_v.nrm, w_o);
  if (w_dot_l < 0.0f) {
    return false;
  }

  const auto& tri = scene.triangles[state.intersection.triangle_index];
  const auto& mat = scene.materials[tri.material_index];
  auto camera_data = BSDFData{spect, state_medium, PathSource::Camera, state.intersection, state.intersection.w_i, w_o};
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
ETX_GPU_CODE void vcm_connect_to_light_path(const Scene& scene, const VCMIteration& iteration, const ArrayView<VCMLightPath>& light_paths,
  const ArrayView<VCMLightVertex>& light_vertices, const VCMOptions& options, const RT& rt, VCMPathState& state) {
  if (options.connect_vertices() == false)
    return;

  const auto& tri = scene.triangles[state.intersection.triangle_index];
  const auto& mat = scene.materials[tri.material_index];
  const auto& light_path = light_paths[state.global_index];

  for (uint64_t i = 0; (i < light_path.count) && (state.total_path_depth + i + 2 <= options.max_depth); ++i) {
    float3 target_position = {};
    SpectralResponse value = {};
    bool connected = vcm_connect_to_light_vertex(scene, state.spect, state, light_vertices[light_path.index + i],  //
      options, iteration.vm_weight, state.medium_index, target_position, value);

    if (connected) {
      float3 p0 = shading_pos(scene.vertices, tri, state.intersection.barycentric, normalize(target_position - state.intersection.pos));
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

  ETX_GPU_CODE float3 gather_index(const Scene& scene, VCMPathState& state, const ArrayView<VCMLightVertex>& samples, const VCMOptions& options, float vc_weight,
    uint32_t index) const {
    uint32_t range_begin = (index == 0) ? 0 : cell_ends[index - 1llu];

    float3 merged = {};
    for (uint32_t j = range_begin, range_end = cell_ends[index]; j < range_end; ++j) {
      const auto& light_vertex = samples[indices[j]];

      auto d = light_vertex.position(scene) - state.intersection.pos;
      float distance_squared = dot(d, d);
      if ((distance_squared > radius_squared) || (light_vertex.path_length + state.total_path_depth > options.max_depth)) {
        continue;
      }

      if (dot(state.intersection.nrm, light_vertex.normal(scene)) <= kEpsilon) {
        continue;
      }

      const auto& tri = scene.triangles[state.intersection.triangle_index];
      const auto& mat = scene.materials[tri.material_index];
      auto camera_data = BSDFData{state.spect, state.medium_index, PathSource::Camera, state.intersection, state.ray.d, -light_vertex.w_i};
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
    return merged;
  }

  ETX_GPU_CODE float3 gather(const Scene& scene, VCMPathState& state, const ArrayView<VCMLightVertex>& samples, const VCMOptions& options, float vc_weight) const {
    if ((options.merge_vertices() == false) || (indices.count == 0) || (state.total_path_depth + 1 > options.max_depth)) {
      return {};
    }

    if (bounding_box.contains(state.intersection.pos) == false) {
      return {};
    }

    float3 m = (state.intersection.pos - bounding_box.p_min) / cell_size;
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
      merged += gather_index(scene, state, samples, options, vc_weight, cell_indices[i]);
    }

    return merged;
  }
};

ETX_GPU_CODE void vcm_gather_vertices(const Scene& scene, const VCMIteration& iteration, const ArrayView<VCMLightVertex> light_vertices, const VCMSpatialGridData& spatial_grid,
  const VCMOptions& options, VCMPathState& state) {
  if ((options.merge_vertices() == false) || (state.total_path_depth + 1 > options.max_depth)) {
    return;
  }

  state.merged += spatial_grid.gather(scene, state, light_vertices, options, iteration.vc_weight);
}

template <class RT>
ETX_GPU_CODE bool vcm_camera_step(const Scene& scene, const VCMIteration& iteration, const VCMOptions& options, const ArrayView<VCMLightPath>& light_paths,
  const ArrayView<VCMLightVertex>& light_vertices, VCMPathState& state, const RT& rt, const VCMSpatialGridData& spatial_grid) {
  bool found_intersection = rt.trace(scene, state.ray, state.intersection, state.sampler);

  Medium::Sample medium_sample = vcm_try_sampling_medium(scene, state);
  if (medium_sample.sampled_medium()) {
    return vcm_handle_sampled_medium(scene, medium_sample, options, state);
  }

  if (found_intersection == false) {
    vcm_cam_handle_miss(scene, options, state);
    return false;
  }

  if (vcm_handle_boundary_bsdf(scene, PathSource::Camera, state)) {
    return true;
  }

  vcm_update_camera_vcm(state);
  vcm_handle_direct_hit(scene, options, state);
  vcm_gather_vertices(scene, iteration, light_vertices, spatial_grid, options, state);

  const auto& tri = scene.triangles[state.intersection.triangle_index];
  const auto& mat = scene.materials[tri.material_index];
  if (bsdf::is_delta(mat, state.intersection.tex, scene, state.sampler) == false) {
    vcm_connect_to_light(scene, iteration, options, rt, state);
    vcm_connect_to_light_path(scene, iteration, light_paths, light_vertices, options, rt, state);
  }

  return vcm_next_ray(scene, PathSource::Camera, options, state, iteration);
}

struct ETX_ALIGNED LightStepResult {
  VCMLightVertex vertex_to_add = {};
  float3 value_to_splat = {};
  float2 splat_uv = {};
  bool add_vertex = false;
  bool continue_tracing = false;
};

template <class RT>
ETX_GPU_CODE LightStepResult vcm_light_step(const Scene& scene, const VCMIteration& iteration, const VCMOptions& options, const uint32_t path_index, VCMPathState& state,
  const RT& rt) {
  LightStepResult result = {};
  bool found_intersection = rt.trace(scene, state.ray, state.intersection, state.sampler);

  Medium::Sample medium_sample = vcm_try_sampling_medium(scene, state);
  if (medium_sample.sampled_medium()) {
    result.continue_tracing = vcm_handle_sampled_medium(scene, medium_sample, options, state);
    return result;
  }

  if (found_intersection == false) {
    return result;
  }

  const auto& tri = scene.triangles[state.intersection.triangle_index];
  const auto& mat = scene.materials[tri.material_index];

  if (vcm_handle_boundary_bsdf(scene, PathSource::Light, state)) {
    result.continue_tracing = true;
    return result;
  }

  vcm_update_light_vcm(state);

  if (bsdf::is_delta(mat, state.intersection.tex, scene, state.sampler) == false) {
    result.add_vertex = true;
    result.vertex_to_add = {state, state.intersection.pos, state.intersection.barycentric, state.intersection.triangle_index, path_index};

    if (options.connect_to_camera() && (state.total_path_depth + 1 <= options.max_depth)) {
      result.value_to_splat = vcm_connect_to_camera(rt, scene, mat, tri, iteration, options, state, result.splat_uv);
    }
  }

  if (vcm_next_ray(scene, PathSource::Light, options, state, iteration) == false) {
    return result;
  }

  result.continue_tracing = true;
  return result;
}

struct ETX_ALIGNED VCMGlobal {
  Scene scene ETX_EMPTY_INIT;
  ArrayView<VCMPathState> input_state ETX_EMPTY_INIT;
  ArrayView<VCMPathState> output_state ETX_EMPTY_INIT;
  ArrayView<VCMLightPath> light_paths ETX_EMPTY_INIT;
  ArrayView<VCMLightVertex> light_vertices ETX_EMPTY_INIT;
  ArrayView<float4> camera_iteration_image ETX_EMPTY_INIT;
  ArrayView<float4> camera_final_image ETX_EMPTY_INIT;
  ArrayView<float4> light_iteration_image ETX_EMPTY_INIT;
  ArrayView<float4> light_final_image ETX_EMPTY_INIT;
  VCMSpatialGridData spatial_grid ETX_EMPTY_INIT;
  VCMOptions options ETX_EMPTY_INIT;
  VCMIteration* iteration ETX_EMPTY_INIT;
  uint32_t active_rays ETX_EMPTY_INIT;
  uint32_t max_light_path_length ETX_EMPTY_INIT;
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
