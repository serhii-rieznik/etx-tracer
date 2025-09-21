#pragma once

#include <etx/render/shared/scene.hxx>
#include <etx/rt/shared/path_tracing_shared.hxx>
#include <etx/rt/rt.hxx>

namespace etx {

struct Options;
struct Raytracing;

struct ETX_ALIGNED VCMOptions {
  enum Kernel : uint32_t {
    Tophat = 0u,
    Epanechnikov = 1u,
  };

  uint32_t options ETX_EMPTY_INIT;
  uint32_t radius_decay ETX_EMPTY_INIT;
  uint32_t kernel ETX_INIT_WITH(Epanechnikov);
  float initial_radius ETX_EMPTY_INIT;
  bool blue_noise ETX_INIT_WITH(true);

  enum : uint32_t {
    ConnectToCamera = 1u << 0u,
    DirectHit = 1u << 1u,
    ConnectToLight = 1u << 2u,
    ConnectVertices = 1u << 3u,
    MergeVertices = 1u << 4u,
    EnableMis = 1u << 5u,
    EnableMerging = 1u << 6u,

    ConnectOnlyOptions = DirectHit | ConnectToLight | ConnectToCamera | ConnectVertices | EnableMis,
    FullOptions = ConnectOnlyOptions | EnableMerging | MergeVertices,

    DefaultOptions = FullOptions,
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
  ETX_GPU_CODE bool enable_mis() const {
    return options & EnableMis;
  }
  ETX_GPU_CODE bool merge_vertices() const {
    return enable_merging() && (options & MergeVertices);
  }
  ETX_GPU_CODE bool enable_merging() const {
    return options & EnableMerging;
  }
  ETX_GPU_CODE bool smooth_kernel() const {
    return kernel == Epanechnikov;
  }

  static VCMOptions default_values();

  void store(Options&) const;
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
    LocalEmitter = 1u << 3u,
    Valid = 1u << 4u,
  };

  SpectralResponse throughput = {};
  SpectralResponse gathered = {};
  Ray ray = {};

  float3 merged = {};
  Sampler sampler = {};

  float2 uv = {};
  uint2 pixel_coord = {};
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

  ETX_GPU_CODE bool local_emitter() const {
    return (flags & LocalEmitter) == LocalEmitter;
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

constexpr uint64_t kVCMPathStateSize = sizeof(VCMPathState);

struct ETX_ALIGNED VCMLightVertex {
  VCMLightVertex() = default;

  ETX_GPU_CODE VCMLightVertex(const VCMPathState& s, const Intersection& i, uint32_t index)
    : throughput(s.throughput)
    , w_i(s.ray.d)
    , d_vcm(s.d_vcm)
    , bc(i.barycentric)
    , d_vc(s.d_vc)
    , pos(i.pos)
    , d_vm(s.d_vm)
    , nrm(i.nrm)
    , triangle_index(i.triangle_index)
    , material_index(i.material_index)
    , medium_index(s.medium_index)
    , is_medium(false)
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

  float3 nrm = {};
  uint32_t triangle_index = kInvalidIndex;
  uint32_t material_index = kInvalidIndex;
  uint32_t medium_index = kInvalidIndex;
  bool is_medium = false;

  uint32_t path_length = 0;
  uint32_t path_index = 0;

  ETX_GPU_CODE Vertex vertex(const Scene& s) const {
    return lerp_vertex(s.vertices, s.triangles[triangle_index], bc);
  }
};

struct VCMLightPath {
  SpectralQuery spect ETX_EMPTY_INIT;
  uint32_t index ETX_EMPTY_INIT;
  uint32_t count ETX_EMPTY_INIT;
  uint32_t pixel_index ETX_EMPTY_INIT;
};

ETX_GPU_CODE bool vcm_can_extend_path(const Scene& scene, const VCMPathState& state) {
  return (state.total_path_depth + 1 <= scene.max_path_length);
}

ETX_GPU_CODE bool vcm_medium_explicit(const Scene& scene, const VCMPathState& state) {
  return (state.medium_index != kInvalidIndex) && scene.mediums[state.medium_index].enable_explicit_connections;
}

ETX_GPU_CODE SpectralResponse vcm_transmittance(const Raytracing& rt, const Scene& scene, VCMPathState& state, const float3& origin, const float3& target) {
  return rt.trace_transmittance(state.spect, scene, origin, target, {.index = state.medium_index}, state.sampler);
}

ETX_GPU_CODE bool vcm_next_ray(const Scene& scene, const PathSource path_source, const VCMOptions& options, VCMPathState& state, const VCMIteration& it,
  const Intersection& intersection, const BSDFData& bsdf_data, const BSDFSample& bsdf_sample, bool subsurface_sample) {
  if (state.total_path_depth + 1 > scene.max_path_length)
    return false;

  if (bsdf_sample.valid() == false) {
    return false;
  }

  const auto& tri = scene.triangles[intersection.triangle_index];
  const auto& mat = scene.materials[intersection.material_index];

  state.throughput *= bsdf_sample.weight;
  ETX_VALIDATE(state.throughput);

  if (path_source == PathSource::Light) {
    state.throughput *= fix_shading_normal(tri.geo_n, intersection.nrm, intersection.w_i, bsdf_sample.w_o);
  }

  if (state.throughput.is_zero()) {
    return false;
  }

  if (random_continue(state.total_path_depth, scene.random_path_termination, state.eta, state.sampler, state.throughput) == false) {
    return false;
  }

  if (bsdf_sample.properties & BSDFSample::MediumChanged) {
    state.medium_index = bsdf_sample.medium_index;
  }

  float cos_theta_bsdf = fabsf(dot(intersection.nrm, bsdf_sample.w_o));

  if (bsdf_sample.is_delta()) {
    state.d_vc *= cos_theta_bsdf;
    ETX_VALIDATE(state.d_vc);

    state.d_vm *= cos_theta_bsdf;
    ETX_VALIDATE(state.d_vm);

    state.d_vcm = 0.0f;
  } else {
    auto rev_sample_pdf = subsurface_sample                                      //
                            ? fabsf(dot(bsdf_data.w_i, intersection.nrm)) / kPi  //
                            : bsdf::reverse_pdf(bsdf_data, bsdf_sample.w_o, mat, scene, state.sampler);
    ETX_VALIDATE(rev_sample_pdf);

    state.d_vc = (cos_theta_bsdf / bsdf_sample.pdf) * (state.d_vc * rev_sample_pdf + state.d_vcm + it.vm_weight);
    ETX_VALIDATE(state.d_vc);

    state.d_vm = (cos_theta_bsdf / bsdf_sample.pdf) * (state.d_vm * rev_sample_pdf + state.d_vcm * it.vc_weight + 1.0f);
    ETX_VALIDATE(state.d_vm);

    state.d_vcm = 1.0f / bsdf_sample.pdf;
    ETX_VALIDATE(state.d_vcm);
  }

  state.ray.d = bsdf_sample.w_o;
  state.ray.o = shading_pos(scene.vertices, tri, intersection.barycentric, bsdf_sample.w_o);
  state.ray.max_t = kMaxFloat;
  state.ray.min_t = kRayEpsilon;
  state.eta *= bsdf_sample.eta;
  state.total_path_depth += 1u;

  return true;
}

ETX_GPU_CODE SpectralResponse vcm_get_radiance(const Scene& scene, const Emitter& emitter, const VCMPathState& state, const VCMOptions& options, const Intersection& intersection) {
  float pdf_emitter_area = 0.0f;
  float pdf_emitter_dir = 0.0f;
  float pdf_emitter_dir_out = 0.0f;

  EmitterRadianceQuery q = {
    .source_position = state.ray.o,
    .target_position = intersection.pos,
    .direction = state.ray.d,
    .uv = intersection.tex,
    .directly_visible = state.total_path_depth == 1,
  };

  SpectralResponse radiance = emitter_get_radiance(emitter, state.spect, q, pdf_emitter_area, pdf_emitter_dir, pdf_emitter_dir_out, scene);

  if (pdf_emitter_dir <= kEpsilon) {
    return {state.spect, 0.0f};
  }

  float emitter_sample_pdf = emitter_discrete_pdf(emitter, scene.emitters_distribution);
  float w_camera = state.d_vcm * pdf_emitter_area * emitter_sample_pdf + state.d_vc * (pdf_emitter_dir_out * emitter_sample_pdf);
  float weight = (options.enable_mis() && (state.total_path_depth > 1)) ? (1.0f / (1.0f + w_camera)) : 1.0f;
  return weight * (state.throughput * radiance);
}

ETX_GPU_CODE VCMPathState vcm_generate_emitter_state(uint32_t index, const Scene& scene, const VCMIteration& it) {
  VCMPathState state = {};
  state.sampler.init(index, it.iteration);
  state.spect = scene.spectral() ? SpectralQuery::spectral_sample(state.sampler.next()) : SpectralQuery::sample();
  state.global_index = index;

  auto emitter_sample = sample_emission(scene, state.spect, state.sampler);
  if (emitter_sample.pdf_dir <= 0.0f) {
    return state;
  }

  float cos_t = dot(emitter_sample.direction, emitter_sample.normal);
  ETX_ASSERT(cos_t > 0.0f);

  state.throughput = emitter_sample.value * (cos_t / (emitter_sample.pdf_dir * emitter_sample.pdf_area * emitter_sample.pdf_sample));
  ETX_VALIDATE(state.throughput);

  state.ray = {emitter_sample.origin, emitter_sample.direction};
  if (emitter_sample.triangle_index != kInvalidIndex) {
    state.ray.o = shading_pos(scene.vertices, scene.triangles[emitter_sample.triangle_index], emitter_sample.barycentric, state.ray.d);
  }

  state.d_vcm = emitter_sample.is_distant ? 1.0f / emitter_sample.pdf_area : 1.0f / emitter_sample.pdf_dir;
  ETX_VALIDATE(state.d_vcm);

  if (emitter_sample.is_delta == false) {
    state.d_vc = (emitter_sample.is_distant ? 1.0f : cos_t) / (emitter_sample.pdf_dir * emitter_sample.pdf_area * emitter_sample.pdf_sample);
    ETX_VALIDATE(state.d_vc);
  }

  state.d_vm = state.d_vc * it.vc_weight;
  ETX_VALIDATE(state.d_vm);

  state.eta = 1.0f;
  state.medium_index = emitter_sample.medium_index;
  state.set_flags(VCMPathState::DeltaEmitter, emitter_sample.is_delta);
  state.set_flags(VCMPathState::LocalEmitter, emitter_sample.is_distant == false);
  state.set_flags(VCMPathState::Valid, true);
  return state;
}

ETX_GPU_CODE VCMPathState vcm_generate_camera_state(const uint2& coord, const uint32_t index, const Scene& scene, const Camera& camera, const VCMIteration& it,
  const SpectralQuery spect) {
  VCMPathState state = {};
  state.global_index = index;
  state.pixel_coord = coord;  // Store pixel coordinate for blue noise

  state.sampler.init(state.global_index, it.iteration);
  auto sampled_spectrum = spect.spectral() ? SpectralQuery::spectral_sample(state.sampler.next()) : SpectralQuery::sample();
  state.spect = (spect.wavelength == 0.0f) ? sampled_spectrum : spect;

  state.uv = get_jittered_uv(state.sampler, coord, camera.film_size);
  state.ray = generate_ray(scene, camera, state.uv, state.sampler.next_2d());
  state.throughput = {state.spect, 1.0f};
  state.gathered = {state.spect, 0.0f};
  state.merged = {};

  auto film_eval = film_evaluate_out(state.spect, camera, state.ray);
  state.d_vcm = 1.0f / film_eval.pdf_dir;
  state.d_vc = 0.0f;
  state.d_vm = 0.0f;
  state.medium_index = camera.medium_index;
  state.eta = 1.0f;
  state.path_distance = 0.0f;

  state.total_path_depth = 1;
  return state;
}

ETX_GPU_CODE Medium::Sample vcm_try_sampling_medium(const Scene& scene, VCMPathState& state, float max_t) {
  if (state.medium_index == kInvalidIndex)
    return {};

  auto medium_sample = scene.mediums[state.medium_index].sample(state.spect, state.throughput, state.sampler, state.ray.o, state.ray.d, max_t);
  state.throughput *= medium_sample.weight;

  ETX_VALIDATE(state.throughput);
  return medium_sample;
}

ETX_GPU_CODE bool vcm_handle_sampled_medium(const Scene& scene, const Medium::Sample& medium_sample, const VCMIteration& it, const VCMOptions& /*options*/,
  const PathSource path_source, VCMPathState& state, const float2& rnd_phase) {
  // Fold pending boundary segment (if any) plus medium segment (no cosine) before recurrences
  bool apply_fold = true;
  if (apply_fold) {
    float seg = state.path_distance + medium_sample.t;
    state.d_vcm *= sqr(seg);
    ETX_VALIDATE(state.d_vcm);
    state.path_distance = 0.0f;
  }

  // Medium scattering increases path length
  state.total_path_depth += 1;

  const auto& medium = scene.mediums[state.medium_index];

  // Sample new direction using phase function with pre-allocated samples
  float3 w_i = state.ray.d;
  float3 w_o = medium.sample_phase_function(rnd_phase, w_i);

  float pdf_fwd = medium.phase_function(w_i, w_o);
  ETX_VALIDATE(pdf_fwd);
  float pdf_rev = medium.phase_function(w_o, w_i);
  ETX_VALIDATE(pdf_rev);

  // Update MIS recurrence for non-delta event with cos_theta = 1 for media
  // Medium vertices are not mergeable: exclude merging competitors in d_vm
  state.d_vc = (1.0f / pdf_fwd) * (state.d_vc * pdf_rev + state.d_vcm);
  ETX_VALIDATE(state.d_vc);
  state.d_vm = (1.0f / pdf_fwd) * (state.d_vm * pdf_rev + 0.0f);
  ETX_VALIDATE(state.d_vm);
  state.d_vcm = 1.0f / pdf_fwd;
  ETX_VALIDATE(state.d_vcm);

  // Advance ray to medium position and set new direction
  state.ray.o = medium_sample.pos;
  state.ray.d = w_o;
  state.ray.max_t = kMaxFloat;
  state.ray.min_t = kRayEpsilon;

  if (state.total_path_depth + 1 > scene.max_path_length)
    return false;

  return random_continue(state.total_path_depth, scene.random_path_termination, state.eta, state.sampler, state.throughput);
}

ETX_GPU_CODE bool vcm_handle_boundary_bsdf(const Scene& scene, const PathSource path_source, const Intersection& intersection, VCMPathState& state) {
  const auto& mat = scene.materials[intersection.material_index];
  if (mat.cls != Material::Class::Boundary)
    return false;

  const auto& tri = scene.triangles[intersection.triangle_index];
  uint32_t new_medium = (dot(intersection.nrm, state.ray.d) < 0.0f) ? mat.int_medium : mat.ext_medium;
  state.path_distance += intersection.t;
  state.medium_index = new_medium;
  state.ray.o = shading_pos(scene.vertices, tri, intersection.barycentric, state.ray.d);
  return true;
}

ETX_GPU_CODE void vcm_update_light_vcm(const Intersection& intersection, VCMPathState& state) {
  if ((state.total_path_depth > 0) || state.local_emitter()) {
    state.d_vcm *= sqr(state.path_distance + intersection.t);
  }

  float cos_to_prev = fabsf(dot(intersection.nrm, -state.ray.d));
  state.d_vcm /= cos_to_prev;
  state.d_vc /= cos_to_prev;
  state.d_vm /= cos_to_prev;
  state.path_distance = 0.0f;
}

ETX_GPU_CODE SpectralResponse vcm_connect_to_camera(const Raytracing& rt, const Scene& scene, const Camera& camera, const VCMIteration& vcm_iteration, const VCMOptions& options,
  bool camera_at_medium, const Intersection* isect, const float3& medium_pos, VCMPathState& state, float2& uv) {
  if ((options.connect_to_camera() == false) || (state.total_path_depth + 2 > scene.max_path_length) || (state.total_path_depth + 2 < scene.min_path_length)) {
    return {};
  }

  float3 sample_pos = camera_at_medium ? medium_pos : isect->pos;
  auto camera_sample = sample_film(state.sampler, scene, camera, sample_pos);
  if (camera_sample.pdf_dir <= 0.0f) {
    return {};
  }

  auto direction = camera_sample.position - sample_pos;
  float dist2 = dot(direction, direction);
  if (dist2 <= kEpsilon) {
    return {};
  }
  auto w_o = normalize(direction);

  SpectralResponse scatter = {state.spect, 0.0f};
  float reverse_pdf = 0.0f;

  float3 origin = sample_pos;
  if (camera_at_medium == false) {
    const auto& mat = scene.materials[isect->material_index];
    auto data = BSDFData{state.spect, state.medium_index, PathSource::Light, *isect, isect->w_i};
    auto eval = bsdf::evaluate(data, w_o, mat, scene, state.sampler);
    if (eval.valid() == false) {
      return {};
    }
    scatter = eval.bsdf;
    reverse_pdf = bsdf::reverse_pdf(data, w_o, mat, scene, state.sampler);

    const auto& tri = scene.triangles[isect->triangle_index];
    origin = shading_pos(scene.vertices, tri, isect->barycentric, w_o);
  } else {
    const auto& medium = scene.mediums[state.medium_index];
    float p = medium.phase_function(state.ray.d, w_o);
    if (p <= 0.0f) {
      return {};
    }
    scatter = {state.spect, p};
    reverse_pdf = medium.phase_function(w_o, state.ray.d);
    ETX_VALIDATE(reverse_pdf);
  }

  auto tr = vcm_transmittance(rt, scene, state, origin, camera_sample.position);
  if (tr.is_zero()) {
    return {};
  }

  uv = camera_sample.uv;

  float camera_pdf = camera_sample.pdf_dir_out * (camera_at_medium ? 1.0f : fabsf(dot(isect->nrm, w_o))) / dist2;
  ETX_VALIDATE(camera_pdf);

  float vmW_cam = camera_at_medium ? 0.0f : vcm_iteration.vm_weight;
  float w_light = camera_pdf * (vmW_cam + state.d_vcm + state.d_vc * reverse_pdf);
  ETX_VALIDATE(w_light);

  float weight = options.enable_mis() ? (1.0f / (1.0f + w_light)) : 1.0f;
  ETX_VALIDATE(weight);

  if (camera_at_medium == false) {
    const auto& tri = scene.triangles[isect->triangle_index];
    weight *= fix_shading_normal(tri.geo_n, isect->nrm, isect->w_i, w_o);
  }

  return tr * scatter * state.throughput * camera_sample.weight * weight;
}

ETX_GPU_CODE void vcm_cam_handle_miss(const Scene& scene, const VCMOptions& options, const Intersection& intersection, VCMPathState& state) {
  if (options.direct_hit() == false)
    return;

  if (state.path_distance > 0.0f) {
    state.d_vcm *= sqr(state.path_distance);
    ETX_VALIDATE(state.d_vcm);
    state.path_distance = 0.0f;
  }

  SpectralResponse accumulated_value = {state.spect, 0.0f};
  float sum_pdf_dir_out = 0.0f;
  float sum_pdf_dir = 0.0f;

  for (uint32_t ie = 0; ie < scene.environment_emitters.count; ++ie) {
    const auto& emitter = scene.emitters[scene.environment_emitters.emitters[ie]];

    EmitterRadianceQuery q = {
      .direction = state.ray.d,
      .directly_visible = state.total_path_depth <= 1,
    };

    float pdf_area = 0.0f;
    float pdf_dir = 0.0f;
    float pdf_dir_out = 0.0f;
    SpectralResponse value = emitter_get_radiance(emitter, state.spect, q, pdf_area, pdf_dir, pdf_dir_out, scene);
    ETX_VALIDATE(value);

    if (pdf_dir > kEpsilon) {
      float pdf_discrete = emitter_discrete_pdf(emitter, scene.emitters_distribution);
      sum_pdf_dir_out += pdf_dir_out * pdf_discrete;
      sum_pdf_dir += pdf_dir * pdf_discrete;
      accumulated_value += value;
      ETX_VALIDATE(accumulated_value);
    }
  }

  if (accumulated_value.maximum() > kEpsilon) {
    float inv_count = (scene.environment_emitters.count > 0u) ? (1.0f / float(scene.environment_emitters.count)) : 0.0f;
    sum_pdf_dir *= inv_count;
    sum_pdf_dir_out *= inv_count;
    float w_camera_sum = state.d_vcm * sum_pdf_dir + state.d_vc * sum_pdf_dir_out;
    ETX_VALIDATE(w_camera_sum);
    float weight = options.enable_mis() && (state.total_path_depth > 1) ? (1.0f / (1.0f + w_camera_sum)) : 1.0f;
    ETX_VALIDATE(weight);
    SpectralResponse add = state.throughput * accumulated_value * weight;
    ETX_VALIDATE(add);
    state.gathered += add;
    ETX_VALIDATE(state.gathered);
  }
}

ETX_GPU_CODE void vcm_update_camera_vcm(const Intersection& intersection, VCMPathState& state) {
  float cos_to_prev = fabsf(dot(intersection.nrm, -state.ray.d));
  state.d_vcm *= sqr(state.path_distance + intersection.t) / cos_to_prev;
  state.d_vc /= cos_to_prev;
  state.d_vm /= cos_to_prev;
  state.path_distance = 0.0f;
}

ETX_GPU_CODE void vcm_handle_direct_hit(const Scene& scene, const VCMOptions& options, const Intersection& intersection, VCMPathState& state) {
  if ((options.direct_hit() == false) || (intersection.emitter_index == kInvalidIndex))
    return;

  if ((state.total_path_depth > scene.max_path_length) || (state.total_path_depth < scene.min_path_length))
    return;

  const auto& emitter = scene.emitters[intersection.emitter_index];
  state.gathered += vcm_get_radiance(scene, emitter, state, options, intersection);
}

ETX_GPU_CODE SpectralResponse vcm_connect_to_light(const Scene& scene, const VCMIteration& vcm_iteration, const VCMOptions& options, bool camera_at_medium,
  const Intersection* isect, const float3& medium_pos, const Raytracing& rt, VCMPathState& state) {
  if ((options.connect_to_light() == false) || (state.total_path_depth + 1 > scene.max_path_length) || (state.total_path_depth + 1 < scene.min_path_length))
    return {state.spect, 0.0f};

  float3 sample_pos = camera_at_medium ? medium_pos : isect->pos;
  uint32_t emitter_index = sample_emitter_index(scene, state.sampler.fixed_w);
  auto emitter_sample = sample_emitter(state.spect, emitter_index, {state.sampler.fixed_u, state.sampler.fixed_v}, sample_pos, scene);
  if (emitter_sample.pdf_dir <= 0.0f)
    return {state.spect, 0.0f};

  float3 w_o = emitter_sample.direction;

  SpectralResponse scatter = {state.spect, 0.0f};
  float reverse_pdf = 0.0f;
  float3 origin = sample_pos;
  float camera_factor = 1.0f;

  if (camera_at_medium) {
    const auto& medium = scene.mediums[state.medium_index];
    float p = medium.phase_function(state.ray.d, w_o);
    if (p <= 0.0f)
      return {state.spect, 0.0f};
    scatter = {state.spect, p};
    reverse_pdf = medium.phase_function(w_o, state.ray.d);
    ETX_VALIDATE(reverse_pdf);
  } else {
    const auto& mat = scene.materials[isect->material_index];
    BSDFData connection_data = {state.spect, state.medium_index, PathSource::Camera, *isect, isect->w_i};
    BSDFEval connection_eval = bsdf::evaluate(connection_data, w_o, mat, scene, state.sampler);
    if (connection_eval.valid() == false)
      return {state.spect, 0.0f};
    scatter = connection_eval.bsdf;
    reverse_pdf = bsdf::reverse_pdf(connection_data, w_o, mat, scene, state.sampler);
    const auto& tri = scene.triangles[isect->triangle_index];
    origin = shading_pos(scene.vertices, tri, isect->barycentric, normalize(emitter_sample.origin - isect->pos));
    camera_factor = fabsf(dot(w_o, tri.geo_n));
  }

  auto tr = vcm_transmittance(rt, scene, state, origin, emitter_sample.origin);
  if (tr.is_zero())
    return {state.spect, 0.0f};

  float l_dot_e = fabsf(dot(emitter_sample.direction, emitter_sample.normal));
  float w_light = 0.0f;
  if (emitter_sample.is_delta == false) {
    if (camera_at_medium) {
      w_light = (scatter.value) / (emitter_sample.pdf_dir * emitter_sample.pdf_sample);
    } else {
      const auto& mat = scene.materials[isect->material_index];
      BSDFData data = {state.spect, state.medium_index, PathSource::Camera, *isect, isect->w_i};
      float conn_pdf = bsdf::pdf(data, w_o, mat, scene, state.sampler);
      ETX_VALIDATE(conn_pdf);
      w_light = conn_pdf / (emitter_sample.pdf_dir * emitter_sample.pdf_sample);
    }
  }

  float vmW_nee = camera_at_medium ? 0.0f : vcm_iteration.vm_weight;
  float w_camera = (emitter_sample.pdf_dir_out * camera_factor) / (emitter_sample.pdf_dir * l_dot_e) * (vmW_nee + state.d_vcm + state.d_vc * reverse_pdf);
  float weight = options.enable_mis() ? 1.0f / (1.0f + w_light + w_camera) : 1.0f;
  ETX_VALIDATE(weight);

  return tr * state.throughput * scatter * emitter_sample.value * (weight / (emitter_sample.pdf_dir * emitter_sample.pdf_sample));
}

ETX_GPU_CODE bool vcm_connect_to_light_vertex(const Scene& scene, const SpectralQuery& spect, VCMPathState& state, const VCMLightVertex& light_vertex, const VCMOptions& options,
  bool camera_at_medium, const Intersection* camera_isect, const float3& medium_pos, float vm_weight, uint32_t state_medium, float3& target_position, SpectralResponse& value) {
  Vertex light_v = light_vertex.is_medium ? Vertex{} : light_vertex.vertex(scene);
  target_position = light_vertex.is_medium ? light_vertex.pos : light_v.pos;

  float3 w_o = target_position - (camera_at_medium ? medium_pos : camera_isect->pos);
  float distance_squared = dot(w_o, w_o);
  if (distance_squared <= kEpsilon)
    return false;
  w_o /= sqrtf(distance_squared);

  float w_dot_l = 1.0f;
  if (light_vertex.is_medium == false) {
    w_dot_l = -dot(light_v.nrm, w_o);
  }

  float camera_area_pdf = 0.0f;
  float camera_rev_pdf = 0.0f;
  SpectralResponse camera_scatter = {spect, 0.0f};

  if (camera_at_medium) {
    const auto& medium = scene.mediums[state_medium];
    float p = medium.phase_function(state.ray.d, w_o);
    if (p <= 0.0f)
      return false;
    float p_rev = medium.phase_function(w_o, state.ray.d);
    ETX_VALIDATE(p_rev);
    camera_area_pdf = p * fabsf(w_dot_l) / distance_squared;
    ETX_VALIDATE(camera_area_pdf);
    camera_rev_pdf = p_rev;
    camera_scatter = {spect, p};
  } else {
    float w_dot_c = dot(camera_isect->nrm, w_o);
    const auto& mat = scene.materials[camera_isect->material_index];
    auto camera_data = BSDFData{spect, state_medium, PathSource::Camera, *camera_isect, camera_isect->w_i};
    auto camera_bsdf = bsdf::evaluate(camera_data, w_o, mat, scene, state.sampler);
    if (camera_bsdf.valid() == false)
      return false;
    camera_area_pdf = camera_bsdf.pdf * fabsf(w_dot_l) / distance_squared;
    ETX_VALIDATE(camera_area_pdf);
    camera_rev_pdf = bsdf::reverse_pdf(camera_data, w_o, mat, scene, state.sampler);
    ETX_VALIDATE(camera_rev_pdf);
    camera_scatter = camera_bsdf.bsdf;
  }

  // Light side: surface or medium
  float light_area_pdf = 0.0f;
  float light_rev_pdf = 0.0f;
  SpectralResponse light_scatter = {spect, 0.0f};

  if (light_vertex.is_medium) {
    const auto& med = scene.mediums[light_vertex.medium_index];
    float p = med.phase_function(light_vertex.w_i, -w_o);
    if (p <= 0.0f)
      return false;
    float p_rev = med.phase_function(-w_o, light_vertex.w_i);
    ETX_VALIDATE(p_rev);
    light_area_pdf = p * (camera_at_medium ? 1.0f : fabsf(dot(camera_isect->nrm, w_o))) / distance_squared;
    ETX_VALIDATE(light_area_pdf);
    light_rev_pdf = p_rev;
    light_scatter = {spect, p};
  } else {
    const auto& light_mat = scene.materials[light_vertex.material_index];
    const auto& light_tri = scene.triangles[light_vertex.triangle_index];
    auto light_data = BSDFData{spect, camera_at_medium ? light_vertex.medium_index : state_medium, PathSource::Light, light_v, light_vertex.w_i};
    auto light_bsdf = bsdf::evaluate(light_data, -w_o, light_mat, scene, state.sampler);
    if (light_bsdf.valid() == false)
      return false;
    if (camera_at_medium) {
      light_area_pdf = light_bsdf.pdf / distance_squared;
    } else {
      float w_dot_c = dot(camera_isect->nrm, w_o);
      light_area_pdf = light_bsdf.pdf * fabsf(w_dot_c) / distance_squared;
    }
    ETX_VALIDATE(light_area_pdf);
    light_rev_pdf = bsdf::reverse_pdf(light_data, -w_o, light_mat, scene, state.sampler);
    ETX_VALIDATE(light_rev_pdf);
    light_scatter = light_bsdf.bsdf * fix_shading_normal(light_tri.geo_n, light_data.nrm, light_data.w_i, -w_o);
  }

  float vmW_pair = (camera_at_medium || light_vertex.is_medium) ? 0.0f : vm_weight;
  float w_light = camera_area_pdf * (vmW_pair + light_vertex.d_vcm + light_vertex.d_vc * light_rev_pdf);
  ETX_VALIDATE(w_light);
  float w_camera = light_area_pdf * (vmW_pair + state.d_vcm + state.d_vc * camera_rev_pdf);
  ETX_VALIDATE(w_camera);
  float weight = options.enable_mis() ? 1.0f / (1.0f + w_light + w_camera) : 1.0f;
  ETX_VALIDATE(weight);

  value = (camera_scatter * state.throughput) * (light_scatter * light_vertex.throughput) * (weight / distance_squared);
  return true;
}

ETX_GPU_CODE SpectralResponse vcm_connect_to_light_path(const Scene& scene, const VCMIteration& iteration, const ArrayView<VCMLightPath>& light_paths,
  const ArrayView<VCMLightVertex>& light_vertices, const VCMOptions& options, bool camera_at_medium, const Intersection* isect, const float3& medium_pos, const Raytracing& rt,
  VCMPathState& state) {
  if (options.connect_vertices() == false)
    return {state.spect, 0.0f};

  const auto& light_path = light_paths[state.global_index];
  SpectralResponse result = {state.spect, 0.0f};
  for (uint64_t i = 0; i < light_path.count; ++i) {
    const uint64_t target_path_length = state.total_path_depth + i + 2u;
    if (target_path_length < scene.min_path_length)
      continue;
    if (target_path_length > scene.max_path_length)
      break;

    float3 target_position = {};
    SpectralResponse value = {};
    bool connected = vcm_connect_to_light_vertex(scene, state.spect, state, light_vertices[light_path.index + i], options, camera_at_medium, isect, medium_pos, iteration.vm_weight,
      state.medium_index, target_position, value);
    if (connected) {
      if (camera_at_medium) {
        auto tr = vcm_transmittance(rt, scene, state, medium_pos, light_vertices[light_path.index + i].pos);
        if (tr.is_zero() == false) {
          result += tr * value;
          ETX_VALIDATE(result);
        }
      } else {
        const auto& tri = scene.triangles[isect->triangle_index];
        float3 p0 = shading_pos(scene.vertices, tri, isect->barycentric, normalize(target_position - isect->pos));
        auto tr = vcm_transmittance(rt, scene, state, p0, target_position);
        if (tr.is_zero() == false) {
          result += tr * value;
          ETX_VALIDATE(result);
        }
      }
    }
  }
  return result;
}

struct ETX_ALIGNED VCMSpatialGridData {
  ArrayView<uint32_t> cell_ends ETX_EMPTY_INIT;
  ArrayView<float3> positions ETX_EMPTY_INIT;
  ArrayView<float3> normals ETX_EMPTY_INIT;
  ArrayView<float3> w_in ETX_EMPTY_INIT;
  ArrayView<float> d_vcm ETX_EMPTY_INIT;
  ArrayView<float> d_vm ETX_EMPTY_INIT;
  ArrayView<uint32_t> path_lengths ETX_EMPTY_INIT;
  ArrayView<float3> throughput_rgb_div_pdf ETX_EMPTY_INIT;
  BoundingBox bounding_box ETX_EMPTY_INIT;
  uint32_t hash_table_mask ETX_EMPTY_INIT;
  float cell_size ETX_EMPTY_INIT;
  float radius_squared ETX_EMPTY_INIT;
  float inv_radius_squared ETX_EMPTY_INIT;

  ETX_GPU_CODE uint32_t cell_index(int32_t x, int32_t y, int32_t z) const {
    return ((x * 73856093u) ^ (y * 19349663) ^ (z * 83492791)) & hash_table_mask;
  }

  ETX_GPU_CODE uint32_t position_to_index(const float3& pos) const {
    auto m = floor((pos - bounding_box.p_min) / cell_size);
    return cell_index(static_cast<int32_t>(m.x), static_cast<int32_t>(m.y), static_cast<int32_t>(m.z));
  }

  ETX_GPU_CODE float3 gather_index(const Scene& scene, const Intersection& intersection, const VCMOptions& options, float vc_weight, uint32_t index, VCMPathState& state) const {
    const auto& mat = scene.materials[intersection.material_index];

    const auto camera_data = BSDFData{state.spect, state.medium_index, PathSource::Camera, intersection, intersection.w_i};
    const auto t_camera = state.throughput / state.spect.sampling_pdf();
    const float w_camera_base = state.d_vcm * vc_weight;
    const bool use_mis = options.enable_mis();
    const bool is_spectral = state.spect.spectral();
    const bool use_epan = (options.kernel == VCMOptions::Epanechnikov);
    const uint32_t range_begin = (index == 0) ? 0 : cell_ends[index - 1llu];

    float3 merged = {};
    for (uint32_t j = range_begin, range_end = cell_ends[index]; j < range_end; ++j) {
      auto d = positions[j] - intersection.pos;
      float distance_squared = dot(d, d);
      if ((distance_squared > radius_squared) || (path_lengths[j] + state.total_path_depth + 1 > scene.max_path_length)) {
        continue;
      }
      if (dot(intersection.nrm, normals[j]) <= kEpsilon) {
        continue;
      }

      const float3 wi = w_in[j];
      auto camera_bsdf = bsdf::evaluate(camera_data, -wi, mat, scene, state.sampler);
      if (camera_bsdf.valid() == false) {
        continue;
      }

      auto camera_rev_pdf = bsdf::reverse_pdf(camera_data, -wi, mat, scene, state.sampler);

      float w_light = d_vcm[j] * vc_weight + d_vm[j] * camera_bsdf.pdf;
      float w_camera = w_camera_base + state.d_vm * camera_rev_pdf;
      float weight = use_mis ? (1.0f / (1.0f + w_light + w_camera)) : 1.0f;

      float kernel_weight = 1.0f;
      float u2 = distance_squared * inv_radius_squared;
      if (use_epan) {
        float one_minus = 1.0f - u2;
        kernel_weight = fmaxf(2.0f * one_minus, 0.0f);
      }

      auto c_value = (camera_bsdf.func * t_camera).to_rgb();
      ETX_VALIDATE(c_value);

      auto l_value = throughput_rgb_div_pdf[j];
      ETX_VALIDATE(l_value);

      if (is_spectral) {
        l_value *= SpectralDistribution::kRGBLuminanceScale;
      }

      merged += (c_value * l_value) * (kernel_weight * weight);
      ETX_VALIDATE(merged);
    }
    return merged;
  }

  ETX_GPU_CODE float3 gather(const Scene& scene, VCMPathState& state, const VCMOptions& options, const Intersection& intersection, float vc_weight) const {
    if (positions.count == 0) {
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
      merged += gather_index(scene, intersection, options, vc_weight, cell_indices[i], state);
    }

    return merged;
  }
};

ETX_GPU_CODE bool vcm_camera_step(const Scene& scene, const VCMIteration& iteration, const VCMOptions& options, const ArrayView<VCMLightPath>& light_paths,
  const ArrayView<VCMLightVertex>& light_vertices, VCMPathState& state, const Raytracing& rt, const VCMSpatialGridData& spatial_grid) {
  Intersection intersection = {};
  bool found_intersection = rt.trace(scene, state.ray, intersection, state.sampler);

  // Try sampling medium BEFORE allocating per-event samples to match BDPT ordering
  Medium::Sample medium_sample = vcm_try_sampling_medium(scene, state, found_intersection ? intersection.t : kMaxFloat);
  if (medium_sample.sampled_medium()) {
    // Allocate samples AFTER medium sampling to match BDPT
    float2 rnd_bsdf = state.sampler.next_2d();        // BSDF/phase function sampling
    float2 rnd_connection = state.sampler.next_2d();  // Light/camera connections
    float2 rnd_support = state.sampler.next_2d();     // Support operations

    // Override with blue noise for first camera interaction (limit early iterations for parity with BDPT)
    if (options.blue_noise && (state.total_path_depth == 1) && (iteration.iteration < 256u)) {
      rnd_bsdf = sample_blue_noise(state.pixel_coord, scene.samples, iteration.iteration, 0);
      rnd_connection = sample_blue_noise(state.pixel_coord, scene.samples, iteration.iteration, 2);
      rnd_support = sample_blue_noise(state.pixel_coord, scene.samples, iteration.iteration, 4);
    }
    // Fold pending boundary + medium segment before connections
    float seg = state.path_distance + medium_sample.t;
    state.d_vcm *= sqr(seg);
    ETX_VALIDATE(state.d_vcm);
    state.path_distance = 0.0f;

    const auto& med = scene.mediums[state.medium_index];

    // Sample phase function BEFORE explicit connections to match BDPT RNG order
    float3 w_o_smp = med.sample_phase_function(rnd_bsdf, state.ray.d);
    float pdf_fwd = med.phase_function(state.ray.d, w_o_smp);
    ETX_VALIDATE(pdf_fwd);
    float pdf_rev = med.phase_function(w_o_smp, state.ray.d);
    ETX_VALIDATE(pdf_rev);

    // Explicit connections from medium BEFORE sampling continuation (match surface ordering)
    if (med.enable_explicit_connections && (state.total_path_depth + 1 <= scene.max_path_length)) {
      if (options.connect_to_light()) {
        state.sampler.push_fixed(rnd_connection.x, rnd_connection.y, rnd_support.y);
        state.gathered += vcm_connect_to_light(scene, iteration, options, true, nullptr, medium_sample.pos, rt, state);
        state.sampler.pop_fixed();
      }
      if (options.connect_vertices()) {
        state.gathered += vcm_connect_to_light_path(scene, iteration, light_paths, light_vertices, options, true, nullptr, medium_sample.pos, rt, state);
      }
    }

    // Update MIS recurrences for continuation

    state.d_vc = (1.0f / pdf_fwd) * (state.d_vc * pdf_rev + state.d_vcm);
    ETX_VALIDATE(state.d_vc);
    state.d_vm = (1.0f / pdf_fwd) * (state.d_vm * pdf_rev + 0.0f);
    ETX_VALIDATE(state.d_vm);
    state.d_vcm = 1.0f / pdf_fwd;
    ETX_VALIDATE(state.d_vcm);

    // Advance ray to the sampled medium position/direction
    state.ray.o = medium_sample.pos;
    state.ray.d = w_o_smp;
    state.ray.max_t = kMaxFloat;
    state.ray.min_t = kRayEpsilon;

    // Medium scattering increases path length
    state.total_path_depth += 1;

    if (state.total_path_depth + 1 > scene.max_path_length)
      return false;

    return random_continue(state.total_path_depth, scene.random_path_termination, state.eta, state.sampler, state.throughput);
  }

  if (found_intersection == false) {
    vcm_cam_handle_miss(scene, options, intersection, state);
    return false;
  }

  if (vcm_handle_boundary_bsdf(scene, PathSource::Camera, intersection, state)) {
    // TODO : infinite loop
    return true;
  }

  const auto& mat = scene.materials[intersection.material_index];
  auto bsdf_data = BSDFData{state.spect, state.medium_index, PathSource::Camera, intersection, intersection.w_i};

  // Allocate samples AFTER confirming no medium event to match BDPT
  float2 rnd_bsdf = state.sampler.next_2d();        // BSDF sampling
  float2 rnd_connection = state.sampler.next_2d();  // Connections
  float2 rnd_support = state.sampler.next_2d();     // Support

  // Override with blue noise for first camera interaction (limit early iterations for parity with BDPT)
  if (options.blue_noise && (state.total_path_depth == 1) && (iteration.iteration < 256u)) {
    rnd_bsdf = sample_blue_noise(state.pixel_coord, scene.samples, iteration.iteration, 0);
    rnd_connection = sample_blue_noise(state.pixel_coord, scene.samples, iteration.iteration, 2);
    rnd_support = sample_blue_noise(state.pixel_coord, scene.samples, iteration.iteration, 4);
  }

  // Use fixed sample allocation for BSDF sampling
  state.sampler.push_fixed(rnd_bsdf.x, rnd_bsdf.y, rnd_support.x);
  auto bsdf_sample = bsdf::sample(bsdf_data, mat, scene, state.sampler);
  bool is_connectible = (bsdf_sample.properties & BSDFSample::Delta) == 0;  // Store connectibility like BDPT
  state.sampler.pop_fixed();

  vcm_update_camera_vcm(intersection, state);
  vcm_handle_direct_hit(scene, options, intersection, state);

  subsurface::Gather ss_gather = {};
  bool subsurface_path = (bsdf_sample.properties & BSDFSample::Diffuse) && (mat.subsurface.cls != SubsurfaceMaterial::Class::Disabled);
  bool subsurface_sampled = subsurface_path && (subsurface::gather(state.spect, scene, intersection, rt, state.sampler, ss_gather) == subsurface::GatherResult::Succeedded);

  if (is_connectible) {  // Use stored connectibility instead of calling is_delta with live sampler
    if (subsurface_sampled) {
      for (uint32_t i = 0; i < ss_gather.intersection_count; ++i) {
        Intersection out_isect = ss_gather.intersections[i];
        out_isect.material_index = scene.subsurface_exit_material;
        state.gathered += ss_gather.weights[i] * vcm_connect_to_light_path(scene, iteration, light_paths, light_vertices, options, false, &out_isect, {}, rt, state);
        // Use fixed samples for each subsurface connection
        state.sampler.push_fixed(rnd_connection.x, rnd_connection.y, rnd_support.y);
        state.gathered += ss_gather.weights[i] * vcm_connect_to_light(scene, iteration, options, false, &out_isect, {}, rt, state);
        state.sampler.pop_fixed();
      }
    } else {
      state.gathered += vcm_connect_to_light_path(scene, iteration, light_paths, light_vertices, options, false, &intersection, {}, rt, state);
      // Use fixed samples for light connection
      state.sampler.push_fixed(rnd_connection.x, rnd_connection.y, rnd_support.y);
      state.gathered += vcm_connect_to_light(scene, iteration, options, false, &intersection, {}, rt, state);
      state.sampler.pop_fixed();
    }
  }

  if (subsurface_sampled) {
    state.throughput *= ss_gather.weights[ss_gather.selected_intersection] * ss_gather.selected_sample_weight;
    intersection = ss_gather.intersections[ss_gather.selected_intersection];
    intersection.material_index = scene.subsurface_exit_material;
    // Use the already allocated samples for subsurface cosine distribution
    state.sampler.push_fixed(rnd_bsdf.x, rnd_bsdf.y, 0.0f);
    bsdf_sample.w_o = sample_cosine_distribution(state.sampler.next_2d(), intersection.nrm, 1.0f);
    state.sampler.pop_fixed();
    bsdf_sample.pdf = fabsf(dot(bsdf_sample.w_o, intersection.nrm)) / kPi;
    bsdf_sample.eta = 1.0f;
    bsdf_data = BSDFData{state.spect, state.medium_index, PathSource::Camera, intersection, intersection.w_i};
  }

  if (is_connectible && options.merge_vertices() && (state.total_path_depth + 1 <= scene.max_path_length)) {
    state.merged += spatial_grid.gather(scene, state, options, intersection, iteration.vc_weight);
  }

  if (subsurface_path && (subsurface_sampled == false)) {
    return false;
  }

  return vcm_next_ray(scene, PathSource::Camera, options, state, iteration, intersection, bsdf_data, bsdf_sample, subsurface_sampled);
}

struct ETX_ALIGNED LightStepResult {
  VCMLightVertex vertex_to_add = {};
  SpectralResponse values_to_splat[subsurface::kTotalIntersections] = {};
  float2 splat_uvs[subsurface::kTotalIntersections] = {};
  uint32_t splat_count = 0;
  bool add_vertex = false;
  bool continue_tracing = false;
};

ETX_GPU_CODE LightStepResult vcm_light_step(const Scene& scene, const Camera& camera, const VCMIteration& iteration, const VCMOptions& options, const uint32_t path_index,
  VCMPathState& state, const Raytracing& rt) {
  Intersection intersection = {};
  bool found_intersection = rt.trace(scene, state.ray, intersection, state.sampler);

  LightStepResult result = {};
  Medium::Sample medium_sample = vcm_try_sampling_medium(scene, state, found_intersection ? intersection.t : kMaxFloat);
  if (medium_sample.sampled_medium()) {
    // Allocate samples AFTER medium sampling to match BDPT
    float2 rnd_bsdf = state.sampler.next_2d();        // Phase function sampling
    float2 rnd_connection = state.sampler.next_2d();  // Camera connection
    float2 rnd_support = state.sampler.next_2d();     // Support operations
    // Fold pending boundary + medium segment before connection/MIS
    float seg = state.path_distance + medium_sample.t;
    state.d_vcm *= sqr(seg);
    ETX_VALIDATE(state.d_vcm);
    state.path_distance = 0.0f;

    const auto& med = scene.mediums[state.medium_index];
    // Register medium light vertex for camera connections BEFORE sampling continuation
    if (options.connect_vertices() && (state.total_path_depth + 1 <= scene.max_path_length)) {
      VCMLightVertex v = {};
      v.throughput = state.throughput;
      v.w_i = state.ray.d;  // incoming direction before scattering
      v.d_vcm = state.d_vcm;
      v.d_vc = state.d_vc;
      v.d_vm = state.d_vm;
      v.pos = medium_sample.pos;
      v.nrm = {0.0f, 0.0f, 0.0f};
      v.triangle_index = kInvalidIndex;
      v.material_index = kInvalidIndex;
      v.medium_index = state.medium_index;
      v.is_medium = true;
      v.path_length = state.total_path_depth;
      v.path_index = path_index;
      result.add_vertex = true;
      result.vertex_to_add = v;
    }

    // Connect to camera from medium BEFORE sampling continuation (align with BDPT)
    if (options.connect_to_camera() && med.enable_explicit_connections && (state.total_path_depth + 1 <= scene.max_path_length)) {
      float2 uv = {};
      state.sampler.push_fixed(rnd_connection.x, rnd_connection.y, rnd_support.y);
      auto value = vcm_connect_to_camera(rt, scene, camera, iteration, options, true, nullptr, medium_sample.pos, state, uv);
      state.sampler.pop_fixed();
      ETX_VALIDATE(value);
      if (value.maximum() > kEpsilon) {
        result.values_to_splat[0] = value;
        result.splat_uvs[0] = uv;
        result.splat_count = 1;
      }
    }

    // Now sample phase and update MIS recurrences for continuation
    float3 w_i = state.ray.d;
    float3 w_o_smp = med.sample_phase_function(rnd_bsdf, w_i);
    float pdf_fwd = med.phase_function(w_i, w_o_smp);
    ETX_VALIDATE(pdf_fwd);
    float pdf_rev = med.phase_function(w_o_smp, w_i);
    ETX_VALIDATE(pdf_rev);

    state.d_vc = (1.0f / pdf_fwd) * (state.d_vc * pdf_rev + state.d_vcm);
    ETX_VALIDATE(state.d_vc);
    state.d_vm = (1.0f / pdf_fwd) * (state.d_vm * pdf_rev + 0.0f);
    ETX_VALIDATE(state.d_vm);
    state.d_vcm = 1.0f / pdf_fwd;
    ETX_VALIDATE(state.d_vcm);

    // Advance ray
    state.ray.o = medium_sample.pos;
    state.ray.d = w_o_smp;
    state.ray.max_t = kMaxFloat;
    state.ray.min_t = kRayEpsilon;

    state.total_path_depth += 1;

    result.continue_tracing =
      (state.total_path_depth + 1 <= scene.max_path_length) && random_continue(state.total_path_depth, scene.random_path_termination, state.eta, state.sampler, state.throughput);

    return result;
  }

  if (found_intersection == false) {
    return result;
  }

  if (vcm_handle_boundary_bsdf(scene, PathSource::Light, intersection, state)) {
    result.continue_tracing = true;
    return result;
  }

  const auto& mat = scene.materials[intersection.material_index];
  auto bsdf_data = BSDFData{state.spect, state.medium_index, PathSource::Light, intersection, intersection.w_i};

  // Allocate samples when no medium event (post medium test) to match BDPT
  float2 rnd_bsdf = state.sampler.next_2d();        // BSDF sampling
  float2 rnd_connection = state.sampler.next_2d();  // Camera connection
  float2 rnd_support = state.sampler.next_2d();     // Support operations

  // Use fixed sample allocation for BSDF sampling
  state.sampler.push_fixed(rnd_bsdf.x, rnd_bsdf.y, rnd_support.x);
  auto bsdf_sample = bsdf::sample(bsdf_data, mat, scene, state.sampler);
  bool is_connectible = (bsdf_sample.properties & BSDFSample::Delta) == 0;  // Store connectibility like BDPT
  state.sampler.pop_fixed();
  ETX_VALIDATE(bsdf_sample.weight);

  vcm_update_light_vcm(intersection, state);

  subsurface::Gather ss_gather = {};
  bool subsurface_path = (bsdf_sample.properties & BSDFSample::Diffuse) && (mat.subsurface.cls != SubsurfaceMaterial::Class::Disabled);
  bool subsurface_sampled = subsurface_path && (subsurface::gather(state.spect, scene, intersection, rt, state.sampler, ss_gather) == subsurface::GatherResult::Succeedded);

  if (is_connectible) {  // Use stored connectibility instead of calling is_delta with live sampler
    result.add_vertex = true;
    result.vertex_to_add = {state, intersection, path_index};
    result.splat_count = 0;

    if (options.connect_to_camera() && (state.total_path_depth + 1 <= scene.max_path_length)) {
      if (subsurface_sampled) {
        for (uint32_t i = 0; i < ss_gather.intersection_count; ++i) {
          // Use fixed samples for each subsurface camera connection
          state.sampler.push_fixed(rnd_connection.x, rnd_connection.y, rnd_support.y);
          Intersection out_isect = ss_gather.intersections[i];
          out_isect.material_index = scene.subsurface_exit_material;
          auto value = vcm_connect_to_camera(rt, scene, camera, iteration, options, false, &out_isect, {}, state, result.splat_uvs[result.splat_count]);
          state.sampler.pop_fixed();
          ETX_VALIDATE(value);
          if (value.maximum() > kEpsilon) {
            float ss_scale = ss_gather.weights[i].average() / ss_gather.total_weight;
            result.values_to_splat[result.splat_count] = ss_gather.weights[i] * value;  // * ss_scale;
            result.splat_count++;
          }
        }
      } else {
        // Use fixed samples for camera connection
        state.sampler.push_fixed(rnd_connection.x, rnd_connection.y, rnd_support.y);
        auto value = vcm_connect_to_camera(rt, scene, camera, iteration, options, false, &intersection, {}, state, result.splat_uvs[0]);
        state.sampler.pop_fixed();
        ETX_VALIDATE(value);
        if (value.maximum() > kEpsilon) {
          result.values_to_splat[0] = value;
          result.splat_count = 1;
        }
      }
    }
  }

  if (subsurface_sampled) {
    state.throughput *= ss_gather.weights[ss_gather.selected_intersection] * ss_gather.selected_sample_weight;
    ETX_VALIDATE(state.throughput);
    intersection = ss_gather.intersections[ss_gather.selected_intersection];
    // Use the already allocated samples for subsurface cosine distribution
    state.sampler.push_fixed(rnd_bsdf.x, rnd_bsdf.y, 0.0f);
    bsdf_sample.w_o = sample_cosine_distribution(state.sampler.next_2d(), intersection.nrm, 1.0f);
    state.sampler.pop_fixed();
    bsdf_sample.pdf = fabsf(dot(bsdf_sample.w_o, intersection.nrm)) / kPi;
    bsdf_sample.eta = 1.0f;
    bsdf_data = BSDFData{state.spect, state.medium_index, PathSource::Light, intersection, intersection.w_i};
  }

  if (subsurface_path && (subsurface_sampled == false)) {
    return result;
  }

  if (vcm_next_ray(scene, PathSource::Light, options, state, iteration, intersection, bsdf_data, bsdf_sample, subsurface_sampled) == false) {
    return result;
  }

  result.continue_tracing = state.total_path_depth + 1u < scene.max_path_length;
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
