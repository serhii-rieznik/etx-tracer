#include <etx/rt/shared/optix.hxx>
#include <etx/rt/shared/vcm_shared.hxx>

using namespace etx;

static __constant__ VCMGlobal global;

ETX_GPU_CODE void continue_tracing(VCMIteration& iteration, VCMPathState& state) {
  int i = atomicAdd(&iteration.active_paths, 1u);
  global.output_state[i] = state;
}

ETX_GPU_CODE void push_light_vertex(VCMIteration& iteration, const VCMLightVertex& v) {
  uint32_t i = atomicAdd(&iteration.light_vertices, 1u);
  global.light_vertices[i] = v;
}

ETX_GPU_CODE void finish_ray(VCMIteration& iteration, VCMPathState& state) {
  float3 result = state.merged * iteration.vm_normalization + (state.gathered / spectrum::sample_pdf()).to_xyz();

  uint32_t x = state.global_index % global.scene.camera.image_size.x;
  uint32_t y = state.global_index / global.scene.camera.image_size.x;
  uint32_t c = x + (global.scene.camera.image_size.y - 1 - y) * global.scene.camera.image_size.x;
  float4& current = global.camera_iteration_image[c];
  atomicAdd(&current.x, result.x);
  atomicAdd(&current.y, result.y);
  atomicAdd(&current.z, result.z);
}

ETX_GPU_CODE void project(const float2& ndc_coord, const float3& value) {
  float2 uv = ndc_coord * 0.5f + 0.5f;
  uint2 dim = global.scene.camera.image_size;
  uint32_t ax = static_cast<uint32_t>(uv.x * float(dim.x));
  uint32_t ay = static_cast<uint32_t>(uv.y * float(dim.y));
  if ((ax < dim.x) && (ay < dim.y)) {
    uint32_t i = ax + ay * dim.x;
    float4& current = global.light_iteration_image[i];
    atomicAdd(&current.x, value.x);
    atomicAdd(&current.y, value.y);
    atomicAdd(&current.z, value.z);
  }
}

ETX_GPU_CODE void continue_tracing(VCMIteration& iteration, const VCMPathState& state) {
  int i = atomicAdd(&iteration.active_paths, 1u);
  global.output_state[i] = state;
}

RAYGEN(light_main) {
  uint3 idx = optixGetLaunchIndex();
  auto& state = global.input_state[idx.x];
  auto& iteration = *global.iteration;

  Raytracing rt = {};
  auto step_result = vcm_light_step(global.scene, iteration, global.options, state.global_index, state, rt);

  if (step_result.add_vertex) {
    push_light_vertex(iteration, step_result.vertex_to_add);
  }

  project(step_result.splat_uv, step_result.value_to_splat);

  if (step_result.continue_tracing) {
    continue_tracing(iteration, state);
  }
}

RAYGEN(camera_main) {
  uint3 idx = optixGetLaunchIndex();
  auto& state = global.input_state[idx.x];
  const auto& scene = global.scene;
  const auto& options = global.options;

  Raytracing rt;
  bool found_intersection = rt.trace(scene, state.ray, state.intersection, state.sampler);

  state.clear_ray_action();

  Medium::Sample medium_sample = vcm_try_sampling_medium(scene, state);
  if (medium_sample.sampled_medium()) {
    bool continue_ray = vcm_handle_sampled_medium(scene, medium_sample, options, state);
    state.continue_ray(continue_ray);
  } else if (found_intersection == false) {
    vcm_cam_handle_miss(scene, options, state);
    state.continue_ray(false);
  } else if (vcm_handle_boundary_bsdf(scene, PathSource::Camera, state)) {
    state.continue_ray(true);
  }
}

RAYGEN(camera_to_light) {
  uint3 idx = optixGetLaunchIndex();
  auto& state = global.input_state[idx.x];
  if (state.ray_action_set()) {
    return;
  }

  const auto& scene = global.scene;
  const auto& options = global.options;
  auto& iteration = *global.iteration;

  Raytracing rt;
  vcm_connect_to_light(scene, iteration, options, rt, state);
}

RAYGEN(camera_to_vertices) {
  uint3 idx = optixGetLaunchIndex();
  auto& state = global.input_state[idx.x];
  if (state.ray_action_set()) {
    return;
  }

  const auto& options = global.options;
  const auto& light_paths = global.light_paths;
  const auto& light_path = light_paths[state.global_index];
  uint32_t i = idx.y;
  if ((i >= light_path.count) || (state.total_path_depth + i + 2 > options.max_depth)) {
    return;
  }

  const auto& scene = global.scene;
  const auto& light_vertices = global.light_vertices;
  auto& iteration = *global.iteration;

  Raytracing rt;

  float3 target_position = {};
  SpectralResponse value = {};
  bool connected = vcm_connect_to_light_vertex(scene, state.spect, state, light_vertices[light_path.index + i],  //
    options, iteration.vm_weight, state.medium_index, target_position, value);

  if (connected == false) {
    return;
  }

  const auto& tri = scene.triangles[state.intersection.triangle_index];
  float3 p0 = shading_pos(scene.vertices, tri, state.intersection.barycentric, normalize(target_position - state.intersection.pos));
  auto tr = value * transmittance(state.spect, state.sampler, p0, target_position, state.medium_index, scene, rt);

  atomicAdd(&state.gathered.components.x, tr.components.x);
  atomicAdd(&state.gathered.components.y, tr.components.y);
  atomicAdd(&state.gathered.components.z, tr.components.z);
}
