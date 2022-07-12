#include <etx/rt/shared/optix.hxx>
#include <etx/rt/shared/vcm_shared.hxx>

using namespace etx;

static __constant__ VCMGlobal global;

ETX_GPU_CODE void finish_ray(VCMIteration& iteration, VCMPathState& state) {
  atomicAdd(&iteration.terminated_paths, 1u);

  state.merged *= iteration.vm_normalization;
  state.merged += (state.gathered / spectrum::sample_pdf()).to_xyz();

  uint32_t x = state.index % global.scene.camera.image_size.x;
  uint32_t y = state.index / global.scene.camera.image_size.x;
  uint32_t c = x + (global.scene.camera.image_size.y - 1 - y) * global.scene.camera.image_size.x;

  float4 old_value = global.camera_image[c];
  float4 new_value = {state.merged.x, state.merged.y, state.merged.z, 1.0f};

  float t = float(iteration.iteration) / float(iteration.iteration + 1);
  global.camera_image[c] = lerp(new_value, old_value, t);
}

ETX_GPU_CODE void continue_tracing(VCMIteration& iteration, VCMPathState& state, uint32_t path_len_offset) {
  state.path_length += path_len_offset;
  if (state.path_length < global.options.max_depth) {
    int i = atomicAdd(&iteration.active_paths, 1u);
    global.output_state[i] = state;
  } else {
    finish_ray(iteration, state);
  }
}

RAYGEN(main) {
  uint3 idx = optixGetLaunchIndex();
  auto& state = global.input_state[idx.x];
  auto& iteration = *global.iteration;

  state.state = 1u;

  Raytracing rt;
  Intersection intersection;
  bool found_intersection = rt.trace(global.scene, state.ray, intersection, state.sampler);

  Medium::Sample medium_sample = vcm_try_sampling_medium(global.scene, state, found_intersection ? intersection.t : kMaxFloat);

  if (medium_sample.sampled_medium()) {
    vcm_handle_sampled_medium(global.scene, medium_sample, state);
    continue_tracing(iteration, state, 1u);
    return;
  }

  if (found_intersection == false) {
    vcm_cam_handle_miss(global.scene, intersection, global.options, state);
    finish_ray(iteration, state);
    return;
  }

  state.path_distance += intersection.t;
  const auto& tri = global.scene.triangles[intersection.triangle_index];
  const auto& mat = global.scene.materials[tri.material_index];

  if (vcm_handle_boundary_bsdf(global.scene, mat, intersection, PathSource::Camera, state)) {
    continue_tracing(iteration, state, 0u);
    return;
  }

  vcm_update_camera_vcm(intersection, state);
  vcm_handle_direct_hit(global.scene, tri, intersection, global.options, state);

  if (bsdf::is_delta(mat, intersection.tex, global.scene, state.sampler) == false) {
    vcm_connect_to_light(global.scene, tri, mat, intersection, iteration, global.options, rt, state);
    vcm_connect_to_light_path(global.scene, tri, mat, intersection, iteration, global.light_paths[state.index], global.light_vertices, global.options, rt, state);
  }

  state.merged += global.spatial_grid.gather(global.scene, state, global.light_vertices.a, intersection, global.options, iteration.vc_weight);

  if (vcm_next_ray(global.scene, PathSource::Camera, intersection, global.options.rr_start, state, iteration) == false) {
    finish_ray(iteration, state);
    return;
  }

  continue_tracing(iteration, state, 1u);
}
