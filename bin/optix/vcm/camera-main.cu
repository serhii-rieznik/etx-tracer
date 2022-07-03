#include <etx/rt/shared/optix.hxx>
#include <etx/rt/shared/vcm_shared.hxx>

using namespace etx;

static __constant__ VCMGlobal global;

ETX_GPU_CODE void continue_tracing(VCMIteration& iteration, const VCMPathState& state) {
  int i = atomicAdd(&iteration.active_camera_paths, 1);
  global.output_state[i] = state;
}

ETX_GPU_CODE void finish_ray(VCMIteration& iteration, VCMPathState& state) {
  state.merged *= iteration.vm_normalization;
  state.merged += (state.gathered / spectrum::sample_pdf()).to_xyz();

  uint32_t x = state.index % global.scene.camera.image_size.y;
  uint32_t y = state.index / global.scene.camera.image_size.y;
  uint32_t c = x + (global.scene.camera.image_size.y - 1 - y) * global.scene.camera.image_size.x;

  float4 old_value = global.camera_image[c];
  float4 new_value = {state.merged.x, state.merged.y, state.merged.z, 1.0f};

  float t = float(iteration.iteration) / float(iteration.iteration + 1);
  global.camera_image[c] = lerp(new_value, old_value, t);
}

RAYGEN(main) {
  uint3 idx = optixGetLaunchIndex();
  uint3 dim = optixGetLaunchDimensions();
  uint32_t index = idx.x + idx.y * dim.x;
  auto& state = global.input_state[index];
  auto& iteration = *global.iteration;

  Raytracing rt;
  Intersection intersection;
  bool found_intersection = rt.trace(global.scene, state.ray, intersection, state.sampler);

  /*
  Medium::Sample medium_sample = vcm_try_sampling_medium(global.scene, state, found_intersection ? intersection.t : kMaxFloat);

  if (medium_sample.sampled_medium()) {
    vcm_handle_sampled_medium(global.scene, medium_sample, state);
    continue_tracing(iteration, state);
    return;
  }
  // */

  if (found_intersection == false) {
    vcm_cam_handle_miss(global.scene, intersection, state);
    finish_ray(iteration, state);
    return;
  }

  const auto& tri = global.scene.triangles[intersection.triangle_index];
  const auto& mat = global.scene.materials[tri.material_index];

  if (vcm_handle_boundary_bsdf(global.scene, mat, intersection, PathSource::Camera, state)) {
    continue_tracing(iteration, state);
    return;
  }

  vcm_update_camera_vcm(intersection, state);
  vcm_handle_direct_hit(global.scene, tri, intersection, state);
  vcm_connect_to_light(global.scene, tri, mat, intersection, iteration, rt, state);
  finish_ray(iteration, state);
  return;

  /*
  if (found_intersection == false) {
    vcm_cam_handle_miss(global.scene, intersection, state);
  }

  state.path_distance += intersection.t;

  const auto& tri = global.scene.triangles[intersection.triangle_index];
  const auto& mat = global.scene.materials[tri.material_index];

  vcm_update_camera_vcm(intersection, state);
  vcm_handle_direct_hit(global.scene, tri, intersection, state);


  if (bsdf::is_delta(mat, intersection.tex, global.scene, state.sampler) == false) {
    vcm_connect_to_light(global.scene, tri, mat, intersection, iteration, rt, state);
    // vcm_connect_to_light_path(scene, tri, mat, intersection, vcm_iteration, light_path, light_vertices, rt, state);
  }

  if (vcm_next_ray(global.scene, PathSource::Camera, intersection, kVCMRRStart, state, iteration) == false) {
    return;
  }

  state.path_length += 1;
  continue_tracing(iteration, state);
  */
}
