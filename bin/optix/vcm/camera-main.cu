#include <etx/rt/shared/optix.hxx>
#include <etx/rt/shared/vcm_shared.hxx>

using namespace etx;

static __constant__ VCMGlobal global;

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

ETX_GPU_CODE void continue_tracing(VCMIteration& iteration, const VCMPathState& state) {
  int i = atomicAdd(&iteration.active_paths, 1u);
  global.output_state[i] = state;
}

RAYGEN(main) {
  Raytracing rt;
  uint3 idx = optixGetLaunchIndex();
  auto& state = global.input_state[idx.x];
  auto& iteration = *global.iteration;

  const auto& light_path = global.light_paths[state.global_index];
  if (vcm_camera_step(global.scene, iteration, global.options, light_path, global.light_vertices, state, rt, global.spatial_grid)) {
    continue_tracing(iteration, state);
  } else {
    finish_ray(iteration, state);
  }
}
