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

RAYGEN(main) {
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
