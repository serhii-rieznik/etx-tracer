#include <etx/render/shared/scene.hxx>
#include <etx/rt/shared/optix.hxx>
#include <etx/rt/shared/path_tracing_shared.hxx>

using namespace etx;

static __constant__ PTGPUData global;

RAYGEN(raygen) {
  uint3 idx = optixGetLaunchIndex();
  uint3 dim = optixGetLaunchDimensions();

  uint32_t index = idx.x + idx.y * dim.x;
  global.payloads[index] = make_ray_payload(global.scene, {idx.x, dim.y - idx.y - 1u}, {dim.x, dim.y}, 0u);
  global.output[index] = {};
}

RAYGEN(main) {
  uint3 idx = optixGetLaunchIndex();
  uint3 dim = optixGetLaunchDimensions();
  uint32_t index = idx.x + idx.y * dim.x;

  auto& payload = global.payloads[index];
  if (payload.iteration > global.options.iterations)
    return;

  Raytracing rt;
  uint32_t path_len = max(1u, global.options.path_per_iteration);

  for (uint32_t i = 0; (i < path_len); ++i) {
    bool continue_iteration = run_path_iteration(global.scene, global.options, rt, payload);
    if (continue_iteration == false) {
      float3 xyz = (payload.accumulated / spectrum::sample_pdf()).to_xyz();

      float t = float(payload.iteration) / float(payload.iteration + 1u);
      float3 new_value = lerp(xyz, to_float3(global.output[index]), t);
      global.output[index] = to_float4(new_value);

      payload = make_ray_payload(global.scene, {idx.x, dim.y - idx.y - 1u}, {dim.x, dim.y}, payload.iteration + 1);
      break;
    }
  }
}
