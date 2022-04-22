#include <etx/rt/shared/optix.hxx>
#include <etx/rt/shared/path_tracing_shared.hxx>

using namespace etx;

static __constant__ PathTracingGPUData global;

RAYGEN(main) {
  uint3 idx = optixGetLaunchIndex();
  uint3 dim = optixGetLaunchDimensions();
  uint32_t index = idx.x + idx.y * dim.x;

  Sampler smp(index, 0);

  float2 uv = get_jittered_uv(smp, uint2{idx.x, dim.y - 1u - idx.y}, uint2{dim.x, dim.y});
  auto ray = generate_ray(smp, global.scene, uv);

  uint32_t p = 0u;
  optixTrace(OptixTraversableHandle(global.acceleration_structure), ray.o, ray.d, ray.min_t, ray.max_t, 0.0f, //
    OptixVisibilityMask(255), OptixRayFlags(OPTIX_RAY_FLAG_DISABLE_ANYHIT), 0u, 0u, 0u, p);

  //
  float t = to_float(p) / 5.0f;
  float3 rgb = (p == 0) ? float3{} : float3{t*t * (uv.x * 0.5f + 0.5f), t*t * (uv.y * 0.5f + 0.5f), 1.0};
  float3 xyz = spectrum::rgb_to_xyz(rgb);
  global.output[index] = {xyz.x, xyz.y, xyz.z, 1.0f};
  // */
}

CLOSEST_HIT(main_closest_hit) {
  float d = optixGetRayTmax();
  optixSetPayload_0(to_uint(d));
}

MISS(main_miss) {
}

CLOSEST_HIT(env_closest_hit) {
}

MISS(env_miss) {
}
