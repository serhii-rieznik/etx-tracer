#include <etx/render/shared/scene.hxx>
#include <etx/rt/shared/optix.hxx>
#include <etx/rt/shared/path_tracing_shared.hxx>

using namespace etx;

static __constant__ PTGPUData global;

RAYGEN(main) {
  uint3 idx = optixGetLaunchIndex();
  uint3 dim = optixGetLaunchDimensions();
  uint32_t index = idx.x + idx.y * dim.x;

  Sampler smp(index, 0);

  auto payload = make_ray_payload(smp, global.scene, {idx.x, dim.y - idx.y - 1u}, {dim.x, dim.y});

  Raytracing rt;
  Intersection intersection;
  if (rt.trace(global.scene, payload.ray, intersection, smp)) {
    handle_hit_ray(smp, global.scene, intersection, global.options, rt, payload);
    float3 xyz = payload.throughput.to_xyz();
      // spectrum::rgb_to_xyz(intersection.barycentric);
    global.output[index] = {xyz.x, xyz.y, xyz.z, 1.0f};
  } else {
    global.output[index] = {};
  }
}
