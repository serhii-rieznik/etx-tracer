#pragma once

#include <cuda.h>
#include <optix.h>

#include <etx/render/shared/sampler.hxx>
#include <etx/render/shared/scene.hxx>

#define RAYGEN(name) extern "C" __global__ void __raygen__##name()
#define ANY_HIT(name) extern "C" __global__ void __anyhit__##name()
#define CLOSEST_HIT(name) extern "C" __global__ void __closesthit__##name()
#define MISS(name) extern "C" __global__ void __miss__##name()
#define EXCEPTION(name) extern "C" __global__ void __exception__##name()

#if (ETX_NVCC_COMPILER == 0)
// Fixes issue with InteliSense
#define __global__
#define __constant__
uint3 optixGetLaunchIndex();
uint3 optixGetLaunchDimensions();
uint32_t optixGetPayload_0();
uint32_t optixGetPayload_1();
uint32_t optixGetPrimitiveIndex();
float2 optixGetTriangleBarycentrics();
float optixGetRayTmax();
void optixIgnoreIntersection();
void optixTrace(...);
template <class T>
T atomicAdd(T*, T);
#endif

namespace etx {

struct Scene;

struct ETX_ALIGNED Raytracing {
  struct TraceData {
    const Scene& scene;
    Intersection& i;
    Sampler& smp;

    ETX_GPU_CODE TraceData(const Scene& as, Intersection& ai, Sampler& asmp)
      : scene(as)
      , i(ai)
      , smp(asmp) {
    }
  };

  static ETX_GPU_CODE TraceData* unpack_data(uint64_t ptr_lo, uint64_t ptr_hi) {
    uint64_t ptr = ptr_lo | (ptr_hi << 32llu);
    return reinterpret_cast<TraceData*>(ptr);
  }

  ETX_GPU_CODE bool trace(const Scene& scene, const Ray& ray, Intersection& i, Sampler& smp) const {
    ETX_CHECK_FINITE(ray.d);
    TraceData d = {scene, i, smp};

    uint64_t ptr = reinterpret_cast<uint64_t>(&d);
    uint32_t ptr_lo = static_cast<uint32_t>((ptr & 0x00000000ffffffff) >> 0llu);
    uint32_t ptr_hi = static_cast<uint32_t>((ptr & 0xffffffff00000000) >> 32llu);

    i.t = -kMaxFloat;
    optixTrace(OptixTraversableHandle(scene.acceleration_structure), ray.o, ray.d, ray.min_t, ray.max_t, 0.0f,  //
      OptixVisibilityMask(255), OptixRayFlags(OPTIX_RAY_FLAG_ENFORCE_ANYHIT), 0u, 0u, 0u, ptr_lo, ptr_hi);

    if (i.t < 0.0f)
      return false;

    const auto& tri = scene.triangles[i.triangle_index];
    lerp_vertex(i, scene.vertices, tri, i.barycentric);

    const auto& mat = scene.materials[tri.material_index];
    if ((mat.normal_image_index != kInvalidIndex) && (mat.normal_scale > 0.0f)) {
      auto sampled_normal = scene.images[mat.normal_image_index].evaluate_normal(i.tex, mat.normal_scale);
      float3x3 from_local = {
        float3{i.tan.x, i.tan.y, i.tan.z},
        float3{i.btn.x, i.btn.y, i.btn.z},
        float3{i.nrm.x, i.nrm.y, i.nrm.z},
      };
      i.nrm = normalize(from_local * sampled_normal);
      i.tan = normalize(i.tan - i.nrm * dot(i.tan, i.nrm));
      i.btn = normalize(cross(i.nrm, i.tan));
    }

    i.w_i = ray.d;

    return true;
  }
};

}  // namespace etx

CLOSEST_HIT(main_closest_hit) {
  auto data = etx::Raytracing::unpack_data(optixGetPayload_0(), optixGetPayload_1());

  data->i.barycentric = etx::barycentrics(optixGetTriangleBarycentrics());
  data->i.triangle_index = optixGetPrimitiveIndex();
  data->i.t = optixGetRayTmax();
}

ANY_HIT(main_any_hit) {
  auto data = etx::Raytracing::unpack_data(optixGetPayload_0(), optixGetPayload_1());
  const auto& tri = data->scene.triangles[optixGetPrimitiveIndex()];
  const auto& mat = data->scene.materials[tri.material_index];
  auto bc = etx::barycentrics(optixGetTriangleBarycentrics());
  auto uv = lerp_uv(data->scene.vertices, tri, bc);

  if (etx::bsdf::continue_tracing(mat, uv, data->scene, data->smp)) {
    optixIgnoreIntersection();
  }
}

MISS(main_miss) {
}
