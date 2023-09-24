#pragma once

#include <optix.h>

#include <etx/render/shared/sampler.hxx>
#include <etx/render/shared/scene.hxx>

#define RAYGEN(name)      extern "C" __global__ void __raygen__##name()
#define ANY_HIT(name)     extern "C" __global__ void __anyhit__##name()
#define CLOSEST_HIT(name) extern "C" __global__ void __closesthit__##name()
#define MISS(name)        extern "C" __global__ void __miss__##name()
#define EXCEPTION(name)   extern "C" __global__ void __exception__##name()

#if (ETX_NVCC_COMPILER == 0)
// Fixes issue with InteliSense
# define __global__
# define __constant__
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

namespace etx {}  // namespace etx
