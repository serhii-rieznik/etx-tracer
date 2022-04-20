#pragma once

#include <optix.h>
#include <etx/render/shared/sampler.hxx>

#define RAYGEN(name) extern "C" __global__ void __raygen__##name()
#define ANY_HIT(name) extern "C" __global__ void __anyhit__##name()
#define CLOSEST_HIT(name) extern "C" __global__ void __closesthit__##name()
#define MISS(name) extern "C" __global__ void __miss__##name()
#define EXCEPTION(name) extern "C" __global__ void __exception__##name()

namespace etx {}
