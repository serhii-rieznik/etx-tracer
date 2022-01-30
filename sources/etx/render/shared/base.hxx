#pragma once

#include <etx/core/handle.hxx>

#if defined(__NVCC__)
#define ETX_NVCC_COMPILER 1
#else
#define ETX_NVCC_COMPILER 0
#endif

#if (ETX_NVCC_COMPILER)
#define ETX_GPU_CODE inline __device__
#define ETX_INIT_WITH(S)
#else
#define ETX_GPU_CODE inline
#define ETX_INIT_WITH(S) = S
#endif

#define ETX_EMPTY_INIT ETX_INIT_WITH({})

#define ETX_RENDER_BASE_INCLUDED 1
#include <etx/render/shared/math.hxx>
#undef ETX_RENDER_BASE_INCLUDED

namespace etx {}
