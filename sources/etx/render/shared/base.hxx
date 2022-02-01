#pragma once

#include <etx/core/handle.hxx>

#if defined(__NVCC__)
#define ETX_NVCC_COMPILER 1
#else
#define ETX_NVCC_COMPILER 0
#endif

#if (ETX_NVCC_COMPILER)
#define ETX_GPU_CODE inline __device__
#define ETX_GPU_DATA __device__
#define ETX_INIT_WITH(S)
#else
#define ETX_GPU_CODE inline
#define ETX_GPU_DATA
#define ETX_INIT_WITH(S) = S
#endif

#define ETX_EMPTY_INIT ETX_INIT_WITH({})

#define ETX_RENDER_BASE_INCLUDED 1
#include <etx/render/shared/math.hxx>
#undef ETX_RENDER_BASE_INCLUDED

#define ETX_VALIDATE(value)

namespace etx {

template <class T>
struct alignas(16) ArrayView {
  T* a ETX_EMPTY_INIT;
  uint64_t count ETX_EMPTY_INIT;

  ArrayView() = default;

  ETX_GPU_CODE ArrayView(T* p, uint64_t c)
    : a(p)
    , count(c) {
  }

  ETX_GPU_CODE const T& operator[](uint64_t i) const {
    ETX_ASSERT(count > 0);
    ETX_ASSERT(a != nullptr);
    ETX_ASSERT(i < count);
    return a[i];
  }

  ETX_GPU_CODE T& operator[](uint64_t i) {
    ETX_ASSERT(count > 0);
    ETX_ASSERT(a != nullptr);
    ETX_ASSERT(i < count);
    return a[i];
  }
};

}  // namespace etx
