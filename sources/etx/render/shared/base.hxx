#pragma once

#include <etx/core/debug.hxx>

#define ETX_ALIGNED alignas(16)

#define ETX_EMPTY_INIT ETX_INIT_WITH({})

#define ETX_RENDER_BASE_INCLUDED 1
#include <etx/render/shared/math.hxx>
#undef ETX_RENDER_BASE_INCLUDED

#define ETX_FORCE_VALIDATION 0

#if (ETX_DEBUG || ETX_FORCE_VALIDATION)

# define ETX_VALIDATE(VALUE)                                     \
   do {                                                          \
     if (valid_value((VALUE)) == false) {                        \
       if (ETX_ASSERT_ATOMIC_CHECK()) {                          \
         print_invalid_value(#VALUE, VALUE, __FILE__, __LINE__); \
         ETX_DEBUG_BREAK();                                      \
       }                                                         \
     }                                                           \
   } while (0)

# define ETX_CHECK_FINITE(VALUE)                                 \
   do {                                                          \
     if (isfinite((VALUE)) == false) {                           \
       if (ETX_ASSERT_ATOMIC_CHECK()) {                          \
         print_invalid_value(#VALUE, VALUE, __FILE__, __LINE__); \
         ETX_DEBUG_BREAK();                                      \
       }                                                         \
     }                                                           \
   } while (0)

#else

# define ETX_VALIDATE(VALUE) \
   do {                      \
   } while (0)

# define ETX_CHECK_FINITE(VALUE) \
   do {                          \
   } while (0)

#endif

namespace etx {

template <class T>
struct ETX_ALIGNED ArrayView {
  T* a ETX_EMPTY_INIT;
  uint64_t count ETX_EMPTY_INIT;

  ArrayView() = default;

  ETX_GPU_CODE ArrayView(T* p, uint64_t c)
    : a(p)
    , count(c) {
  }

  ETX_GPU_CODE const T& operator[](uint64_t i) const {
    ETX_ASSERT_GREATER(count, 0llu);
    ETX_ASSERT(a != nullptr);
    ETX_ASSERT_LESS(i, count);
    return a[i];
  }

  ETX_GPU_CODE T& operator[](uint64_t i) {
    ETX_ASSERT_GREATER(count, 0llu);
    ETX_ASSERT(a != nullptr);
    ETX_ASSERT_LESS(i, count);
    return a[i];
  }

  ETX_GPU_CODE T* begin() const {
    ETX_ASSERT_GREATER(count, 0llu);
    ETX_ASSERT(a != nullptr);
    return a;
  }

  ETX_GPU_CODE T* end() const {
    ETX_ASSERT_GREATER(count, 0llu);
    ETX_ASSERT(a != nullptr);
    return a + count;
  }
};

template <class T>
struct Pointer {
  T* ptr ETX_EMPTY_INIT;

  Pointer() = default;

  ETX_GPU_CODE Pointer(T* p)
    : ptr(p) {
  }

  ETX_GPU_CODE T* operator->() {
    ETX_ASSERT(ptr != nullptr);
    return ptr;
  }

  ETX_GPU_CODE T* operator->() const {
    ETX_ASSERT(ptr != nullptr);
    return ptr;
  }
};

template <class T>
ETX_GPU_CODE ArrayView<T> make_array_view(void* p, uint64_t count) {
  return {reinterpret_cast<T*>(p), count};
}

template <class T>
ETX_GPU_CODE ArrayView<T> make_array_view(uint64_t p, uint64_t count) {
  return {reinterpret_cast<T*>(p), count};
}

}  // namespace etx
