#include <etx/core/core.hxx>

#include <atomic>

#if (ETX_PLATFORM_APPLE)
#include <libkern/OSAtomic.h>
#endif

namespace etx {

uint32_t atomic_compare_exchange(int32_t* ptr, int32_t old_value, int32_t new_value) {
#if (ETX_PLATFORM_APPLE)
  return OSAtomicCompareAndSwap32(old_value, new_value, ptr);
#elif (ETX_PLATFORM_WINDOWS)
#error NO
#endif
}

uint32_t atomic_inc(int32_t* ptr) {
#if (ETX_PLATFORM_APPLE)
  return OSAtomicAdd32(1, ptr);
#elif (ETX_PLATFORM_WINDOWS)
#error NO
#endif
}

void atomic_add_float(float* ptr, float value) {
  auto iptr = std::bit_cast<int32_t*>(ptr);
  int32_t old_value = 0u;
  int32_t new_value = 0u;
  do {
    old_value = std::bit_cast<int32_t>(*ptr);
    new_value = std::bit_cast<int32_t>(*ptr + value);
  } while (atomic_compare_exchange(iptr, new_value, old_value) != old_value);
}

}
