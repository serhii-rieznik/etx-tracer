#include <etx/core/core.hxx>

#include <atomic>

#if (ETX_PLATFORM_APPLE)
# include <libkern/OSAtomic.h>
#endif

namespace etx {

uint32_t atomic_compare_exchange(int32_t* ptr, int32_t old_value, int32_t new_value) {
#if (ETX_PLATFORM_APPLE)
  return OSAtomicCompareAndSwap32(old_value, new_value, ptr);
#elif (ETX_PLATFORM_WINDOWS)
  static_assert(sizeof(long) == sizeof(int32_t));
  return _InterlockedCompareExchange(reinterpret_cast<volatile long*>(ptr), new_value, old_value);
#endif
}

uint32_t atomic_inc(int32_t* ptr) {
#if (ETX_PLATFORM_APPLE)
  return OSAtomicAdd32(1, ptr);
#elif (ETX_PLATFORM_WINDOWS)
  static_assert(sizeof(long) == sizeof(int32_t));
  return _InterlockedIncrement(reinterpret_cast<volatile long*>(ptr));
#endif
}

uint64_t atomic_inc(int64_t* ptr) {
#if (ETX_PLATFORM_APPLE)
  return OSAtomicAdd64(1, ptr);
#elif (ETX_PLATFORM_WINDOWS)
  static_assert(sizeof(long long) == sizeof(int64_t));
  return _InterlockedIncrement64(reinterpret_cast<volatile long long*>(ptr));
#endif
}

int64_t atomic_add_int64(int64_t* ptr, int64_t value) {
#if (ETX_PLATFORM_APPLE)
  return OSAtomicAdd64(value, ptr);
#elif (ETX_PLATFORM_WINDOWS)
  static_assert(sizeof(long) == sizeof(int32_t));
  return _InterlockedExchangeAdd64(reinterpret_cast<volatile long long*>(ptr), value);
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

}  // namespace etx
