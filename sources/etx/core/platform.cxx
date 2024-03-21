#if (ETX_PLATFORM_APPLE)
# include <libkern/OSAtomic.h>
#endif

namespace etx {

uint32_t atomic_inc(int32_t* ptr) {
#if (ETX_PLATFORM_APPLE)
  return OSAtomicIncrement32(ptr);
#elif (ETX_PLATFORM_WINDOWS)
  static_assert(sizeof(long) == sizeof(int32_t));
  return _InterlockedIncrement(reinterpret_cast<volatile long*>(ptr));
#endif
}

uint64_t atomic_inc(int64_t* ptr) {
#if (ETX_PLATFORM_APPLE)
  return OSAtomicIncrement64(ptr);
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
#if (ETX_PLATFORM_WINDOWS)
  volatile long* iptr = std::bit_cast<volatile long*>(ptr);
  long old_value, new_value;
  do {
    old_value = std::bit_cast<long>(*ptr);
    new_value = std::bit_cast<long>(*ptr + value);
  } while (_InterlockedCompareExchange(iptr, new_value, old_value) != old_value);
#elif (ETX_PLATFORM_APPLE)
# error Implement
#endif
}

}  // namespace etx
