#pragma once

#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#if defined(NDEBUG) || defined(_NDEBUG)
#define ETX_DEBUG 0
#else
#define ETX_DEBUG 1
#endif

#if defined(__NVCC__)

#define ETX_NVCC_COMPILER 1
#define ETX_CPU_CODE __host__
#define ETX_GPU_CODE inline __device__
#define ETX_SHARED_CODE ETX_GPU_CODE ETX_CPU_CODE
#define ETX_GPU_DATA __device__
#define ETX_GPU_CALLABLE extern "C" __global__
#define ETX_INIT_WITH(S)

#else

#define ETX_NVCC_COMPILER 0
#define ETX_CPU_CODE
#define ETX_GPU_CODE inline
#define ETX_SHARED_CODE ETX_GPU_CODE ETX_CPU_CODE
#define ETX_GPU_CALLABLE
#define ETX_GPU_DATA
#define ETX_INIT_WITH(S) = S

template <class T>
T atomicAdd(T*, T);

#endif

#if (ETX_NVCC_COMPILER)

#define ETX_DEBUG_BREAK() \
  do {                    \
    __threadfence();      \
    asm("trap;");         \
  } while (0)

#define ETX_ABORT() assert(false)

#else

#define ETX_DEBUG_BREAK() __debugbreak()
#define ETX_ABORT() abort()

#endif

ETX_GPU_CODE void print_value(const char* name, const char* tag, float t) {
  printf("%s : %s %f\n", name, tag, t);
}

ETX_GPU_CODE void print_value_no_tag(const char* name, float t) {
  printf("%s : %f", name, t);
}
ETX_GPU_CODE void print_value_no_tag(const char* name, uint32_t t) {
  printf("%s : %u", name, t);
}
ETX_GPU_CODE void print_value_no_tag(const char* name, int32_t t) {
  printf("%s : %d", name, t);
}
ETX_GPU_CODE void print_value_no_tag(const char* name, uint64_t t) {
  printf("%s : %llu", name, t);
}
ETX_GPU_CODE void print_value_no_tag(const char* name, int64_t t) {
  printf("%s : %lld", name, t);
}

#define ETX_FORCE_ASSERTS 0

#if (ETX_DEBUG || ETX_FORCE_ASSERTS)

#define ETX_ASSERT_EQUAL(A, B)                             \
  do {                                                     \
    if ((A) != (B)) {                                      \
      printf("Equal condition: (");                        \
      print_value_no_tag(#A, A);                           \
      printf(") != (");                                    \
      print_value_no_tag(#B, B);                           \
      printf(") failed at %s [%u]\n", __FILE__, __LINE__); \
      ETX_DEBUG_BREAK();                                   \
    }                                                      \
  } while (0)

#define ETX_ASSERT_LESS(A, B)                              \
  do {                                                     \
    if (((A) < (B)) == false) {                            \
      printf("Less condition: (");                         \
      print_value_no_tag(#A, A);                           \
      printf(") < (");                                     \
      print_value_no_tag(#B, B);                           \
      printf(") failed at %s [%u]\n", __FILE__, __LINE__); \
      ETX_DEBUG_BREAK();                                   \
    }                                                      \
  } while (0)

#define ETX_ASSERT_GREATER(A, B)                           \
  do {                                                     \
    if (((A) > (B)) == false) {                            \
      ETX_DEBUG_BREAK();                                   \
      printf("Greater condition: (");                      \
      print_value_no_tag(#A, A);                           \
      printf(") > (");                                     \
      print_value_no_tag(#B, B);                           \
      printf(") failed at %s [%u]\n", __FILE__, __LINE__); \
      ETX_DEBUG_BREAK();                                   \
    }                                                      \
  } while (0)

#define ETX_ASSERT(condition)                                                     \
  do {                                                                            \
    if (!(condition)) {                                                           \
      printf("Condition %s failed at %s [%u]\n", #condition, __FILE__, __LINE__); \
      ETX_DEBUG_BREAK();                                                          \
    }                                                                             \
  } while (0)

#else

#define ETX_ASSERT(condition) \
  do {                        \
  } while (0)

#define ETX_ASSERT_EQUAL(A, B) \
  do {                         \
  } while (0)
#define ETX_ASSERT_LESS(A, B) \
  do {                        \
  } while (0)
#define ETX_ASSERT_GREATER(A, B) \
  do {                           \
  } while (0)

#endif

#define ETX_CRITICAL(condition)                                                            \
  do {                                                                                     \
    if (!(condition)) {                                                                    \
      printf("Critical condition %s failed at %s [%u]\n", #condition, __FILE__, __LINE__); \
      ETX_DEBUG_BREAK();                                                                   \
    }                                                                                      \
  } while (0)

#define ETX_FAIL(msg)                   \
  do {                                  \
    printf("Critical fail: %s\n", msg); \
    ETX_ABORT();                        \
  } while (0)

#define ETX_FAIL_FMT(fmt, ...) \
  do {                         \
    printf(fmt, __VA_ARGS__);  \
    ETX_ABORT();               \
  } while (0)
