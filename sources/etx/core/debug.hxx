#pragma once

#include <stdio.h>
#include <stdint.h>

#if defined(NDEBUG) || defined(_NDEBUG)
#define ETX_DEBUG 0
#else
#define ETX_DEBUG 1
#endif

#if defined(__NVCC__)
#define ETX_NVCC_COMPILER 1
#else
#define ETX_NVCC_COMPILER 0
#endif

#if (ETX_NVCC_COMPILER)
#define ETX_DEBUG_BREAK() assert(false)
#define ETX_ABORT() assert(false)
#else
#define ETX_DEBUG_BREAK() __debugbreak()
#define ETX_ABORT() abort()
#endif

#define ETX_FORCE_ASSERTS 0

#if (ETX_DEBUG || ETX_FORCE_ASSERTS)

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
