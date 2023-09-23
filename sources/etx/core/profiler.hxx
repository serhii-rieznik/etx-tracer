#pragma once

#include <etx/core/pimpl.hxx>

#include <chrono>

namespace etx {

#define ETX_PROFILER_ENABLED 0

#if (ETX_PROFILER_ENABLED)

#error NOT IMPLEMENTED

#else

# define ETX_EMPTY_STATEMENT \
   do {                      \
   } while (0)

# define ETX_PROFILER_REGISTER_THREAD  ETX_EMPTY_STATEMENT
# define ETX_FUNCTION_SCOPE()          ETX_EMPTY_STATEMENT
# define ETX_PROFILER_RESET_COUNTERS() ETX_EMPTY_STATEMENT

#endif

}  // namespace etx
