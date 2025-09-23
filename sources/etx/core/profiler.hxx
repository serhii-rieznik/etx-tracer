#pragma once

#include <etx/core/pimpl.hxx>

#include <chrono>

namespace etx {

#define ETX_PROFILER_ENABLED 0

#if (ETX_PROFILER_ENABLED)

# include <microprofile.h>

# define ETX_PROFILER_MAIN_THREAD()        \
   do {                                    \
     MicroProfileOnThreadCreate("main");   \
     MicroProfileSetEnableAllGroups(true); \
   } while (0)

# define ETX_PROFILER_REGISTER_THREAD(name) \
   do {                                     \
     MicroProfileOnThreadCreate(name);      \
   } while (0)

# define ETX_PROFILER_EXIT_THREAD() \
   do {                             \
     MicroProfileOnThreadExit();    \
   } while (0)

# define ETX_PROFILER_SCOPE()                                                                                                                                      \
   static MicroProfileToken MICROPROFILE_TOKEN_PASTE(_, __LINE__) = MicroProfileGetToken("CPU", __FUNCTION__, fnv1a32(__FUNCTION__), MicroProfileTokenTypeCpu, 0); \
   MICROPROFILE_SCOPE_TOKEN(MICROPROFILE_TOKEN_PASTE(_, __LINE__));

# define ETX_PROFILER_NAMED_SCOPE(name)                                                                                                                        \
   static MicroProfileToken MICROPROFILE_TOKEN_PASTE(profile_token, __LINE__) = MicroProfileGetToken("CPU", name, fnv1a32(name), MicroProfileTokenTypeCpu, 0); \
   MICROPROFILE_SCOPE_TOKEN(MICROPROFILE_TOKEN_PASTE(profile_token, __LINE__));

# define ETX_END_PROFILER_FRAME() MicroProfileFlip(nullptr)

#else

# define ETX_EMPTY_STATEMENT \
   do {                      \
   } while (0)

# define ETX_PROFILER_MAIN_THREAD(...)     ETX_EMPTY_STATEMENT
# define ETX_PROFILER_REGISTER_THREAD(...) ETX_EMPTY_STATEMENT
# define ETX_PROFILER_EXIT_THREAD(...)     ETX_EMPTY_STATEMENT
# define ETX_PROFILER_SCOPE(...)           ETX_EMPTY_STATEMENT
# define ETX_PROFILER_NAMED_SCOPE(...)     ETX_EMPTY_STATEMENT
# define ETX_PROFILER_FRAME(...)           ETX_EMPTY_STATEMENT
# define ETX_END_PROFILER_FRAME(...)       ETX_EMPTY_STATEMENT

#endif

}  // namespace etx
// ...