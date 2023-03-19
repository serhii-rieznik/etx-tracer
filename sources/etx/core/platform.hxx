#pragma once

#include <inttypes.h>

#if defined(_WIN32)

#define ETX_PLATFORM_WINDOWS 1

#elif defined(__MACH__)

#define ETX_PLATFORM_APPLE 1
#define _stricmp strcasecmp

#endif
