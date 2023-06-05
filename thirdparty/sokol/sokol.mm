#if defined(__APPLE__)

#define SOKOL_METAL 1
#define SOKOL_NO_ENTRY 1

#define SOKOL_APP_IMPL
#include "sokol_app.h"

#define SOKOL_GFX_IMPL
#include "sokol_gfx.h"

#define SOKOL_IMGUI_IMPL
#include <imgui.h>
#include "util/sokol_imgui.h"

#endif
