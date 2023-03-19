#if defined(_MSC_VER)
#define SOKOL_D3D11 1
#else
#define SOKOL_GLCORE33 1
#endif

#include <sokol_gfx.h>
#include <sokol_app.h>

#include <imgui.h>
#include <imgui_internal.h>
#include "cimgui.h"

#define SOKOL_IMGUI_IMPL 1
#include "sokol_imgui.h"
