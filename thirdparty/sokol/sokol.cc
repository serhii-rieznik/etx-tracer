#if defined(_WIN32)

# include <imgui.h>

# define SOKOL_NO_ENTRY 1
# define SOKOL_APP_IMPL
# define SOKOL_GFX_IMPL
# define SOKOL_IMGUI_IMPL

# if defined(ETX_SOKOL_NEW)
#  define SOKOL_NOAPI
#  include "sokol_app_new.h"
# else
#  define SOKOL_D3D11
#  include "sokol_app.h"
#  include "sokol_gfx.h"
#  include "util/sokol_imgui.h"
# endif

#endif
