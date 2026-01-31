#if defined(_WIN32)

# define SOKOL_NO_ENTRY 1
# define SOKOL_APP_IMPL
# define SOKOL_NOAPI
# include "sokol_app.h"

#endif
