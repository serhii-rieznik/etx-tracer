#pragma once

#if defined(__cplusplus)

# include <cstdint>
# include <etx/render/shared/base.hxx>
# define ETX_SHADER_U32 uint32_t
#else
# define ETX_SHADER_U32 uint

#endif

#if !defined(ETX_ALIGNED)
# define ETX_ALIGNED
#endif

#if !defined(ETX_GPU_CODE)
# define ETX_GPU_CODE
#endif
