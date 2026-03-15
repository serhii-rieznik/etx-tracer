#pragma once

#if defined(__cplusplus)
# define ETX_CPP 1
#else
# define ETX_CPP 0
#endif

#if defined(__HLSL_VERSION)
# define ETX_HLSL 1
#else
# define ETX_HLSL 0
#endif

#if (ETX_CPP)
# define ETX_STD std::
#else
# define ETX_STD
#endif

#if (ETX_CPP)
# define ETX_ENUM_U32(name)           enum class name : uint32_t
# define ETX_ENUM_U32_TO_UINT32(value) static_cast<uint32_t>(value)
# define ETX_STATIC_ASSERT(cond, msg) static_assert((cond), msg)
#else
# define ETX_ENUM_U32(name)           enum name
# define ETX_ENUM_U32_TO_UINT32(value) (value)
# define ETX_STATIC_ASSERT(cond, msg)
#endif

#if (ETX_CPP)
# include <cmath>
# include <complex>
# include <cstring>
# include <cstdio>
# include <cstdint>
#endif

#if (ETX_CPP)
# define ETX_SHARED_INLINE     inline
# define ETX_IN(type, name)    const type& name
# define ETX_OUT(type, name)   type& name
# define ETX_INOUT(type, name) type& name
# define ETX_ALIGNED           alignas(16)
# define ETX_INIT(...)         = __VA_ARGS__
# define ETX_ZERO(type) \
  {                    \
  }
# define ETX_ZERO_INIT(type, name) type name = ETX_ZERO(type)
# define ETX_STATIC_CONST          constexpr
#else
# define ETX_SHARED_INLINE         inline
# define ETX_IN(type, name)        in type name
# define ETX_OUT(type, name)       out type name
# define ETX_INOUT(type, name)     inout type name
# define ETX_ALIGNED               /* */
# define ETX_INIT(...)             /* */
# define ETX_ZERO(type)            (type)0
# define ETX_ZERO_INIT(type, name) type name = ETX_ZERO(type)
# define ETX_STATIC_CONST          static const
#endif
