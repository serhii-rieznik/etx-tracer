#pragma once

#if defined(__cplusplus) && !defined(ETX_RENDER_BASE_INCLUDED)
# error This file should not be included separately. Use etx/render/shared/base.hxx or an interop header.
#endif

#if !defined(__cplusplus)
# include "interop.hxx"
#endif

#if defined(__cplusplus)
# include <cstddef>
# include <type_traits>
#endif

#if defined(__cplusplus) && !defined(ETX_INIT)
# define ETX_INIT(...) ETX_INIT_WITH(__VA_ARGS__)
# define ETX_RAY_LOCAL_ETX_INIT 1
#endif

struct ETX_ALIGNED Ray {
#if defined(__cplusplus)
  Ray() = default;

  ETX_SHARED_INLINE Ray(const float3& origin, const float3& direction)
    : o(origin)
    , d(direction) {
  }

  ETX_SHARED_INLINE Ray(const float3& origin, const float3& direction, float t_min, float t_max)
    : o(origin)
    , min_t(t_min)
    , d(direction)
    , max_t(t_max) {
  }
#endif
  float3 o ETX_INIT({});
  float min_t ETX_INIT(kRayEpsilon);
  float3 d ETX_INIT({});
  float max_t ETX_INIT(kMaxFloat);
};

#if defined(ETX_RAY_LOCAL_ETX_INIT)
# undef ETX_RAY_LOCAL_ETX_INIT
# undef ETX_INIT
#endif

#if defined(__cplusplus)
static_assert(std::is_standard_layout_v<Ray>, "Ray must stay standard layout for C++/HLSL interop");
static_assert(alignof(Ray) == 16, "Ray alignment must match HLSL packing");
static_assert(sizeof(Ray) == 32, "Ray size changed; update shared ABI or padding");
static_assert(offsetof(Ray, d) == 16, "Ray::d offset changed");
#endif
