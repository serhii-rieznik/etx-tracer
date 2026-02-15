#pragma once

#include "interop.hxx"

#if defined(__cplusplus)
# include <cstddef>
# include <type_traits>
#endif

struct Vertex {
  float3 pos;
  float3 nrm;
  float3 tan;
  float3 btn;
  float2 tex;
};

struct Triangle {
  uint32_t i[3] ETX_INIT({kInvalidIndex, kInvalidIndex, kInvalidIndex});
  uint32_t material_index ETX_INIT(kInvalidIndex);
  float3 geo_n ETX_INIT({});
  uint32_t emitter_index ETX_INIT(kInvalidIndex);
};

#if defined(__cplusplus)
static_assert(std::is_standard_layout_v<Vertex>, "Vertex must stay standard layout for C++/HLSL interop");
static_assert(std::is_trivially_copyable_v<Vertex>, "Vertex must stay trivially copyable for C++/HLSL interop");
static_assert(sizeof(Vertex) == 56, "Vertex size changed; update shared ABI");
static_assert(offsetof(Vertex, pos) == 0, "Vertex::pos offset changed; update shared ABI");
static_assert(offsetof(Vertex, nrm) == 12, "Vertex::nrm offset changed; update shared ABI");
static_assert(offsetof(Vertex, tan) == 24, "Vertex::tan offset changed; update shared ABI");
static_assert(offsetof(Vertex, btn) == 36, "Vertex::btn offset changed; update shared ABI");
static_assert(offsetof(Vertex, tex) == 48, "Vertex::tex offset changed; update shared ABI");

static_assert(std::is_standard_layout_v<Triangle>, "Triangle must stay standard layout for C++/HLSL interop");
static_assert(std::is_trivially_copyable_v<Triangle>, "Triangle must stay trivially copyable for C++/HLSL interop");
static_assert(sizeof(Triangle) == 32, "Triangle size changed; update shared ABI");
static_assert(offsetof(Triangle, i) == 0, "Triangle::i offset changed; update shared ABI");
static_assert(offsetof(Triangle, material_index) == 12, "Triangle::material_index offset changed; update shared ABI");
static_assert(offsetof(Triangle, geo_n) == 16, "Triangle::geo_n offset changed; update shared ABI");
static_assert(offsetof(Triangle, emitter_index) == 28, "Triangle::emitter_index offset changed; update shared ABI");
#endif
