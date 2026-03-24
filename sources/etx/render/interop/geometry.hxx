#pragma once

#include "interop.hxx"

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
