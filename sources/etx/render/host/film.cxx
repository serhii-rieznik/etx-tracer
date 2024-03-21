#include <etx/core/core.hxx>
#include <etx/render/host/film.hxx>
#include <etx/render/shared/spectrum.hxx>

#include <algorithm>

namespace etx {

void Film::allocate(const uint2& dim, uint32_t layers) {
  _dimensions = dim;
  _buffer.resize(1llu * _dimensions.x * _dimensions.y);
  clear();
}

void Film::atomic_add(const float4& value, const float2& ndc_coord) {
  float2 uv = ndc_coord * 0.5f + 0.5f;
  uint32_t ax = static_cast<uint32_t>(uv.x * float(_dimensions.x));
  uint32_t ay = static_cast<uint32_t>(uv.y * float(_dimensions.y));
  atomic_add(value, ax, ay);
}

void Film::atomic_add(const float4& value, uint32_t x, uint32_t y) {
  if ((x >= _dimensions.x) || (y >= _dimensions.y)) {
    return;
  }

  ETX_VALIDATE(value);
  uint32_t index = x + y * _dimensions.x;
  auto ptr = _buffer.data() + index;
  atomic_add_float(&ptr->x, value.x);
  atomic_add_float(&ptr->y, value.y);
  atomic_add_float(&ptr->z, value.z);
  atomic_add_float(&ptr->w, value.w);
}

void Film::accumulate(const float4& value, uint32_t x, uint32_t y, float t) {
  if ((x >= _dimensions.x) || (y >= _dimensions.y)) {
    return;
  }
  ETX_VALIDATE(value);

  uint32_t i = x + (_dimensions.y - 1 - y) * _dimensions.x;

  float lum_sq = sqr(luminance({value.x, value.y, value.z}));
  float4 existing_value = _buffer[i];

  float4 new_value = {value.x, value.y, value.z, lum_sq};
  if (t > 0.0f) {
    new_value.x = lerp(value.x, existing_value.x, t);
    new_value.y = lerp(value.y, existing_value.y, t);
    new_value.z = lerp(value.z, existing_value.z, t);
    new_value.w = existing_value.w + lum_sq;
  }
  _buffer[i] = new_value;
}

void Film::accumulate(const float4& value, const float2& ndc_coord, float t) {
  float2 uv = ndc_coord * 0.5f + 0.5f;
  uint32_t ax = static_cast<uint32_t>(uv.x * float(_dimensions.x));
  uint32_t ay = static_cast<uint32_t>(uv.y * float(_dimensions.y));
  accumulate(value, ax, ay, t);
}

void Film::flush_to(Film& other, float t) {
  ETX_ASSERT(_dimensions == other._dimensions);

  auto dst = other._buffer.data();
  auto src = _buffer.data();
  uint64_t pixel_count = count();
  for (uint64_t i = 0; i < pixel_count; ++i) {
    dst[i] = (t == 0.0f) ? src[i] : lerp(src[i], dst[i], t);
  }
}

void Film::clear() {
  std::fill(_buffer.begin(), _buffer.end(), float4{});
}

}  // namespace etx
