#include <etx/render/host/film.hxx>

namespace etx {

inline void atomic_add_impl(volatile float* ptr, float value) {
  volatile long* iptr = std::bit_cast<volatile long*>(ptr);
  long old_value = {};
  long new_value = {};
  do {
    old_value = std::bit_cast<long>(*ptr);
    new_value = std::bit_cast<long>(*ptr + value);
  } while (_InterlockedCompareExchange(iptr, new_value, old_value) != old_value);
}

void Film::resize(const uint2& dim) {
  _dimensions = dim;
  _data.resize(1llu * _dimensions.x * _dimensions.y);
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
  auto ptr = _data[index].data.data;
  atomic_add_impl(ptr + 0, value.x);
  atomic_add_impl(ptr + 1, value.y);
  atomic_add_impl(ptr + 2, value.z);
}

void Film::accumulate(const float4& value, uint32_t x, uint32_t y, float t) {
  if ((x >= _dimensions.x) || (y >= _dimensions.y)) {
    return;
  }
  ETX_VALIDATE(value);
  uint32_t i = x + (_dimensions.y - 1 - y) * _dimensions.x;
  _data[i] = (t <= 0.0f) ? value : lerp(value, _data[i], t);
}

void Film::accumulate(const float4& value, const float2& ndc_coord, float t) {
  float2 uv = ndc_coord * 0.5f + 0.5f;
  uint32_t ax = static_cast<uint32_t>(uv.x * float(_dimensions.x));
  uint32_t ay = static_cast<uint32_t>(uv.y * float(_dimensions.y));
  accumulate(value, ax, ay, t);
}

void Film::merge(const Film& other, float t) {
  ETX_ASSERT(_dimensions == other._dimensions);
  for (uint64_t i = 0, e = _data.size(); i < e; ++i) {
    _data[i] = (t == 0.0f) ? other._data[i] : lerp(other._data[i], _data[i], t);
  }
}

void Film::clear() {
  std::fill(_data.begin(), _data.end(), float4{});
}

}  // namespace etx
