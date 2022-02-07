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

  uint32_t size = _dimensions.x * _dimensions.y;
  if (size == 0) {
    _data.clear();
    _data_ptr = nullptr;
  }

  _data.resize(size);
  for (uint32_t i = 0; i < size; ++i) {
    _data[i] = {1.0f, 1.0f, 1.0f, 1.0f};
  }
  _data_ptr = _data.data();
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
  uint32_t index = x + (_dimensions.y - 1 - y) * _dimensions.x;
  auto ptr = (_data_ptr + index)->data.data;
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

}  // namespace etx
