#include <etx/core/core.hxx>
#include <etx/render/host/film.hxx>
#include <etx/render/shared/spectrum.hxx>

#include <algorithm>

namespace etx {

void Film::resize(const uint2& dim, uint32_t threads) {
  _dimensions = dim;
  _thread_count = threads;

  uint32_t pixel_count = _dimensions.x * _dimensions.y;
  _buffer.resize(1llu * _thread_count * pixel_count);

  _sequence.resize(pixel_count);
  for (uint32_t i = 0; i < pixel_count; ++i) {
    _sequence[i] = i;
  }

  std::sort(_sequence.begin(), _sequence.end(), [w = _dimensions.x, h = _dimensions.y, s = float(_dimensions.x) / float(_dimensions.y)](uint32_t a, uint32_t b) {
    float ax = s * (float(a % w) / w * 2.0f - 1.0f);
    float ay = float(a / w) / h * 2.0f - 1.0f;
    float bx = s * (float(b % w) / w * 2.0f - 1.0f);
    float by = float(b / w) / h * 2.0f - 1.0f;
    return (ax * ax + ay * ay) < (bx * bx + by * by);
  });

  clear();
}

void Film::atomic_add(const float4& value, const float2& ndc_coord, uint32_t thread_id) {
  float2 uv = ndc_coord * 0.5f + 0.5f;
  uint32_t ax = static_cast<uint32_t>(uv.x * float(_dimensions.x));
  uint32_t ay = static_cast<uint32_t>(uv.y * float(_dimensions.y));
  atomic_add(value, ax, ay, thread_id);
}

void Film::atomic_add(const float4& value, uint32_t x, uint32_t y, uint32_t thread_id) {
  ETX_ASSERT(thread_id < _thread_count);

  if ((x >= _dimensions.x) || (y >= _dimensions.y)) {
    return;
  }

  ETX_VALIDATE(value);
  uint32_t index = x + y * _dimensions.x;
  _buffer[index * _thread_count + thread_id] += value;
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
  ETX_ASSERT(other._thread_count == 1);

  auto dst = other._buffer.data();
  auto src = _buffer.data();
  uint64_t pixel_count = count();
  for (uint32_t i = 0; i < pixel_count; ++i) {
    uint32_t base = i * _thread_count;
    float4 sum = src[base];
    for (uint32_t t = 1; t < _thread_count; ++t) {
      sum += src[base + t];
    }
    memset(src + base, 0, sizeof(float4) * _thread_count);
    dst[i] = (t == 0.0f) ? sum : lerp(sum, dst[i], t);
  }
}

void Film::clear() {
  std::fill(_buffer.begin(), _buffer.end(), float4{});
}

}  // namespace etx
