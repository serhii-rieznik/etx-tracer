#pragma once

#include <etx/render/shared/base.hxx>
#include <vector>

namespace etx {

struct Film {
  struct Layer {
    enum : uint32_t {
      CameraRays = 1u << 0u,
      LightRays = 1u << 1u,
      InfoNormals = 1u << 2u,
      InfoAlbedo = 1u << 3u,
    };
  };

  Film() = default;
  ~Film() = default;

  void allocate(const uint2& dim, uint32_t layers, uint32_t threads);

  void atomic_add(const float4& value, const float2& ndc_coord, uint32_t thread_id);
  void atomic_add(const float4& value, uint32_t x, uint32_t y, uint32_t thread_id);

  void accumulate(const float4& value, const float2& ndc_coord, float t);
  void accumulate(const float4& value, uint32_t x, uint32_t y, float t);

  void flush_to(Film& other, float t);

  void clear();

  const uint2& dimensions() const {
    return _dimensions;
  }

  const uint32_t count() const {
    return _dimensions.x * _dimensions.y;
  }

  const float4* data() const {
    return _buffer.data();
  }

  uint32_t pixel_at(uint32_t i) const {
    return _sequence[i];
  }

  static float calculate_ev(float f, float s) {
    return log2f(f * f / s);
  }

 private:
  Film(const Film&) = delete;
  Film& operator=(const Film&) = delete;
  Film(Film&&) = delete;
  Film& operator=(Film&&) = delete;

 private:
  uint2 _dimensions = {};
  uint32_t _thread_count = 0;
  std::vector<float4> _buffer = {};
  std::vector<uint32_t> _sequence = {};
};

}  // namespace etx
