#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/shared/base.hxx>

namespace etx {

struct TaskScheduler;
struct FilmImpl;
struct Film {
  enum : uint32_t {
    Camera,
    LightImage,
    LightIteration,
    Normals,
    Albedo,
    Result,
    Denoised,

    LayerCount,
  };

  using Layers = std::initializer_list<uint32_t>;

  Film(TaskScheduler&);
  ~Film();

  void allocate(const uint2& dim, const Layers& layers);

  void atomic_add(uint32_t layer, const float4& value, const float2& ndc_coord);
  void atomic_add(uint32_t layer, const float4& value, uint32_t x, uint32_t y);

  void accumulate(uint32_t layer, const float4& value, const float2& ndc_coord, float t);
  void accumulate(uint32_t layer, const float4& value, uint32_t x, uint32_t y, float t);

  void flush_to(Film& other, float t, const Layers& layers);

  void commit_light_iteration(uint32_t i);
  const float4* combined_result() const;

  void clear(const Layers& layers);
  void clear();

  const uint2& dimensions() const;
  const uint32_t count() const;
  const float4* layer(uint32_t layer) const;

  float4* mutable_layer(uint32_t layer);

  static float calculate_ev(float f, float s) {
    return log2f(f * f / s);
  }

  static const char* layer_name(uint32_t layer);

 private:
  ETX_DECLARE_PIMPL(Film, 256);
};

}  // namespace etx
