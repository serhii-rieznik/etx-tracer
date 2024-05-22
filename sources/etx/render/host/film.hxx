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

  static constexpr Layers kAllLayers = {Camera, LightImage, LightIteration, Normals, Albedo, Result, Denoised};

  Film(TaskScheduler&);
  ~Film();

  void allocate(const uint2& dim);

  void atomic_add(uint32_t layer, const float4& value, const float2& ndc_coord);
  void atomic_add(uint32_t layer, const float4& value, uint32_t x, uint32_t y);

  void accumulate(uint32_t layer, const float4& value, const float2& ndc_coord, float t);
  void accumulate(uint32_t layer, const float4& value, uint32_t x, uint32_t y, float t);

  void flush_to(Film& other, float t, const Layers& layers) const;

  void commit_light_iteration(uint32_t i);

  void clear(const Layers& layers);
  void clear();

  const uint2& dimensions() const;
  const uint32_t count() const;

  const float4* layer(uint32_t layer) const;
  float4* mutable_layer(uint32_t layer) const;

  const float4* combined_result() const;
  float4* mutable_combined_result() const;

  void denoise();

  static float calculate_ev(float f, float s) {
    return log2f(f * f / s);
  }

  static const char* layer_name(uint32_t layer);

 private:
  ETX_DECLARE_PIMPL(Film, 512);
};

}  // namespace etx
