#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/shared/camera.hxx>

namespace etx {

struct TaskScheduler;
struct Scene;

struct FilmImpl;
struct Film {
  enum : uint32_t {
    CameraImage,
    LightImage,
    LightIteration,
    Normals,
    Albedo,
    Result,
    Denoised,
    Adaptive,
    Internal,
    Debug,

    LayerCount,
  };

  enum : uint32_t {
    PixelFilterBlackmanHarris,
    PixelFilterCount,

    PixelFilterSize = 128u,
  };

  using Layers = std::initializer_list<uint32_t>;

  static constexpr Layers kAllLayers = {CameraImage, LightImage, LightIteration, Normals, Albedo, Result, Denoised, Adaptive, Internal, Debug};
  static constexpr float kFilmHorizontalSize = 36.0f;
  static constexpr float kFilmVerticalSize = 24.0f;

  struct LayerValue {
    float3 value = {};
    uint32_t layer = 0;
  };

  Film(TaskScheduler&);
  ~Film();

  void allocate(const uint2& dim);

  float2 sample(const Scene& scene, const PixelFilter& sampler, const uint2& pixel, const float2& rnd) const;

  void atomic_add(uint32_t layer, const float3& value, const float2& ndc_coord);
  void atomic_add(uint32_t layer, const float3& value, uint32_t x, uint32_t y);

  void accumulate(uint32_t layer, const float3& value, const uint2& pixel);
  void accumulate(const uint2& pixel, const std::initializer_list<LayerValue>& lv);

  void commit_light_iteration(uint32_t i);

  void clear(const Layers& layers);
  void clear();

  // total size of the film in pixels
  const uint2& size() const;

  // current size of the film in pixels, accounting for pixel size
  uint2 dimensions() const;

  const float4* layer(uint32_t layer) const;
  float4* mutable_layer(uint32_t layer) const;

  const float4* combined_result() const;
  float4* mutable_combined_result() const;

  void denoise();

  uint32_t pixel_size() const;
  void set_pixel_size(uint32_t size);

  /*
   * Adaptive sampling
   */
  uint32_t pixel_count() const;
  uint32_t active_pixel_count() const;

  bool active_pixel(uint32_t linear_index, uint2& location) const;
  void estimate_noise_levels(uint32_t sample_index, uint32_t total_samples, float threshold);
  float noise_level() const;

  static void generate_filter_image(uint32_t filter, std::vector<float4>&);

  static float calculate_ev(float f, float s) {
    return log2f(f * f / s);
  }

  static const char* layer_name(uint32_t layer);

 private:
  ETX_DECLARE_PIMPL(Film, 512);
};

}  // namespace etx
