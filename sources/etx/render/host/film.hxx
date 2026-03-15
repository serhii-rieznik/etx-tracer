#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/shared/camera.hxx>

namespace etx {

struct TaskScheduler;

struct FilmImpl;
struct Film {
  enum ClearOptions : uint32_t {
    ClearIteration = 1u << 0u,
    ClearEverything = 1u << 1u,
  };

  enum : uint32_t {
    PixelFilterBlackmanHarris,
    PixelFilterCount,

    PixelFilterSize = 128u,
  };

  static constexpr float kFilmHorizontalSize = 36.0f;
  static constexpr float kFilmVerticalSize = 24.0f;

  struct LayerValue {
    float3 value = {};
    uint32_t layer = 0;
  };

  Film(TaskScheduler&);
  ~Film();

  void allocate(const uint2& dim);

  float2 sample(const PixelFilter& sampler, const uint2& pixel, const float2& rnd) const;

  void submit(const float3& value, const float2& ndc_coord);
  void submit(const float3& value, const float3& normal, const float3& albedo, const uint2& pixel);
  void commit_iteration(uint32_t sample_index, uint32_t total_samples, float noise_threshold, float radiance_clamp);

  void clear(uint32_t clear_options);

  const uint2& size() const;         // total size of the film in pixels
  uint2 current_dimensions() const;  // current size of the film in pixels, accounting for pixel size
  uint2 base_dimensions() const;     // current size of the film in pixels, accounting for pixel size

  float4* layer(uint32_t layer, float radiance_clamp) const;
  void denoise(uint32_t layer_to_denoise, float radiance_clamp);

  uint32_t pixel_size() const;
  void set_pixel_size(uint32_t size);

  /*
   * Adaptive sampling
   */
  uint32_t total_pixel_count() const;
  uint32_t current_pixel_count() const;
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
  ETX_DECLARE_PIMPL(Film, 640);
};

}  // namespace etx
