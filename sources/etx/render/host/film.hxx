#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/shared/camera.hxx>

namespace etx {

struct TaskScheduler;
struct Scene;

struct FilmImpl;
struct Film {
  enum : uint32_t {
    Result,
    Denoised,
    Albedo,
    Normals,
    CameraImage,
    LightImage,
    LightIteration,
    CameraAdaptive,
    LightAdaptive,
    Debug,

    LayerCount,
  };

  enum ClearOptions : uint32_t {
    ClearCameraData = 1u << 0u,
    ClearLightData = 1u << 1u,
    ClearEverything = 1u << 2u,
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

  float2 sample(const Scene& scene, const PixelFilter& sampler, const uint2& pixel, const float2& rnd) const;

  void accumulate_camera_image(const uint2& pixel, const float3& color, const float3& normal, const float3& albedo);
  void atomic_add_light_iteration(const float3& value, const float2& ndc_coord);
  void commit_light_iteration(uint32_t i);

  void clear(uint32_t clear_options);

  const uint2& size() const;  // total size of the film in pixels
  uint2 dimensions() const;   // current size of the film in pixels, accounting for pixel size

  float4* layer(uint32_t layer) const;
  void denoise(uint32_t layer_to_denoise);

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
  ETX_DECLARE_PIMPL(Film, 640);
};

}  // namespace etx
