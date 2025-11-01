#include <etx/render/host/film.hxx>

#include <etx/core/core.hxx>
#include <etx/render/host/tasks.hxx>
#include <etx/render/host/denoiser.hxx>

#include <etx/render/shared/scene.hxx>

#define ETX_LOG_NOISE_LEVEL 0

namespace etx {

namespace {

constexpr uint32_t kMinSamples = 32u;

enum StorageLayers : uint32_t {
  StorageCameraImage,
  StorageLightImage,
  StorageCameraAdaptive,
  StorageLightAdaptive,
  StorageLightIteration,
  StorageNormals,
  StorageAlbedo,
  StorageDenoised,

  StorageLayerCount,
};

struct InternalData {
  float error_level = 0.0f;
  uint32_t sample_count : 30;
  uint32_t converged    : 1;
  uint32_t tmp          : 1;
};

struct LayerInfo {
  uint32_t layer_id = 0;
  uint32_t storage = 0;
} layer_info[Film::LayerCount] = {
  {Film::Result, kInvalidIndex},
  {Film::Denoised, StorageDenoised},
  {Film::Albedo, StorageAlbedo},
  {Film::Normals, StorageNormals},
  {Film::CameraImage, StorageCameraImage},
  {Film::LightImage, StorageLightImage},
  {Film::LightIteration, StorageLightIteration},
  {Film::CameraAdaptive, StorageCameraAdaptive},
  {Film::LightAdaptive, StorageLightAdaptive},
  {Film::Debug, kInvalidIndex},
};

float filter_box(const float2& p, float radius) {
  return float(fabsf(p.x) < radius) * float(fabsf(p.y) < radius);
}

float filter_tent(const float2& p, float radius) {
  float dx = fmaxf(0.0, radius - fabsf(p.x)) / radius;
  float dy = fmaxf(0.0, radius - fabsf(p.y)) / radius;
  return dx * dy;
}

float filter_blackman_harris(const float2& p, float radius) {
  float sample_distance = sqrtf(p.x * p.x + p.y * p.y);
  float r = kDoublePi * saturate(0.5f + sample_distance / (2.0f * radius));
  return 0.35875f - 0.48829f * cosf(r) + 0.14128f * cosf(2.0f * r) - 0.01168f * cosf(3.0f * r);
}

using filter_function = float (*)(const float2&, float);

}  // namespace

struct FilmImpl {
  FilmImpl(TaskScheduler& t)
    : tasks(t) {
    denoiser.init();
  }

  Denoiser denoiser;
  TaskScheduler& tasks;
  uint2 dimensions = {};
  std::vector<float3> storage_buffers[StorageLayerCount] = {};
  std::vector<float4> output_data = {};
  std::vector<InternalData> internal_data = {};
  std::atomic<float> last_noise_level = {};
  std::atomic<uint32_t> active_pixels = {};
  uint32_t max_sample_count = 0u;
  uint32_t pixel_size = 1u;
  uint32_t target_pixel_size = 1u;

  uint32_t total_pixel_count() const {
    return dimensions.x * dimensions.y;
  }
};

Film::Film(TaskScheduler& t) {
  ETX_PIMPL_INIT(Film, t);
}

Film::~Film() {
  ETX_PIMPL_CLEANUP(Film);
}

void Film::allocate(const uint2& dim) {
  if (_private->dimensions != dim) {
    _private->dimensions = {max(1u, dim.x), max(1u, dim.y)};
    for (auto& buffer : _private->storage_buffers) {
      buffer.clear();
      buffer.resize(1llu * _private->dimensions.x * _private->dimensions.y);
    }
    _private->internal_data.clear();
    _private->internal_data.resize(1llu * _private->dimensions.x * _private->dimensions.y);
    _private->output_data.clear();
    _private->output_data.resize(1llu * _private->dimensions.x * _private->dimensions.y);

    float3* albedo = _private->storage_buffers[StorageAlbedo].data();
    float3* normals = _private->storage_buffers[StorageNormals].data();
    _private->denoiser.allocate_buffers(albedo, normals, _private->dimensions);
  }
  clear(ClearEverything);
}

void Film::generate_filter_image(uint32_t filter, std::vector<float4>& data) {
  constexpr float2 center = {float(PixelFilterSize) * 0.5f, float(PixelFilterSize) * 0.5f};
  constexpr float radius = float(PixelFilterSize) * 0.5f;

  data.resize(PixelFilterSize * PixelFilterSize);
  for (uint32_t y = 0; y < PixelFilterSize; ++y) {
    for (uint32_t x = 0; x < PixelFilterSize; ++x) {
      float2 pos = {float(x), float(y)};
      float value = filter_blackman_harris(pos - center, radius);
      data[x + y * PixelFilterSize] = {value, value, value, 1.0f};
    }
  }
}

float2 Film::sample(const Scene& scene, const PixelFilter& sampler, const uint2& pixel, const float2& rnd) const {
  float2 jitter = rnd * 2.0f - 1.0f;
  if (sampler.image_index != kInvalidIndex) {
    jitter = scene.images[sampler.image_index].sample(rnd) * 2.0f - 1.0f;
  }
  float u = (float(pixel.x) + 0.5f + sampler.radius * jitter.x) / float(_private->dimensions.x) * 2.0f - 1.0f;
  float v = (float(pixel.y) + 0.5f + sampler.radius * jitter.y) / float(_private->dimensions.y) * 2.0f - 1.0f;
  return {u, v};
}

void Film::atomic_add_light_iteration(const float3& value, const float2& ndc_coord) {
  if (dot(value, value) == 0.0f)
    return;

  float2 uv = ndc_coord * 0.5f + 0.5f;
  uint32_t x = static_cast<uint32_t>(uv.x * dimensions().x) * _private->pixel_size;
  uint32_t y = static_cast<uint32_t>(uv.y * dimensions().y) * _private->pixel_size;
  if ((x >= _private->dimensions.x) || (y >= _private->dimensions.y)) {
    return;
  }

  const uint32_t p_base_x = (x / _private->pixel_size) * _private->pixel_size;
  const uint32_t p_base_y = (y / _private->pixel_size) * _private->pixel_size;
  auto ptr = _private->storage_buffers[StorageLightIteration].data();
  for (uint32_t v = 0; v < _private->pixel_size; ++v) {
    const uint32_t py = min(p_base_y + v, _private->dimensions.y - 1u);
    for (uint32_t u = 0; u < _private->pixel_size; ++u) {
      const uint32_t px = min(p_base_x + u, _private->dimensions.x - 1u);
      const uint32_t i = px + (_private->dimensions.y - 1u - py) * _private->dimensions.x;
      atomic_add_float(&ptr[i].x, value.x);
      atomic_add_float(&ptr[i].y, value.y);
      atomic_add_float(&ptr[i].z, value.z);
    }
  }
}

void Film::accumulate_camera_image(const uint2& pixel, const float3& color, const float3& normal, const float3& albedo) {
  if ((pixel.x >= _private->dimensions.x) || (pixel.y >= _private->dimensions.y)) {
    return;
  }

  auto var_data = _private->storage_buffers[StorageCameraAdaptive].data();
  auto int_data = _private->internal_data.data();

  auto colors_data = _private->storage_buffers[StorageCameraImage].data();
  auto normal_data = _private->storage_buffers[StorageNormals].data();
  auto albedo_data = _private->storage_buffers[StorageAlbedo].data();

  uint32_t p_base_x = (pixel.x / _private->pixel_size) * _private->pixel_size;
  uint32_t p_base_y = (pixel.y / _private->pixel_size) * _private->pixel_size;
  for (uint32_t py = p_base_y, pye = min(p_base_y + _private->pixel_size, _private->dimensions.y); py < pye; ++py) {
    for (uint32_t px = p_base_x, pxe = min(p_base_x + _private->pixel_size, _private->dimensions.x); px < pxe; ++px) {
      uint32_t i = px + (_private->dimensions.y - 1u - py) * _private->dimensions.x;

      uint32_t sample_index = int_data[i].sample_count;
      double ds = double(sample_index);

      if (sample_index == 0) {
        colors_data[i] = color;
        normal_data[i] = normal;
        albedo_data[i] = albedo;
        var_data[i] = color;
      } else {
        float t = float(ds / (ds + 1.0));

        colors_data[i] = {
          .x = lerp(color.x, colors_data[i].x, t),
          .y = lerp(color.y, colors_data[i].y, t),
          .z = lerp(color.z, colors_data[i].z, t),
        };
        normal_data[i] = {
          .x = lerp(normal.x, normal_data[i].x, t),
          .y = lerp(normal.y, normal_data[i].y, t),
          .z = lerp(normal.z, normal_data[i].z, t),
        };
        albedo_data[i] = {
          .x = lerp(albedo.x, albedo_data[i].x, t),
          .y = lerp(albedo.y, albedo_data[i].y, t),
          .z = lerp(albedo.z, albedo_data[i].z, t),
        };

        if ((sample_index % 2) == 0) {
          t = float(ds / (ds + 2.0));
          var_data[i] = {
            .x = lerp(color.x, var_data[i].x, t),
            .y = lerp(color.y, var_data[i].y, t),
            .z = lerp(color.z, var_data[i].z, t),
          };
        }
      }

      int_data[i].sample_count += 1u;
    }
  }
}

void Film::estimate_noise_levels(uint32_t sample_index, uint32_t total_samples, float threshold) {
  _private->max_sample_count = total_samples;

  if ((threshold == 0.0f) || (sample_index < kMinSamples) || (sample_index % 2) != 0)
    return;

#if (ETX_LOG_NOISE_LEVEL)
  auto t0 = std::chrono::steady_clock::now();
#endif

  auto var_data = _private->storage_buffers[StorageCameraAdaptive].data();
  auto cam_data = _private->storage_buffers[StorageCameraImage].data();
  auto int_data = _private->internal_data.data();

  _private->active_pixels = 0;
  _private->last_noise_level = 0.0f;
  _private->tasks.execute(_private->total_pixel_count(), [&](uint32_t begin, uint32_t end, uint32_t) {
    float total_noise = 0.0f;
    for (uint32_t i = begin; i < end; ++i) {
      if (int_data[i].converged)
        continue;

      const float3& v_i = cam_data[i];
      const float3& v_a = var_data[i];
      float error_diff = dot(abs(v_i - v_a), 1.0f);
      float error_norm = dot(abs(v_i), 1.0f);
      float error_level = error_diff / (((error_norm < 1.0f) ? sqrtf(error_norm) : error_norm) + kEpsilon);
      uint32_t converged = error_level < threshold ? 1u : 0u;

      int_data[i].error_level = error_level;
      int_data[i].converged = converged;
      int_data[i].tmp = converged;

      _private->active_pixels += converged;
      total_noise += error_level;
    }
    _private->last_noise_level.fetch_add(total_noise);
  });

#if (ETX_LOG_NOISE_LEVEL)
  auto t1 = std::chrono::steady_clock::now();
#endif

  constexpr uint32_t kBlockSize = 5u;

  if (_private->active_pixels > 0.0f) {
    _private->last_noise_level = _private->last_noise_level / float(_private->active_pixels);
  }

  _private->tasks.execute(_private->total_pixel_count(), [&](uint32_t begin, uint32_t end, uint32_t) {
    for (uint32_t i = begin; i < end; ++i) {
      if (int_data[i].converged) {
        continue;
      }

      uint32_t w = _private->dimensions.x;
      uint32_t x = i % w;
      uint32_t y = i / w;
      uint32_t begin_x = x >= kBlockSize ? x - kBlockSize : 0u;
      uint32_t end_x = min(w, x + kBlockSize);
      for (uint32_t p = begin_x; p < end_x; ++p) {
        int_data[p + y * w].tmp = 0;
      }
    }
  });

#if (ETX_LOG_NOISE_LEVEL)
  auto t2 = std::chrono::steady_clock::now();
#endif

  _private->tasks.execute(_private->total_pixel_count(), [&](uint32_t begin, uint32_t end, uint32_t) {
    for (uint32_t i = begin; i < end; ++i) {
      if (int_data[i].tmp) {
        continue;
      }

      uint32_t w = _private->dimensions.x;
      uint32_t h = _private->dimensions.y;
      uint32_t x = i % w;
      uint32_t y = i / w;

      uint32_t begin_y = y >= kBlockSize ? y - kBlockSize : 0u;
      uint32_t end_y = min(h, y + kBlockSize);
      for (uint32_t p = begin_y; p < end_y; ++p) {
        int_data[x + p * w].converged = 0;
      }
    }
  });

#if (ETX_LOG_NOISE_LEVEL)
  auto t3 = std::chrono::steady_clock::now();
  auto a0 = (t1 - t0).count() / 1.0e+6;
  auto a1 = (t2 - t1).count() / 1.0e+6;
  auto a2 = (t3 - t2).count() / 1.0e+6;
  auto a3 = (t3 - t0).count() / 1.0e+6;
  log::info("[%u] Estimated noise level in %.2fms (%.2f + %.2f + %.2f) -> %u active pixels", sample_index, a2, a0, a1, a2, _private->active_pixels.load());
#endif
}

void Film::commit_light_iteration(uint32_t i) {
  float t = float(double(i) / double(i + 1u));

  auto sptr = _private->storage_buffers[StorageLightIteration].data();
  auto dptr = _private->storage_buffers[StorageLightImage].data();

  uint64_t pixel_count = _private->total_pixel_count();
  for (uint64_t i = 0; i < pixel_count; ++i) {
    dptr[i] = (t == 0.0f) ? sptr[i] : lerp(sptr[i], dptr[i], t);
    sptr[i] = {};
  }
}

void Film::clear(uint32_t options) {
  if ((options & ClearCameraData) || (options & ClearEverything)) {
    memset(_private->internal_data.data(), 0, _private->internal_data.size() * sizeof(_private->internal_data[0]));
  }

  bool clear[StorageLayerCount] = {};
  clear[StorageLightImage] = options & ClearLightData;
  clear[StorageLightIteration] = clear[StorageLightImage] || (options & ClearLightIteration);
  for (auto id = 0; (options & ClearEverything) && (id < StorageLayerCount); ++id) {
    clear[id] = true;
  }

  auto& buffers = _private->storage_buffers;
  for (auto id = 0; id < StorageLayerCount; ++id) {
    if (clear[id]) {
      auto& buffer = buffers[id];
      memset(buffer.data(), 0, buffer.size() * sizeof(buffer[0]));
    }
  }

  _private->pixel_size = _private->target_pixel_size;
  _private->last_noise_level = {};
  _private->active_pixels = pixel_count();
}

const uint2& Film::size() const {
  return _private->dimensions;
}

uint2 Film::dimensions() const {
  return {
    (_private->dimensions.x + _private->pixel_size - 1u) / _private->pixel_size,
    (_private->dimensions.y + _private->pixel_size - 1u) / _private->pixel_size,
  };
}

float4* Film::layer(uint32_t layer) const {
  ETX_PROFILER_SCOPE();

  const auto layer_ref = layer_info[layer].storage;
  auto output = _private->output_data.data();

  if (layer == Debug) {
    const auto int_data = _private->internal_data.data();
    bool total_valid = _private->max_sample_count > kMinSamples;
    _private->tasks.execute(_private->total_pixel_count(), [&](uint32_t begin, uint32_t end, uint32_t) {
      for (uint32_t i = begin; i < end; ++i) {
        uint32_t pixel_sample_count = int_data[i].sample_count;
        double t = total_valid && (pixel_sample_count >= kMinSamples) ? double(pixel_sample_count - kMinSamples) / double(_private->max_sample_count - kMinSamples) : 0.0;
        float h = lerp(2.0f / 3.0f, 0.0f, float(t));
        output[i] = to_float4(hsv_to_rgb({h, 1.0f, 1.0f}));
      }
    });
  } else if (layer == Result) {
    ETX_PROFILER_SCOPE();
    auto c_buf = _private->storage_buffers[StorageCameraImage].data();
    auto l_buf = _private->storage_buffers[StorageLightImage].data();
    _private->tasks.execute(_private->total_pixel_count(), [&](uint32_t begin, uint32_t end, uint32_t) {
      for (uint32_t i = begin; i < end; ++i) {
        output[i] = to_float4(max({}, c_buf[i] + l_buf[i]));
      }
    });
  } else if (layer_ref != kInvalidIndex) {
    auto buf = _private->storage_buffers[layer_ref].data();
    _private->tasks.execute(_private->total_pixel_count(), [&](uint32_t begin, uint32_t end, uint32_t) {
      for (uint32_t i = begin; i < end; ++i) {
        float3 out = (layer == Normals) ? buf[i] * 0.5f + 0.5f : buf[i];
        output[i] = to_float4(out);
      }
    });
  }

  return output;
}

void Film::denoise(uint32_t layer_to_denoise) {
  const auto source = layer(layer_to_denoise);
  _private->denoiser.denoise(source, _private->storage_buffers[StorageDenoised].data());
}

uint32_t Film::pixel_count() const {
  uint2 dim = dimensions();
  return dim.x * dim.y;
}

uint32_t Film::active_pixel_count() const {
  return _private->active_pixels.load();
}

bool Film::active_pixel(uint32_t index, uint2& location) const {
  ETX_ASSERT(index < _private->total_pixel_count());

  uint32_t linear_index = index;
  const uint2& film_size = _private->dimensions;

  if (_private->pixel_size > 1) {
    uint2 dim = dimensions();
    uint2 a_location = {
      (index % dim.x) * _private->pixel_size,
      (index / dim.x) * _private->pixel_size,
    };
    a_location.x += rand() % _private->pixel_size;
    a_location.y += rand() % _private->pixel_size;
    linear_index = min(a_location.x, film_size.x - 1u) + min(a_location.y, film_size.y - 1u) * film_size.x;
    ETX_ASSERT(linear_index < _private->total_pixel_count());
  }

  location = {
    linear_index % film_size.x,
    linear_index / film_size.x,
  };
  uint32_t i = location.x + (_private->dimensions.y - 1u - location.y) * film_size.x;
  auto int_data = _private->internal_data.data();
  return int_data[i].converged == 0;
}

float Film::noise_level() const {
  return _private->last_noise_level;
}

const char* Film::layer_name(uint32_t layer) {
  static const char* names[] = {
    "Result",
    "Denoised",
    "Albedo",
    "Normals",
    "Camera Image",
    "Light Image",
    "Light Iteration",
    "Camera Adaptive",
    "Light Adaptive",
    "Debug",
  };
  static_assert(std::size(names) == LayerCount);
  ETX_ASSERT(layer < LayerCount);
  return names[layer];
}

void Film::set_pixel_size(uint32_t size) {
  _private->target_pixel_size = clamp(size, 1u, 1024u);
}

uint32_t Film::pixel_size() const {
  return _private->pixel_size;
}

}  // namespace etx
