#include <etx/render/host/film.hxx>

#include <etx/core/core.hxx>
#include <etx/render/host/tasks.hxx>
#include <etx/render/host/denoiser.hxx>

#include <etx/render/shared/scene.hxx>

// TODO : make better option
#include <../../../bin/shaders/shared/render_options.hxx>

#define ETX_LOG_NOISE_LEVEL 0

namespace etx {

namespace {

constexpr uint32_t kMinSamples = 32u;

enum StorageLayers : uint32_t {
  StorageAccumulation,
  StorageAdaptive,
  StorageNormals,
  StorageAlbedo,
  StorageDenoised,

  StorageLayerCount,
};

struct InternalData {
  float3 color = {};
  float error_level = 0.0f;
  uint32_t sample_count = 0;
  uint8_t written = 0;
  uint8_t converged = 0;
  uint8_t tmp = 0;
  uint8_t pad = 0;
};

struct LayerInfo {
  uint32_t layer_id = 0;
  uint32_t storage = 0;
} layer_info[ViewLayer::Count] = {
  {ViewLayer::Result, kInvalidIndex},
  {ViewLayer::Denoised, StorageDenoised},
  {ViewLayer::CurrentFrame, kInvalidIndex},
  {ViewLayer::Accumulation, StorageAccumulation},
  {ViewLayer::AdaptiveAccumulation, StorageAdaptive},
  {ViewLayer::Albedo, StorageAlbedo},
  {ViewLayer::Normals, StorageNormals},
  {ViewLayer::Debug, kInvalidIndex},
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

  void commit_iteration(const Scene& scene);
  void estimate_noise(uint32_t sample_index, uint32_t total_samples, float threshold);
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

void Film::submit(const float3& value, const float3& normal, const float3& albedo, const uint2& pixel) {
  if ((pixel.x >= _private->dimensions.x) || (pixel.y >= _private->dimensions.y)) {
    return;
  }

  auto int_data = _private->internal_data.data();
  auto normal_data = _private->storage_buffers[StorageNormals].data();
  auto albedo_data = _private->storage_buffers[StorageAlbedo].data();

  uint32_t p_base_x = (pixel.x / _private->pixel_size) * _private->pixel_size;
  uint32_t p_base_y = (pixel.y / _private->pixel_size) * _private->pixel_size;
  for (uint32_t py = p_base_y, pye = min(p_base_y + _private->pixel_size, _private->dimensions.y); py < pye; ++py) {
    for (uint32_t px = p_base_x, pxe = min(p_base_x + _private->pixel_size, _private->dimensions.x); px < pxe; ++px) {
      uint32_t i = px + (_private->dimensions.y - 1u - py) * _private->dimensions.x;
      auto& target = int_data[i].color;
      atomic_add_float(&target.x, value.x);
      atomic_add_float(&target.y, value.y);
      atomic_add_float(&target.z, value.z);

      const uint32_t sample_count = int_data[i].sample_count;
      if (sample_count == 0) {
        normal_data[i] = normal;
        albedo_data[i] = albedo;
      } else {
        float t = float(double(sample_count) / double(sample_count + 1u));
        normal_data[i] = lerp(normal, normal_data[i], t);
        albedo_data[i] = lerp(albedo, albedo_data[i], t);
      }

      int_data[i].written = 1;
    }
  }
}

void Film::submit(const float3& value, const float2& ndc_coord) {
  if (dot(value, value) < kEpsilon)
    return;

  float2 uv = ndc_coord * 0.5f + 0.5f;
  uint32_t x = static_cast<uint32_t>(uv.x * dimensions().x) * _private->pixel_size;
  uint32_t y = static_cast<uint32_t>(uv.y * dimensions().y) * _private->pixel_size;
  if ((x >= _private->dimensions.x) || (y >= _private->dimensions.y)) {
    return;
  }

  auto int_data = _private->internal_data.data();

  uint32_t p_base_x = (x / _private->pixel_size) * _private->pixel_size;
  uint32_t p_base_y = (y / _private->pixel_size) * _private->pixel_size;
  for (uint32_t py = p_base_y, pye = min(p_base_y + _private->pixel_size, _private->dimensions.y); py < pye; ++py) {
    for (uint32_t px = p_base_x, pxe = min(p_base_x + _private->pixel_size, _private->dimensions.x); px < pxe; ++px) {
      uint32_t i = px + (_private->dimensions.y - 1u - py) * _private->dimensions.x;
      auto& target = int_data[i].color;
      atomic_add_float(&target.x, value.x);
      atomic_add_float(&target.y, value.y);
      atomic_add_float(&target.z, value.z);
      int_data[i].written = 1;
    }
  }
}

void FilmImpl::estimate_noise(uint32_t sample_index, uint32_t total_samples, float threshold) {
  max_sample_count = total_samples;

  if ((threshold == 0.0f) || (sample_index < kMinSamples) || (sample_index % 2) != 0)
    return;

#if (ETX_LOG_NOISE_LEVEL)
  auto t0 = std::chrono::steady_clock::now();
#endif

  auto var_data = storage_buffers[StorageAdaptive].data();
  auto cam_data = storage_buffers[StorageAccumulation].data();
  auto int_data = internal_data.data();

  active_pixels = 0;
  last_noise_level = 0.0f;
  tasks.execute(total_pixel_count(), [&](uint32_t begin, uint32_t end, uint32_t) {
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

      active_pixels += converged;
      total_noise += error_level;
    }
    last_noise_level.fetch_add(total_noise);
  });

#if (ETX_LOG_NOISE_LEVEL)
  auto t1 = std::chrono::steady_clock::now();
#endif

  constexpr uint32_t kBlockSize = 5u;

  if (active_pixels > 0.0f) {
    last_noise_level = last_noise_level / float(active_pixels);
  }

  tasks.execute(total_pixel_count(), [&](uint32_t begin, uint32_t end, uint32_t) {
    for (uint32_t i = begin; i < end; ++i) {
      if (int_data[i].converged) {
        continue;
      }

      uint32_t w = dimensions.x;
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

  tasks.execute(total_pixel_count(), [&](uint32_t begin, uint32_t end, uint32_t) {
    for (uint32_t i = begin; i < end; ++i) {
      if (int_data[i].tmp) {
        continue;
      }

      uint32_t w = dimensions.x;
      uint32_t h = dimensions.y;
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

void FilmImpl::commit_iteration(const Scene& scene) {
  auto int_data = internal_data.data();
  auto accumumlation = storage_buffers[StorageAccumulation].data();
  auto adaptive = storage_buffers[StorageAdaptive].data();

  uint64_t pixel_count = total_pixel_count();
  for (uint64_t i = 0; i < pixel_count; ++i) {
    auto& idata = int_data[i];
    if (idata.written == 0)
      continue;

    const uint32_t sample_count = idata.sample_count;

    // Apply radiance clamping to accumulated color before blending
    if (scene.options.radiance_clamp > 0.0f) {
      float lum = luminance(idata.color);
      if (lum > scene.options.radiance_clamp) {
        idata.color *= scene.options.radiance_clamp / lum;
      }
    }

    if (sample_count == 0) {
      accumumlation[i] = idata.color;
      adaptive[i] = idata.color;
    } else {
      float t = float(double(sample_count) / double(sample_count + 1u));
      accumumlation[i] = lerp(idata.color, accumumlation[i], t);
      if (sample_count % 2 == 0) {
        uint32_t adaptive_sample_count = sample_count / 2u;
        t = float(double(adaptive_sample_count) / double(adaptive_sample_count + 1u));
        adaptive[i] = lerp(idata.color, adaptive[i], t);
      }
    }

    idata.color = {};
    idata.written = 0;
    idata.sample_count++;
  }
}

void Film::commit_iteration(uint32_t sample_index, const Scene& scene) {
  _private->commit_iteration(scene);
  _private->estimate_noise(sample_index, scene.options.samples, scene.options.noise_threshold);
}

void Film::clear(uint32_t options) {
  bool clear_all = options & ClearEverything;
  bool clear_frame = clear_all || (options & ClearIteration);

  if (clear_all) {
    memset(_private->internal_data.data(), 0, _private->internal_data.size() * sizeof(_private->internal_data[0]));
  }

  if (clear_frame) {
    for (auto& i : _private->internal_data) {
      i.color = {};
      i.written = 0;
    }
  }

  bool clear[StorageLayerCount] = {};
  clear[StorageAccumulation] = clear_all;
  clear[StorageAdaptive] = clear_all;
  clear[StorageNormals] = clear_all;
  clear[StorageAlbedo] = clear_all;
  clear[StorageDenoised] = clear_all;

  auto& buffers = _private->storage_buffers;
  for (auto id = 0; id < StorageLayerCount; ++id) {
    if (clear[id]) {
      auto& buffer = buffers[id];
      memset(buffer.data(), 0, buffer.size() * sizeof(buffer[0]));
    }
  }

  if (clear_all) {
    _private->last_noise_level = {};
    _private->active_pixels = pixel_count();
  }

  _private->pixel_size = _private->target_pixel_size;
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

float4* Film::layer(uint32_t layer, const Scene& scene) const {
  ETX_PROFILER_SCOPE();

  const auto layer_ref = layer_info[layer].storage;
  auto output = _private->output_data.data();

  if (layer == ViewLayer::Debug) {
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
  } else if (layer == ViewLayer::Result) {
    ETX_PROFILER_SCOPE();
    auto accum = _private->storage_buffers[StorageAccumulation].data();
    auto current = _private->internal_data.data();
    _private->tasks.execute(_private->total_pixel_count(), [&](uint32_t begin, uint32_t end, uint32_t) {
      for (uint32_t i = begin; i < end; ++i) {
        float3 color;
        const auto& curr = current[i];

        // Clamp current iteration's color before blending
        float3 clamped_curr_color = curr.color;
        if (scene.options.radiance_clamp > 0.0f) {
          float lum = luminance(clamped_curr_color);
          if (lum > scene.options.radiance_clamp) {
            clamped_curr_color *= scene.options.radiance_clamp / lum;
          }
        }

        if (curr.sample_count == 0) {
          color = max({}, clamped_curr_color);
        } else if (current[i].written == 0) {
          color = max({}, accum[i]);
        } else {
          uint32_t effective_sample_count = current[i].sample_count + 1u;
          float t = float(double(effective_sample_count) / double(effective_sample_count + 1u));
          color = max({}, lerp(clamped_curr_color, accum[i], t));
        }

        output[i] = to_float4(color);
      }
    });
  } else if (layer == ViewLayer::CurrentFrame) {
    ETX_PROFILER_SCOPE();
    auto frame = _private->internal_data.data();
    _private->tasks.execute(_private->total_pixel_count(), [&](uint32_t begin, uint32_t end, uint32_t) {
      for (uint32_t i = begin; i < end; ++i) {
        output[i] = to_float4(frame[i].color);
      }
    });
  } else if (layer_ref != kInvalidIndex) {
    ETX_PROFILER_SCOPE();
    auto buf = _private->storage_buffers[layer_ref].data();
    _private->tasks.execute(_private->total_pixel_count(), [&](uint32_t begin, uint32_t end, uint32_t) {
      for (uint32_t i = begin; i < end; ++i) {
        output[i] = to_float4(buf[i]);
      }
    });
  }

  return output;
}

void Film::denoise(uint32_t layer_to_denoise, const Scene& scene) {
  const auto source = layer(layer_to_denoise, scene);
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
    "Current Frame",
    "Accumulation",
    "Adaptive Accumulation",
    "Albedo",
    "Normals",
    "Debug",
  };
  static_assert(std::size(names) == ViewLayer::Count);
  ETX_ASSERT(layer < ViewLayer::Count);
  return names[layer];
}

void Film::set_pixel_size(uint32_t size) {
  _private->target_pixel_size = clamp(size, 1u, 1024u);
}

uint32_t Film::pixel_size() const {
  return _private->pixel_size;
}

}  // namespace etx
