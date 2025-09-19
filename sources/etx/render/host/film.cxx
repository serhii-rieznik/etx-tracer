#include <etx/render/host/film.hxx>

#include <etx/core/core.hxx>
#include <etx/render/host/tasks.hxx>
#include <etx/render/host/denoiser.hxx>

#include <etx/render/shared/scene.hxx>

#define ETX_LOG_NOISE_LEVEL 0

namespace etx {

namespace {

constexpr uint32_t kMinSamples = 32u;

struct InternalData {
  float sample_count = 0.0f;
  float tmp0 = 0.0f;
  float tmp1 = 0.0f;
  float converged = 0.0f;
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
  std::vector<float4> buffers[Film::LayerCount] = {};
  std::atomic<float> last_noise_level = {};
  std::atomic<uint32_t> active_pixels = {};
  uint32_t max_sample_count = 0u;
  uint32_t pixel_size = 1u;
  uint32_t target_pixel_size = 1u;

  float4* layer(uint32_t layer) {
    ETX_ASSERT(layer < Film::LayerCount);
    auto& buffer = buffers[layer];
    return buffer.empty() ? nullptr : buffer.data();
  }

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
    for (auto& buffer : _private->buffers) {
      buffer.clear();
      buffer.resize(1llu * _private->dimensions.x * _private->dimensions.y);
    }
    float4* source = mutable_layer(Film::Result);
    float4* albedo = mutable_layer(Film::Albedo);
    float4* normals = mutable_layer(Film::Normals);
    float4* denoised = mutable_layer(Film::Denoised);
    _private->denoiser.allocate_buffers(source, albedo, normals, denoised, _private->dimensions);
  }
  clear(kAllLayers);
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

void Film::atomic_add(uint32_t layer, const float3& value, const float2& ndc_coord) {
  if (dot(value, value) == 0.0f)
    return;

  float2 uv = ndc_coord * 0.5f + 0.5f;
  uint32_t ax = static_cast<uint32_t>(uv.x * dimensions().x) * _private->pixel_size;
  uint32_t ay = static_cast<uint32_t>(uv.y * dimensions().y) * _private->pixel_size;
  atomic_add(layer, value, ax, ay);
}

void Film::atomic_add(uint32_t layer, const float3& value, uint32_t x, uint32_t y) {
  if ((x >= _private->dimensions.x) || (y >= _private->dimensions.y)) {
    return;
  }

  const uint32_t p_base_x = (x / _private->pixel_size) * _private->pixel_size;
  const uint32_t p_base_y = (y / _private->pixel_size) * _private->pixel_size;
  auto ptr = _private->layer(layer);
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

void Film::accumulate(uint32_t layer, const float3& value, const uint2& a_pixel) {
  if ((a_pixel.x >= _private->dimensions.x) || (a_pixel.y >= _private->dimensions.y) || (layer >= LayerCount)) {
    return;
  }

  auto layer_data = _private->layer(layer);
  auto var_data = _private->layer(Adaptive);
  auto int_data = reinterpret_cast<InternalData*>(_private->layer(Internal));

  uint32_t p_base_x = (a_pixel.x / _private->pixel_size) * _private->pixel_size;
  uint32_t p_base_y = (a_pixel.y / _private->pixel_size) * _private->pixel_size;
  for (uint32_t py = p_base_y, pye = min(p_base_y + _private->pixel_size, _private->dimensions.y); py < pye; ++py) {
    for (uint32_t px = p_base_x, pxe = min(p_base_x + _private->pixel_size, _private->dimensions.x); px < pxe; ++px) {
      uint32_t i = px + (_private->dimensions.y - 1u - py) * _private->dimensions.x;
      float sample_index = int_data[i].sample_count;
      float4 new_data_value = {value.x, value.y, value.z, 0.0f};
      if ((layer == CameraImage) && (uint32_t(sample_index) % 2 == 0)) {
        if (sample_index > 0.0f) {
          float t = float(0.5f * sample_index) / float(0.5f * sample_index + 1.0f);
          const float4& existing_value = var_data[i];
          new_data_value.x = lerp(new_data_value.x, existing_value.x, t);
          new_data_value.y = lerp(new_data_value.y, existing_value.y, t);
          new_data_value.z = lerp(new_data_value.z, existing_value.z, t);
          new_data_value.w = existing_value.w;
        }
        var_data[i] = new_data_value;
      }

      float4 new_value = {value.x, value.y, value.z, 0.0f};
      if (sample_index > 0) {
        float t = sample_index / (sample_index + 1.0f);
        const float4& existing_value = layer_data[i];
        new_value.x = lerp(value.x, existing_value.x, t);
        new_value.y = lerp(value.y, existing_value.y, t);
        new_value.z = lerp(value.z, existing_value.z, t);
      }
      layer_data[i] = new_value;

      if (layer == CameraImage) {
        int_data[i].sample_count += 1.0f;
      }
    }
  }
}

void Film::accumulate(const uint2& pixel, const std::initializer_list<LayerValue>& lv) {
  const auto& ptr = lv.begin();
  for (uint64_t i = 0, e = lv.size(); i < e; ++i) {
    accumulate(ptr[i].layer, ptr[i].value, pixel);
  }
}

void Film::estimate_noise_levels(uint32_t sample_index, uint32_t total_samples, float threshold) {
  _private->max_sample_count = total_samples;

  if ((threshold == 0.0f) || (sample_index < kMinSamples) || (sample_index % 2) != 0)
    return;

#if (ETX_LOG_NOISE_LEVEL)
  auto t0 = std::chrono::steady_clock::now();
#endif

  float4* var_data = _private->layer(Adaptive);
  float4* cam_data = _private->layer(CameraImage);
  auto int_data = reinterpret_cast<InternalData*>(_private->layer(Internal));

  _private->active_pixels = 0;
  _private->last_noise_level = 0.0f;
  _private->tasks.execute(_private->total_pixel_count(), [&](uint32_t begin, uint32_t end, uint32_t) {
    float total_noise = 0.0f;
    for (uint32_t i = begin; i < end; ++i) {
      if (int_data[i].converged > 0.0f)
        continue;

      float3 v_i = to_float3(cam_data[i]);
      float3 v_a = to_float3(var_data[i]);
      float error_diff = dot(abs(v_i - v_a), 1.0f);
      float error_norm = dot(abs(v_i), 1.0f);
      float error_level = error_diff / (((error_norm < 1.0f) ? sqrtf(error_norm) : error_norm) + kEpsilon);
      bool converged = error_level < threshold;

      var_data[i].w = error_level;
      int_data[i].converged = float(converged);
      int_data[i].tmp0 = float(converged);

      _private->active_pixels += converged ? 0u : 1u;
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
      if (int_data[i].converged > 0.0f) {
        continue;
      }

      uint32_t w = _private->dimensions.x;
      uint32_t x = i % w;
      uint32_t y = i / w;
      uint32_t begin_x = x >= kBlockSize ? x - kBlockSize : 0u;
      uint32_t end_x = min(w, x + kBlockSize);
      for (uint32_t p = begin_x; p < end_x; ++p) {
        int_data[p + y * w].tmp0 = 0.0f;
      }
    }
  });

#if (ETX_LOG_NOISE_LEVEL)
  auto t2 = std::chrono::steady_clock::now();
#endif

  _private->tasks.execute(_private->total_pixel_count(), [&](uint32_t begin, uint32_t end, uint32_t) {
    for (uint32_t i = begin; i < end; ++i) {
      if (int_data[i].tmp0 > 0.0f) {
        continue;
      }

      uint32_t w = _private->dimensions.x;
      uint32_t h = _private->dimensions.y;
      uint32_t x = i % w;
      uint32_t y = i / w;

      uint32_t begin_y = y >= kBlockSize ? y - kBlockSize : 0u;
      uint32_t end_y = min(h, y + kBlockSize);
      for (uint32_t p = begin_y; p < end_y; ++p) {
        int_data[x + p * w].converged = 0.0f;
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

  auto sptr = _private->layer(LightIteration);
  auto dptr = _private->layer(LightImage);

  uint64_t pixel_count = _private->total_pixel_count();
  for (uint64_t i = 0; i < pixel_count; ++i) {
    dptr[i] = (t == 0.0f) ? sptr[i] : lerp(sptr[i], dptr[i], t);
    sptr[i] = {};
  }
}

const float4* Film::combined_result() const {
  return mutable_combined_result();
}

float4* Film::mutable_combined_result() const {
  ETX_PROFILER_SCOPE();
  auto c_buf = _private->buffers[CameraImage].data();
  auto l_buf = _private->buffers[LightImage].data();
  auto r_buf = _private->buffers[Result].data();
  _private->tasks.execute(_private->total_pixel_count(), [&](uint32_t begin, uint32_t end, uint32_t) {
    for (uint32_t i = begin; i < end; ++i) {
      r_buf[i] = _private->buffers[Result][i] = max({}, c_buf[i] + l_buf[i]);
    }
  });
  return r_buf;
}

void Film::clear(const Layers& layers) {
  for (uint32_t l : layers) {
    if (_private->buffers[l].empty() == false) {
      memset(_private->buffers[l].data(), 0, _private->buffers[l].size() * sizeof(float4));
    }
  }
  _private->pixel_size = _private->target_pixel_size;
  _private->last_noise_level = {};
  _private->active_pixels = pixel_count();
}

void Film::clear() {
  for (auto& buffer : _private->buffers) {
    if (buffer.empty() == false) {
      memset(buffer.data(), 0, buffer.size() * sizeof(float4));
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

const float4* Film::layer(uint32_t layer) const {
  ETX_PROFILER_SCOPE();
  if (layer == Debug) {
    const auto int_data = reinterpret_cast<InternalData*>(_private->layer(Internal));
    float4* dbg_data = _private->layer(Debug);

    _private->tasks.execute(_private->total_pixel_count(), [&](uint32_t begin, uint32_t end, uint32_t) {
      for (uint32_t i = begin; i < end; ++i) {
        float pixel_sample_count = int_data[i].sample_count;

        float t = (pixel_sample_count >= kMinSamples) && (_private->max_sample_count > kMinSamples)
                    ? float(pixel_sample_count - kMinSamples) / float(_private->max_sample_count - kMinSamples)
                    : 0.0f;

        float h = lerp(2.0f / 3.0f, 0.0f, t);
        dbg_data[i] = to_float4(hsv_to_rgb({h, 1.0f, 1.0f}));
      }
    });
  }

  switch (layer) {
    case Result:
      return combined_result();
    default:
      return _private->layer(layer);
  }
}

float4* Film::mutable_layer(uint32_t layer) const {
  switch (layer) {
    case Result:
      return mutable_combined_result();
    default:
      return _private->layer(layer);
  }
}

void Film::denoise() {
  _private->denoiser.denoise();
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
  const InternalData* int_data = reinterpret_cast<InternalData*>(_private->layer(Internal));
  return int_data[i].converged == 0.0f;
}

float Film::noise_level() const {
  return _private->last_noise_level;
}

const char* Film::layer_name(uint32_t layer) {
  static const char* names[] = {
    "Camera Image",
    "Light Image",
    "Light Iteration",
    "Normals",
    "Albedo",
    "Result",
    "Denoised",
    "Adaptive",
    "Internal",
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
