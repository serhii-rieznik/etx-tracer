#include <etx/render/host/film.hxx>

#include <etx/core/core.hxx>
#include <etx/render/host/tasks.hxx>
#include <etx/render/host/denoiser.hxx>

#include <etx/render/shared/scene.hxx>

namespace etx {

namespace {

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

  TaskScheduler& tasks;
  Denoiser denoiser;
  uint2 dimensions = {};
  std::vector<float4> buffers[Film::LayerCount] = {};

  float4* layer(uint32_t layer) {
    ETX_ASSERT(layer < Film::LayerCount);
    auto& buffer = buffers[layer];
    return buffer.empty() ? nullptr : buffer.data();
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
    _private->dimensions = dim;
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

void Film::atomic_add(uint32_t layer, const float4& value, const float2& ndc_coord) {
  float2 uv = ndc_coord * 0.5f + 0.5f;
  uint32_t ax = static_cast<uint32_t>(uv.x * float(_private->dimensions.x));
  uint32_t ay = static_cast<uint32_t>(uv.y * float(_private->dimensions.y));
  atomic_add(layer, value, ax, ay);
}

void Film::atomic_add(uint32_t layer, const float4& value, uint32_t x, uint32_t y) {
  if ((x >= _private->dimensions.x) || (y >= _private->dimensions.y)) {
    return;
  }

  uint32_t i = x + (_private->dimensions.y - 1 - y) * _private->dimensions.x;
  auto ptr = _private->layer(layer) + i;
  atomic_add_float(&ptr->x, value.x);
  atomic_add_float(&ptr->y, value.y);
  atomic_add_float(&ptr->z, value.z);
  atomic_add_float(&ptr->w, value.w);
}

void Film::accumulate(uint32_t layer, const float4& value, const uint2& pixel, uint32_t sample_index) {
  if ((pixel.x >= _private->dimensions.x) || (pixel.y >= _private->dimensions.y) || (layer >= LayerCount)) {
    return;
  }

  uint32_t i = pixel.x + (_private->dimensions.y - 1u - pixel.y) * _private->dimensions.x;

  float4* layer_data = _private->layer(layer);
  float4* var_data = _private->layer(Data);

  float4 new_data_value = {value.x, value.y, value.z, 0.0f};
  if ((layer == CameraImage) && (sample_index % 2 == 0)) {
    if (sample_index > 0) {
      float t = float(sample_index / 2) / float(sample_index / 2 + 1u);
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
    float t = float(sample_index) / float(sample_index + 1u);
    const float4& existing_value = layer_data[i];
    new_value.x = lerp(value.x, existing_value.x, t);
    new_value.y = lerp(value.y, existing_value.y, t);
    new_value.z = lerp(value.z, existing_value.z, t);
  }
  layer_data[i] = new_value;
}

void Film::estimate_noise_levels(uint32_t sample_index, float threshold) {
  float4* var_data = _private->layer(Data);
  float4* cam_data = _private->layer(CameraImage);

  _private->tasks.execute(total_pixel_count(), [var_data, cam_data, sample_index](uint32_t begin, uint32_t end, uint32_t) {
    for (uint32_t i = begin; i < end; ++i) {
      const float3& v_i = *reinterpret_cast<const float3*>(var_data + i);
      const float3& v_a = *reinterpret_cast<const float3*>(cam_data + i);
      float error_diff =
        // fabsf(luminance(v_i) - luminance(v_a));
        dot(abs(v_i - v_a), 1.0f);
      float error_norm =
        // sqrtf(luminance({v_i.x, v_i.y, v_i.z}));
        sqrtf(dot(v_i, 1.0f));
      var_data[i].w = error_diff / (error_norm + kEpsilon);
    }
  });

  constexpr uint32_t block_size = 8u;
  uint2 blocks = _private->dimensions / block_size;

  _private->tasks.execute(blocks.x * blocks.y, [var_data, cam_data, sample_index, threshold, this](uint32_t begin, uint32_t end, uint32_t) {
    for (uint32_t block_i = begin; block_i < end; ++block_i) {
      uint2 b_loc = {
        block_i % (_private->dimensions.x / block_size),
        block_i / (_private->dimensions.x / block_size),
      };

      float avg_err = 0.0f;
      uint32_t x0 = b_loc.x * block_size;
      uint32_t x1 = min(x0 + block_size, _private->dimensions.x);

      uint32_t y0 = b_loc.y * block_size;
      uint32_t y1 = min(y0 + block_size, _private->dimensions.y);

      float area = 0.0f;
      for (uint32_t y = y0; y < y1; ++y) {
        for (uint32_t x = x0; x < x1; ++x) {
          avg_err += var_data[x + y * _private->dimensions.x].w;
          area += 1.0f;
        }
      }
      // float area = sqrtf(w / float(total_pixel_count()));
      avg_err *= 1.0f / (area + kEpsilon);

      float error_level = avg_err;
      for (uint32_t y = y0; y < y1; ++y) {
        for (uint32_t x = x0; x < x1; ++x) {
          cam_data[x + y * _private->dimensions.x].w = (sample_index < 64) || (error_level > threshold) ? 0.0f : 1.0f;
        }
      }
    }
  });
}

void Film::accumulate(uint32_t layer, const float4& value, const float2& ndc_coord, uint32_t sample_index) {
  float2 uv = ndc_coord * 0.5f + 0.5f;
  uint32_t ax = static_cast<uint32_t>(uv.x * float(_private->dimensions.x));
  uint32_t ay = static_cast<uint32_t>(uv.y * float(_private->dimensions.y));
  accumulate(layer, value, uint2{ax, ay}, sample_index);
}

void Film::commit_light_iteration(uint32_t i) {
  float t = float(double(i) / double(i + 1u));

  auto sptr = _private->layer(LightIteration);
  auto dptr = _private->layer(LightImage);

  uint64_t pixel_count = total_pixel_count();
  for (uint64_t i = 0; i < pixel_count; ++i) {
    dptr[i] = (t == 0.0f) ? sptr[i] : lerp(sptr[i], dptr[i], t);
    sptr[i] = {};
  }
}

const float4* Film::combined_result() const {
  return mutable_combined_result();
}

float4* Film::mutable_combined_result() const {
  _private->tasks.execute(total_pixel_count(), [this](uint32_t begin, uint32_t end, uint32_t) {
    for (uint32_t i = begin; i < end; ++i) {
      float4 c = _private->buffers[CameraImage][i];
      float4 l = _private->buffers[LightImage][i];
      _private->buffers[Result][i] = max({}, c + l);
      _private->buffers[Result][i] = max({}, c + l);
    }
  });
  return _private->buffers[Result].data();
}

void Film::clear(const Layers& layers) {
  for (uint32_t l : layers) {
    if (_private->buffers[l].empty() == false) {
      memset(_private->buffers[l].data(), 0, _private->buffers[l].size() * sizeof(float4));
    }
  }
}

void Film::clear() {
  for (auto& buffer : _private->buffers) {
    if (buffer.empty() == false) {
      memset(buffer.data(), 0, buffer.size() * sizeof(float4));
    }
  }
}

const uint2& Film::dimensions() const {
  return _private->dimensions;
}

const float4* Film::layer(uint32_t layer) const {
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

uint32_t Film::total_pixel_count() const {
  return _private->dimensions.x * _private->dimensions.y;
}

uint32_t Film::active_pixel_count() const {
  return _private->dimensions.x * _private->dimensions.y;
}

bool Film::active_pixel(uint32_t linear_index, uint2& location) {
  uint32_t sz = _private->dimensions.x;

  location = {
    linear_index % sz,
    linear_index / sz,
  };

  uint32_t i = location.x + (_private->dimensions.y - 1u - location.y) * _private->dimensions.x;
  return (_private->layer(CameraImage) + i)->w == 0.0f;
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
    "Data",
    "(data storage)",
  };
  static_assert(std::size(names) == LayerCount);
  ETX_ASSERT(layer < LayerCount);
  return names[layer];
}

}  // namespace etx
