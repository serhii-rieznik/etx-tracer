#include <etx/render/host/film.hxx>

#include <etx/core/core.hxx>
#include <etx/render/host/tasks.hxx>
#include <etx/render/host/denoiser.hxx>

#include <etx/render/shared/scene.hxx>
#include <etx/render/shared/scene_camera.hxx>

// TODO : make better option
#include <interop/render_options.hxx>

namespace etx {

namespace {

enum StorageLayers : uint32_t {
  StorageAccumulation,
  StorageNormals,
  StorageAlbedo,
  StorageDenoised,

  StorageLayerCount,
};

struct InternalData {
  float3 color = {};
  uint32_t sample_count = 0;
  uint8_t written = 0;
};

struct LayerInfo {
  uint32_t layer_id = 0;
  uint32_t storage = 0;
} layer_info[ViewLayer::Count] = {
  {ViewLayer::Result, kInvalidIndex},
  {ViewLayer::Denoised, StorageDenoised},
  {ViewLayer::CurrentFrame, kInvalidIndex},
  {ViewLayer::Accumulation, StorageAccumulation},
  {kInvalidIndex, kInvalidIndex},
  {ViewLayer::Albedo, StorageAlbedo},
  {ViewLayer::Normals, StorageNormals},
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
  uint32_t pixel_size = 1u;
  uint32_t target_pixel_size = 1u;
  uint2 render_window_origin = {};
  uint2 render_window_size = {};

  uint32_t total_pixel_count() const {
    return dimensions.x * dimensions.y;
  }

  uint2 active_dimensions() const {
    if ((render_window_size.x == 0u) || (render_window_size.y == 0u)) {
      return dimensions;
    }

    return render_window_size;
  }

  void commit_iteration(float radiance_clamp);
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
    if ((_private->render_window_size.x > _private->dimensions.x) || (_private->render_window_size.y > _private->dimensions.y) ||
        (_private->render_window_origin.x >= _private->dimensions.x) || (_private->render_window_origin.y >= _private->dimensions.y) ||
        (_private->render_window_size.x > (_private->dimensions.x - _private->render_window_origin.x)) ||
        (_private->render_window_size.y > (_private->dimensions.y - _private->render_window_origin.y))) {
      _private->render_window_origin = {};
      _private->render_window_size = _private->dimensions;
    }

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

void Film::release() {
  _private->denoiser.release_buffers();
  for (auto& buffer : _private->storage_buffers) {
    std::vector<float3>().swap(buffer);
  }
  std::vector<float4>().swap(_private->output_data);
  std::vector<InternalData>().swap(_private->internal_data);
  _private->dimensions = {};
  _private->render_window_origin = {};
  _private->render_window_size = {};
}

void Film::reset_render_window() {
  _private->render_window_origin = {};
  _private->render_window_size = _private->dimensions;
}

bool Film::set_render_window(const uint2& origin, const uint2& size) {
  if ((size.x == 0u) || (size.y == 0u)) {
    return false;
  }

  _private->render_window_origin = origin;
  _private->render_window_size = size;
  return true;
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

float2 Film::sample(const PixelFilter& sampler, const uint2& pixel, const float2& pixel_rnd, const float2& filter_rnd) const {
  return pixel_filter_sample_uv(pixel, _private->dimensions, pixel_rnd, sample_pixel_filter_offset(sampler, filter_rnd));
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
  if ((value.x == 0.0f) && (value.y == 0.0f) && (value.z == 0.0f)) {
    return;
  }

  if (pixel_filter_contains_uv(ndc_coord) == false) {
    return;
  }
  float2 uv = ndc_coord * 0.5f + 0.5f;
  uint32_t x = static_cast<uint32_t>(uv.x * current_dimensions().x) * _private->pixel_size;
  uint32_t y = static_cast<uint32_t>(uv.y * current_dimensions().y) * _private->pixel_size;
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

void FilmImpl::commit_iteration(float radiance_clamp) {
  auto int_data = internal_data.data();
  auto accumumlation = storage_buffers[StorageAccumulation].data();

  uint64_t pixel_count = total_pixel_count();
  for (uint64_t i = 0; i < pixel_count; ++i) {
    auto& idata = int_data[i];
    if (idata.written == 0)
      continue;

    const uint32_t sample_count = idata.sample_count;

    // Apply radiance clamping to accumulated color before blending
    if (radiance_clamp > 0.0f) {
      float lum = luminance(idata.color);
      if (lum > radiance_clamp) {
        idata.color *= radiance_clamp / lum;
      }
    }

    if (sample_count == 0) {
      accumumlation[i] = idata.color;
    } else {
      float t = float(double(sample_count) / double(sample_count + 1u));
      accumumlation[i] = lerp(idata.color, accumumlation[i], t);
    }

    idata.color = {};
    idata.written = 0;
    idata.sample_count++;
  }
}

void Film::commit_iteration(float radiance_clamp) {
  _private->commit_iteration(radiance_clamp);
}

void Film::clear(uint32_t options) {
  bool clear_all = options & ClearEverything;
  bool clear_frame = clear_all || (options & ClearIteration);

  if (clear_all) {
    memset(_private->internal_data.data(), 0, _private->internal_data.size() * sizeof(_private->internal_data[0]));
  }

  if (clear_frame && (clear_all == false)) {
    for (auto& i : _private->internal_data) {
      i.color = {};
      i.written = 0;
    }
  }

  bool clear[StorageLayerCount] = {};
  clear[StorageAccumulation] = clear_all;
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

  _private->pixel_size = _private->target_pixel_size;
}

uint2 Film::base_dimensions() const {
  return _private->dimensions;
}

uint2 Film::current_dimensions() const {
  return {
    (_private->dimensions.x + _private->pixel_size - 1u) / _private->pixel_size,
    (_private->dimensions.y + _private->pixel_size - 1u) / _private->pixel_size,
  };
}

float4* Film::layer(uint32_t layer, float radiance_clamp) const {
  ETX_PROFILER_SCOPE();

  ETX_ASSERT(layer_name(layer) != nullptr);
  const auto layer_ref = layer_info[layer].storage;
  auto output = _private->output_data.data();

  if (layer == ViewLayer::Result) {
    ETX_PROFILER_SCOPE();
    auto accum = _private->storage_buffers[StorageAccumulation].data();
    auto current = _private->internal_data.data();
    _private->tasks.execute(_private->total_pixel_count(), [&](uint32_t begin, uint32_t end, uint32_t) {
      for (uint32_t i = begin; i < end; ++i) {
        float3 color;
        const auto& curr = current[i];

        // Clamp current iteration's color before blending
        float3 clamped_curr_color = curr.color;
        if (radiance_clamp > 0.0f) {
          float lum = luminance(clamped_curr_color);
          if (lum > radiance_clamp) {
            clamped_curr_color *= radiance_clamp / lum;
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

void Film::denoise(uint32_t layer_to_denoise, float radiance_clamp) {
  const auto source = layer(layer_to_denoise, radiance_clamp);
  _private->denoiser.denoise(source, _private->storage_buffers[StorageDenoised].data());
}

uint32_t Film::total_pixel_count() const {
  uint2 dim = base_dimensions();
  return dim.x * dim.y;
}

uint32_t Film::current_pixel_count() const {
  uint2 dim = _private->active_dimensions();
  if (_private->pixel_size > 1u) {
    dim = {
      (dim.x + _private->pixel_size - 1u) / _private->pixel_size,
      (dim.y + _private->pixel_size - 1u) / _private->pixel_size,
    };
  }
  return dim.x * dim.y;
}

Film::MemoryStats Film::memory_stats() const {
  MemoryStats result = {};
  result.accumulation_bytes = _private->storage_buffers[StorageAccumulation].capacity() * sizeof(float3);
  result.normals_bytes = _private->storage_buffers[StorageNormals].capacity() * sizeof(float3);
  result.albedo_bytes = _private->storage_buffers[StorageAlbedo].capacity() * sizeof(float3);
  result.denoised_bytes = _private->storage_buffers[StorageDenoised].capacity() * sizeof(float3);
  result.output_bytes = _private->output_data.capacity() * sizeof(float4);
  result.internal_bytes = _private->internal_data.capacity() * sizeof(InternalData);
  return result;
}

uint2 Film::pixel_location(uint32_t index) const {
  ETX_ASSERT(index < current_pixel_count());

  uint32_t linear_index = index;
  const uint2& film_size = _private->dimensions;
  const uint2 render_window_origin = _private->render_window_origin;
  const uint2 render_window_size = _private->active_dimensions();
  const uint32_t render_window_origin_y = film_size.y - render_window_origin.y - render_window_size.y;

  if (_private->pixel_size > 1) {
    const uint2 dim = {
      (render_window_size.x + _private->pixel_size - 1u) / _private->pixel_size,
      (render_window_size.y + _private->pixel_size - 1u) / _private->pixel_size,
    };
    uint2 a_location = {
      render_window_origin.x + (index % dim.x) * _private->pixel_size,
      render_window_origin_y + (index / dim.x) * _private->pixel_size,
    };
    a_location.x += rand() % _private->pixel_size;
    a_location.y += rand() % _private->pixel_size;
    linear_index = min(a_location.x, render_window_origin.x + render_window_size.x - 1u) + min(a_location.y, render_window_origin_y + render_window_size.y - 1u) * film_size.x;
    ETX_ASSERT(linear_index < _private->total_pixel_count());
  } else {
    const uint32_t local_x = index % render_window_size.x;
    const uint32_t local_y = index / render_window_size.x;
    linear_index = (render_window_origin.x + local_x) + (render_window_origin_y + local_y) * film_size.x;
  }

  return {
    linear_index % film_size.x,
    linear_index / film_size.x,
  };
}

const char* Film::layer_name(uint32_t layer) {
  static const char* names[] = {
    "Result",
    "Denoised",
    "Current Frame",
    "Accumulation",
    nullptr,
    "Albedo",
    "Normals",
  };
  static_assert(std::size(names) == ViewLayer::Count);
  return (layer < ViewLayer::Count) ? names[layer] : nullptr;
}

void Film::set_pixel_size(uint32_t size) {
  _private->target_pixel_size = clamp(size, 1u, 1024u);
}

uint32_t Film::pixel_size() const {
  return _private->pixel_size;
}

}  // namespace etx
