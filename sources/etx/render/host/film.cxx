#include <etx/core/core.hxx>
#include <etx/render/host/tasks.hxx>
#include <etx/render/host/film.hxx>
#include <etx/render/shared/spectrum.hxx>

#include <vector>

namespace etx {

struct FilmImpl {
  FilmImpl(TaskScheduler& t)
    : tasks(t) {
  }

  TaskScheduler& tasks;
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

void Film::allocate(const uint2& dim, const Layers& layers) {
  _private->dimensions = dim;

  for (auto& buffer : _private->buffers) {
    buffer.clear();
  }

  for (uint32_t l : layers) {
    if (l < LayerCount) {
      _private->buffers[l].resize(1llu * _private->dimensions.x * _private->dimensions.y);
    }
  }

  clear(layers);
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

  ETX_VALIDATE(value);
  uint32_t index = x + y * _private->dimensions.x;
  auto ptr = _private->layer(layer) + index;
  atomic_add_float(&ptr->x, value.x);
  atomic_add_float(&ptr->y, value.y);
  atomic_add_float(&ptr->z, value.z);
  atomic_add_float(&ptr->w, value.w);
}

void Film::accumulate(uint32_t layer, const float4& value, uint32_t x, uint32_t y, float t) {
  if ((x >= _private->dimensions.x) || (y >= _private->dimensions.y) || (layer >= LayerCount)) {
    return;
  }
  ETX_VALIDATE(value);

  float4* layer_data = _private->layer(layer);

  uint32_t i = x + (_private->dimensions.y - 1 - y) * _private->dimensions.x;
  float4 new_value = {value.x, value.y, value.z, 1.0f};
  if (t > 0.0f) {
    const float4& existing_value = layer_data[i];
    new_value.x = lerp(value.x, existing_value.x, t);
    new_value.y = lerp(value.y, existing_value.y, t);
    new_value.z = lerp(value.z, existing_value.z, t);
  }
  layer_data[i] = new_value;
}

void Film::accumulate(uint32_t layer, const float4& value, const float2& ndc_coord, float t) {
  float2 uv = ndc_coord * 0.5f + 0.5f;
  uint32_t ax = static_cast<uint32_t>(uv.x * float(_private->dimensions.x));
  uint32_t ay = static_cast<uint32_t>(uv.y * float(_private->dimensions.y));
  accumulate(layer, value, ax, ay, t);
}

void Film::commit_light_iteration(uint32_t i) {
  float t = float(double(i) / double(i + 1u));

  auto sptr = _private->layer(LightIteration);
  auto dptr = _private->layer(LightImage);

  uint64_t pixel_count = count();
  for (uint64_t i = 0; i < pixel_count; ++i) {
    dptr[i] = (t == 0.0f) ? sptr[i] : lerp(sptr[i], dptr[i], t);
    sptr[i] = {};
  }
}

const float4* Film::combined_result() const {
  _private->tasks.execute(count(), [this](uint32_t begin, uint32_t end, uint32_t) {
    for (uint32_t i = begin; i < end; ++i) {
      _private->buffers[Result][i] = _private->buffers[Camera][i] + _private->buffers[LightImage][i];
    }
  });
  return _private->buffers[Result].data();
}

void Film::flush_to(Film& other, float t, const Layers& layers) {
  ETX_ASSERT(_private->dimensions == other._private->dimensions);

  const float4* src[LayerCount] = {
    _private->layer(Camera),
    _private->layer(LightImage),
    _private->layer(Normals),
    _private->layer(Albedo),
  };

  float4* dst[LayerCount] = {
    other._private->layer(Camera),
    other._private->layer(LightImage),
    other._private->layer(Normals),
    other._private->layer(Albedo),
  };

  uint64_t pixel_count = count();
  for (uint64_t i = 0; i < pixel_count; ++i) {
    for (uint32_t l : layers) {
      const auto sptr = src[l];
      auto dptr = dst[l];
      if ((sptr != nullptr) && (dptr != nullptr)) {
        dptr[i] = (t == 0.0f) ? sptr[i] : lerp(sptr[i], dptr[i], t);
      }
    }
  }
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

const uint32_t Film::count() const {
  return _private->dimensions.x * _private->dimensions.y;
}

const float4* Film::layer(uint32_t layer) const {
  return (layer == Result) ? combined_result() : _private->layer(layer);
}

float4* Film::mutable_layer(uint32_t layer) {
  return _private->layer(layer);
}

const char* Film::layer_name(uint32_t layer) {
  static const char* names[] = {
    "Camera",
    "Light Image",
    "Light Iteration",
    "Normals",
    "Albedo",
    "Result",
    "Denoised",
  };
  static_assert(std::size(names) == LayerCount);
  ETX_ASSERT(layer < LayerCount);
  return names[layer];
}

}  // namespace etx
