#include <etx/core/core.hxx>

#include <etx/render/host/image_pool.hxx>
#include <etx/render/host/distribution_builder.hxx>
#include <etx/render/host/pool.hxx>

#include <tinyexr.hxx>
#include <stb_image.hxx>

#include <vector>
#include <unordered_map>
#include <functional>

namespace etx {

bool load_pfm(const char* path, uint2& size, std::vector<uint8_t>& data);

struct ImagePoolImpl {
  ImagePoolImpl(TaskScheduler& s)
    : scheduler(s) {
  }

  void init(uint32_t capacity) {
    image_pool.init(capacity);
  }

  void cleanup() {
    ETX_ASSERT(image_pool.count_alive() == 0);
    image_pool.cleanup();
  }

  uint32_t add_from_file(const std::string& path, uint32_t image_options) {
    auto i = mapping.find(path);
    if (i != mapping.end()) {
      return i->second;
    }

    auto handle = image_pool.alloc();
    mapping[path] = handle;

    auto& image = image_pool.get(handle);
    image.options = image_options;
    if ((image.options & Image::DelayLoad) == 0) {
      perform_loading(handle, image);
    }

    return handle;
  }

  uint32_t add_from_data(const float4 data[], const uint2& dimensions) {
    uint64_t data_size = sizeof(float4) * dimensions.x * dimensions.y;
    uint32_t hash = fnv1a32(reinterpret_cast<const uint8_t*>(data), data_size);
    char buffer[32] = {};
    snprintf(buffer, sizeof(buffer), "#%u", hash);
    auto i = mapping.find(buffer);
    if (i != mapping.end()) {
      return i->second;
    }

    auto handle = image_pool.alloc();
    mapping[buffer] = handle;
    auto& image = image_pool.get(handle);
    image.format = Image::Format::RGBA32F;
    image.pixels.f32 = make_array_view<float4>(calloc(1llu * dimensions.x * dimensions.y, sizeof(float4)), dimensions.x * dimensions.y);
    memcpy(image.pixels.f32.a, data, data_size);
    image.isize = dimensions;
    image.fsize = {float(dimensions.x), float(dimensions.y)};
    return handle;
  }

  void perform_loading(uint32_t handle, Image& image) {
    for (auto& cache : mapping) {
      if (cache.second != handle)
        continue;

      load_image(image, cache.first.c_str());
      if (image.options & Image::BuildSamplingTable) {
        build_sampling_table(image);
      }
      image.options &= ~Image::DelayLoad;
      break;
    }
  }

  void delay_load() {
    auto objects = image_pool.data();
    uint32_t object_count = image_pool.latest_alive_index() + 1u;
    scheduler.execute(object_count, [this, objects](uint32_t begin, uint32_t end, uint32_t) {
      for (uint32_t i = begin; i < end; ++i) {
        if (image_pool.alive(i) && (objects[i].options & Image::DelayLoad)) {
          perform_loading(i, objects[i]);
        }
      }
    });
  }

  const Image& get(uint32_t handle) const {
    return image_pool.get(handle);
  }

  void remove(uint32_t handle) {
    if (handle == kInvalidIndex) {
      return;
    }

    free_image(image_pool.get(handle));
    image_pool.free(handle);

    for (auto i = mapping.begin(), e = mapping.end(); i != e; ++i) {
      if (i->second == handle) {
        mapping.erase(i);
        break;
      }
    }
  }

  void remove_all() {
    image_pool.free_all(std::bind(&ImagePoolImpl::free_image, this, std::placeholders::_1));
    mapping.clear();
  }

  void load_image(Image& img, const char* file_name) {
    ETX_ASSERT(img.pixels.f32.a == nullptr);
    ETX_ASSERT(img.pixels.f32.count == 0);
    ETX_ASSERT(img.x_distributions.a == nullptr);
    ETX_ASSERT(img.y_distribution.values.count == 0);
    ETX_ASSERT(img.y_distribution.values.a == nullptr);

    std::vector<uint8_t> source_data = {};
    img.format = load_data(file_name, source_data, img.isize);

    if ((img.format == Image::Format::Undefined) || (img.isize.x * img.isize.y == 0)) {
      source_data.resize(sizeof(float4));
      *(float4*)(source_data.data()) = {1.0f, 1.0f, 1.0f, 1.0f};
      img.format = Image::Format::RGBA32F;
      img.options = img.options & (~Image::Linear);
      img.options = img.options & (~Image::RepeatU);
      img.options = img.options & (~Image::Linear);
      img.options = img.options | Image::Linear | Image::RepeatU | Image::RepeatV;
      img.isize.x = 1;
      img.isize.y = 1;
    }

    img.fsize.x = static_cast<float>(img.isize.x);
    img.fsize.y = static_cast<float>(img.isize.y);

    if (img.format == Image::Format::RGBA8) {
      bool srgb = (img.options & Image::Linear) == 0;
      img.pixels.u8.count = 1llu * img.isize.x * img.isize.y;
      img.pixels.u8.a = reinterpret_cast<ubyte4*>(calloc(img.pixels.u8.count, sizeof(ubyte4)));
      auto src_data = reinterpret_cast<const ubyte4*>(source_data.data());
      for (uint32_t y = 0; y < img.isize.y; ++y) {
        for (uint32_t x = 0; x < img.isize.x; ++x) {
          uint32_t i = x + y * img.isize.x;
          uint32_t j = x + (img.isize.y - 1 - y) * img.isize.x;
          float4 f = to_float4(src_data[i]);
          if (srgb) {
            float3 linear = gamma_to_linear({f.x, f.y, f.z});
            f = {linear.x, linear.y, linear.z, f.w};
          }
          auto val = to_ubyte4(f);
          img.pixels.u8[j] = val;
        }
      }
    } else if (img.format == Image::Format::RGBA32F) {
      img.pixels.f32.count = 1llu * img.isize.x * img.isize.y;
      img.pixels.f32.a = reinterpret_cast<float4*>(calloc(img.pixels.f32.count, sizeof(float4)));
      memcpy(img.pixels.f32.a, source_data.data(), source_data.size());
    } else {
      ETX_FAIL_FMT("Unsupported image format %u", img.format);
      return;
    }

    for (uint32_t i = 0, e = img.isize.x * img.isize.y; i < e; ++i) {
      if (img.pixel(i).w < 1.0f) {
        img.options = img.options | Image::HasAlphaChannel;
        break;
      }
    }
  }

  void build_sampling_table(Image& img) {
    bool uniform_sampling = (img.options & Image::UniformSamplingTable) == Image::UniformSamplingTable;
    DistributionBuilder y_dist(img.y_distribution, img.isize.y);
    img.x_distributions.count = img.isize.y;
    img.x_distributions.a = reinterpret_cast<Distribution*>(calloc(img.x_distributions.count, sizeof(Distribution)));

    std::atomic<float> total_weight = {0.0f};
    scheduler.execute_linear(img.isize.y, [&img, uniform_sampling, &total_weight, &y_dist](uint32_t begin, uint32_t end, uint32_t) {
      for (uint32_t y = begin; y < end; ++y) {
        float v = (float(y) + 0.5f) / img.fsize.y;
        float row_value = 0.0f;

        DistributionBuilder d_x(img.x_distributions[y], img.isize.x);
        for (uint32_t x = 0; x < img.isize.x; ++x) {
          float u = (float(x) + 0.5f) / img.fsize.x;
          float4 px = img.read(img.fsize * float2{u, v});
          float lum = luminance(to_float3(px));
          row_value += lum;
          d_x.add(lum);
        }
        d_x.finalize();

        float row_weight = uniform_sampling ? 1.0f : std::sin(v * kPi);
        row_value *= row_weight;

        total_weight = total_weight + row_value;
        y_dist.set(y, row_value);
      }
    });
    y_dist.set_size(img.isize.y);
    y_dist.finalize();

    img.normalization = total_weight / (img.fsize.x * img.fsize.y);
  }

  void free_image(Image& img) {
    free(img.pixels.f32.a);
    for (uint64_t i = 0; (img.x_distributions.a != nullptr) && (i < img.y_distribution.values.count); ++i) {
      free(img.x_distributions[i].values.a);
    }
    free(img.x_distributions.a);
    free(img.y_distribution.values.a);
    img = {};
  }

  Image::Format load_data(const char* source, std::vector<uint8_t>& data, uint2& dimensions) {
    if (source == nullptr)
      return Image::Format::Undefined;

    const char* ext = nullptr;
    if (uint64_t l = strlen(source)) {
      while ((l > 0) && (source[--l] != '.')) {
      }
      ext = source + l;
    } else {
      return Image::Format::Undefined;
    }

    if (strcmp(ext, ".exr") == 0) {
      int w = 0;
      int h = 0;
      const char* error = nullptr;
      float* rgba_data = nullptr;
      if (LoadEXR(&rgba_data, &w, &h, source, &error) != TINYEXR_SUCCESS) {
        printf("Failed to load EXR from file: %s\n", error);
        return Image::Format::Undefined;
      }

      scheduler.execute(4 * w * h, [&rgba_data](uint32_t begin, uint32_t end, uint32_t) {
        for (uint32_t i = begin; i < end; ++i) {
          if (std::isinf(rgba_data[i])) {
            rgba_data[i] = 65504.0f;  // max value in half-float
          }
          if (std::isnan(rgba_data[i]) || (rgba_data[i] < 0.0f)) {
            rgba_data[i] = 0.0f;
          }
        }
      });

      dimensions = {uint32_t(w), uint32_t(h)};
      data.resize(sizeof(float4) * w * h);
      memcpy(data.data(), rgba_data, sizeof(float4) * w * h);
      free(rgba_data);

      return Image::Format::RGBA32F;
    }

    if (strcmp(ext, ".hdr") == 0) {
      int w = 0;
      int h = 0;
      int c = 0;
      stbi_set_flip_vertically_on_load(false);
      auto image = stbi_loadf(source, &w, &h, &c, 0);
      if (image == nullptr) {
        return Image::Format::Undefined;
      }

      dimensions = {uint32_t(w), uint32_t(h)};
      data.resize(sizeof(float4) * w * h);
      auto ptr = reinterpret_cast<float4*>(data.data());
      if (c == 4) {
        memcpy(ptr, image, sizeof(float4) * w * h);
      } else {
        for (int i = 0; i < w * h; ++i) {
          ptr[i] = {image[3 * i + 0], image[3 * i + 1], image[3 * i + 1], 1.0f};
        }
      }
      free(image);
      return Image::Format::RGBA32F;
    }

    if (strcmp(ext, ".pfm") == 0) {
      return load_pfm(source, dimensions, data) ? Image::Format::RGBA32F : Image::Format::Undefined;
    }

    int w = 0;
    int h = 0;
    int c = 0;
    stbi_set_flip_vertically_on_load(false);
    auto image = stbi_load(source, &w, &h, &c, 0);
    if (image == nullptr) {
      return Image::Format::Undefined;
    }

    dimensions = {uint32_t(w), uint32_t(h)};
    data.resize(4llu * w * h);
    uint8_t* ptr = reinterpret_cast<uint8_t*>(data.data());
    switch (c) {
      case 4: {
        memcpy(ptr, image, 4llu * w * h);
        break;
      }

      case 3: {
        for (int i = 0; i < w * h; ++i) {
          ptr[4 * i + 0] = image[3 * i + 0];
          ptr[4 * i + 1] = image[3 * i + 1];
          ptr[4 * i + 2] = image[3 * i + 2];
          ptr[4 * i + 3] = 255;
        }
        break;
      }

      case 1: {
        for (int i = 0; i < w * h; ++i) {
          ptr[4 * i + 0] = image[i];
          ptr[4 * i + 1] = image[i];
          ptr[4 * i + 2] = image[i];
          ptr[4 * i + 3] = 255;
        }
        break;
      }

      default:
        break;
    }

    free(image);
    return Image::Format::RGBA8;
  }

  TaskScheduler& scheduler;
  ObjectIndexPool<Image> image_pool;
  std::unordered_map<std::string, uint32_t> mapping;
  Image empty;
};

ETX_PIMPL_IMPLEMENT(ImagePool, Impl);

ImagePool::ImagePool(TaskScheduler& s) {
  ETX_PIMPL_INIT(ImagePool, s);
}

ImagePool::~ImagePool() {
  ETX_PIMPL_CLEANUP(ImagePool);
}

void ImagePool::init(uint32_t capacity) {
  _private->init(capacity);
}

void ImagePool::cleanup() {
  _private->cleanup();
}

uint32_t ImagePool::add_from_file(const std::string& path, uint32_t image_options) {
  return _private->add_from_file(path, image_options);
}

uint32_t ImagePool::add_from_data(const float4* data, const uint2& dimensions) {
  return _private->add_from_data(data, dimensions);
}

const Image& ImagePool::get(uint32_t handle) {
  return _private->get(handle);
}

void ImagePool::remove(uint32_t handle) {
  _private->remove(handle);
}

void ImagePool::remove_all() {
  _private->remove_all();
}

Image* ImagePool::as_array() {
  return _private->image_pool.alive_objects_count() > 0 ? _private->image_pool.data() : nullptr;
}

uint64_t ImagePool::array_size() {
  return _private->image_pool.alive_objects_count() > 0 ? 1llu + _private->image_pool.latest_alive_index() : 0;
}

bool load_pfm(const char* path, uint2& size, std::vector<uint8_t>& data) {
  FILE* in_file = fopen(path, "rb");
  if (in_file == nullptr) {
    return false;
  }

  char buffer[16] = {};

  auto read_line = [&]() {
    memset(buffer, 0, sizeof(buffer));
    char c = {};
    int p = 0;
    while ((p < 16) && (fread(&c, 1, 1, in_file) == 1)) {
      if (c == '\n') {
        return;
      } else {
        buffer[p++] = c;
      }
    }
  };

  read_line();
  if ((buffer[0] != 'P') && (buffer[1] != 'f') && (buffer[1] != 'F')) {
    fclose(in_file);
    return false;
  }
  char format = buffer[1];

  read_line();
  if (sscanf(buffer, "%d", &size.x) != 1) {
    fclose(in_file);
    return false;
  }

  read_line();
  if (sscanf(buffer, "%d", &size.y) != 1) {
    fclose(in_file);
    return false;
  }

  read_line();
  float scale = 0.0f;
  if (sscanf(buffer, "%f", &scale) != 1) {
    fclose(in_file);
    return false;
  }

  data.resize(sizeof(float4) * size.x * size.y);
  auto data_ptr = reinterpret_cast<float4*>(data.data());

  if (format == 'f') {
    for (uint32_t i = 0; i < size.y; ++i) {
      for (uint32_t j = 0; j < size.x; ++j) {
        float value = 0.0f;
        if (fread(&value, sizeof(float), 1, in_file) != 1) {
          fclose(in_file);
          return false;
        }
        data_ptr[j + i * size.x] = {value, value, value, 1.0f};
      }
    }
  } else if (format == 'F') {
    for (uint32_t i = 0; i < size.y; ++i) {
      for (uint32_t j = 0; j < size.x; ++j) {
        float3 value = {};
        if (fread(&value, 3 * sizeof(float), 1, in_file) != 1) {
          fclose(in_file);
          return false;
        }
        data_ptr[j + i * size.x] = {value.x, value.y, value.z, 1.0f};
      }
    }
  } else {
    fclose(in_file);
    return false;
  }

  fclose(in_file);
  return true;
}

void ImagePool::load_images() {
  _private->delay_load();
}

}  // namespace etx
