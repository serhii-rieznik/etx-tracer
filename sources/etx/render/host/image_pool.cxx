#include <etx/core/core.hxx>

#include <etx/render/host/image_pool.hxx>
#include <etx/render/host/image_loaders.hxx>
#include <etx/render/host/distribution_builder.hxx>

#include <atomic>
#include <vector>
#include <unordered_map>
#include <functional>
#include <filesystem>

namespace etx {

struct ImagePoolImpl {
  ImagePoolImpl(TaskScheduler& s)
    : scheduler(s) {
  }

  void init(uint32_t capacity) {
    images.reserve(capacity);
    paths.reserve(capacity);
  }

  void cleanup() {
    remove_all();
  }

  uint32_t add_copy(const Image& img) {
    ETX_ASSERT((img.options & Image::BuildSamplingTable) == 0);
    ETX_ASSERT(img.y_distribution.values.count == 0);
    ETX_ASSERT(img.x_distributions.count == 0);

    std::string path = "~mem" + std::to_string(1u + counter++);
    uint32_t handle = create_entry(path);

    auto ptr_pixels = reinterpret_cast<ubyte4*>(malloc(img.data_size));
    ETX_CRITICAL(ptr_pixels != nullptr);
    memcpy(ptr_pixels, img.pixels.u8.a, img.data_size);

    auto& image = images[handle];
    image = img;
    image.pixels.u8.a = ptr_pixels;

    return handle;
  }

  uint32_t add_from_file(std::string path, uint32_t image_options, const float2& offset, const float2& scale) {
    auto i = mapping.find(path);
    if (i != mapping.end()) {
      return i->second;
    }

    uint32_t handle = create_entry(path);
    auto& image = images[handle];
    image.offset = offset;
    image.scale = scale;
    image.options = Image::PerformLoading | image_options;
    if ((image.options & Image::Delay) == 0) {
      perform_loading(handle, image);
    }

    return handle;
  }

  uint32_t add_from_data(const float4 data[], const uint2& dimensions, uint32_t image_options, const float2& offset, const float2& scale) {
    std::string path = "~mem" + std::to_string(1u + counter++);
    uint32_t handle = create_entry(path);
    auto& image = images[handle];
    image.offset = offset;
    image.scale = scale;
    image.format = Image::Format::RGBA32F;
    image.data_size = static_cast<uint32_t>(dimensions.x * dimensions.y * sizeof(float4));
    image.pixels.f32 = make_array_view<float4>(calloc(image.data_size, 1u), dimensions.x * dimensions.y);
    image.options = image_options;
    image.isize = dimensions;
    image.fsize = {float(dimensions.x), float(dimensions.y)};

    if (data != nullptr) {
      memcpy(image.pixels.f32.a, data, image.data_size);
    }

    if ((image.options & Image::BuildSamplingTable) && ((image.options & Image::Delay) == 0)) {
      build_image_sampling_table(image, scheduler);
    }

    return handle;
  }

  void perform_loading(uint32_t handle, Image& image) {
    if (handle >= paths.size())
      return;

    const std::string& path = paths[handle];
    if ((image.options & Image::PerformLoading) && (path.empty() == false)) {
      load_image(image, path.c_str());
    }

    if (image.options & Image::BuildSamplingTable) {
      build_image_sampling_table(image, scheduler);
    }

    image.options &= ~Image::Delay;
  }

  void delay_load() {
    if (images.empty())
      return;

    scheduler.execute(static_cast<uint32_t>(images.size()), [this](uint32_t begin, uint32_t end, uint32_t) {
      for (uint32_t i = begin; i < end; ++i) {
        if (i >= images.size())
          break;

        Image& image = images[i];
        if (image.options & Image::Delay) {
          perform_loading(i, image);
        }
      }
    });
  }

  const Image& get(uint32_t handle) const {
    ETX_CRITICAL(handle < images.size());
    return images[handle];
  }

  std::string path(uint32_t handle) const {
    return (handle < paths.size()) ? paths[handle] : std::string{};
  }

  void remove(uint32_t handle) {
    if (handle == kInvalidIndex) {
      return;
    }

    if (handle >= images.size())
      return;

    free_image(images[handle]);

    const std::string& path = paths[handle];
    auto it = mapping.find(path);
    if ((it != mapping.end()) && (it->second == handle)) {
      mapping.erase(it);
    }

    paths[handle].clear();
  }

  void remove_all() {
    for (auto& image : images) {
      free_image(image);
    }
    images.clear();
    paths.clear();
    mapping.clear();
    counter = 0;
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
      log::error("Failed to load image from file: %s", file_name);
      source_data.resize(sizeof(float4));
      *(float4*)(source_data.data()) = {1.0f, 1.0f, 1.0f, 1.0f};
      img.format = Image::Format::RGBA32F;
      img.options = img.options & (~Image::SkipSRGBConversion);
      img.options = img.options & (~Image::RepeatU);
      img.options = img.options & (~Image::SkipSRGBConversion);
      img.options = img.options | Image::SkipSRGBConversion | Image::RepeatU | Image::RepeatV;
      img.isize.x = 1;
      img.isize.y = 1;
    }

    img.fsize.x = static_cast<float>(img.isize.x);
    img.fsize.y = static_cast<float>(img.isize.y);

    if (img.format == Image::Format::RGBA8) {
      bool convert_from_srgb = (img.options & Image::SkipSRGBConversion) == 0;
      img.pixels.u8.count = 1llu * img.isize.x * img.isize.y;
      img.pixels.u8.a = reinterpret_cast<ubyte4*>(calloc(img.pixels.u8.count, sizeof(ubyte4)));
      img.data_size = static_cast<uint32_t>(img.pixels.u8.count * sizeof(ubyte4));
      auto src_data = reinterpret_cast<const ubyte4*>(source_data.data());
      for (uint32_t y = 0; y < img.isize.y; ++y) {
        for (uint32_t x = 0; x < img.isize.x; ++x) {
          uint32_t i = x + y * img.isize.x;
          uint32_t j = x + (img.isize.y - 1u - y) * img.isize.x;
          float4 f = to_float4(src_data[i]);
          if (convert_from_srgb) {
            float3 linear = gamma_to_linear({f.x, f.y, f.z});
            f = {linear.x, linear.y, linear.z, f.w};
          }
          img.pixels.u8[j] = to_ubyte4(f);
        }
      }
    } else if (img.format == Image::Format::RGBA32F) {
      img.pixels.f32.count = 1llu * img.isize.x * img.isize.y;
      img.pixels.f32.a = reinterpret_cast<float4*>(calloc(img.pixels.f32.count, sizeof(float4)));
      img.data_size = static_cast<uint32_t>(img.pixels.f32.count * sizeof(float4));
      ETX_CRITICAL(img.pixels.f32.a);
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

  void build_image_sampling_table(Image& img, TaskScheduler& scheduler) {
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

  TaskScheduler& scheduler;
  uint64_t counter = 0;
  std::vector<Image> images;
  std::vector<std::string> paths;
  std::unordered_map<std::string, uint32_t> mapping;

  uint32_t create_entry(const std::string& path) {
    uint32_t index = static_cast<uint32_t>(images.size());
    images.emplace_back();
    paths.emplace_back(path);
    mapping[path] = index;
    return index;
  }
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

uint32_t ImagePool::add_copy(const Image& img) {
  return _private->add_copy(img);
}

uint32_t ImagePool::add_from_file(const std::string& path, uint32_t image_options, const float2& offset, const float2& scale) {
  return _private->add_from_file(path, image_options, offset, scale);
}

uint32_t ImagePool::add_from_data(const float4* data, const uint2& dimensions, uint32_t image_options, const float2& offset, const float2& scale) {
  return _private->add_from_data(data, dimensions, image_options, offset, scale);
}

const Image& ImagePool::get(uint32_t handle) {
  return _private->get(handle);
}

std::string ImagePool::path(uint32_t handle) const {
  if (handle == kInvalidIndex) {
    return {};
  }
  return _private->path(handle);
}

void ImagePool::free_image(Image& img) {
  _private->free_image(img);
}

void ImagePool::remove(uint32_t handle) {
  _private->remove(handle);
}

void ImagePool::remove_all() {
  _private->remove_all();
}

Image* ImagePool::as_array() {
  return _private->images.empty() ? nullptr : _private->images.data();
}

uint64_t ImagePool::array_size() {
  return _private->images.size();
}

void ImagePool::add_options(uint32_t index, uint32_t options) {
  ETX_CRITICAL(index < _private->images.size());
  _private->images[index].options |= options;
}

void ImagePool::load_images() {
  _private->delay_load();
}

}  // namespace etx
