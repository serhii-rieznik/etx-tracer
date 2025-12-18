#include <etx/core/core.hxx>

#include <etx/render/host/image_pool.hxx>
#include <etx/render/host/image_loaders.hxx>
#include <etx/render/host/distribution_builder.hxx>
#include <etx/render/shared/math.hxx>

#include <atomic>
#include <vector>
#include <unordered_map>
#include <functional>
#include <filesystem>
#include <algorithm>

namespace etx {

struct ImagePoolImpl {
  ImagePoolImpl(std::vector<Image>& external_images, std::vector<ImageStorage>& external_storage)
    : images(external_images)
    , storage(external_storage) {
  }

  void init(uint32_t capacity) {
    images.reserve(capacity);
    storage.reserve(capacity);
    paths.reserve(capacity);
  }

  void cleanup() {
    remove_all();
  }

  uint32_t add_copy(const Image& img) {
    std::string path = "##mem" + std::to_string(1u + counter++);
    uint32_t handle = create_entry(path);

    auto& image = images[handle];
    auto& img_storage = storage[handle];

    image = img;

    if (img.data_size > 0 && img.format != Image::Format::Undefined) {
      const uint8_t* src_data = (img.format == Image::Format::RGBA32F) ? reinterpret_cast<const uint8_t*>(img.pixels.f32.a) : reinterpret_cast<const uint8_t*>(img.pixels.u8.a);

      if (src_data != nullptr) {
        img_storage.data.assign(src_data, src_data + img.data_size);

        if (img.format == Image::Format::RGBA32F) {
          image.pixels.f32 = {reinterpret_cast<float4*>(img_storage.data.data()), img.pixels.f32.count};
        } else if (img.format == Image::Format::RGBA8) {
          image.pixels.u8 = {reinterpret_cast<ubyte4*>(img_storage.data.data()), img.pixels.u8.count};
        }
      }
    }

    if (img.x_distributions.a != nullptr && img.x_distributions.count > 0) {
      size_t total_x_entries = 0;
      for (uint32_t i = 0; i < img.x_distributions.count; ++i) {
        total_x_entries += img.x_distributions.a[i].values.count + 1;  // +1 for sentinel
      }

      img_storage.x_distributions_storage.resize(total_x_entries);
      img_storage.x_distributions.resize(img.x_distributions.count);

      size_t x_entry_offset = 0;
      for (uint32_t i = 0; i < img.x_distributions.count; ++i) {
        const auto& src_dist = img.x_distributions.a[i];
        auto& dst_dist = img_storage.x_distributions[i];

        std::copy(src_dist.values.a, src_dist.values.a + src_dist.values.count + 1, img_storage.x_distributions_storage.data() + x_entry_offset);

        dst_dist.values = {img_storage.x_distributions_storage.data() + x_entry_offset, src_dist.values.count};
        dst_dist.total_weight = src_dist.total_weight;

        x_entry_offset += src_dist.values.count + 1;
      }

      image.x_distributions = {img_storage.x_distributions.data(), img.x_distributions.count};
    }

    if (img.y_distribution.values.a != nullptr && img.y_distribution.values.count > 0) {
      img_storage.y_distribution_storage.assign(img.y_distribution.values.a, img.y_distribution.values.a + img.y_distribution.values.count + 1);

      image.y_distribution.values = {img_storage.y_distribution_storage.data(), img.y_distribution.values.count};
      image.y_distribution.total_weight = img.y_distribution.total_weight;
    }

    return handle;
  }

  uint32_t add_from_file(const std::string& path, uint32_t image_options, const float2& offset, const float2& scale) {
    auto i = mapping.find(path);
    if (i != mapping.end()) {
      return i->second;
    }

    uint32_t handle = create_entry(path);
    auto& image = images[handle];
    image.offset = offset;
    image.scale = scale;
    image.options = image_options;

    return handle;
  }

  uint32_t add_from_data(const float4 data[], const uint2& dimensions, uint32_t image_options, const float2& offset, const float2& scale) {
    std::string path = "##mem" + std::to_string(1u + counter++);
    uint32_t handle = create_entry(path);
    auto& image = images[handle];
    auto& img_storage = storage[handle];

    image.offset = offset;
    image.scale = scale;
    image.format = Image::Format::RGBA32F;
    image.options = image_options;
    image.isize = dimensions;
    image.fsize = {float(dimensions.x), float(dimensions.y)};
    size_t pixel_count = 1llu * dimensions.x * dimensions.y;
    img_storage.data.resize(pixel_count * sizeof(float4));
    image.data_size = static_cast<uint32_t>(img_storage.data.size());

    if (data != nullptr) {
      auto* pixels_f32 = reinterpret_cast<float4*>(img_storage.data.data());
      memcpy(pixels_f32, data, image.data_size);
    }

    return handle;
  }

  static float evaluate_sh_basis(uint32_t index, const float3& dir) {
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;

    switch (index) {
      case 0:
        return 0.282095f;
      case 1:
        return 0.488603f * y;
      case 2:
        return 0.488603f * z;
      case 3:
        return 0.488603f * x;
      case 4:
        return 1.092548f * x * y;
      case 5:
        return 1.092548f * y * z;
      case 6:
        return 0.315392f * (3.0f * z * z - 1.0f);
      case 7:
        return 1.092548f * x * z;
      case 8:
        return 0.546274f * (x * x - y * y);
      default:
        return 0.0f;
    }
  }

  uint32_t add_from_spherical_harmonics(TaskScheduler& scheduler, const float3 sh_coeffs[9], const uint2& dimensions, uint32_t image_options, const float2& offset,
    const float2& scale) {
    std::vector<float4> image_data(dimensions.x * dimensions.y);

    scheduler.execute(dimensions.x * dimensions.y, [&image_data, &dimensions, &sh_coeffs, &offset, &scale](uint32_t begin, uint32_t end, uint32_t) {
      for (uint32_t i = begin; i < end; ++i) {
        uint32_t x = i % dimensions.x;
        uint32_t y = i / dimensions.x;

        float u = (float(x) + 0.5f) / float(dimensions.x);
        float v = (float(y) + 0.5f) / float(dimensions.y);

        float3 dir = uv_to_direction({u, v}, offset, scale.x, ProjectionType::Equirectangular);

        float3 color = {};
        for (uint32_t j = 0; j < 9; ++j) {
          float basis = evaluate_sh_basis(j, dir);
          color = color + sh_coeffs[j] * basis;
        }

        float3 linear = gamma_to_linear({max(0.0f, color.x), max(0.0f, color.y), max(0.0f, color.z)});
        image_data[i] = {linear.x, linear.y, linear.z, 1.0f};
      }
    });

    return add_from_data(image_data.data(), dimensions, image_options, offset, scale);
  }

  uint32_t add_from_cubemap(TaskScheduler& scheduler, const uint32_t cube_face_images[6], const uint2& dimensions, uint32_t image_options, const float2& offset,
    const float2& scale) {
    bool needs_srgb_conversion[6] = {};
    for (size_t i = 0; i < 6; ++i) {
      const auto& face_img = images[cube_face_images[i]];
      needs_srgb_conversion[i] = (face_img.format == Image::Format::RGBA8);
    }

    std::vector<float4> equirect_data(dimensions.x * dimensions.y);

    scheduler.execute(dimensions.x * dimensions.y,
      [&equirect_data, &dimensions, &cube_face_images, &needs_srgb_conversion, &offset, &scale, this](uint32_t begin, uint32_t end, uint32_t) {
        for (uint32_t i = begin; i < end; ++i) {
          uint32_t x = i % dimensions.x;
          uint32_t y = i / dimensions.x;

          float u = (float(x) + 0.5f) / float(dimensions.x);
          float v = (float(y) + 0.5f) / float(dimensions.y);

          float3 dir = uv_to_direction({u, v}, offset, scale.x, ProjectionType::Equirectangular);

          float3 abs_dir = {fabsf(dir.x), fabsf(dir.y), fabsf(dir.z)};
          int face_idx = 0;
          float2 cube_uv = {};

          if (abs_dir.x >= abs_dir.y && abs_dir.x >= abs_dir.z) {
            if (dir.x >= 0.0f) {
              face_idx = 0;
              cube_uv = {-dir.z / dir.x, -dir.y / dir.x};
            } else {
              face_idx = 1;
              cube_uv = {dir.z / dir.x, -dir.y / dir.x};
            }
          } else if (abs_dir.y >= abs_dir.z) {
            if (dir.y >= 0.0f) {
              face_idx = 2;
              cube_uv = {dir.x / dir.y, dir.z / dir.y};
            } else {
              face_idx = 3;
              cube_uv = {dir.x / dir.y, dir.z / dir.y};
            }
          } else {
            if (dir.z >= 0.0f) {
              face_idx = 4;
              cube_uv = {dir.x / dir.z, -dir.y / dir.z};
            } else {
              face_idx = 5;
              cube_uv = {-dir.x / dir.z, -dir.y / dir.z};
            }
          }

          cube_uv = cube_uv * 0.5f + 0.5f;

          bool flip_v = true;
          switch (face_idx) {
            case 1:
              cube_uv.x = 1.0f - cube_uv.x;
              flip_v = false;
              break;
            case 3:
              cube_uv.x = 1.0f - cube_uv.x;
              break;
            case 5:
              cube_uv.x = 1.0f - cube_uv.x;
              flip_v = false;
              break;
          }
          if (flip_v) {
            cube_uv.y = 1.0f - cube_uv.y;
          }

          const auto& face_image = images[cube_face_images[face_idx]];
          float4 color = face_image.evaluate(cube_uv, nullptr);

          if (needs_srgb_conversion[face_idx]) {
            float3 linear = gamma_to_linear({color.x, color.y, color.z});
            color = {linear.x, linear.y, linear.z, color.w};
          }

          equirect_data[i] = color;
        }
      });

    return add_from_data(equirect_data.data(), dimensions, image_options, offset, scale);
  }

  void rebuild_sampling_table(uint32_t index, TaskScheduler& scheduler) {
    ETX_CRITICAL((index < images.size()));
    Image& image = images[index];
    ImageStorage& img_storage = storage[index];

    img_storage.x_distributions_storage.clear();
    img_storage.y_distribution_storage.clear();
    img_storage.x_distributions.clear();

    image.x_distributions = {};
    image.y_distribution.values = {};
    image.y_distribution.total_weight = 0.0f;

    build_image_sampling_table(image, img_storage, scheduler);
  }

  void delay_load(TaskScheduler& scheduler) {
    if (images.empty())
      return;

    scheduler.execute(static_cast<uint32_t>(images.size()), [this, &scheduler](uint32_t begin, uint32_t end, uint32_t) {
      for (uint32_t i = begin; i < end; ++i) {
        if (i >= images.size())
          break;
        ETX_CRITICAL(i < paths.size());

        Image& image = images[i];
        if (image.options & Image::Committed)
          continue;

        if (paths[i].empty() == false) {
          load_image(image, storage[i], paths[i].c_str());
        }

        if (image.format == Image::Format::RGBA32F) {
          image.pixels.f32 = {reinterpret_cast<float4*>(storage[i].data.data()), static_cast<uint32_t>(storage[i].data.size() / sizeof(float4))};
        } else if (image.format == Image::Format::RGBA8) {
          image.pixels.u8 = {reinterpret_cast<ubyte4*>(storage[i].data.data()), static_cast<uint32_t>(storage[i].data.size() / sizeof(ubyte4))};
        }

        // Detect alpha channel
        for (uint32_t i = 0, e = image.isize.x * image.isize.y; i < e; ++i) {
          if (image.pixel(i).w < 1.0f) {
            image.options = image.options | Image::HasAlphaChannel;
            break;
          }
        }

        if (image.options & Image::BuildSamplingTable) {
          build_image_sampling_table(image, storage[i], scheduler);
        }

        image.options |= Image::Committed;
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
    for (auto& img_storage : storage) {
      img_storage.clear();
    }
    images.clear();
    storage.clear();
    paths.clear();
    mapping.clear();
    counter = 0;
  }

  void load_image(Image& img, ImageStorage& img_storage, const char* file_name) {
    const bool skip_loading = (file_name == nullptr) || (file_name[0] == '\0') || ((file_name[0] == '#') && (file_name[1] == '#'));

    std::vector<uint8_t> source_data = {};

    if (skip_loading == false) {
      img.format = load_data(file_name, source_data, img.isize);
      if ((img.format == Image::Format::Undefined) || (img.isize.x * img.isize.y == 0)) {
        log::error("Failed to load image from file: %s", file_name);
      }
    }

    if ((img.format == Image::Format::Undefined) || (img.isize.x * img.isize.y == 0)) {
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
      size_t pixel_count = 1llu * img.isize.x * img.isize.y;
      img_storage.data.resize(pixel_count * sizeof(ubyte4));
      img.data_size = static_cast<uint32_t>(img_storage.data.size());
      auto* pixels_u8 = reinterpret_cast<ubyte4*>(img_storage.data.data());
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
          pixels_u8[j] = to_ubyte4(f);
        }
      }
    } else if (img.format == Image::Format::RGBA32F) {
      size_t pixel_count = 1llu * img.isize.x * img.isize.y;
      img_storage.data.resize(pixel_count * sizeof(float4));
      img.data_size = static_cast<uint32_t>(img_storage.data.size());
      auto* pixels_f32 = reinterpret_cast<float4*>(img_storage.data.data());
      ETX_CRITICAL(pixels_f32);
      memcpy(pixels_f32, source_data.data(), source_data.size());
    } else {
      ETX_FAIL_FMT("Unsupported image format %u", img.format);
      return;
    }
  }

  void build_image_sampling_table(Image& img, ImageStorage& img_storage, TaskScheduler& scheduler) {
    ETX_ASSERT(img.x_distributions.count == 0);
    ETX_ASSERT(img.y_distribution.values.count == 0);
    ETX_ASSERT(img.y_distribution.values.a == nullptr);
    bool uniform_sampling = (img.options & Image::UniformSamplingTable) == Image::UniformSamplingTable;

    uint32_t x_entries_per_row = img.isize.x + 1;  // +1 for sentinel
    uint32_t y_entries_count = img.isize.y + 1;    // +1 for sentinel
    uint32_t total_x_entries = img.isize.y * x_entries_per_row;

    img_storage.x_distributions_storage.resize(total_x_entries);
    img_storage.y_distribution_storage.resize(y_entries_count);
    img_storage.x_distributions.resize(img.isize.y);

    for (uint32_t y = 0; y < img.isize.y; ++y) {
      auto& dist = img_storage.x_distributions[y];
      dist.values = {img_storage.x_distributions_storage.data() + y * x_entries_per_row, img.isize.x};
      dist.total_weight = 0.0f;  // Will be set by finalize
    }

    img.x_distributions = {img_storage.x_distributions.data(), static_cast<uint32_t>(img_storage.x_distributions.size())};
    img.y_distribution.values = {img_storage.y_distribution_storage.data(), img.isize.y};
    img.y_distribution.total_weight = 0.0f;  // Will be set by finalize

    std::atomic<float> total_weight = {0.0f};
    scheduler.execute(img.isize.y, [&img, &img_storage, uniform_sampling, &total_weight, x_entries_per_row](uint32_t begin, uint32_t end, uint32_t) {
      for (uint32_t y = begin; y < end; ++y) {
        float v = (float(y) + 0.5f) / img.fsize.y;
        float row_value = 0.0f;

        auto* x_entries = img_storage.x_distributions_storage.data() + y * x_entries_per_row;
        for (uint32_t x = 0; x < img.isize.x; ++x) {
          float u = (float(x) + 0.5f) / img.fsize.x;
          float4 px = img.read(img.fsize * float2{u, v});
          float lum = luminance(to_float3(px));
          row_value += lum;
          x_entries[x] = {lum, 0.0f, 0.0f};
        }

        DistributionBuilder::finalize_entries(x_entries, img.isize.x);
        img_storage.x_distributions[y].total_weight = x_entries[img.isize.x].cdf;  // Last entry has total weight

        float row_weight = uniform_sampling ? 1.0f : std::sin(v * kPi);
        row_value *= row_weight;
        total_weight = total_weight + row_value;

        img_storage.y_distribution_storage[y] = {row_value, 0.0f, 0.0f};
      }
    });

    float y_total_weight = DistributionBuilder::finalize_entries(img_storage.y_distribution_storage.data(), img.isize.y);
    img.y_distribution.total_weight = y_total_weight;
    img.normalization = total_weight / (img.fsize.x * img.fsize.y);
  }

  void free_image(Image& img) {
    img.pixels.f32 = {};
    img.pixels.u8 = {};
    img.x_distributions = {};
    img.y_distribution = {};

    img.fsize = {};
    img.offset = {};
    img.scale = {1.0f, 1.0f};
    img.isize = {};
    img.normalization = 0.0f;
    img.options = 0;
    img.format = Image::Format::Undefined;
    img.data_size = 0u;
  }

  std::vector<Image>& images;
  std::vector<ImageStorage>& storage;
  std::vector<std::string> paths;
  std::unordered_map<std::string, uint32_t> mapping;
  uint64_t counter = 0;

  uint32_t create_entry(const std::string& path) {
    uint32_t index = static_cast<uint32_t>(images.size());
    images.emplace_back();
    storage.emplace_back();
    paths.emplace_back(path);
    mapping[path] = index;
    return index;
  }
};

ImagePool::ImagePool(std::vector<Image>& external_images, std::vector<ImageStorage>& external_storage) {
  ETX_PIMPL_CREATE(ImagePool, Impl, external_images, external_storage);
}

ImagePool::~ImagePool() {
  ETX_PIMPL_DESTROY(ImagePool, Impl);
}

std::vector<ImageStorage>& ImagePool::storage() {
  return _private->storage;
}

ETX_PIMPL_IMPLEMENT(ImagePool, Impl);

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

uint32_t ImagePool::add_from_spherical_harmonics(TaskScheduler& scheduler, const float3 sh_coeffs[9], const uint2& dimensions, uint32_t image_options, const float2& offset,
  const float2& scale) {
  return _private->add_from_spherical_harmonics(scheduler, sh_coeffs, dimensions, image_options, offset, scale);
}

uint32_t ImagePool::add_from_cubemap(TaskScheduler& scheduler, const uint32_t cube_face_images[6], const uint2& dimensions, uint32_t image_options, const float2& offset,
  const float2& scale) {
  return _private->add_from_cubemap(scheduler, cube_face_images, dimensions, image_options, offset, scale);
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

void ImagePool::load_images(TaskScheduler& scheduler) {
  _private->delay_load(scheduler);
}

void ImagePool::rebuild_sampling_table(uint32_t index, TaskScheduler& scheduler) {
  _private->rebuild_sampling_table(index, scheduler);
}

}  // namespace etx
