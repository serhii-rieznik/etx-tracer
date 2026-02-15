#include <etx/core/core.hxx>

#include <etx/render/host/image_pool.hxx>
#include <etx/render/host/buffer_pool.hxx>
#include <etx/render/host/image_loaders.hxx>
#include <etx/render/shared/math.hxx>
#include <etx/render/shared/distribution.hxx>
namespace etx {

struct ImagePoolImpl {
  ImagePoolImpl(std::vector<Image>& external_images, BufferPool& external_buffer_pool)
    : images(external_images)
    , buffer_pool(external_buffer_pool) {
  }

  void init(uint32_t capacity) {
    images.reserve(capacity);
    paths.reserve(capacity);
  }

  void cleanup() {
    remove_all();
  }

  uint32_t add_copy(const Image& img) {
    std::string path = "##mem" + std::to_string(1u + counter++);
    uint32_t handle = create_entry(path);

    auto& image = images[handle];
    const BufferHandle dst_pixel_buffer = image.pixel_buffer;
    const BufferHandle dst_distribution_buffer = image.distribution_buffer;

    image = img;
    image.pixel_buffer = dst_pixel_buffer;
    image.distribution_buffer = dst_distribution_buffer;
    image.data = {};
    image.x_distributions_storage = {};
    image.y_distribution_storage = {};
    image.x_distributions_buffer = {};

    if (img.data_size > 0 && img.format != Image::Format::Undefined) {
      const uint8_t* src_data = nullptr;
      if (img.format == Image::Format::RGBA32F) {
        src_data = reinterpret_cast<const uint8_t*>(img.pixels.f32.a);
      } else if (img.format == Image::Format::RGBA8) {
        src_data = reinterpret_cast<const uint8_t*>(img.pixels.u8.a);
      } else if (Image::is_compressed_bc_format(img.format)) {
        src_data = reinterpret_cast<const uint8_t*>(img.pixels.compressed.a);
      }

      if (src_data != nullptr) {
        image.data = buffer_pool.allocate(image.pixel_buffer, img.data_size, 16u);
        buffer_pool.write(image.data, src_data, img.data_size);

        if (img.format == Image::Format::RGBA32F) {
          image.pixels.f32 = {buffer_pool.map<float4>(image.data), img.pixels.f32.count};
        } else if (img.format == Image::Format::RGBA8) {
          image.pixels.u8 = {buffer_pool.map<ubyte4>(image.data), img.pixels.u8.count};
        } else if (Image::is_compressed_bc_format(img.format)) {
          image.pixels.compressed = {buffer_pool.map<uint8_t>(image.data), static_cast<uint32_t>(image.data.byte_size)};
        }
      }
    }

    if (img.x_distributions.a != nullptr && img.x_distributions.count > 0) {
      uint64_t total_x_entries = 0u;
      for (uint32_t i = 0; i < img.x_distributions.count; ++i) {
        total_x_entries += img.x_distributions.a[i].values.count + 1;  // +1 for sentinel
      }

      image.x_distributions_storage = buffer_pool.allocate_elements<Distribution::Entry>(image.distribution_buffer, total_x_entries);
      image.x_distributions_buffer = buffer_pool.allocate_elements<Distribution>(image.distribution_buffer, img.x_distributions.count, alignof(Distribution));

      auto* x_entries_base = buffer_pool.map<Distribution::Entry>(image.x_distributions_storage);
      auto* x_distributions_base = buffer_pool.map<Distribution>(image.x_distributions_buffer);
      ETX_CRITICAL((x_entries_base != nullptr) && (x_distributions_base != nullptr));

      uint64_t x_entry_offset = 0u;
      for (uint32_t i = 0; i < img.x_distributions.count; ++i) {
        const auto& src_dist = img.x_distributions.a[i];
        auto& dst_dist = x_distributions_base[i];

        std::copy(src_dist.values.a, src_dist.values.a + src_dist.values.count + 1, x_entries_base + x_entry_offset);

        dst_dist.values = {x_entries_base + x_entry_offset, src_dist.values.count};
        dst_dist.total_weight = src_dist.total_weight;
        dst_dist.values_buffer = image.distribution_buffer;
        dst_dist.values_storage = {
          .buffer_index = image.x_distributions_storage.buffer_index,
          .byte_offset = image.x_distributions_storage.byte_offset + x_entry_offset * sizeof(Distribution::Entry),
          .byte_size = (src_dist.values.count + 1u) * sizeof(Distribution::Entry),
        };

        x_entry_offset += src_dist.values.count + 1;
      }

      image.x_distributions = {x_distributions_base, img.x_distributions.count};
    }

    if (img.y_distribution.values.a != nullptr && img.y_distribution.values.count > 0) {
      image.y_distribution_storage = buffer_pool.allocate_elements<Distribution::Entry>(image.distribution_buffer, img.y_distribution.values.count + 1u,
        alignof(Distribution::Entry));
      auto* y_entries = buffer_pool.map<Distribution::Entry>(image.y_distribution_storage);
      ETX_CRITICAL(y_entries != nullptr);
      std::copy(img.y_distribution.values.a, img.y_distribution.values.a + img.y_distribution.values.count + 1u, y_entries);

      image.y_distribution.values = {y_entries, img.y_distribution.values.count};
      image.y_distribution.total_weight = img.y_distribution.total_weight;
      image.y_distribution.values_buffer = image.distribution_buffer;
      image.y_distribution.values_storage = image.y_distribution_storage;
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

    image.offset = offset;
    image.scale = scale;
    image.format = Image::Format::RGBA32F;
    image.options = image_options;
    image.isize = dimensions;
    image.fsize = {float(dimensions.x), float(dimensions.y)};
    const uint64_t pixel_count = 1ull * dimensions.x * dimensions.y;
    image.data = buffer_pool.allocate_elements<float4>(image.pixel_buffer, pixel_count, alignof(float4));
    image.data_size = static_cast<uint32_t>(image.data.byte_size);

    if (data != nullptr) {
      buffer_pool.write(image.data, data, image.data_size);
      image.pixels.f32 = {buffer_pool.map<float4>(image.data), static_cast<uint32_t>(pixel_count)};
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

        float3 dir = uv_to_direction({u, v}, offset, scale.x, static_cast<uint32_t>(ProjectionType::Equirectangular));

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

          float3 dir = uv_to_direction({u, v}, offset, scale.x, static_cast<uint32_t>(ProjectionType::Equirectangular));

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

    buffer_pool.reset(image.distribution_buffer);
    image.x_distributions_storage = {};
    image.y_distribution_storage = {};
    image.x_distributions_buffer = {};

    image.x_distributions = {};
    image.y_distribution.values = {};
    image.y_distribution.total_weight = 0.0f;

    build_image_sampling_table(image, scheduler);
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
          load_image(image, paths[i].c_str());
        }

        if (image.format == Image::Format::RGBA32F) {
          image.pixels.f32 = {buffer_pool.map<float4>(image.data), static_cast<uint32_t>(image.data.byte_size / sizeof(float4))};
        } else if (image.format == Image::Format::RGBA8) {
          image.pixels.u8 = {buffer_pool.map<ubyte4>(image.data), static_cast<uint32_t>(image.data.byte_size / sizeof(ubyte4))};
        } else if (Image::is_compressed_bc_format(image.format)) {
          image.pixels.compressed = {buffer_pool.map<uint8_t>(image.data), static_cast<uint32_t>(image.data.byte_size)};
        }

        for (uint32_t i = 0, e = image.isize.x * image.isize.y; i < e; ++i) {
          if (image.pixel(i).w < 1.0f) {
            image.options = image.options | Image::HasAlphaChannel;
            break;
          }
        }

        if (image.options & Image::BuildSamplingTable) {
          build_image_sampling_table(image, scheduler);
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

    Image& image = images[handle];
    buffer_pool.destroy(image.pixel_buffer);
    buffer_pool.destroy(image.distribution_buffer);
    free_image(image);

    const std::string& path = paths[handle];
    auto it = mapping.find(path);
    if ((it != mapping.end()) && (it->second == handle)) {
      mapping.erase(it);
    }

    paths[handle].clear();
  }

  void remove_all() {
    for (auto& image : images) {
      buffer_pool.destroy(image.pixel_buffer);
      buffer_pool.destroy(image.distribution_buffer);
      free_image(image);
    }
    images.clear();
    paths.clear();
    mapping.clear();
    counter = 0;
  }

  void load_image(Image& img, const char* file_name) {
    const bool skip_loading = (file_name == nullptr) || (file_name[0] == '\0') || ((file_name[0] == '#') && (file_name[1] == '#'));

    // In-memory image entries are already filled by add_from_data/add_copy.
    if (skip_loading && img.data.valid() && (img.data_size > 0u) && (img.format != Image::Format::Undefined) && (img.isize.x > 0u) && (img.isize.y > 0u)) {
      img.fsize.x = static_cast<float>(img.isize.x);
      img.fsize.y = static_cast<float>(img.isize.y);
      return;
    }

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
      const uint64_t pixel_count = 1ull * img.isize.x * img.isize.y;
      img.data = buffer_pool.allocate_elements<ubyte4>(img.pixel_buffer, pixel_count, alignof(ubyte4));
      img.data_size = static_cast<uint32_t>(img.data.byte_size);
      auto* pixels_u8 = buffer_pool.map<ubyte4>(img.data);
      ETX_CRITICAL(pixels_u8 != nullptr);
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
      const uint64_t pixel_count = 1ull * img.isize.x * img.isize.y;
      img.data = buffer_pool.allocate_elements<float4>(img.pixel_buffer, pixel_count, alignof(float4));
      img.data_size = static_cast<uint32_t>(img.data.byte_size);
      auto* pixels_f32 = buffer_pool.map<float4>(img.data);
      ETX_CRITICAL(pixels_f32);
      memcpy(pixels_f32, source_data.data(), source_data.size());
#if ETX_STORE_COMPRESSED_BC
    } else if (Image::is_compressed_bc_format(img.format)) {
      // Store compressed BC data directly - no decompression at load time
      img.data = buffer_pool.allocate_elements<uint8_t>(img.pixel_buffer, source_data.size(), alignof(uint8_t));
      img.data_size = static_cast<uint32_t>(img.data.byte_size);
      buffer_pool.write(img.data, source_data.data(), source_data.size());
      // TODO: Setup compressed data view for runtime decompression during sampling
#endif
    } else {
      ETX_FAIL_FMT("Unsupported image format %u", img.format);
      return;
    }
  }

  void build_image_sampling_table(Image& img, TaskScheduler& scheduler) {
    ETX_ASSERT(img.x_distributions.count == 0);
    ETX_ASSERT(img.y_distribution.values.count == 0);
    ETX_ASSERT(img.y_distribution.values.a == nullptr);
    bool uniform_sampling = (img.options & Image::UniformSamplingTable) == Image::UniformSamplingTable;

    uint32_t x_entries_per_row = img.isize.x + 1;  // +1 for sentinel
    uint32_t y_entries_count = img.isize.y + 1;    // +1 for sentinel
    uint32_t total_x_entries = img.isize.y * x_entries_per_row;

    img.x_distributions_storage = buffer_pool.allocate_elements<Distribution::Entry>(img.distribution_buffer, total_x_entries, alignof(Distribution::Entry));
    img.y_distribution_storage = buffer_pool.allocate_elements<Distribution::Entry>(img.distribution_buffer, y_entries_count, alignof(Distribution::Entry));
    img.x_distributions_buffer = buffer_pool.allocate_elements<Distribution>(img.distribution_buffer, img.isize.y, alignof(Distribution));

    auto* x_entries_base = buffer_pool.map<Distribution::Entry>(img.x_distributions_storage);
    auto* y_entries_base = buffer_pool.map<Distribution::Entry>(img.y_distribution_storage);
    auto* x_distributions_base = buffer_pool.map<Distribution>(img.x_distributions_buffer);
    ETX_CRITICAL((x_entries_base != nullptr) && (y_entries_base != nullptr) && (x_distributions_base != nullptr));

    for (uint32_t y = 0; y < img.isize.y; ++y) {
      auto& dist = x_distributions_base[y];
      dist.values = {x_entries_base + y * x_entries_per_row, img.isize.x};
      dist.total_weight = 0.0f;  // Will be set by finalize
      dist.values_buffer = img.distribution_buffer;
      dist.values_storage = {
        .buffer_index = img.x_distributions_storage.buffer_index,
        .byte_offset = img.x_distributions_storage.byte_offset + static_cast<uint64_t>(y) * x_entries_per_row * sizeof(Distribution::Entry),
        .byte_size = static_cast<uint64_t>(x_entries_per_row) * sizeof(Distribution::Entry),
      };
    }

    img.x_distributions = {x_distributions_base, img.isize.y};
    img.y_distribution.values = {y_entries_base, img.isize.y};
    img.y_distribution.total_weight = 0.0f;  // Will be set by finalize
    img.y_distribution.values_buffer = img.distribution_buffer;
    img.y_distribution.values_storage = img.y_distribution_storage;

    std::atomic<float> total_weight = {0.0f};
    scheduler.execute(img.isize.y, [&img, x_entries_base, y_entries_base, x_distributions_base, uniform_sampling, &total_weight, x_entries_per_row](uint32_t begin, uint32_t end,
                              uint32_t) {
      for (uint32_t y = begin; y < end; ++y) {
        float v = (float(y) + 0.5f) / img.fsize.y;
        float row_value = 0.0f;

        auto* x_entries = x_entries_base + y * x_entries_per_row;
        for (uint32_t x = 0; x < img.isize.x; ++x) {
          float u = (float(x) + 0.5f) / img.fsize.x;
          float4 px = img.read(img.fsize * float2{u, v});
          float lum = luminance(to_float3(px));
          row_value += lum;
          x_entries[x] = {lum, 0.0f, 0.0f};
        }

        auto& dist = x_distributions_base[y];
        dist = Distribution::build(x_entries, img.isize.x, dist.values_buffer, dist.values_storage);

        float row_weight = uniform_sampling ? 1.0f : std::sin(v * kPi);
        row_value *= row_weight;
        total_weight = total_weight + row_value;

        y_entries_base[y] = {row_value, 0.0f, 0.0f};
      }
    });

    img.y_distribution = Distribution::build(y_entries_base, img.isize.y, img.y_distribution.values_buffer, img.y_distribution.values_storage);
    img.normalization = total_weight / (img.fsize.x * img.fsize.y);
  }

  void free_image(Image& img) {
    img.pixels.f32 = {};
    img.pixels.u8 = {};
    img.pixels.compressed = {};
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

    img.data = {};
    img.x_distributions_storage = {};
    img.y_distribution_storage = {};
    img.x_distributions_buffer = {};
    img.pixel_buffer = {};
    img.distribution_buffer = {};
  }

  std::vector<Image>& images;
  BufferPool& buffer_pool;
  std::vector<std::string> paths;
  std::unordered_map<std::string, uint32_t> mapping;
  uint64_t counter = 0;

  uint32_t create_entry(const std::string& path) {
    uint32_t index = static_cast<uint32_t>(images.size());
    images.emplace_back();
    images[index].pixel_buffer = buffer_pool.create(0u, "image_pixels");
    images[index].distribution_buffer = buffer_pool.create(0u, "image_distributions");
    paths.emplace_back(path);
    mapping[path] = index;
    return index;
  }
};

ImagePool::ImagePool(std::vector<Image>& external_images, BufferPool& buffer_pool) {
  ETX_PIMPL_CREATE(ImagePool, Impl, external_images, buffer_pool);
}

ImagePool::~ImagePool() {
  ETX_PIMPL_DESTROY(ImagePool, Impl);
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

const Image& ImagePool::get(uint32_t handle) const {
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

const Image* ImagePool::as_array() const {
  return _private->images.empty() ? nullptr : _private->images.data();
}

const uint64_t ImagePool::array_size() const {
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
