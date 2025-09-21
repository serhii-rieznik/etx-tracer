#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/host/tasks.hxx>

#include <etx/render/shared/image.hxx>

#include <string>

namespace etx {

struct ImagePool {
  ImagePool(TaskScheduler&);
  ~ImagePool();

  void init(uint32_t capacity);
  void cleanup();

  uint32_t add_copy(const Image& img);
  uint32_t add_from_file(const std::string& path, uint32_t image_options, const float2& offset, const float2& scale);
  uint32_t add_from_data(const float4* data, const uint2& dimensions, uint32_t image_options, const float2& offset, const float2& scale);

  void remove(uint32_t handle);
  void remove_all();

  void add_options(uint32_t, uint32_t);
  void load_images();

  const Image& get(uint32_t);

  void free_image(Image&);

  Image* as_array();
  uint64_t array_size();

  ETX_DECLARE_PIMPL(ImagePool, 256);
};

}  // namespace etx
