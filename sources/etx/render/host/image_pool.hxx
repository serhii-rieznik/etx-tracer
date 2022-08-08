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

  uint32_t add_from_file(const std::string& path, uint32_t image_options);
  uint32_t add_from_data(const float4* data, const uint2& dimensions);
  void remove(uint32_t handle);
  void remove_all();

  void load_images();

  const Image& get(uint32_t);

  Image* as_array();
  uint64_t array_size();

  ETX_DECLARE_PIMPL(ImagePool, 256);
};

}  // namespace etx
