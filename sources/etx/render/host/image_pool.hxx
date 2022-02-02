#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/shared/image.hxx>

namespace etx {

struct ImagePool {
  ImagePool();
  ~ImagePool();

  void init(uint32_t capacity);
  void cleanup();

  uint32_t add_from_file(const char* path, uint32_t image_options);
  void remove(uint32_t handle);
  void remove_all();

  const Image& get(uint32_t);

  Image* as_array();
  uint64_t array_size();

  ETX_DECLARE_PIMPL(ImagePool, 256);
};

}  // namespace etx
