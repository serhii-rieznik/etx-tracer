#pragma once

#include <etx/render/shared/image.hxx>

namespace etx {

struct ImagePool {
  static void init(uint32_t capacity);
  static void cleanup();

  static Handle add_from_file(const char* path, uint32_t image_options);
  static void remove(Handle handle);

  static const Image& get(Handle);
};

}  // namespace etx
