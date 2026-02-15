#pragma once

#include <etx/render/interop/interop.hxx>

namespace etx {

struct BufferHandle {
  uint32_t index = kInvalidIndex;
  uint32_t generation = 0u;

  ETX_SHARED_INLINE bool valid() const {
    return index != kInvalidIndex;
  }
};

struct BufferView {
  uint32_t buffer_index = kInvalidIndex;
  uint64_t byte_offset = 0u;
  uint64_t byte_size = 0u;

  ETX_SHARED_INLINE bool valid() const {
    return (buffer_index != kInvalidIndex) && (byte_size > 0u);
  }
};

}  // namespace etx
