#pragma once

#include <etx/core/debug.hxx>
#include <stdint.h>

namespace etx {

union Handle {
  enum : uint64_t {
    ClsBits = 8,
    ClsMax = (1 << ClsBits) - 1,
    IndexBits = 28,
    IndexMax = (1 << IndexBits) - 1,
    GenBits = 28,
    GenMax = (1 << GenBits) - 1,
    BitCount = ClsBits + IndexBits + GenBits,
  };
  static_assert(BitCount == 64);

  uint64_t value = 0;
  struct {
    uint64_t cls : ClsBits;
    uint64_t index : IndexBits;
    uint64_t generation : GenBits;
  };

  bool operator==(const Handle& h) const {
    return value == h.value;
  }

  static inline Handle construct(uint64_t a_cls, uint64_t a_index, uint64_t a_generation) {
    ETX_ASSERT(a_cls < ClsMax);
    ETX_ASSERT(a_index < IndexMax);
    ETX_ASSERT(a_generation < GenMax);
    Handle result = {};
    result.cls = a_cls;
    result.index = a_index;
    result.generation = a_generation;
    return result;
  }
};

static_assert(sizeof(Handle) == sizeof(uint64_t));

}  // namespace etx

#include <type_traits>

namespace std {

template <>
struct hash<etx::Handle> {
  size_t operator()(const etx::Handle handle) const {
    return handle.value;
  }
};

}  // namespace std
