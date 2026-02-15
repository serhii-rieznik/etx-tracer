#pragma once

#include <etx/render/shared/buffer_view.hxx>

#include <cstdint>
#include <string>
#include <vector>

namespace etx {

struct BufferPool {
  struct Stats {
    uint64_t buffer_count = 0u;
    uint64_t used_bytes = 0u;
    uint64_t capacity_bytes = 0u;
  };

  BufferHandle create(uint64_t initial_capacity = 0u, const char* debug_name = nullptr);
  void destroy(BufferHandle handle);
  bool valid(BufferHandle handle) const;

  BufferView allocate(BufferHandle handle, uint64_t byte_size, uint64_t alignment = 16u);

  template <typename T>
  BufferView allocate_elements(BufferHandle handle, uint64_t count, uint64_t alignment = alignof(T)) {
    return allocate(handle, count * sizeof(T), alignment);
  }

  bool write(BufferView view, const void* src, uint64_t byte_size, uint64_t write_offset = 0u);

  void* map(BufferView view);
  const void* map(BufferView view) const;

  template <typename T>
  T* map(BufferView view) {
    return reinterpret_cast<T*>(map(view));
  }

  template <typename T>
  const T* map(BufferView view) const {
    return reinterpret_cast<const T*>(map(view));
  }

  template <typename T>
  ArrayView<T> view_as_array(BufferView view) const {
    ETX_ASSERT((view.byte_size % sizeof(T)) == 0u);
    const T* ptr = map<T>(view);
    return {ptr, view.byte_size / sizeof(T)};
  }

  void reset(BufferHandle handle);
  void clear();

  uint64_t used_bytes(BufferHandle handle) const;
  uint64_t capacity_bytes(BufferHandle handle) const;

  Stats stats() const;

 private:
  struct Slot {
    std::string name;
    std::vector<uint8_t> bytes;
    uint64_t used = 0u;
    uint32_t generation = 1u;
    bool alive = false;
  };

  std::vector<Slot> _slots;
  std::vector<uint32_t> _free_indices;

  Slot* resolve(BufferHandle handle);
  const Slot* resolve(BufferHandle handle) const;
  Slot* resolve(uint32_t index);
  const Slot* resolve(uint32_t index) const;

  static uint64_t align_up(uint64_t value, uint64_t alignment);
  static uint64_t next_capacity(uint64_t required, uint64_t current);
};

}  // namespace etx
