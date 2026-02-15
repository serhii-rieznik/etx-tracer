#include <etx/render/host/buffer_pool.hxx>

#include <cstring>

namespace etx {
namespace {

constexpr uint64_t kMinBufferCapacity = 256u;

}

BufferHandle BufferPool::create(uint64_t initial_capacity, const char* debug_name) {
  const uint32_t index = _free_indices.empty() ? static_cast<uint32_t>(_slots.size()) : _free_indices.back();

  if (_free_indices.empty() == false) {
    _free_indices.pop_back();
  } else {
    _slots.emplace_back();
  }

  Slot& slot = _slots[index];
  slot.alive = true;
  slot.used = 0u;
  slot.name = (debug_name != nullptr) ? debug_name : "";
  slot.bytes.clear();
  slot.bytes.resize(static_cast<size_t>(initial_capacity));

  return BufferHandle{index, slot.generation};
}

void BufferPool::destroy(BufferHandle handle) {
  Slot* slot = resolve(handle);
  if (slot == nullptr) {
    return;
  }

  slot->alive = false;
  slot->used = 0u;
  slot->bytes.clear();
  slot->name.clear();
  slot->generation += 1u;

  _free_indices.push_back(handle.index);
}

bool BufferPool::valid(BufferHandle handle) const {
  return resolve(handle) != nullptr;
}

BufferView BufferPool::allocate(BufferHandle handle, uint64_t byte_size, uint64_t alignment) {
  Slot* slot = resolve(handle);
  if ((slot == nullptr) || (byte_size == 0u)) {
    return {};
  }

  const uint64_t aligned_offset = align_up(slot->used, alignment);
  const uint64_t required_size = aligned_offset + byte_size;
  if (required_size < aligned_offset) {
    return {};
  }

  if (required_size > slot->bytes.size()) {
    const uint64_t new_capacity = next_capacity(required_size, slot->bytes.size());
    slot->bytes.resize(static_cast<size_t>(new_capacity));
  }

  slot->used = required_size;
  return {
    .buffer_index = handle.index,
    .byte_offset = aligned_offset,
    .byte_size = byte_size,
  };
}

bool BufferPool::write(BufferView view, const void* src, uint64_t byte_size, uint64_t write_offset) {
  Slot* slot = resolve(view.buffer_index);
  if ((slot == nullptr) || (src == nullptr)) {
    return false;
  }

  if ((write_offset > view.byte_size) || (byte_size > (view.byte_size - write_offset))) {
    return false;
  }

  const uint64_t dst_offset = view.byte_offset + write_offset;
  const uint64_t dst_end = dst_offset + byte_size;
  if ((dst_end > slot->bytes.size()) || (dst_end > slot->used)) {
    return false;
  }

  memcpy(slot->bytes.data() + dst_offset, src, static_cast<size_t>(byte_size));
  return true;
}

void* BufferPool::map(BufferView view) {
  Slot* slot = resolve(view.buffer_index);
  if (slot == nullptr) {
    return nullptr;
  }

  const uint64_t end = view.byte_offset + view.byte_size;
  if ((view.byte_size == 0u) || (end > slot->bytes.size()) || (end > slot->used)) {
    return nullptr;
  }

  return slot->bytes.data() + view.byte_offset;
}

const void* BufferPool::map(BufferView view) const {
  const Slot* slot = resolve(view.buffer_index);
  if (slot == nullptr) {
    return nullptr;
  }

  const uint64_t end = view.byte_offset + view.byte_size;
  if ((view.byte_size == 0u) || (end > slot->bytes.size()) || (end > slot->used)) {
    return nullptr;
  }

  return slot->bytes.data() + view.byte_offset;
}

void BufferPool::reset(BufferHandle handle) {
  Slot* slot = resolve(handle);
  if (slot == nullptr) {
    return;
  }
  slot->used = 0u;
}

void BufferPool::clear() {
  _slots.clear();
  _free_indices.clear();
}

uint64_t BufferPool::used_bytes(BufferHandle handle) const {
  const Slot* slot = resolve(handle);
  return (slot != nullptr) ? slot->used : 0u;
}

uint64_t BufferPool::capacity_bytes(BufferHandle handle) const {
  const Slot* slot = resolve(handle);
  return (slot != nullptr) ? static_cast<uint64_t>(slot->bytes.size()) : 0u;
}

BufferPool::Stats BufferPool::stats() const {
  Stats result = {};
  for (const auto& slot : _slots) {
    if (slot.alive == false) {
      continue;
    }
    result.buffer_count += 1u;
    result.used_bytes += slot.used;
    result.capacity_bytes += static_cast<uint64_t>(slot.bytes.size());
  }
  return result;
}

BufferPool::Slot* BufferPool::resolve(BufferHandle handle) {
  if (handle.index == kInvalidIndex) {
    return nullptr;
  }
  if (handle.index >= _slots.size()) {
    return nullptr;
  }

  Slot& slot = _slots[handle.index];
  if ((slot.alive == false) || (slot.generation != handle.generation)) {
    return nullptr;
  }

  return &slot;
}

const BufferPool::Slot* BufferPool::resolve(BufferHandle handle) const {
  if (handle.index == kInvalidIndex) {
    return nullptr;
  }
  if (handle.index >= _slots.size()) {
    return nullptr;
  }

  const Slot& slot = _slots[handle.index];
  if ((slot.alive == false) || (slot.generation != handle.generation)) {
    return nullptr;
  }

  return &slot;
}

BufferPool::Slot* BufferPool::resolve(uint32_t index) {
  if ((index == kInvalidIndex) || (index >= _slots.size())) {
    return nullptr;
  }

  Slot& slot = _slots[index];
  return slot.alive ? &slot : nullptr;
}

const BufferPool::Slot* BufferPool::resolve(uint32_t index) const {
  if ((index == kInvalidIndex) || (index >= _slots.size())) {
    return nullptr;
  }

  const Slot& slot = _slots[index];
  return slot.alive ? &slot : nullptr;
}

uint64_t BufferPool::align_up(uint64_t value, uint64_t alignment) {
  const uint64_t a = (alignment == 0u) ? 1u : alignment;
  return ((value + a - 1u) / a) * a;
}

uint64_t BufferPool::next_capacity(uint64_t required, uint64_t current) {
  uint64_t capacity = (current > 0u) ? current : kMinBufferCapacity;
  while (capacity < required) {
    const uint64_t grown = capacity + (capacity / 2u);
    capacity = (grown > capacity) ? grown : required;
  }
  return capacity;
}

}  // namespace etx
