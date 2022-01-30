#pragma once

#include <etx/core/handle.hxx>

namespace etx {

template <class T>
struct ObjectPool {
  void init(uint64_t capacity, uint64_t type_id) {
    cleanup();

    _head = 0;
    _capacity = capacity;
    _type_id = type_id;
    _objects = reinterpret_cast<TData*>(::calloc(capacity, sizeof(TData)));

    for (auto i = _objects, e = _objects + capacity; i < e; ++i) {
      i->next = i - _objects + 1;
    }
  }

  void cleanup() {
    ::free(_objects);
    _capacity = 0;
    _type_id = 0;
    _objects = nullptr;
  }

  template <class... Args>
  Handle alloc(Args... args) {
    ETX_CRITICAL(_head != _capacity);

    auto& obj = _objects[_head];
    ETX_ASSERT(obj.alive == 0);

    obj.alive = 1;
    obj.generation += 1;
    ETX_ASSERT(obj.generation != 0);

    new (&obj.object) T(std::forward<Args>(args)...);
    auto result = Handle::construct(_type_id, _head, obj.generation);
    _head = obj.next;
    return result;
  }

  T& get(Handle h) {
    ETX_ASSERT(h.alive);
    ETX_ASSERT(h.index < _capacity);
    auto& obj = _objects[h.index];
    ETX_ASSERT(obj.generation == h.generation);
    return obj.object;
  }

  const T& get(Handle h) const {
    ETX_ASSERT(h.alive);
    ETX_ASSERT(h.index < _capacity);
    const auto& obj = _objects[h.index];
    ETX_ASSERT(obj.generation == h.generation);
    return obj.object;
  }

  void free(Handle h) {
    ETX_ASSERT(h.alive);
    ETX_ASSERT(h.index < _capacity);
    auto& obj = _objects[h.index];
    ETX_ASSERT(obj.generation = h.generation);
    (obj.object).~T();
    obj.alive = 0;
    obj.next = _head;
    _head = h.index;
  }

  uint64_t count_alive() const {
    uint64_t result = 0;
    for (uint64_t i = 0; i < _capacity; ++i) {
      result += _objects[i].alive;
    }
    return result;
  }

 private:
  struct alignas(T) TData {
    T object;
    uint64_t alive : 1;
    uint64_t generation : Handle::GenBits;
    uint64_t next : Handle::IndexBits;
  };
  TData* _objects = nullptr;
  uint64_t _type_id = 0;
  uint64_t _capacity = 0;
  uint64_t _head = 0;
};

}  // namespace etx
