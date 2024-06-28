#pragma once

#include <etx/core/handle.hxx>

namespace etx {

template <class T>
struct ObjectIndexPool {
  void init(uint32_t capacity) {
    cleanup();

    _head = 0;
    _capacity = capacity;
    _objects = reinterpret_cast<T*>(::calloc(capacity, sizeof(T)));
    _info = reinterpret_cast<TData*>(::calloc(capacity, sizeof(TData)));
    for (uint32_t i = 0; (_info != nullptr) && (i < _capacity); ++i) {
      _info[i].next = i + 1;
    }
  }

  void cleanup() {
    ::free(_objects);
    _objects = nullptr;
    ::free(_info);
    _info = nullptr;
    _head = 0;
    _capacity = 0;
  }

  template <class... Args>
  uint32_t alloc(Args&&... args) {
    ETX_ASSERT(_capacity > 0);
    ETX_CRITICAL(_head != _capacity);

    auto& info = _info[_head];
    ETX_ASSERT(info.alive == 0);
    info.alive = 1;

    new (_objects + _head) T(std::forward<Args>(args)...);

    auto result = _head;
    _head = info.next;
    return result;
  }

  T& get(uint32_t h) {
    ETX_ASSERT(h < _capacity);
    ETX_ASSERT(_info[h].alive);
    return _objects[h];
  }

  const T& get(uint32_t h) const {
    ETX_ASSERT(h < _capacity);
    ETX_ASSERT(_info[h].alive);
    return _objects[h];
  }

  void free(uint32_t h) {
    ETX_ASSERT(h < _capacity);
    ETX_ASSERT(_info[h].alive);

    auto& obj = _objects[h];
    obj.~T();

    auto& info = _info[h];
    info.alive = 0;
    info.next = _head;

    _head = h;
  }

  template <class ReleaseFunc>
  void free_all(ReleaseFunc release_func) {
    for (uint32_t i = 0; i < _capacity; ++i) {
      if (_info[i].alive) {
        release_func(_objects[i]);
        _objects[i].~T();
        _info[i].alive = 0;
      }
      _info[i].next = i + 1;
    }
    _head = 0;
  }

  T* data() {
    return _objects;
  }

  uint32_t alive_objects_count() const {
    uint32_t result = 0;
    for (uint32_t i = 0; i < _capacity; ++i) {
      result += _info[i].alive;
    }
    return result;
  }

  uint32_t latest_alive_index() const {
    uint32_t result = 0;
    for (uint32_t i = 0; i < _capacity; ++i) {
      if (_info[i].alive) {
        result = i;
      }
    }
    return result;
  }

  bool alive(uint32_t index) const {
    ETX_ASSERT(index < _capacity);
    return _info[index].alive;
  }

 private:
  struct TData {
    uint32_t alive = 0;
    uint32_t next = uint32_t(-1);
  };
  T* _objects = nullptr;
  TData* _info = nullptr;
  uint32_t _capacity = 0;
  uint32_t _head = 0;
};

}  // namespace etx
