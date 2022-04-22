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
    _head = 0;
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
    ETX_ASSERT(h.index < _capacity);
    auto& obj = _objects[h.index];
    ETX_ASSERT(obj.alive);
    ETX_ASSERT(obj.generation == h.generation);
    return obj.object;
  }

  const T& get(Handle h) const {
    ETX_ASSERT(h.index < _capacity);
    const auto& obj = _objects[h.index];
    ETX_ASSERT(obj.alive);
    ETX_ASSERT(obj.generation == h.generation);
    return obj.object;
  }

  void free(Handle h) {
    ETX_ASSERT(h.index < _capacity);
    auto& obj = _objects[h.index];
    ETX_ASSERT(obj.alive);
    ETX_ASSERT(obj.generation = h.generation);
    (obj.object).~T();
    obj.alive = 0;
    obj.next = _head;
    _head = h.index;
  }

  void free_all() {
    for (auto i = _objects, e = _objects + _capacity; i < e; ++i) {
      if (i->alive) {
        (i->object).~T();
        i->alive = 0;
      }
      i->generation = 0;
      i->next = i - _objects + 1;
    }
  }

  template <class ReleaseFunc>
  void free_all(ReleaseFunc release_func) {
    for (auto i = _objects, e = _objects + _capacity; i < e; ++i) {
      if (i->alive) {
        release_func(i->object);
        (i->object).~T();
        i->alive = 0;
      }
      i->generation = 0;
      i->next = i - _objects + 1;
    }
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
  uint32_t alloc(Args... args) {
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

  void free_all() {
    for (uint32_t i = 0; i < _capacity; ++i) {
      if (_info[i].alive) {
        _objects[i].~T();
        _info[i].alive = 0;
      }
      _info[i].next = i + 1;
    }
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
  }

  uint32_t count_alive() const {
    uint32_t result = 0;
    for (uint32_t i = 0; i < _capacity; ++i) {
      result += _info[i].alive;
    }
    return result;
  }

  T* data() {
    return _objects;
  }

  uint32_t alive_objects_count() const {
    uint32_t result = 0;
    for (uint32_t i = 0; i < _capacity; ++i) {
      result += uint32_t(_info[i].alive);
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
