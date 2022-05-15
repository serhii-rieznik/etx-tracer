#pragma once

#include <etx/core/debug.hxx>

#include <memory.h>
#include <stdint.h>
#include <new>

#define ETX_PIMPL_DECLARE(T, SUFFIX, SZ)      \
 public:                                      \
  T(const T&) = delete;                       \
  T& operator=(const T&) = delete;            \
  T(T&&) noexcept;                            \
  T& operator=(T&&) noexcept;                 \
                                              \
 private:                                     \
  alignas(16) uint8_t _private_storage[SZ]{}; \
  struct T##SUFFIX* _private = nullptr

#define ETX_PIMPL_IMPLEMENT(T, SUFFIX)                                          \
  T::T(T&& other) noexcept {                                                    \
    if (_private) {                                                             \
      _private->~T##SUFFIX();                                                   \
    }                                                                           \
    memcpy(_private_storage, other._private_storage, sizeof(_private_storage)); \
    _private = reinterpret_cast<struct T##SUFFIX*>(_private_storage);           \
    memset(other._private_storage, 0, sizeof(_private_storage));                \
    other._private = nullptr;                                                   \
  }                                                                             \
  T& T::operator=(T&& other) noexcept {                                         \
    if (_private) {                                                             \
      _private->~T##SUFFIX();                                                   \
    }                                                                           \
    memcpy(_private_storage, other._private_storage, sizeof(_private_storage)); \
    _private = reinterpret_cast<struct T##SUFFIX*>(_private_storage);           \
    memset(other._private_storage, 0, sizeof(_private_storage));                \
    other._private = nullptr;                                                   \
    return *this;                                                               \
  }

#define ETX_PIMPL_IMPLEMENT_ALL(T, SUFFIX) \
  ETX_PIMPL_IMPLEMENT(T, SUFFIX)           \
  T::T() {                                 \
    ETX_PIMPL_CREATE(T, SUFFIX);           \
  }                                        \
  T::~T() {                                \
    ETX_PIMPL_DESTROY(T, SUFFIX);          \
  }

#define ETX_PIMPL_CREATE(T, SUFFIX, ...)                                                                         \
  static_assert(sizeof(_private_storage) >= sizeof(T##SUFFIX), "Not enough storage for private implementation"); \
  ETX_ASSERT(_private == nullptr);                                                                               \
  memset(_private_storage, 0, sizeof(_private_storage));                                                         \
  _private = new (_private_storage) T##SUFFIX(__VA_ARGS__)

#define ETX_PIMPL_DESTROY(T, SUFFIX)                       \
  if (_private != nullptr) {                               \
    _private->~T##SUFFIX();                                \
    _private = nullptr;                                    \
    memset(_private_storage, 0, sizeof(_private_storage)); \
  }                                                        \
  do {                                                     \
  } while (false)

#define ETX_DECLARE_PIMPL(T, SZ) ETX_PIMPL_DECLARE(T, Impl, SZ)
#define ETX_IMPLEMENT_PIMPL(T) ETX_PIMPL_IMPLEMENT(T, Impl)
#define ETX_PIMPL_INIT(T, ...) ETX_PIMPL_CREATE(T, Impl, __VA_ARGS__)
#define ETX_PIMPL_CLEANUP(T) ETX_PIMPL_DESTROY(T, Impl)
