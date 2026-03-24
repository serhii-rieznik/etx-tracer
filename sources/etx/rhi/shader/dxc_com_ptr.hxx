#pragma once

namespace etx {

template <typename T>
struct DxcComPtr {
  DxcComPtr() = default;

  DxcComPtr(const DxcComPtr& other)
    : _ptr(other._ptr) {
    internal_add_ref();
  }

  DxcComPtr(DxcComPtr&& other) noexcept
    : _ptr(other._ptr) {
    other._ptr = nullptr;
  }

  ~DxcComPtr() {
    internal_release();
  }

  DxcComPtr& operator=(const DxcComPtr& other) {
    if (this == &other) {
      return *this;
    }

    T* new_ptr = other._ptr;
    if (new_ptr != nullptr) {
      new_ptr->AddRef();
    }

    internal_release();
    _ptr = new_ptr;
    return *this;
  }

  DxcComPtr& operator=(DxcComPtr&& other) noexcept {
    if (this == &other) {
      return *this;
    }

    internal_release();
    _ptr = other._ptr;
    other._ptr = nullptr;
    return *this;
  }

  T* Get() const {
    return _ptr;
  }

  T** GetAddressOf() {
    return &_ptr;
  }

  T** ReleaseAndGetAddressOf() {
    Reset();
    return &_ptr;
  }

  T* Detach() {
    T* result = _ptr;
    _ptr = nullptr;
    return result;
  }

  void Reset() {
    internal_release();
    _ptr = nullptr;
  }

  T* operator->() const {
    return _ptr;
  }

  explicit operator bool() const {
    return _ptr != nullptr;
  }

 private:
  void internal_add_ref() {
    if (_ptr != nullptr) {
      _ptr->AddRef();
    }
  }

  void internal_release() {
    if (_ptr != nullptr) {
      _ptr->Release();
      _ptr = nullptr;
    }
  }

  T* _ptr = nullptr;
};

}  // namespace etx
