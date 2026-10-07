#pragma once

#include <etx/core/core.hxx>
#include <array>
#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace etx {

struct MeshInput {
  explicit MeshInput(const std::filesystem::path& path) {
    _file.open(path, std::ios::binary | std::ios::ate);
    if (_file.is_open() == false)
      throw std::runtime_error("Cannot open mesh: " + path_to_utf8(path));
    const auto size = _file.tellg();
    if (size < 0)
      throw std::runtime_error("Cannot determine mesh size.");
    _size = static_cast<uint64_t>(size);
    _file.seekg(0);
  }

  uint64_t remaining() const {
    return _size - _offset;
  }

  uint64_t size() const {
    return _size;
  }

  void read(void* destination, size_t count) {
    if (count > remaining())
      throw std::runtime_error("Truncated mesh data.");
    auto* bytes = static_cast<char*>(destination);
    while (count > 0u) {
      refill();
      const size_t chunk = std::min(count, _end - _cursor);
      if (bytes != nullptr) {
        std::memcpy(bytes, _buffer.data() + _cursor, chunk);
        bytes += chunk;
      }
      advance(chunk);
      count -= chunk;
    }
  }

  template <class T>
  T binary(bool big_endian) {
    std::array<char, sizeof(T)> bytes;
    read(bytes.data(), bytes.size());
    if ((sizeof(T) > 1u) && (big_endian != (std::endian::native == std::endian::big)))
      std::reverse(bytes.begin(), bytes.end());
    return std::bit_cast<T>(bytes);
  }

  std::string line() {
    std::string result;
    while (remaining() > 0u) {
      refill();
      const char* begin = _buffer.data() + _cursor;
      const char* end = _buffer.data() + _end;
      const char* newline = std::find(begin, end, '\n');
      result.append(begin, newline);
      advance(static_cast<size_t>(newline - begin));
      if (newline != end) {
        advance(1u);
        break;
      }
    }
    if ((result.empty() == false) && (result.back() == '\r'))
      result.pop_back();
    return result;
  }

  std::string_view token() {
    _token.clear();
    while (remaining() > 0u) {
      refill();
      if (space(_buffer[_cursor]) == false)
        break;
      advance(1u);
    }
    while (remaining() > 0u) {
      refill();
      const size_t begin = _cursor;
      while ((_cursor < _end) && (space(_buffer[_cursor]) == false))
        advance(1u);
      const std::string_view part(_buffer.data() + begin, _cursor - begin);
      if ((_cursor < _end) || (remaining() == 0u)) {
        if (_token.empty())
          return part;
        _token.append(part);
        return _token;
      }
      _token.append(part);
    }
    return _token;
  }

  double number() {
    auto value = token();
    if ((value.empty() == false) && (value.front() == '+'))
      value.remove_prefix(1u);
    double result = 0.0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if ((parsed.ec != std::errc()) || (parsed.ptr != (value.data() + value.size())) || (std::isfinite(result) == false))
      throw std::runtime_error("Invalid or non-finite mesh number.");
    return result;
  }

  static bool space(char value) {
    return (value == ' ') || (value == '\t') || (value == '\r') || (value == '\n') || (value == '\v') || (value == '\f');
  }

 private:
  void advance(size_t count) {
    _cursor += count;
    _offset += count;
  }

  void refill() {
    if (_cursor < _end)
      return;
    _file.read(_buffer.data(), _buffer.size());
    _cursor = 0u;
    _end = static_cast<size_t>(_file.gcount());
    if (_end == 0u)
      throw std::runtime_error("Failed to read mesh data.");
  }

  std::ifstream _file;
  std::array<char, 65536u> _buffer;
  std::string _token;
  uint64_t _size = 0u;
  uint64_t _offset = 0u;
  size_t _cursor = 0u;
  size_t _end = 0u;
};

inline float mesh_float(double value) {
  if ((std::isfinite(value) == false) || (std::abs(value) > std::numeric_limits<float>::max()))
    throw std::runtime_error("Mesh attribute exceeds the native floating-point range.");
  return static_cast<float>(value);
}

}  // namespace etx
