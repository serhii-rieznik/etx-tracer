#include <etx/std.hxx>
#include <etx/core/core.hxx>
#include "scene_pbrt_input.hxx"

#include <zlib.h>
#include <array>
#include <fstream>
#include <stdexcept>

namespace etx {
namespace {

struct GzipInflater {
  GzipInflater() {
    if (inflateInit2(&stream, MAX_WBITS + 16) != Z_OK)
      throw std::runtime_error("Cannot initialize the PBRT gzip decoder.");
  }

  ~GzipInflater() {
    inflateEnd(&stream);
  }

  z_stream stream = {};
};

}  // namespace

bool is_pbrt_gzip_file(const std::filesystem::path& file) {
  return _stricmp(path_to_utf8(file.extension()).c_str(), ".gz") == 0;
}

bool is_pbrt_scene_file(const std::filesystem::path& file) {
  const auto extension = is_pbrt_gzip_file(file) ? file.stem().extension() : file.extension();
  return _stricmp(path_to_utf8(extension).c_str(), ".pbrt") == 0;
}

std::string read_pbrt_gzip(const std::filesystem::path& file) {
  const auto fail = [&](const std::string& message) -> void {
    throw std::runtime_error(path_to_utf8(file) + ": " + message);
  };
  std::ifstream input(file, std::ios::binary | std::ios::ate);
  if (input.is_open() == false)
    fail("Cannot open gzip input.");
  const auto size = input.tellg();
  if (size < 0)
    fail("Cannot determine gzip input size.");
  input.seekg(0);
  std::string contents;
  contents.reserve(static_cast<size_t>(size));
  GzipInflater inflater;
  auto& stream = inflater.stream;
  std::array<Bytef, 65536u> compressed, decompressed;
  bool member_complete = false;
  for (;;) {
    if (stream.avail_in == 0u) {
      input.read(reinterpret_cast<char*>(compressed.data()), compressed.size());
      if (input.bad() || (input.fail() && (input.eof() == false)))
        fail("Cannot read gzip input.");
      stream.avail_in = static_cast<uInt>(input.gcount());
      stream.next_in = compressed.data();
      if (stream.avail_in == 0u) {
        if (member_complete)
          return contents;
        fail("Truncated gzip input.");
      }
    }
    if (member_complete) {
      if (inflateReset(&stream) != Z_OK)
        fail("Cannot reset the gzip decoder.");
      member_complete = false;
    }
    stream.avail_out = static_cast<uInt>(decompressed.size());
    stream.next_out = decompressed.data();
    const int result = inflate(&stream, Z_NO_FLUSH);
    contents.append(reinterpret_cast<const char*>(decompressed.data()), decompressed.size() - stream.avail_out);
    if (result == Z_STREAM_END)
      member_complete = true;
    else if ((result != Z_OK) && ((result != Z_BUF_ERROR) || (stream.avail_in != 0u)))
      fail(std::string("Invalid gzip input: ") + (stream.msg == nullptr ? "decompression failed" : stream.msg));
  }
}

}  // namespace etx
