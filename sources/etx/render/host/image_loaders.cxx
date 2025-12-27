#include <etx/render/host/image_loaders.hxx>
#include <etx/render/shared/math.hxx>

#include <tinyexr.hxx>
#include <stb_image.hxx>

#define BCDEC_IMPLEMENTATION
#define BCDEC_BC4BC5_PRECISE
#include <bcdec.h>

#include <atomic>
#include <vector>
#include <functional>
#include <cstring>
#include <cstdio>
#include <algorithm>

namespace etx {

#pragma pack(push, 1)
struct DDS_PIXELFORMAT {
  uint32_t dwSize;
  uint32_t dwFlags;
  uint32_t dwFourCC;
  uint32_t dwRGBBitCount;
  uint32_t dwRBitMask;
  uint32_t dwGBitMask;
  uint32_t dwBBitMask;
  uint32_t dwABitMask;
};

struct DDS_HEADER {
  uint32_t dwSize;
  uint32_t dwFlags;
  uint32_t dwHeight;
  uint32_t dwWidth;
  uint32_t dwPitchOrLinearSize;
  uint32_t dwDepth;
  uint32_t dwMipMapCount;
  uint32_t dwReserved1[11];
  DDS_PIXELFORMAT ddspf;
  uint32_t dwCaps;
  uint32_t dwCaps2;
  uint32_t dwCaps3;
  uint32_t dwCaps4;
  uint32_t dwReserved2;
};

struct DDS_HEADER_DXT10 {
  uint32_t dxgiFormat;
  uint32_t resourceDimension;
  uint32_t miscFlag;
  uint32_t arraySize;
  uint32_t miscFlags2;
};
#pragma pack(pop)

#define DDS_FOURCC_DXT1 0x31545844  // "DXT1"
#define DDS_FOURCC_DXT3 0x33545844  // "DXT3"
#define DDS_FOURCC_DXT5 0x35545844  // "DXT5"
#define DDS_FOURCC_BC4U 0x55344342  // "BC4U"
#define DDS_FOURCC_BC4S 0x53344342  // "BC4S"
#define DDS_FOURCC_BC5U 0x55354342  // "BC5U"
#define DDS_FOURCC_BC5S 0x53354342  // "BC5S"
#define DDS_FOURCC_ATI2 0x32495441  // "ATI2" (alternative name for BC5)
#define DDS_FOURCC_BC6H 0x48433642  // "BC6H"
#define DDS_FOURCC_BC7  0x37433642  // "BC7"

#define DXGI_FORMAT_BC1_UNORM      71
#define DXGI_FORMAT_BC1_UNORM_SRGB 72
#define DXGI_FORMAT_BC2_UNORM      74
#define DXGI_FORMAT_BC2_UNORM_SRGB 75
#define DXGI_FORMAT_BC3_UNORM      77
#define DXGI_FORMAT_BC3_UNORM_SRGB 78
#define DXGI_FORMAT_BC4_UNORM      80
#define DXGI_FORMAT_BC4_SNORM      81
#define DXGI_FORMAT_BC5_UNORM      83
#define DXGI_FORMAT_BC5_SNORM      84
#define DXGI_FORMAT_BC6H_UF16      95
#define DXGI_FORMAT_BC6H_SF16      96
#define DXGI_FORMAT_BC7_UNORM      98
#define DXGI_FORMAT_BC7_UNORM_SRGB 99

enum class BCType { BC1, BC2, BC3, BC4, BC5, BC6H, BC7, Unknown };

struct BCFormatInfo {
  BCType type;
  bool is_signed;
  bool is_srgb;
};

Image::Format bc_type_to_image_format(BCType type, bool is_signed, bool is_srgb) {
  switch (type) {
    case BCType::BC1:
      return is_srgb ? Image::Format::BC1_SRGB : Image::Format::BC1;
    case BCType::BC2:
      return is_srgb ? Image::Format::BC2_SRGB : Image::Format::BC2;
    case BCType::BC3:
      return is_srgb ? Image::Format::BC3_SRGB : Image::Format::BC3;
    case BCType::BC4:
      return Image::Format::BC4;
    case BCType::BC5:
      return Image::Format::BC5;
    case BCType::BC6H:
      return is_signed ? Image::Format::BC6H_SIGNED : Image::Format::BC6H;
    case BCType::BC7:
      return is_srgb ? Image::Format::BC7_SRGB : Image::Format::BC7;
    default:
      return Image::Format::Undefined;
  }
}

BCFormatInfo detect_bc_format(const DDS_HEADER& header, const DDS_HEADER_DXT10* dxt10 = nullptr) {
  if (header.ddspf.dwFourCC == DDS_FOURCC_DXT1)
    return {BCType::BC1, false, true};
  if (header.ddspf.dwFourCC == DDS_FOURCC_DXT3)
    return {BCType::BC2, false, true};
  if (header.ddspf.dwFourCC == DDS_FOURCC_DXT5)
    return {BCType::BC3, false, true};
  if (header.ddspf.dwFourCC == DDS_FOURCC_BC4U)
    return {BCType::BC4, false, false};
  if (header.ddspf.dwFourCC == DDS_FOURCC_BC4S)
    return {BCType::BC4, true, false};
  if (header.ddspf.dwFourCC == DDS_FOURCC_BC5U)
    return {BCType::BC5, false, false};
  if (header.ddspf.dwFourCC == DDS_FOURCC_BC5S)
    return {BCType::BC5, true, false};
  if (header.ddspf.dwFourCC == DDS_FOURCC_ATI2)
    return {BCType::BC5, false, false};  // ATI2 is BC5 unsigned
  if (header.ddspf.dwFourCC == DDS_FOURCC_BC6H)
    return {BCType::BC6H, true, false};  // Default BC6H to signed for BC6S compatibility
  if (header.ddspf.dwFourCC == DDS_FOURCC_BC7)
    return {BCType::BC7, false, true};

  // Check for DX10 header (FourCC "DX10")
  if ((header.ddspf.dwFourCC == 0x30315844) && dxt10) {  // "DX10"
    switch (dxt10->dxgiFormat) {
      case DXGI_FORMAT_BC1_UNORM:
        return {BCType::BC1, false, false};
      case DXGI_FORMAT_BC1_UNORM_SRGB:
        return {BCType::BC1, false, true};
      case DXGI_FORMAT_BC2_UNORM:
        return {BCType::BC2, false, false};
      case DXGI_FORMAT_BC2_UNORM_SRGB:
        return {BCType::BC2, false, true};
      case DXGI_FORMAT_BC3_UNORM:
        return {BCType::BC3, false, false};
      case DXGI_FORMAT_BC3_UNORM_SRGB:
        return {BCType::BC3, false, true};
      case DXGI_FORMAT_BC4_UNORM:
        return {BCType::BC4, false, false};
      case DXGI_FORMAT_BC4_SNORM:
        return {BCType::BC4, true, false};
      case DXGI_FORMAT_BC5_UNORM:
        return {BCType::BC5, false, false};
      case DXGI_FORMAT_BC5_SNORM:
        return {BCType::BC5, true, false};
      case DXGI_FORMAT_BC6H_UF16:
        return {BCType::BC6H, false, false};
      case DXGI_FORMAT_BC6H_SF16:
        return {BCType::BC6H, true, false};
      case DXGI_FORMAT_BC7_UNORM:
        return {BCType::BC7, false, true};
      case DXGI_FORMAT_BC7_UNORM_SRGB:
        return {BCType::BC7, false, true};
      default:
        return {BCType::Unknown, false, false};
    }
  }

  return {BCType::Unknown, false, false};
}

void decompress_bc_block(BCType type, bool is_signed, bool is_bgra, const uint8_t* block_data, float4 decompressed_block[16]) {
  if (type == BCType::BC5) {
    float bc5_rg[32] = {};
    bcdec_bc5_float(block_data, bc5_rg, 4 * 2, is_signed ? 1 : 0);

    for (int i = 0; i < 16; ++i) {
      float r = bc5_rg[i * 2 + 0];
      float g = bc5_rg[i * 2 + 1];
      float nx = r * 2.0f - 1.0f;
      float ny = g * 2.0f - 1.0f;
      float nz = 1.0f;

      float length = sqrtf(nx * nx + ny * ny + nz * nz);
      if (length > 0.0f) {
        nx /= length;
        ny /= length;
        nz /= length;
      }

      decompressed_block[i] = {nx * 0.5f + 0.5f, ny * 0.5f + 0.5f, nz * 0.5f + 0.5f, 1.0f};
    }
    return;
  }

  if (type == BCType::BC6H) {
    // BC6H needs float precision, handle directly
    float decompressed_float[48] = {};
    bcdec_bc6h_float(block_data, decompressed_float, 4 * 3, is_signed ? 1 : 0);

    for (int i = 0; i < 16; ++i) {
      uint32_t pixel_offset = i * 3;
      float r = decompressed_float[pixel_offset + 0];
      float g = decompressed_float[pixel_offset + 1];
      float b = decompressed_float[pixel_offset + 2];

      if (is_bgra) {
        std::swap(r, b);
      }

      decompressed_block[i] = {r, g, b, 1.0f};
    }
    return;
  }

  Image::Format format = bc_type_to_image_format(type, is_signed, false);
  uint8_t rgba8[64] = {};
  Image::decompress_bc_to_rgba(format, block_data, rgba8, is_signed);

  for (int i = 0; i < 16; ++i) {
    uint8_t r = rgba8[i * 4 + 0];
    uint8_t g = rgba8[i * 4 + 1];
    uint8_t b = rgba8[i * 4 + 2];
    uint8_t a = rgba8[i * 4 + 3];

    if (is_bgra) {
      std::swap(r, b);
    }

    decompressed_block[i] = {r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f};
  }
}

float half_to_float(uint16_t half) {
  uint32_t sign = (half >> 15) & 0x1;
  uint32_t exponent = (half >> 10) & 0x1F;
  uint32_t mantissa = half & 0x3FF;

  if (exponent == 0) {
    // Denormalized number
    if (mantissa == 0)
      return sign ? -0.0f : 0.0f;
    return (sign ? -1.0f : 1.0f) * powf(2.0f, -14.0f) * (mantissa / 1024.0f);
  } else if (exponent == 31) {
    // Infinity or NaN
    return mantissa == 0 ? (sign ? -INFINITY : INFINITY) : NAN;
  } else {
    // Normalized number
    return (sign ? -1.0f : 1.0f) * powf(2.0f, exponent - 15.0f) * (1.0f + mantissa / 1024.0f);
  }
}

Image::Format load_dds(const char* source, std::vector<uint8_t>& data, uint2& dimensions) {
  FILE* file = fopen(source, "rb");
  if (!file) {
    log::error("Failed to open DDS file: %s", source);
    return Image::Format::Undefined;
  }

  uint32_t magic = 0;
  if ((fread(&magic, sizeof(uint32_t), 1, file) != 1) || (magic != 0x20534444)) {
    log::error("Invalid DDS magic number in file: %s", source);
    fclose(file);
    return Image::Format::Undefined;
  }

  DDS_HEADER header = {};
  if (fread(&header, sizeof(DDS_HEADER), 1, file) != 1) {
    fclose(file);
    return Image::Format::Undefined;
  }

  if ((header.dwSize != 124) || (header.ddspf.dwSize != 32)) {
    log::error("Invalid DDS header in file: %s", source);
    fclose(file);
    return Image::Format::Undefined;
  }

  DDS_HEADER_DXT10 dxt10_header = {};
  bool has_dxt10 = (header.ddspf.dwFourCC == 0x30315844);  // "DX10"
  if (has_dxt10) {
    if (fread(&dxt10_header, sizeof(DDS_HEADER_DXT10), 1, file) != 1) {
      log::error("Failed to read DX10 header from DDS file: %s", source);
      fclose(file);
      return Image::Format::Undefined;
    }
  }

  BCFormatInfo bc_format_info = detect_bc_format(header, has_dxt10 ? &dxt10_header : nullptr);
  if (bc_format_info.type == BCType::Unknown) {
    if (has_dxt10) {
      log::error("Unsupported DXGI format in DDS file: %s (DXGI format: %u)", source, dxt10_header.dxgiFormat);
    } else {
      log::error("Unsupported BC format in DDS file: %s (FourCC: 0x%08X, Flags: 0x%08X)", source, header.ddspf.dwFourCC, header.ddspf.dwFlags);
    }
    fclose(file);
    return Image::Format::Undefined;
  }

  bool is_srgb = bc_format_info.is_srgb;

  // Check if DDS uses BGRA channel order (common in some DDS files)
  bool is_bgra = false;
  if ((header.ddspf.dwRBitMask == 0x000000FF) && (header.ddspf.dwBBitMask == 0x00FF0000)) {
    is_bgra = true;
  }

  dimensions = {header.dwWidth, header.dwHeight};

  // Calculate BC block dimensions
  uint32_t blocks_x = (header.dwWidth + 3) / 4;
  uint32_t blocks_y = (header.dwHeight + 3) / 4;
  uint32_t block_size = 0;

  switch (bc_format_info.type) {
    case BCType::BC1:
      block_size = 8;
      break;
    case BCType::BC2:
      block_size = 16;
      break;
    case BCType::BC3:
      block_size = 16;
      break;
    case BCType::BC4:
      block_size = 8;
      break;
    case BCType::BC5:
      block_size = 16;
      break;
    case BCType::BC6H:
      block_size = 16;
      break;
    case BCType::BC7:
      block_size = 16;
      break;
    default:
      fclose(file);
      return Image::Format::Undefined;
  }

  size_t compressed_size = blocks_x * blocks_y * block_size;
  std::vector<uint8_t> compressed_data(compressed_size);
  if (fread(compressed_data.data(), 1, compressed_size, file) != compressed_size) {
    log::error("Failed to read DDS compressed data from file: %s", source);
    fclose(file);
    return Image::Format::Undefined;
  }
  fclose(file);

#if ETX_STORE_COMPRESSED_BC
  data = std::move(compressed_data);
  dimensions = {header.dwWidth, header.dwHeight};

  switch (bc_format_info.type) {
    case BCType::BC1:
      return bc_format_info.is_srgb ? Image::Format::BC1_SRGB : Image::Format::BC1;
    case BCType::BC2:
      return bc_format_info.is_srgb ? Image::Format::BC2_SRGB : Image::Format::BC2;
    case BCType::BC3:
      return bc_format_info.is_srgb ? Image::Format::BC3_SRGB : Image::Format::BC3;
    case BCType::BC4:
      return Image::Format::BC4;
    case BCType::BC5:
      return Image::Format::BC5;
    case BCType::BC6H:
      return bc_format_info.is_signed ? Image::Format::BC6H_SIGNED : Image::Format::BC6H;
    case BCType::BC7:
      return bc_format_info.is_srgb ? Image::Format::BC7_SRGB : Image::Format::BC7;
    default:
      return Image::Format::Undefined;
  }
#else
  // Default behavior: decompress to RGBA8/RGBA32F
  if (is_srgb) {
    data.resize(header.dwWidth * header.dwHeight * sizeof(ubyte4));
  } else {
    data.resize(header.dwWidth * header.dwHeight * sizeof(float4));
  }

  float4* output_float = nullptr;
  ubyte4* output_byte = nullptr;
  if (is_srgb) {
    output_byte = reinterpret_cast<ubyte4*>(data.data());
  } else {
    output_float = reinterpret_cast<float4*>(data.data());
  }

  for (uint32_t by = 0; by < blocks_y; ++by) {
    for (uint32_t bx = 0; bx < blocks_x; ++bx) {
      uint32_t block_index = by * blocks_x + bx;
      const uint8_t* block_data = compressed_data.data() + block_index * block_size;

      float4 decompressed_block[16];
      decompress_bc_block(bc_format_info.type, bc_format_info.is_signed, is_bgra, block_data, decompressed_block);

      for (uint32_t y = 0; y < 4; ++y) {
        for (uint32_t x = 0; x < 4; ++x) {
          uint32_t pixel_x = bx * 4 + x;
          uint32_t pixel_y = by * 4 + y;
          uint32_t block_pixel_index = y * 4 + x;

          if (bc_format_info.type == BCType::BC5) {
            pixel_y = (blocks_y - 1 - by) * 4 + (3 - y);
            block_pixel_index = (3 - y) * 4 + x;
          } else if (bc_format_info.type == BCType::BC6H) {
            pixel_y = (header.dwHeight - 1) - (by * 4 + y);
          }

          if ((pixel_x < header.dwWidth) && (pixel_y < header.dwHeight)) {
            uint32_t pixel_index = pixel_y * header.dwWidth + pixel_x;

            float4 val = decompressed_block[block_pixel_index];

            if (is_srgb) {
              output_byte[pixel_index] = {static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val.x * 255.0f))),
                static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val.y * 255.0f))), static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val.z * 255.0f))),
                static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val.w * 255.0f)))};
            } else {
              output_float[pixel_index] = val;
            }
          }
        }
      }
    }
  }

  dimensions = {header.dwWidth, header.dwHeight};
  return is_srgb ? Image::Format::RGBA8 : Image::Format::RGBA32F;
#endif
}

bool load_pfm(const char* path, uint2& size, std::vector<uint8_t>& data) {
  FILE* in_file = fopen(path, "rb");
  if (in_file == nullptr) {
    log::error("Failed to open PFM file: %s", path);
    return false;
  }

  char buffer[16] = {};

  auto read_line = [&]() {
    memset(buffer, 0, sizeof(buffer));
    char c = {};
    int p = 0;
    while ((p < 16) && (fread(&c, 1, 1, in_file) == 1)) {
      if (c == '\n') {
        return;
      } else {
        buffer[p++] = c;
      }
    }
  };

  read_line();
  if (strcmp(buffer, "PF") != 0) {
    log::error("Invalid PFM format identifier in file: %s", path);
    fclose(in_file);
    return false;
  }

  read_line();
  int w = 0;
  int h = 0;
  if (sscanf(buffer, "%d %d", &w, &h) != 2) {
    log::error("Invalid PFM dimensions in file: %s", path);
    fclose(in_file);
    return false;
  }

  read_line();
  float scale = 0.0f;
  if (sscanf(buffer, "%f", &scale) != 1) {
    log::error("Invalid PFM scale in file: %s", path);
    fclose(in_file);
    return false;
  }

  size = {uint32_t(w), uint32_t(h)};
  data.resize(sizeof(float4) * w * h);

  auto ptr = reinterpret_cast<float4*>(data.data());
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float r = 0.0f, g = 0.0f, b = 0.0f;
      bool read_r = (fread(&r, sizeof(float), 1, in_file) == 1);
      bool read_g = (fread(&g, sizeof(float), 1, in_file) == 1);
      bool read_b = (fread(&b, sizeof(float), 1, in_file) == 1);
      if ((read_r == false) || (read_g == false) || (read_b == false)) {
        log::error("Failed to read PFM pixel data from file: %s", path);
        fclose(in_file);
        return false;
      }

      int target_y = h - 1 - y;
      ptr[target_y * w + x] = {r, g, b, 1.0f};
    }
  }

  fclose(in_file);
  return true;
}

Image::Format load_data(const char* source, std::vector<uint8_t>& data, uint2& dimensions) {
  if (source == nullptr)
    return Image::Format::Undefined;

  const char* ext = nullptr;
  if (uint64_t l = strlen(source)) {
    while ((l > 0) && (source[--l] != '.')) {
    }
    ext = source + l;
  } else {
    return Image::Format::Undefined;
  }

  if ((strcmp(ext, ".dds") == 0) || (strcmp(ext, ".DDS") == 0)) {
    return load_dds(source, data, dimensions);
  }

  if (strcmp(ext, ".exr") == 0) {
    int w = 0;
    int h = 0;
    const char* error = nullptr;
    float* rgba_data = nullptr;
    if (LoadEXR(&rgba_data, &w, &h, source, &error) != TINYEXR_SUCCESS) {
      printf("Failed to load EXR from file: %s\n", error);
      return Image::Format::Undefined;
    }

    for (int i = 0; i < 4 * w * h; ++i) {
      if (std::isinf(rgba_data[i])) {
        rgba_data[i] = 65504.0f;  // max value in half-float
      }
      if (std::isnan(rgba_data[i]) || (rgba_data[i] < 0.0f)) {
        rgba_data[i] = 0.0f;
      }
    }

    dimensions = {uint32_t(w), uint32_t(h)};
    data.resize(sizeof(float4) * w * h);
    memcpy(data.data(), rgba_data, sizeof(float4) * w * h);
    free(rgba_data);

    return Image::Format::RGBA32F;
  }

  if (strcmp(ext, ".hdr") == 0) {
    int w = 0;
    int h = 0;
    int c = 0;
    stbi_set_flip_vertically_on_load(false);
    auto image = stbi_loadf(source, &w, &h, &c, 0);
    if (image == nullptr) {
      log::error("Failed to load HDR image: %s", source);
      return Image::Format::Undefined;
    }

    dimensions = {uint32_t(w), uint32_t(h)};
    data.resize(sizeof(float4) * w * h);
    auto ptr = reinterpret_cast<float4*>(data.data());
    if (c == 4) {
      memcpy(ptr, image, sizeof(float4) * w * h);
    } else {
      for (int i = 0; i < w * h; ++i) {
        ptr[i] = {image[3 * i + 0], image[3 * i + 1], image[3 * i + 2], 1.0f};
      }
    }
    free(image);
    return Image::Format::RGBA32F;
  }

  if (strcmp(ext, ".pfm") == 0) {
    return load_pfm(source, dimensions, data) ? Image::Format::RGBA32F : Image::Format::Undefined;
  }

  if ((strcmp(ext, ".tga") == 0) || (strcmp(ext, ".TGA") == 0)) {
    int w = 0;
    int h = 0;
    int c = 0;
    stbi_set_flip_vertically_on_load(false);
    auto image = stbi_load(source, &w, &h, &c, 4);
    if (image == nullptr) {
      return Image::Format::Undefined;
    }

    dimensions = {uint32_t(w), uint32_t(h)};
    data.resize(4llu * w * h);
    memcpy(data.data(), image, 4llu * w * h);
    free(image);
    return Image::Format::RGBA8;
  }

  int w = 0;
  int h = 0;
  int c = 0;
  stbi_set_flip_vertically_on_load(true);
  auto image = stbi_load(source, &w, &h, &c, 0);
  if (image == nullptr) {
    const char* image_pos = strstr(source, "##image-");
    if (image_pos == nullptr) {
      log::error("Failed to load image: %s", source);
    }
    return Image::Format::Undefined;
  }

  dimensions = {uint32_t(w), uint32_t(h)};
  data.resize(4llu * w * h);
  uint8_t* ptr = reinterpret_cast<uint8_t*>(data.data());
  switch (c) {
    case 4: {
      memcpy(ptr, image, 4llu * w * h);
      break;
    }

    case 3: {
      for (int i = 0; i < w * h; ++i) {
        ptr[4 * i + 0] = image[3 * i + 0];
        ptr[4 * i + 1] = image[3 * i + 1];
        ptr[4 * i + 2] = image[3 * i + 2];
        ptr[4 * i + 3] = 255;
      }
      break;
    }

    case 1: {
      for (int i = 0; i < w * h; ++i) {
        ptr[4 * i + 0] = image[i];
        ptr[4 * i + 1] = image[i];
        ptr[4 * i + 2] = image[i];
        ptr[4 * i + 3] = 255;
      }
      break;
    }

    default: {
      free(image);
      return Image::Format::Undefined;
    }
  }

  free(image);
  return Image::Format::RGBA8;
}

}  // namespace etx
