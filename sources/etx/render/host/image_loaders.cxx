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

// DDS structures for BC compressed textures
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

// DDS loading constants
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

enum class BCFormat { BC1, BC2, BC3, BC4, BC5, BC6H, BC7, Unknown };

struct BCFormatInfo {
  BCFormat format;
  bool is_signed;
};

BCFormatInfo detect_bc_format(const DDS_HEADER& header, const DDS_HEADER_DXT10* dxt10 = nullptr) {
  // Check FourCC codes first (legacy DDS)
  if (header.ddspf.dwFourCC == DDS_FOURCC_DXT1)
    return {BCFormat::BC1, false};
  if (header.ddspf.dwFourCC == DDS_FOURCC_DXT3)
    return {BCFormat::BC2, false};
  if (header.ddspf.dwFourCC == DDS_FOURCC_DXT5)
    return {BCFormat::BC3, false};
  if (header.ddspf.dwFourCC == DDS_FOURCC_BC4U)
    return {BCFormat::BC4, false};
  if (header.ddspf.dwFourCC == DDS_FOURCC_BC4S)
    return {BCFormat::BC4, true};
  if (header.ddspf.dwFourCC == DDS_FOURCC_BC5U)
    return {BCFormat::BC5, false};
  if (header.ddspf.dwFourCC == DDS_FOURCC_BC5S)
    return {BCFormat::BC5, true};
  if (header.ddspf.dwFourCC == DDS_FOURCC_ATI2)
    return {BCFormat::BC5, false};  // ATI2 is BC5 unsigned
  if (header.ddspf.dwFourCC == DDS_FOURCC_BC6H)
    return {BCFormat::BC6H, false};  // BC6H has signed variant but we'll handle it separately
  if (header.ddspf.dwFourCC == DDS_FOURCC_BC7)
    return {BCFormat::BC7, false};

  // Check for DX10 header (FourCC "DX10")
  if (header.ddspf.dwFourCC == 0x30315844 && dxt10) {  // "DX10"
    // Map DXGI format to BC format
    switch (dxt10->dxgiFormat) {
      case DXGI_FORMAT_BC1_UNORM:
      case DXGI_FORMAT_BC1_UNORM_SRGB:
        return {BCFormat::BC1, false};
      case DXGI_FORMAT_BC2_UNORM:
      case DXGI_FORMAT_BC2_UNORM_SRGB:
        return {BCFormat::BC2, false};
      case DXGI_FORMAT_BC3_UNORM:
      case DXGI_FORMAT_BC3_UNORM_SRGB:
        return {BCFormat::BC3, false};
      case DXGI_FORMAT_BC4_UNORM:
        return {BCFormat::BC4, false};
      case DXGI_FORMAT_BC4_SNORM:
        return {BCFormat::BC4, true};
      case DXGI_FORMAT_BC5_UNORM:
        return {BCFormat::BC5, false};
      case DXGI_FORMAT_BC5_SNORM:
        return {BCFormat::BC5, true};
      case DXGI_FORMAT_BC6H_UF16:
        return {BCFormat::BC6H, false};
      case DXGI_FORMAT_BC6H_SF16:
        return {BCFormat::BC6H, true};
      case DXGI_FORMAT_BC7_UNORM:
      case DXGI_FORMAT_BC7_UNORM_SRGB:
        return {BCFormat::BC7, false};
      default:
        return {BCFormat::Unknown, false};
    }
  }

  return {BCFormat::Unknown, false};
}

// Convert half-float (16-bit) to float (32-bit)
float half_to_float(uint16_t half) {
  // IEEE 754 half-float conversion
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

bool load_dds(const char* source, std::vector<uint8_t>& data, uint2& dimensions, bool& out_is_srgb) {
  FILE* file = fopen(source, "rb");
  if (!file) {
    log::error("Failed to open DDS file: %s", source);
    return false;
  }

  // Read DDS magic
  uint32_t magic = 0;
  if (fread(&magic, sizeof(uint32_t), 1, file) != 1 || magic != 0x20534444) {
    log::error("Invalid DDS magic number in file: %s", source);
    fclose(file);
    return false;
  }

  // Read DDS header
  DDS_HEADER header = {};
  if (fread(&header, sizeof(DDS_HEADER), 1, file) != 1) {
    fclose(file);
    return false;
  }

  // Validate header
  if (header.dwSize != 124 || header.ddspf.dwSize != 32) {
    log::error("Invalid DDS header in file: %s", source);
    fclose(file);
    return false;
  }

  // Read DX10 header if present
  DDS_HEADER_DXT10 dxt10_header = {};
  bool has_dxt10 = (header.ddspf.dwFourCC == 0x30315844);  // "DX10"
  if (has_dxt10) {
    if (fread(&dxt10_header, sizeof(DDS_HEADER_DXT10), 1, file) != 1) {
      log::error("Failed to read DX10 header from DDS file: %s", source);
      fclose(file);
      return false;
    }
  }

  BCFormatInfo bc_format_info = detect_bc_format(header, has_dxt10 ? &dxt10_header : nullptr);
  if (bc_format_info.format == BCFormat::Unknown) {
    if (has_dxt10) {
      log::error("Unsupported DXGI format in DDS file: %s (DXGI format: %u)", source, dxt10_header.dxgiFormat);
    } else {
      log::error("Unsupported BC format in DDS file: %s (FourCC: 0x%08X, Flags: 0x%08X)", source, header.ddspf.dwFourCC, header.ddspf.dwFlags);
    }
    fclose(file);
    return false;
  }

  // Check if this is an sRGB format
  // BC5/BC4 are excluded from sRGB detection as they're used for normal/roughness maps, not colors
  // For DX10 files, check the explicit SRGB format codes
  // For legacy files, assume color textures (BC1/BC2/BC3/BC7) are sRGB by default
  out_is_srgb = false;
  if (has_dxt10) {
    // DX10 format with explicit sRGB flags
    if (bc_format_info.format != BCFormat::BC5 && bc_format_info.format != BCFormat::BC4) {
      switch (dxt10_header.dxgiFormat) {
        case DXGI_FORMAT_BC1_UNORM_SRGB:
        case DXGI_FORMAT_BC2_UNORM_SRGB:
        case DXGI_FORMAT_BC3_UNORM_SRGB:
        case DXGI_FORMAT_BC7_UNORM_SRGB:
          out_is_srgb = true;
          break;
      }
    }
  } else {
    // Legacy DDS format - assume color textures are sRGB
    // BC5/BC4 are typically used for normal/roughness maps and should be linear
    if (bc_format_info.format != BCFormat::BC5 && bc_format_info.format != BCFormat::BC4) {
      out_is_srgb = true;  // Assume legacy DDS color textures are sRGB
    }
  }

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

  switch (bc_format_info.format) {
    case BCFormat::BC1:
      block_size = 8;
      break;  // 8 bytes per 4x4 block
    case BCFormat::BC2:
      block_size = 16;
      break;  // 16 bytes per 4x4 block
    case BCFormat::BC3:
      block_size = 16;
      break;  // 16 bytes per 4x4 block
    case BCFormat::BC4:
      block_size = 8;
      break;  // 8 bytes per 4x4 block
    case BCFormat::BC5:
      block_size = 16;
      break;  // 16 bytes per 4x4 block
    case BCFormat::BC6H:
      block_size = 16;
      break;  // 16 bytes per 4x4 block
    case BCFormat::BC7:
      block_size = 16;
      break;  // 16 bytes per 4x4 block
    default:
      fclose(file);
      return false;
  }

  // Read compressed data
  size_t compressed_size = blocks_x * blocks_y * block_size;
  std::vector<uint8_t> compressed_data(compressed_size);
  if (fread(compressed_data.data(), 1, compressed_size, file) != compressed_size) {
    log::error("Failed to read DDS compressed data from file: %s", source);
    fclose(file);
    return false;
  }
  fclose(file);

  if (out_is_srgb) {
    data.resize(header.dwWidth * header.dwHeight * sizeof(ubyte4));
  } else {
    data.resize(header.dwWidth * header.dwHeight * sizeof(float4));
  }

  // Decompress blocks
  float4* output_float = nullptr;
  ubyte4* output_byte = nullptr;
  if (out_is_srgb) {
    output_byte = reinterpret_cast<ubyte4*>(data.data());
  } else {
    output_float = reinterpret_cast<float4*>(data.data());
  }

  for (uint32_t by = 0; by < blocks_y; ++by) {
    for (uint32_t bx = 0; bx < blocks_x; ++bx) {
      uint32_t block_index = by * blocks_x + bx;
      const uint8_t* block_data = compressed_data.data() + block_index * block_size;

      // Decompress 4x4 block
      uint8_t decompressed_rgba8[64] = {};   // 4x4 RGBA8 block (16 pixels * 4 bytes) for BC1-BC5, BC7
      float4 decompressed_rgba32f[16] = {};  // 4x4 RGBA32F block (16 pixels) for BC6H

      switch (bc_format_info.format) {
        case BCFormat::BC1:
          bcdec_bc1(block_data, decompressed_rgba8, 4 * 4);  // pitch in bytes
          break;
        case BCFormat::BC2:
          bcdec_bc2(block_data, decompressed_rgba8, 4 * 4);
          break;
        case BCFormat::BC3:
          bcdec_bc3(block_data, decompressed_rgba8, 4 * 4);
          break;
        case BCFormat::BC4:
          bcdec_bc4(block_data, decompressed_rgba8, 4 * 4, bc_format_info.is_signed ? 1 : 0);
          break;
        case BCFormat::BC5: {
          float bc5_rg[128] = {};
          bcdec_bc5_float(block_data, bc5_rg, 4 * 2 * sizeof(float), bc_format_info.is_signed ? 1 : 0);
          for (int i = 0; i < 16; ++i) {
            float r = bc5_rg[i * 2 + 0];
            float g = bc5_rg[i * 2 + 1];
            decompressed_rgba8[i * 4 + 0] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, r * 255.0f)));
            decompressed_rgba8[i * 4 + 1] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, g * 255.0f)));
            decompressed_rgba8[i * 4 + 2] = 128;  // Default B
            decompressed_rgba8[i * 4 + 3] = 255;  // Default A
          }
          break;
        }
        case BCFormat::BC6H: {
          float bc6h_rgb[4 * 4 * 3];  // 4x4 RGB floats
          bcdec_bc6h_float(block_data, bc6h_rgb, 4 * 3 * sizeof(float), bc_format_info.is_signed ? 1 : 0);
          // Convert to RGBA32F
          for (int i = 0; i < 16; ++i) {
            float r = bc6h_rgb[i * 3 + 0];
            float g = bc6h_rgb[i * 3 + 1];
            float b = bc6h_rgb[i * 3 + 2];

            // Swap R and B if DDS uses BGRA format
            if (is_bgra) {
              std::swap(r, b);
            }

            decompressed_rgba32f[i].x = r;
            decompressed_rgba32f[i].y = g;
            decompressed_rgba32f[i].z = b;
            decompressed_rgba32f[i].w = 1.0f;
          }
          break;
        }
        case BCFormat::BC7:
          bcdec_bc7(block_data, decompressed_rgba8, 4 * 4);
          break;
        default:
          return false;
      }

      // Convert to float4 and copy to output
      for (uint32_t y = 0; y < 4; ++y) {
        for (uint32_t x = 0; x < 4; ++x) {
          uint32_t pixel_x = bx * 4 + x;
          uint32_t pixel_y = by * 4 + y;  // Don't flip during decompression

          if (pixel_x < header.dwWidth && pixel_y < header.dwHeight) {
            uint32_t pixel_index = pixel_y * header.dwWidth + pixel_x;
            uint32_t block_pixel_index = y * 4 + x;

            if (bc_format_info.format == BCFormat::BC6H) {
              // BC6H is already in float format
              if (out_is_srgb) {
                // Convert float to byte for sRGB DDS files
                float4 val = decompressed_rgba32f[block_pixel_index];
                output_byte[pixel_index] = {static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val.x * 255.0f))),
                  static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val.y * 255.0f))), static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val.z * 255.0f))),
                  static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val.w * 255.0f)))};
              } else {
                output_float[pixel_index] = decompressed_rgba32f[block_pixel_index];
              }
            } else {
              // Convert RGBA8 to float
              uint8_t r = decompressed_rgba8[block_pixel_index * 4 + 0];
              uint8_t g = decompressed_rgba8[block_pixel_index * 4 + 1];
              uint8_t b = decompressed_rgba8[block_pixel_index * 4 + 2];
              uint8_t a = decompressed_rgba8[block_pixel_index * 4 + 3];

              // Swap R and B if DDS uses BGRA format
              if (is_bgra) {
                std::swap(r, b);
              }

              if (bc_format_info.format == BCFormat::BC5) {
                // BC5 normal maps: R=X, G=Y in [0,1] range where 0.5 = neutral
                // Convert to [-1,1] tangent space
                float nx = (r / 255.0f) * 2.0f - 1.0f;
                float ny = (g / 255.0f) * 2.0f - 1.0f;
                float nz = 1.0f;

                // Normalize the vector to ensure unit length
                float length = sqrtf(nx * nx + ny * ny + nz * nz);
                if (length > 0.0f) {
                  nx /= length;
                  ny /= length;
                  nz /= length;
                }

                // Store in [0,1] range for evaluate_normal()
                float fx = nx * 0.5f + 0.5f;
                float fy = ny * 0.5f + 0.5f;
                float fz = nz * 0.5f + 0.5f;
                float fw = 1.0f;

                if (out_is_srgb) {
                  output_byte[pixel_index] = {static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, fx * 255.0f))),
                    static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, fy * 255.0f))), static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, fz * 255.0f))), 255};
                } else {
                  output_float[pixel_index] = {fx, fy, fz, fw};
                }
              } else {
                // Regular color data
                if (out_is_srgb) {
                  output_byte[pixel_index] = {r, g, b, a};
                } else {
                  output_float[pixel_index] = {r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f};
                }
              }
            }
          }
        }
      }
    }
  }

  return true;
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
      if (fread(&r, sizeof(float), 1, in_file) != 1 || fread(&g, sizeof(float), 1, in_file) != 1 || fread(&b, sizeof(float), 1, in_file) != 1) {
        log::error("Failed to read PFM pixel data from file: %s", path);
        fclose(in_file);
        return false;
      }

      // PFM stores pixels bottom-to-top, but we want top-to-bottom
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

  if (strcmp(ext, ".dds") == 0 || strcmp(ext, ".DDS") == 0) {
    bool is_srgb = false;
    if (load_dds(source, data, dimensions, is_srgb)) {
      // If DDS is sRGB, return as RGBA8 so it goes through sRGB conversion
      // Otherwise return as RGBA32F (assuming it's already linear)
      return is_srgb ? Image::Format::RGBA8 : Image::Format::RGBA32F;
    }
    return Image::Format::Undefined;
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

    // Process infinite/NaN values (this would need access to scheduler, but for now inline)
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

  if (strcmp(ext, ".tga") == 0 || strcmp(ext, ".TGA") == 0) {
    int w = 0;
    int h = 0;
    int c = 0;
    stbi_set_flip_vertically_on_load(false);        // TGA files can have different orientations
    auto image = stbi_load(source, &w, &h, &c, 4);  // Force 4 channels (RGBA)
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
    // Don't log errors for placeholder image names (e.g., "image-0" or paths containing "/image-")
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
