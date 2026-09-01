#pragma once

#include <etx/render/shared/image.hxx>
#include <etx/render/shared/medium.hxx>

#include <cstring>

namespace etx {

struct PackedPayloadLocation {
  uint32_t chunk_index = kInvalidIndex;
  uint32_t offset = kInvalidIndex;
};

ETX_SHARED_INLINE uint32_t gpu_image_pixel_stride(const Image& image) {
  if (image.format == Image::Format::RGBA32F) {
    return sizeof(float4);
  }
  if (image.format == Image::Format::RGBA8) {
    return sizeof(ubyte4);
  }
  if (image.format == Image::Format::R32F) {
    return sizeof(float);
  }
  if (Image::is_compressed_bc_format(image.format)) {
    return Image::get_bc_block_size(image.format);
  }
  return 0u;
}

ETX_SHARED_INLINE ::Image make_gpu_image_descriptor(const Image& image, PackedPayloadLocation pixel_payload = {}, PackedPayloadLocation x_distribution_payload = {},
  PackedPayloadLocation y_distribution_payload = {}) {
  ::Image result;
  std::memset(&result, 0, sizeof(result));
  result.fsize = image.fsize;
  result.offset = image.offset;
  result.scale = image.scale;
  result.normalization = image.normalization;
  result.isize = image.isize;
  result.options = image.options;
  result.format = image.format;
  result.data_size = image.data_size;
  result.pixel_data_offset = pixel_payload.offset;
  result.x_distribution_entries_offset = x_distribution_payload.offset;
  result.y_distribution_entries_offset = y_distribution_payload.offset;
  result.x_entries_stride = (image.x_distributions_storage.valid() && (image.isize.x > 0u)) ? (image.isize.x + 1u) : 0u;
  result.x_distribution_count = image.x_distributions_storage.valid() ? image.isize.y : 0u;
  result.y_entries_count = image.y_distribution_storage.valid() ? (image.isize.y + 1u) : 0u;
  result.y_distribution_total_weight = image.y_distribution.total_weight;
  result.pixel_data_stride = gpu_image_pixel_stride(image);
  result.pixel_data_chunk_index = pixel_payload.chunk_index;
  result.x_distribution_chunk_index = x_distribution_payload.chunk_index;
  result.y_distribution_chunk_index = y_distribution_payload.chunk_index;
  return result;
}

ETX_SHARED_INLINE ::Medium make_gpu_medium_descriptor(const Medium& medium, PackedPayloadLocation density_payload = {}) {
  ::Medium result;
  std::memset(&result, 0, sizeof(result));
  result.grid.dimensions = medium.grid.dimensions;
  result.grid.type = medium.grid.type;
  result.grid.noise_type = medium.grid.noise_type;
  result.grid.density_data_offset = density_payload.offset;
  result.grid.density_count = (medium.grid.density_image_index != kInvalidIndex) ? 1u : static_cast<uint32_t>(medium.density_view.count);
  result.grid.noise_seed = medium.grid.noise_seed;
  result.grid.noise_offset = medium.grid.noise_offset;
  result.grid.noise_enable_border_fade = medium.grid.noise_enable_border_fade;
  result.grid.noise_octaves = medium.grid.noise_octaves;
  result.grid.noise_scale = medium.grid.noise_scale;
  result.grid.noise_lacunarity = medium.grid.noise_lacunarity;
  result.grid.noise_persistence = medium.grid.noise_persistence;
  result.grid.noise_power = medium.grid.noise_power;
  result.grid.noise_sharpness = medium.grid.noise_sharpness;
  result.grid.noise_border_fade_distance = medium.grid.noise_border_fade_distance;
  result.grid.density_data_chunk_index = density_payload.chunk_index;
  result.grid.density_image_index = medium.grid.density_image_index;
  result.bounds = medium.bounds;
  result.absorption_index = medium.absorption_index;
  result.scattering_index = medium.scattering_index;
  result.phase_function_g = medium.phase_function_g;
  result.enable_explicit_connections = medium.enable_explicit_connections;
  result.cls = medium.cls;
  result.world_to_object = medium.world_to_object;
  result.local_bounds = medium.local_bounds;
  return result;
}

}  // namespace etx
