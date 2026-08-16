#include <etx/render/host/exr.hxx>

#include <exr.h>

#include <cstdlib>
#include <cstring>
#include <limits>

namespace etx {

namespace {

int find_channel(const exr_part& part, const char* name) {
  for (int i = 0; i < part.header.num_channels; ++i) {
    if (strcmp(part.header.channels[i].name, name) == 0) {
      return i;
    }
  }
  return -1;
}

float read_channel(const exr_part& part, int channel, size_t pixel_index) {
  const void* source = part.images[channel];
  switch (part.header.channels[channel].pixel_type) {
    case EXR_PIXEL_HALF: {
      float value = 0.0f;
      exr_half_to_float(static_cast<const uint16_t*>(source) + pixel_index, &value, 1);
      return value;
    }
    case EXR_PIXEL_FLOAT:
      return static_cast<const float*>(source)[pixel_index];
    case EXR_PIXEL_UINT:
      return static_cast<float>(static_cast<const uint32_t*>(source)[pixel_index]);
  }
  return 0.0f;
}

void set_error(std::string* error, exr_result result) {
  if (error != nullptr) {
    *error = exr_result_string(result);
  }
}

}  // namespace

bool load_exr_image(const char* path, std::vector<float4>& pixels, uint2& dimensions, std::string* error) {
  pixels.clear();
  dimensions = {};
  if (error != nullptr) {
    error->clear();
  }
  if (path == nullptr) {
    set_error(error, EXR_ERROR_INVALID_ARGUMENT);
    return false;
  }

  exr_image image = {};
  const exr_result load_result = exr_load_from_file(path, nullptr, &image);
  if (!EXR_OK(load_result)) {
    set_error(error, load_result);
    return false;
  }

  if ((image.num_parts < 1) || (image.parts == nullptr)) {
    exr_image_free(&image);
    set_error(error, EXR_ERROR_INVALID_FILE);
    return false;
  }

  const exr_part& part = image.parts[0];
  if (part.is_deep || (part.images == nullptr) || (part.width <= 0) || (part.height <= 0)) {
    exr_image_free(&image);
    set_error(error, EXR_ERROR_UNSUPPORTED);
    return false;
  }

  const size_t width = static_cast<size_t>(part.width);
  const size_t height = static_cast<size_t>(part.height);
  if ((height > std::numeric_limits<size_t>::max() / width) || (width > std::numeric_limits<uint32_t>::max()) ||
      (height > std::numeric_limits<uint32_t>::max())) {
    exr_image_free(&image);
    set_error(error, EXR_ERROR_INVALID_FILE);
    return false;
  }

  if (exr_part_is_luminance_chroma(&part)) {
    float* rgba = nullptr;
    int rgba_width = 0;
    int rgba_height = 0;
    const exr_result convert_result = exr_part_yc_to_rgba_float(nullptr, &part, &rgba, &rgba_width, &rgba_height);
    if (EXR_OK(convert_result) && (rgba != nullptr) && (rgba_width > 0) && (rgba_height > 0)) {
      dimensions = {static_cast<uint32_t>(rgba_width), static_cast<uint32_t>(rgba_height)};
      pixels.resize(static_cast<size_t>(rgba_width) * static_cast<size_t>(rgba_height));
      memcpy(pixels.data(), rgba, pixels.size() * sizeof(float4));
      free(rgba);
      exr_image_free(&image);
      return true;
    }
    free(rgba);
  }

  const int channels[4] = {
    find_channel(part, "R"),
    find_channel(part, "G"),
    find_channel(part, "B"),
    find_channel(part, "A"),
  };

  const size_t pixel_count = width * height;
  if (part.header.num_channels == 1) {
    pixels.resize(pixel_count);
    for (size_t pixel_index = 0; pixel_index < pixel_count; ++pixel_index) {
      const float value = read_channel(part, 0, pixel_index);
      pixels[pixel_index] = {value, value, value, value};
    }
    dimensions = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    exr_image_free(&image);
    return true;
  }

  if ((channels[0] < 0) || (channels[1] < 0) || (channels[2] < 0)) {
    exr_image_free(&image);
    set_error(error, EXR_ERROR_INVALID_FILE);
    return false;
  }

  pixels.resize(pixel_count, float4{0.0f, 0.0f, 0.0f, 1.0f});
  for (size_t pixel_index = 0; pixel_index < pixel_count; ++pixel_index) {
    float* rgba = &pixels[pixel_index].x;
    for (int component = 0; component < 4; ++component) {
      if (channels[component] >= 0) {
        rgba[component] = read_channel(part, channels[component], pixel_index);
      }
    }
  }

  dimensions = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
  exr_image_free(&image);
  return true;
}

bool save_exr_image(const char* path, const float4* pixels, uint2 dimensions, std::string* error) {
  if (error != nullptr) {
    error->clear();
  }
  if ((path == nullptr) || (pixels == nullptr) || (dimensions.x == 0u) || (dimensions.y == 0u) ||
      (dimensions.x > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) ||
      (dimensions.y > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))) {
    set_error(error, EXR_ERROR_INVALID_ARGUMENT);
    return false;
  }

  const size_t pixel_count = static_cast<size_t>(dimensions.x) * static_cast<size_t>(dimensions.y);
  std::vector<float> planes[4];
  for (auto& plane : planes) {
    plane.resize(pixel_count);
  }
  for (size_t pixel_index = 0; pixel_index < pixel_count; ++pixel_index) {
    planes[0][pixel_index] = pixels[pixel_index].w;
    planes[1][pixel_index] = pixels[pixel_index].z;
    planes[2][pixel_index] = pixels[pixel_index].y;
    planes[3][pixel_index] = pixels[pixel_index].x;
  }

  exr_channel channels[4] = {};
  constexpr const char* channel_names[4] = {"A", "B", "G", "R"};
  void* images[4] = {};
  for (int i = 0; i < 4; ++i) {
    strcpy(channels[i].name, channel_names[i]);
    channels[i].pixel_type = EXR_PIXEL_FLOAT;
    channels[i].x_sampling = 1;
    channels[i].y_sampling = 1;
    images[i] = planes[i].data();
  }

  exr_part part = {};
  part.header.part_type = EXR_PART_SCANLINE;
  part.header.data_window = {0, 0, static_cast<int32_t>(dimensions.x - 1u), static_cast<int32_t>(dimensions.y - 1u)};
  part.header.display_window = part.header.data_window;
  part.header.pixel_aspect_ratio = 1.0f;
  part.header.screen_window_width = 1.0f;
  part.header.num_channels = 4;
  part.header.channels = channels;
  part.width = static_cast<int32_t>(dimensions.x);
  part.height = static_cast<int32_t>(dimensions.y);
  part.images = images;

  exr_image image = {};
  image.num_parts = 1;
  image.parts = &part;

  const exr_compression compression = ((dimensions.x < 16u) && (dimensions.y < 16u)) ? EXR_COMPRESSION_NONE : EXR_COMPRESSION_ZIP;
  const exr_result save_result = exr_save_to_file(path, &image, compression);
  if (!EXR_OK(save_result)) {
    set_error(error, save_result);
    return false;
  }
  return true;
}

}  // namespace etx
