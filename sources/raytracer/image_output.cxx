#include "image_output.hxx"

#include <etx/core/environment.hxx>
#include <etx/core/log.hxx>

#include <algorithm>
#include <tinyexr.hxx>
#include <stb_image.hxx>
#include <stb_image_write.hxx>

#include <cmath>
#include <cstring>
#include <vector>

namespace etx {

namespace {

float compare_space_value(float value) {
  const float clamped = fmaxf(value, 0.0f);
  return clamped / (1.0f + clamped);
}

float3 compare_space_rgb(const float4& value) {
  return {compare_space_value(value.x), compare_space_value(value.y), compare_space_value(value.z)};
}

float luminance_from_rgb(const float3& value) {
  return dot(value, float3{0.2126f, 0.7152f, 0.0722f});
}

uint32_t clamp_image_coordinate(int32_t value, uint32_t limit) {
  if (value < 0) {
    return 0u;
  }

  const int32_t max_value = static_cast<int32_t>(limit) - 1;
  if (value > max_value) {
    return limit - 1u;
  }

  return static_cast<uint32_t>(value);
}

void gaussian_blur_compare_space_image(const std::vector<float3>& input, const uint2& image_size, std::vector<float3>& output) {
  static constexpr float kernel[5] = {1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f};

  const size_t pixel_count = static_cast<size_t>(image_size.x) * static_cast<size_t>(image_size.y);
  std::vector<float3> temp(pixel_count);
  output.resize(pixel_count);

  for (uint32_t y = 0u; y < image_size.y; ++y) {
    for (uint32_t x = 0u; x < image_size.x; ++x) {
      float3 value = {0.0f, 0.0f, 0.0f};
      for (int32_t dx = -2; dx <= 2; ++dx) {
        const uint32_t sx = clamp_image_coordinate(static_cast<int32_t>(x) + dx, image_size.x);
        const float weight = kernel[dx + 2];
        const float3 sample = input[static_cast<size_t>(y) * static_cast<size_t>(image_size.x) + sx];
        value += sample * weight;
      }
      temp[static_cast<size_t>(y) * static_cast<size_t>(image_size.x) + x] = value;
    }
  }

  for (uint32_t y = 0u; y < image_size.y; ++y) {
    for (uint32_t x = 0u; x < image_size.x; ++x) {
      float3 value = {0.0f, 0.0f, 0.0f};
      for (int32_t dy = -2; dy <= 2; ++dy) {
        const uint32_t sy = clamp_image_coordinate(static_cast<int32_t>(y) + dy, image_size.y);
        const float weight = kernel[dy + 2];
        const float3 sample = temp[static_cast<size_t>(sy) * static_cast<size_t>(image_size.x) + x];
        value += sample * weight;
      }
      output[static_cast<size_t>(y) * static_cast<size_t>(image_size.x) + x] = value;
    }
  }
}

float3 heatmap_color(float t) {
  const float clamped_t = saturate(t);

  if (clamped_t <= 0.2f) {
    const float local_t = clamped_t / 0.2f;
    return lerp(float3{0.0f, 0.0f, 0.0f}, float3{0.0f, 0.0f, 1.0f}, local_t);
  }

  if (clamped_t <= 0.4f) {
    const float local_t = (clamped_t - 0.2f) / 0.2f;
    return lerp(float3{0.0f, 0.0f, 1.0f}, float3{0.0f, 1.0f, 1.0f}, local_t);
  }

  if (clamped_t <= 0.6f) {
    const float local_t = (clamped_t - 0.4f) / 0.2f;
    return lerp(float3{0.0f, 1.0f, 1.0f}, float3{0.0f, 1.0f, 0.0f}, local_t);
  }

  if (clamped_t <= 0.8f) {
    const float local_t = (clamped_t - 0.6f) / 0.2f;
    return lerp(float3{0.0f, 1.0f, 0.0f}, float3{1.0f, 1.0f, 0.0f}, local_t);
  }

  if (clamped_t <= 0.95f) {
    const float local_t = (clamped_t - 0.8f) / 0.15f;
    return lerp(float3{1.0f, 1.0f, 0.0f}, float3{1.0f, 0.0f, 0.0f}, local_t);
  }

  const float local_t = (clamped_t - 0.95f) / 0.05f;
  return lerp(float3{1.0f, 0.0f, 0.0f}, float3{1.0f, 1.0f, 1.0f}, local_t);
}

float percentile_from_sorted_values(const std::vector<float>& sorted_values, float percentile) {
  if (sorted_values.empty()) {
    return 0.0f;
  }

  const float clamped_percentile = saturate(percentile);
  const size_t max_index = sorted_values.size() - 1u;
  const float scaled_index = clamped_percentile * static_cast<float>(max_index);
  const size_t index_0 = static_cast<size_t>(scaled_index);
  const size_t index_1 = min(index_0 + 1u, max_index);
  const float t = scaled_index - static_cast<float>(index_0);
  return lerp(sorted_values[index_0], sorted_values[index_1], t);
}

}  // namespace

SaveImageMode save_image_mode_from_file_name(const std::string& file_name) {
  const char* extension = get_file_ext(file_name.c_str());
  if ((extension != nullptr) && ((strcmp(extension, ".png") == 0) || (strcmp(extension, ".PNG") == 0))) {
    return SaveImageMode::TonemappedLDR;
  }

  return SaveImageMode::RGB;
}

bool save_image_to_file(const std::string& file_name, const float4* output, const uint2& image_size, const ImageOutputParameters& params) {
  if ((output == nullptr) || (image_size.x == 0u) || (image_size.y == 0u)) {
    log::error("Invalid image data for saving");
    return false;
  }

  std::string target_file_name = file_name;
  if (params.mode == SaveImageMode::TonemappedLDR) {
    if (std::strlen(get_file_ext(target_file_name.c_str())) == 0u) {
      target_file_name += ".png";
    }

    std::vector<ubyte4> tonemapped(static_cast<size_t>(image_size.x) * static_cast<size_t>(image_size.y));
    for (uint32_t i = 0u, e = image_size.x * image_size.y; i < e; ++i) {
      float3 tm = {
        1.0f - expf(-params.exposure * output[i].x),
        1.0f - expf(-params.exposure * output[i].y),
        1.0f - expf(-params.exposure * output[i].z),
      };
      float3 gamma = linear_to_gamma(tm);
      tonemapped[i].x = static_cast<uint8_t>(255.0f * saturate(gamma.x));
      tonemapped[i].y = static_cast<uint8_t>(255.0f * saturate(gamma.y));
      tonemapped[i].z = static_cast<uint8_t>(255.0f * saturate(gamma.z));
      tonemapped[i].w = 255u;
    }

    if (stbi_write_png(target_file_name.c_str(), image_size.x, image_size.y, 4, tonemapped.data(), 0) != 1) {
      log::error("Failed to save PNG image to %s", target_file_name.c_str());
      return false;
    }
  } else {
    if (std::strlen(get_file_ext(target_file_name.c_str())) == 0u) {
      target_file_name += ".exr";
    }

    const char* error = nullptr;
    if (SaveEXR(reinterpret_cast<const float*>(output), image_size.x, image_size.y, 4, false, target_file_name.c_str(), &error) != TINYEXR_SUCCESS) {
      log::error("Failed to save EXR image to %s: %s", target_file_name.c_str(), (error != nullptr) ? error : "unknown error");
      return false;
    }
  }

  log::info("Saved image to %s", target_file_name.c_str());
  return true;
}

bool load_hdr_image_from_file(const std::string& file_name, std::vector<float4>& output, uint2& image_size) {
  output.clear();
  image_size = {};

  const char* extension = get_file_ext(file_name.c_str());
  if ((extension == nullptr) || (extension[0] == 0)) {
    log::error("Unsupported image file: %s", file_name.c_str());
    return false;
  }

  if ((strcmp(extension, ".exr") == 0) || (strcmp(extension, ".EXR") == 0)) {
    int width = 0;
    int height = 0;
    float* rgba_data = nullptr;
    const char* error = nullptr;
    if (LoadEXR(&rgba_data, &width, &height, file_name.c_str(), &error) != TINYEXR_SUCCESS) {
      log::error("Failed to load EXR image from %s: %s", file_name.c_str(), (error != nullptr) ? error : "unknown error");
      if (error != nullptr) {
        FreeEXRErrorMessage(error);
      }
      return false;
    }

    image_size = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    output.resize(static_cast<size_t>(image_size.x) * static_cast<size_t>(image_size.y));
    memcpy(output.data(), rgba_data, output.size() * sizeof(float4));
    free(rgba_data);
    return true;
  }

  if ((strcmp(extension, ".hdr") == 0) || (strcmp(extension, ".HDR") == 0)) {
    int width = 0;
    int height = 0;
    int components = 0;
    stbi_set_flip_vertically_on_load(false);
    float* image = stbi_loadf(file_name.c_str(), &width, &height, &components, 0);
    if (image == nullptr) {
      log::error("Failed to load HDR image from %s", file_name.c_str());
      return false;
    }

    image_size = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    output.resize(static_cast<size_t>(image_size.x) * static_cast<size_t>(image_size.y));
    for (size_t i = 0, e = output.size(); i < e; ++i) {
      float4 pixel = {0.0f, 0.0f, 0.0f, 1.0f};
      pixel.x = image[i * static_cast<size_t>(components) + 0u];
      pixel.y = image[i * static_cast<size_t>(components) + ((components > 1) ? 1u : 0u)];
      pixel.z = image[i * static_cast<size_t>(components) + ((components > 2) ? 2u : 0u)];
      pixel.w = (components > 3) ? image[i * static_cast<size_t>(components) + 3u] : 1.0f;
      output[i] = pixel;
    }
    stbi_image_free(image);
    return true;
  }

  log::error("Unsupported HDR comparison image format: %s", file_name.c_str());
  return false;
}

bool compare_images(const float4* reference, const float4* result, const uint2& image_size, std::vector<float4>& difference_image, ImageComparisonResult& comparison) {
  if ((reference == nullptr) || (result == nullptr) || (image_size.x == 0u) || (image_size.y == 0u)) {
    log::error("Invalid image data for comparison");
    return false;
  }

  difference_image.resize(static_cast<size_t>(image_size.x) * static_cast<size_t>(image_size.y));
  comparison = {};

  double absolute_error_sum = 0.0;
  double squared_error_sum = 0.0;
  double signed_error_sum = 0.0;
  double reference_energy_sum = 0.0;
  double reference_luminance_sum = 0.0;
  double result_luminance_sum = 0.0;
  float max_absolute_error = 0.0f;

  double linear_absolute_error_sum = 0.0;
  double linear_squared_error_sum = 0.0;
  double linear_signed_error_sum = 0.0;
  double linear_reference_energy_sum = 0.0;
  double linear_reference_luminance_sum = 0.0;
  double linear_result_luminance_sum = 0.0;
  float linear_max_absolute_error = 0.0f;
  const double channel_count = 3.0 * static_cast<double>(image_size.x) * static_cast<double>(image_size.y);
  const double pixel_count = static_cast<double>(image_size.x) * static_cast<double>(image_size.y);
  std::vector<float3> reference_compare_space = {};
  std::vector<float3> result_compare_space = {};
  std::vector<float> absolute_error_distribution = {};
  std::vector<float> linear_absolute_error_distribution = {};
  reference_compare_space.reserve(static_cast<size_t>(image_size.x) * static_cast<size_t>(image_size.y));
  result_compare_space.reserve(static_cast<size_t>(image_size.x) * static_cast<size_t>(image_size.y));
  absolute_error_distribution.reserve(static_cast<size_t>(image_size.x) * static_cast<size_t>(image_size.y));
  linear_absolute_error_distribution.reserve(static_cast<size_t>(image_size.x) * static_cast<size_t>(image_size.y));

  for (uint32_t i = 0u, e = image_size.x * image_size.y; i < e; ++i) {
    const float3 reference_linear_rgb = {reference[i].x, reference[i].y, reference[i].z};
    const float3 result_linear_rgb = {result[i].x, result[i].y, result[i].z};
    const float3 reference_rgb = compare_space_rgb(reference[i]);
    const float3 result_rgb = compare_space_rgb(result[i]);
    const float3 diff_rgb = {
      fabsf(result_rgb.x - reference_rgb.x),
      fabsf(result_rgb.y - reference_rgb.y),
      fabsf(result_rgb.z - reference_rgb.z),
    };
    const float3 signed_diff_rgb = {
      result_rgb.x - reference_rgb.x,
      result_rgb.y - reference_rgb.y,
      result_rgb.z - reference_rgb.z,
    };
    const float3 linear_diff_rgb = {
      fabsf(result_linear_rgb.x - reference_linear_rgb.x),
      fabsf(result_linear_rgb.y - reference_linear_rgb.y),
      fabsf(result_linear_rgb.z - reference_linear_rgb.z),
    };
    const float3 linear_signed_diff_rgb = {
      result_linear_rgb.x - reference_linear_rgb.x,
      result_linear_rgb.y - reference_linear_rgb.y,
      result_linear_rgb.z - reference_linear_rgb.z,
    };
    const float diff_value = fmaxf(diff_rgb.x, fmaxf(diff_rgb.y, diff_rgb.z));
    const float linear_diff_value = fmaxf(linear_diff_rgb.x, fmaxf(linear_diff_rgb.y, linear_diff_rgb.z));
    const float3 diff_heatmap = heatmap_color(diff_value);

    reference_compare_space.push_back(reference_rgb);
    result_compare_space.push_back(result_rgb);
    difference_image[i] = {diff_heatmap.x, diff_heatmap.y, diff_heatmap.z, 1.0f};

    absolute_error_sum += diff_rgb.x + diff_rgb.y + diff_rgb.z;
    squared_error_sum += (diff_rgb.x * diff_rgb.x) + (diff_rgb.y * diff_rgb.y) + (diff_rgb.z * diff_rgb.z);
    signed_error_sum += signed_diff_rgb.x + signed_diff_rgb.y + signed_diff_rgb.z;
    reference_energy_sum += (reference_rgb.x * reference_rgb.x) + (reference_rgb.y * reference_rgb.y) + (reference_rgb.z * reference_rgb.z);
    reference_luminance_sum += luminance_from_rgb(reference_rgb);
    result_luminance_sum += luminance_from_rgb(result_rgb);
    max_absolute_error = fmaxf(max_absolute_error, fmaxf(diff_rgb.x, fmaxf(diff_rgb.y, diff_rgb.z)));
    absolute_error_distribution.push_back(diff_value);

    linear_absolute_error_sum += linear_diff_rgb.x + linear_diff_rgb.y + linear_diff_rgb.z;
    linear_squared_error_sum += (linear_diff_rgb.x * linear_diff_rgb.x) + (linear_diff_rgb.y * linear_diff_rgb.y) + (linear_diff_rgb.z * linear_diff_rgb.z);
    linear_signed_error_sum += linear_signed_diff_rgb.x + linear_signed_diff_rgb.y + linear_signed_diff_rgb.z;
    linear_reference_energy_sum +=
      (reference_linear_rgb.x * reference_linear_rgb.x) + (reference_linear_rgb.y * reference_linear_rgb.y) + (reference_linear_rgb.z * reference_linear_rgb.z);
    linear_reference_luminance_sum += luminance_from_rgb(reference_linear_rgb);
    linear_result_luminance_sum += luminance_from_rgb(result_linear_rgb);
    linear_max_absolute_error = fmaxf(linear_max_absolute_error, linear_diff_value);
    linear_absolute_error_distribution.push_back(linear_diff_value);
  }

  std::sort(absolute_error_distribution.begin(), absolute_error_distribution.end());
  std::sort(linear_absolute_error_distribution.begin(), linear_absolute_error_distribution.end());

  comparison.mean_absolute_error = static_cast<float>(absolute_error_sum / channel_count);
  comparison.root_mean_squared_error = static_cast<float>(sqrt(squared_error_sum / channel_count));
  comparison.relative_root_mean_squared_error = static_cast<float>(sqrt(squared_error_sum / fmax(reference_energy_sum, 1.0e-12)));
  comparison.max_absolute_error = max_absolute_error;
  comparison.mean_signed_error = static_cast<float>(signed_error_sum / channel_count);
  comparison.percentile_95_absolute_error = percentile_from_sorted_values(absolute_error_distribution, 0.95f);
  comparison.percentile_99_absolute_error = percentile_from_sorted_values(absolute_error_distribution, 0.99f);
  comparison.reference_mean_luminance = static_cast<float>(reference_luminance_sum / pixel_count);
  comparison.result_mean_luminance = static_cast<float>(result_luminance_sum / pixel_count);
  comparison.brightness_ratio = comparison.result_mean_luminance / fmaxf(comparison.reference_mean_luminance, 1.0e-12f);
  comparison.brightness_relative_error = (comparison.result_mean_luminance - comparison.reference_mean_luminance) / fmaxf(comparison.reference_mean_luminance, 1.0e-12f);
  comparison.similarity = 100.0f * (1.0f - saturate(comparison.root_mean_squared_error));

  std::vector<float3> low_frequency_reference = {};
  std::vector<float3> low_frequency_result = {};
  gaussian_blur_compare_space_image(reference_compare_space, image_size, low_frequency_reference);
  gaussian_blur_compare_space_image(result_compare_space, image_size, low_frequency_result);

  double low_frequency_squared_error_sum = 0.0;
  for (uint32_t i = 0u, e = image_size.x * image_size.y; i < e; ++i) {
    const float3 diff_rgb = {
      low_frequency_result[i].x - low_frequency_reference[i].x,
      low_frequency_result[i].y - low_frequency_reference[i].y,
      low_frequency_result[i].z - low_frequency_reference[i].z,
    };
    low_frequency_squared_error_sum += (diff_rgb.x * diff_rgb.x) + (diff_rgb.y * diff_rgb.y) + (diff_rgb.z * diff_rgb.z);
  }

  comparison.low_frequency_root_mean_squared_error = static_cast<float>(sqrt(low_frequency_squared_error_sum / channel_count));
  comparison.low_frequency_similarity = 100.0f * (1.0f - saturate(comparison.low_frequency_root_mean_squared_error));

  comparison.linear_mean_absolute_error = static_cast<float>(linear_absolute_error_sum / channel_count);
  comparison.linear_root_mean_squared_error = static_cast<float>(sqrt(linear_squared_error_sum / channel_count));
  comparison.linear_relative_root_mean_squared_error = static_cast<float>(sqrt(linear_squared_error_sum / fmax(linear_reference_energy_sum, 1.0e-12)));
  comparison.linear_max_absolute_error = linear_max_absolute_error;
  comparison.linear_mean_signed_error = static_cast<float>(linear_signed_error_sum / channel_count);
  comparison.linear_percentile_95_absolute_error = percentile_from_sorted_values(linear_absolute_error_distribution, 0.95f);
  comparison.linear_percentile_99_absolute_error = percentile_from_sorted_values(linear_absolute_error_distribution, 0.99f);
  comparison.linear_reference_mean_luminance = static_cast<float>(linear_reference_luminance_sum / pixel_count);
  comparison.linear_result_mean_luminance = static_cast<float>(linear_result_luminance_sum / pixel_count);
  comparison.linear_brightness_ratio = comparison.linear_result_mean_luminance / fmaxf(comparison.linear_reference_mean_luminance, 1.0e-12f);
  comparison.linear_brightness_relative_error =
    (comparison.linear_result_mean_luminance - comparison.linear_reference_mean_luminance) / fmaxf(comparison.linear_reference_mean_luminance, 1.0e-12f);

  return true;
}

std::string comparison_file_name_from_output(const std::string& output_file_name) {
  return output_file_name + ".cmp.exr";
}

}  // namespace etx
