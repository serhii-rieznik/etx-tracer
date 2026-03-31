#pragma once

#include "options.hxx"

#include <vector>

namespace etx {

struct ImageOutputParameters {
  SaveImageMode mode = SaveImageMode::RGB;
  float exposure = 1.0f;
};

struct ImageComparisonResult {
  float similarity = 0.0f;
  float low_frequency_similarity = 0.0f;
  float mean_absolute_error = 0.0f;
  float root_mean_squared_error = 0.0f;
  float low_frequency_root_mean_squared_error = 0.0f;
  float relative_root_mean_squared_error = 0.0f;
  float max_absolute_error = 0.0f;
  float mean_signed_error = 0.0f;
  float percentile_95_absolute_error = 0.0f;
  float percentile_99_absolute_error = 0.0f;
  float reference_mean_luminance = 0.0f;
  float result_mean_luminance = 0.0f;
  float brightness_ratio = 0.0f;
  float brightness_relative_error = 0.0f;

  float linear_mean_absolute_error = 0.0f;
  float linear_root_mean_squared_error = 0.0f;
  float linear_relative_root_mean_squared_error = 0.0f;
  float linear_max_absolute_error = 0.0f;
  float linear_mean_signed_error = 0.0f;
  float linear_percentile_95_absolute_error = 0.0f;
  float linear_percentile_99_absolute_error = 0.0f;
  float linear_reference_mean_luminance = 0.0f;
  float linear_result_mean_luminance = 0.0f;
  float linear_brightness_ratio = 0.0f;
  float linear_brightness_relative_error = 0.0f;
};

bool save_image_to_file(const std::string& file_name, const float4* output, const uint2& image_size, const ImageOutputParameters& params);
SaveImageMode save_image_mode_from_file_name(const std::string& file_name);
bool load_hdr_image_from_file(const std::string& file_name, std::vector<float4>& output, uint2& image_size);
bool compare_images(const float4* reference, const float4* result, const uint2& image_size, std::vector<float4>& difference_image, ImageComparisonResult& comparison);
std::string comparison_file_name_from_output(const std::string& output_file_name);

}  // namespace etx
