#include "image_output.hxx"

#include <etx/core/environment.hxx>
#include <etx/core/log.hxx>

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
  double reference_energy_sum = 0.0;
  float max_absolute_error = 0.0f;
  const double channel_count = 3.0 * static_cast<double>(image_size.x) * static_cast<double>(image_size.y);

  for (uint32_t i = 0u, e = image_size.x * image_size.y; i < e; ++i) {
    const float3 reference_rgb = {compare_space_value(reference[i].x), compare_space_value(reference[i].y), compare_space_value(reference[i].z)};
    const float3 result_rgb = {compare_space_value(result[i].x), compare_space_value(result[i].y), compare_space_value(result[i].z)};
    const float3 diff_rgb = {
      fabsf(result_rgb.x - reference_rgb.x),
      fabsf(result_rgb.y - reference_rgb.y),
      fabsf(result_rgb.z - reference_rgb.z),
    };
    const float diff_value = fmaxf(diff_rgb.x, fmaxf(diff_rgb.y, diff_rgb.z));
    const float3 diff_heatmap = heatmap_color(diff_value);

    difference_image[i] = {diff_heatmap.x, diff_heatmap.y, diff_heatmap.z, 1.0f};

    absolute_error_sum += diff_rgb.x + diff_rgb.y + diff_rgb.z;
    squared_error_sum += (diff_rgb.x * diff_rgb.x) + (diff_rgb.y * diff_rgb.y) + (diff_rgb.z * diff_rgb.z);
    reference_energy_sum += (reference_rgb.x * reference_rgb.x) + (reference_rgb.y * reference_rgb.y) + (reference_rgb.z * reference_rgb.z);
    max_absolute_error = fmaxf(max_absolute_error, fmaxf(diff_rgb.x, fmaxf(diff_rgb.y, diff_rgb.z)));
  }

  comparison.mean_absolute_error = static_cast<float>(absolute_error_sum / channel_count);
  comparison.root_mean_squared_error = static_cast<float>(sqrt(squared_error_sum / channel_count));
  comparison.relative_root_mean_squared_error = static_cast<float>(sqrt(squared_error_sum / fmax(reference_energy_sum, 1.0e-12)));
  comparison.max_absolute_error = max_absolute_error;
  comparison.similarity = 100.0f * (1.0f - saturate(comparison.root_mean_squared_error));

  return true;
}

std::string comparison_file_name_from_output(const std::string& output_file_name) {
  return output_file_name + ".cmp.exr";
}

}  // namespace etx
