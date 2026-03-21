#include "batch_mode.hxx"

#include <etx/core/environment.hxx>
#include <etx/core/log.hxx>
#include <etx/render/host/film.hxx>
#include <etx/render/host/scene_global.hxx>
#include <etx/render/host/scene_representation.hxx>
#include <etx/render/shared/ior_database.hxx>
#include <etx/rt/integrators/integrator.hxx>
#include <etx/rt/rt.hxx>

#include "cpu_renderer.hxx"
#include "gpu_renderer.hxx"
#include "headless_render_context.hxx"
#include "image_output.hxx"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <thread>

namespace etx {

namespace {

bool parse_u32_argument(const char* value, uint32_t& result) {
  if ((value == nullptr) || (value[0] == 0)) {
    return false;
  }

  char* end_ptr = nullptr;
  unsigned long parsed_value = std::strtoul(value, &end_ptr, 10);
  if ((end_ptr == nullptr) || (end_ptr[0] != 0)) {
    return false;
  }
  if (parsed_value > 0xFFFFFFFFul) {
    return false;
  }

  result = static_cast<uint32_t>(parsed_value);
  return true;
}

bool parse_f32_argument(const char* value, float& result) {
  if ((value == nullptr) || (value[0] == 0)) {
    return false;
  }

  char* end_ptr = nullptr;
  result = std::strtof(value, &end_ptr);
  if ((end_ptr == nullptr) || (end_ptr[0] != 0)) {
    return false;
  }

  return true;
}

bool parse_u32_pair_argument(const char* value, char separator, uint32_t& first, uint32_t& second) {
  if ((value == nullptr) || (value[0] == 0)) {
    return false;
  }

  const char* separator_ptr = std::strchr(value, separator);
  if ((separator_ptr == nullptr) && (separator == 'x')) {
    separator_ptr = std::strchr(value, 'X');
  }
  if ((separator_ptr == nullptr) || (separator_ptr == value) || (separator_ptr[1] == 0)) {
    return false;
  }

  const std::string first_value(value, static_cast<size_t>(separator_ptr - value));
  const std::string second_value(separator_ptr + 1);
  if ((parse_u32_argument(first_value.c_str(), first) == false) || (parse_u32_argument(second_value.c_str(), second) == false)) {
    return false;
  }

  return true;
}

bool parse_resolution_argument(const char* value, uint32_t& width, uint32_t& height) {
  if (parse_u32_pair_argument(value, 'x', width, height) == false) {
    return false;
  }

  return (width > 0u) && (height > 0u);
}

bool parse_crop_argument(const char* value, uint32_t& x, uint32_t& y, uint32_t& width, uint32_t& height) {
  if ((value == nullptr) || (value[0] == 0)) {
    return false;
  }

  const std::string text(value);
  const size_t first_separator = text.find(',');
  if ((first_separator == std::string::npos) || (first_separator == 0u)) {
    return false;
  }

  const size_t second_separator = text.find(',', first_separator + 1u);
  if ((second_separator == std::string::npos) || (second_separator == (first_separator + 1u))) {
    return false;
  }

  const size_t third_separator = text.find(',', second_separator + 1u);
  if ((third_separator == std::string::npos) || (third_separator == (second_separator + 1u)) || ((third_separator + 1u) >= text.size())) {
    return false;
  }

  const std::string x_text = text.substr(0u, first_separator);
  const std::string y_text = text.substr(first_separator + 1u, second_separator - first_separator - 1u);
  const std::string width_text = text.substr(second_separator + 1u, third_separator - second_separator - 1u);
  const std::string height_text = text.substr(third_separator + 1u);

  if ((parse_u32_argument(x_text.c_str(), x) == false) || (parse_u32_argument(y_text.c_str(), y) == false) || (parse_u32_argument(width_text.c_str(), width) == false) ||
      (parse_u32_argument(height_text.c_str(), height) == false)) {
    return false;
  }

  return (width > 0u) && (height > 0u);
}

std::string normalize_strategy_flag_name(const std::string& value) {
  std::string result = value;
  for (char& ch : result) {
    if (ch == '-') {
      ch = '_';
    } else {
      ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
  }
  return result;
}

bool parse_strategy_flags_argument(const char* value, uint32_t& result) {
  if ((value == nullptr) || (value[0] == 0)) {
    return false;
  }

  const std::string normalized_text = normalize_strategy_flag_name(value);
  if (normalized_text == "none") {
    result = 0u;
    return true;
  }

  uint32_t flags = 0u;
  size_t token_begin = 0u;
  while (token_begin < normalized_text.size()) {
    const size_t token_end = normalized_text.find(',', token_begin);
    const std::string token =
      (token_end == std::string::npos) ? normalized_text.substr(token_begin) : normalized_text.substr(token_begin, token_end - token_begin);
    if (token.empty()) {
      return false;
    }

    if (token == "direct_hit") {
      flags |= Scene::Strategy::DirectHit;
    } else if (token == "connect_to_light") {
      flags |= Scene::Strategy::ConnectToLight;
    } else if (token == "connect_to_camera") {
      flags |= Scene::Strategy::ConnectToCamera;
    } else if (token == "connect_vertices") {
      flags |= Scene::Strategy::ConnectVertices;
    } else if (token == "merge_vertices") {
      flags |= Scene::Strategy::MergeVertices;
    } else {
      return false;
    }

    if (token_end == std::string::npos) {
      break;
    }
    token_begin = token_end + 1u;
  }

  result = flags;
  return true;
}

struct FullComparisonTechniqueInfo {
  const char* file_tag = "";
  uint32_t bdpt_mode = 0u;
  uint32_t strategy_flags = 0u;
};

constexpr uint32_t kBDPTModePathTracing = 0u;
constexpr uint32_t kBDPTModeLightTracing = 1u;
constexpr uint32_t kBDPTModeFast = 2u;

const FullComparisonTechniqueInfo kFullComparisonTechniques[] = {
  {
    .file_tag = "pt",
    .bdpt_mode = kBDPTModePathTracing,
    .strategy_flags = Scene::Strategy::DirectHit | Scene::Strategy::ConnectToLight,
  },
  {
    .file_tag = "pt-direct-hit",
    .bdpt_mode = kBDPTModePathTracing,
    .strategy_flags = Scene::Strategy::DirectHit,
  },
  {
    .file_tag = "pt-connect-to-light",
    .bdpt_mode = kBDPTModePathTracing,
    .strategy_flags = Scene::Strategy::ConnectToLight,
  },
  {
    .file_tag = "lt",
    .bdpt_mode = kBDPTModeLightTracing,
    .strategy_flags = Scene::Strategy::ConnectToCamera,
  },
  {
    .file_tag = "bdpt-fast",
    .bdpt_mode = kBDPTModeFast,
    .strategy_flags = Scene::Strategy::DirectHit | Scene::Strategy::ConnectToLight | Scene::Strategy::ConnectToCamera | Scene::Strategy::ConnectVertices,
  },
  {
    .file_tag = "bdpt-fast-direct-hit",
    .bdpt_mode = kBDPTModeFast,
    .strategy_flags = Scene::Strategy::DirectHit,
  },
  {
    .file_tag = "bdpt-fast-connect-to-light",
    .bdpt_mode = kBDPTModeFast,
    .strategy_flags = Scene::Strategy::ConnectToLight,
  },
  {
    .file_tag = "bdpt-fast-connect-to-camera",
    .bdpt_mode = kBDPTModeFast,
    .strategy_flags = Scene::Strategy::ConnectToCamera,
  },
};

const char* batch_usage_string() {
  return
    "Usage:\n"
    "  raytracer --render --scene <scene-file> --output <output-file> [options]\n"
    "  raytracer --full-comparison --scene <scene-file> [options]\n"
    "\n"
    "Options:\n"
    "  --full-comparison\n"
    "  --integrator <debug|pt|bdpt|vcm|bdpt_distilled>\n"
    "  --renderer <cpu|gpu>\n"
    "  --samples <count>\n"
    "  --max-path-length <count>\n"
    "  --random-seed <value>\n"
    "  --resolution <width>x<height>\n"
    "  --crop <x>,<y>,<width>,<height>\n"
    "  --strategy-flags <flag[,flag...]>\n"
    "  --gpu-compile-only\n"
    "  --gpu-compile-stage <entry-point>\n"
    "  --reference <reference-image>\n"
    "  --compare <render>\n"
    "  --denoise\n"
    "  --exposure <value>\n"
    "  --help\n";
}

std::string resolve_input_path(const std::string& path) {
  if (path.empty()) {
    return {};
  }

  std::filesystem::path input_path(path);
  std::error_code ec = {};
  if (input_path.is_absolute()) {
    const std::filesystem::path canonical_path = std::filesystem::weakly_canonical(input_path, ec);
    if (ec.value() == 0) {
      return canonical_path.generic_string();
    }
    const std::filesystem::path absolute_path = std::filesystem::absolute(input_path, ec);
    return absolute_path.generic_string();
  }

  if (std::filesystem::exists(input_path, ec)) {
    const std::filesystem::path canonical_path = std::filesystem::weakly_canonical(input_path, ec);
    if (ec.value() == 0) {
      return canonical_path.generic_string();
    }
    const std::filesystem::path absolute_path = std::filesystem::absolute(input_path, ec);
    return absolute_path.generic_string();
  }

  return env().resolve_to_absolute(path);
}

std::string full_comparison_output_file_name(const std::string& scene_file, const char* technique_tag, const char* renderer_tag) {
  const std::filesystem::path scene_path(resolve_input_path(scene_file));
  const std::filesystem::path output_directory = scene_path.parent_path() / (scene_path.stem().generic_string() + ".comparison");
  std::string result = output_directory.generic_string();
  result += "/";
  result += technique_tag;
  result += ".";
  result += renderer_tag;
  result += ".exr";
  return result;
}

std::string full_comparison_report_file_name(const std::string& scene_file) {
  const std::filesystem::path scene_path(resolve_input_path(scene_file));
  const std::filesystem::path output_directory = scene_path.parent_path() / (scene_path.stem().generic_string() + ".comparison");
  return (output_directory / "comparison.html").generic_string();
}

std::string full_comparison_ai_report_file_name(const std::string& scene_file) {
  const std::filesystem::path scene_path(resolve_input_path(scene_file));
  const std::filesystem::path output_directory = scene_path.parent_path() / (scene_path.stem().generic_string() + ".comparison");
  return (output_directory / "comparison.ai.json").generic_string();
}

bool ensure_parent_directory_exists(const std::string& file_name) {
  const std::filesystem::path file_path(file_name);
  const std::filesystem::path parent_path = file_path.parent_path();
  if (parent_path.empty()) {
    return true;
  }

  std::error_code ec = {};
  std::filesystem::create_directories(parent_path, ec);
  if (ec.value() != 0) {
    log::error("Failed to create output directory: %s", parent_path.generic_string().c_str());
    return false;
  }

  return true;
}

std::string png_file_name_from_output_file(const std::string& output_file) {
  std::filesystem::path path(output_file);
  path.replace_extension(".png");
  return path.generic_string();
}

std::string html_escape(const std::string& value) {
  std::string result = {};
  result.reserve(value.size());

  for (const char ch : value) {
    switch (ch) {
      case '&':
        result += "&amp;";
        break;
      case '<':
        result += "&lt;";
        break;
      case '>':
        result += "&gt;";
        break;
      case '"':
        result += "&quot;";
        break;
      case '\'':
        result += "&#39;";
        break;
      default:
        result.push_back(ch);
        break;
    }
  }

  return result;
}

std::string html_file_name_only(const std::string& file_name) {
  return html_escape(std::filesystem::path(file_name).filename().generic_string());
}

float clamp_metric_score(float value) {
  return clamp(value, 0.0f, 1.0f);
}

float inverse_error_metric_score(float value, float reference_scale) {
  const float safe_scale = max(reference_scale, 1.0e-6f);
  return clamp_metric_score(1.0f - (value / safe_scale));
}

const char* metric_tooltip_text(const char* label, bool linear_space) {
  if (strcmp(label, "Similarity") == 0) {
    return linear_space ?
             "Overall similarity score derived from linear-space RMSE. Higher is better. 100% means identical images." :
             "Overall similarity score derived from compare-space RMSE. Higher is better. 100% means identical images.";
  }
  if (strcmp(label, "RMSE") == 0) {
    return linear_space ?
             "Root mean squared error in linear HDR space. Penalizes larger errors more strongly than MAE. Lower is better." :
             "Root mean squared error in compare space. Penalizes larger errors more strongly than MAE. Lower is better.";
  }
  if (strcmp(label, "MAE") == 0) {
    return linear_space ? "Mean absolute error in linear HDR space. Average per-channel absolute difference. Lower is better."
                        : "Mean absolute error in compare space. Average per-channel absolute difference. Lower is better.";
  }
  if (strcmp(label, "Relative RMSE") == 0) {
    return linear_space ? "RMSE normalized by reference image energy in linear HDR space. Helps compare scenes with different brightness. Lower is better."
                        : "RMSE normalized by reference image energy in compare space. Helps compare scenes with different brightness. Lower is better.";
  }
  if (strcmp(label, "Max Abs") == 0) {
    return linear_space ? "Maximum absolute per-channel error anywhere in the image, measured in linear HDR space. Lower is better."
                        : "Maximum absolute per-channel error anywhere in the image, measured in compare space. Lower is better.";
  }
  if (strcmp(label, "Mean Signed") == 0) {
    return linear_space ? "Average signed difference in linear HDR space. Values near zero are best; positive means GPU tends brighter, negative means darker."
                        : "Average signed difference in compare space. Values near zero are best; positive means GPU tends brighter, negative means darker.";
  }
  if (strcmp(label, "Ref Avg Luma") == 0) {
    return linear_space ? "Average luminance of the CPU reference image in linear HDR space."
                        : "Average luminance of the CPU reference image in compare space.";
  }
  if (strcmp(label, "GPU Avg Luma") == 0) {
    return linear_space ? "Average luminance of the GPU result image in linear HDR space."
                        : "Average luminance of the GPU result image in compare space.";
  }
  if (strcmp(label, "Brightness Ratio") == 0) {
    return linear_space ? "GPU average luminance divided by CPU average luminance in linear HDR space. A perfect match is 1.0."
                        : "GPU average luminance divided by CPU average luminance in compare space. A perfect match is 1.0.";
  }
  if (strcmp(label, "Brightness Rel") == 0) {
    return linear_space ? "Signed relative average-luminance difference in linear HDR space. Positive means GPU is brighter, negative means darker."
                        : "Signed relative average-luminance difference in compare space. Positive means GPU is brighter, negative means darker.";
  }
  if (strcmp(label, "P95 Abs") == 0) {
    return linear_space ? "95th percentile absolute error in linear HDR space. Shows how bad the worst 5% of pixels are. Lower is better."
                        : "95th percentile absolute error in compare space. Shows how bad the worst 5% of pixels are. Lower is better.";
  }
  if (strcmp(label, "P99 Abs") == 0) {
    return linear_space ? "99th percentile absolute error in linear HDR space. Highlights near-outlier error without using only the single worst pixel. Lower is better."
                        : "99th percentile absolute error in compare space. Highlights near-outlier error without using only the single worst pixel. Lower is better.";
  }

  return linear_space ? "Metric measured in linear HDR space. Lower error is generally better unless stated otherwise."
                      : "Metric measured in compare space. Lower error is generally better unless stated otherwise.";
}

std::string metric_label_cell(const char* label, float score, bool linear_space) {
  const float clamped_score = clamp_metric_score(score);
  const float fill_percent = clamped_score * 100.0f;
  const std::string tooltip = html_escape(metric_tooltip_text(label, linear_space));

  const float fill_r = lerp(138.0f, 92.0f, clamped_score);
  const float fill_g = lerp(82.0f, 126.0f, clamped_score);
  const float fill_b = lerp(82.0f, 104.0f, clamped_score);

  char buffer[512] = {};
  std::snprintf(buffer, sizeof(buffer),
    "<td class=\"metric-label\" style=\"background: linear-gradient(90deg, rgba(%.0f, %.0f, %.0f, 0.34) 0%%, rgba(%.0f, %.0f, %.0f, 0.34) %.2f%%, "
    "rgba(72, 53, 53, 0.16) %.2f%%, rgba(72, 53, 53, 0.16) 100%%);\" title=\"%s\">%s</td>",
    fill_r, fill_g, fill_b, fill_r, fill_g, fill_b, fill_percent, fill_percent, tooltip.c_str(), html_escape(label).c_str());
  return buffer;
}

std::string metric_row_percent(const char* label, float value, float score, bool linear_space) {
  char buffer[768] = {};
  const std::string label_cell = metric_label_cell(label, score, linear_space);
  std::snprintf(buffer, sizeof(buffer), "<tr>%s<td>%.2f%%</td></tr>\n", label_cell.c_str(), value);
  return buffer;
}

std::string metric_row_value(const char* label, float value, float score, bool linear_space) {
  char buffer[768] = {};
  const std::string label_cell = metric_label_cell(label, score, linear_space);
  std::snprintf(buffer, sizeof(buffer), "<tr>%s<td>%.6f</td></tr>\n", label_cell.c_str(), value);
  return buffer;
}

std::string json_escape(const std::string& text) {
  std::string result = {};
  result.reserve(text.size() + 8u);

  for (const char ch : text) {
    switch (ch) {
      case '\\':
        result += "\\\\";
        break;
      case '"':
        result += "\\\"";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20u) {
          char buffer[8] = {};
          std::snprintf(buffer, sizeof(buffer), "\\u%04x", static_cast<unsigned int>(static_cast<unsigned char>(ch)));
          result += buffer;
        } else {
          result.push_back(ch);
        }
        break;
    }
  }

  return result;
}

std::string format_comparison_report(const char* technique_tag, const ImageComparisonResult& comparison) {
  char buffer[1536] = {};
  std::snprintf(
    buffer, sizeof(buffer),
    "[%s] compare_space{similarity=%.2f%%, rmse=%.6f, mae=%.6f, relative_rmse=%.6f, max_abs=%.6f, mean_signed=%.6f, ref_avg_luma=%.6f, gpu_avg_luma=%.6f, "
    "brightness_ratio=%.6f, brightness_rel=%.6f, p95_abs=%.6f, p99_abs=%.6f} linear{rmse=%.6f, mae=%.6f, relative_rmse=%.6f, max_abs=%.6f, "
    "mean_signed=%.6f, ref_avg_luma=%.6f, gpu_avg_luma=%.6f, brightness_ratio=%.6f, brightness_rel=%.6f, p95_abs=%.6f, p99_abs=%.6f}\n",
    technique_tag, comparison.similarity, comparison.root_mean_squared_error, comparison.mean_absolute_error, comparison.relative_root_mean_squared_error,
    comparison.max_absolute_error, comparison.mean_signed_error, comparison.reference_mean_luminance, comparison.result_mean_luminance, comparison.brightness_ratio,
    comparison.brightness_relative_error, comparison.percentile_95_absolute_error, comparison.percentile_99_absolute_error,
    comparison.linear_root_mean_squared_error, comparison.linear_mean_absolute_error, comparison.linear_relative_root_mean_squared_error,
    comparison.linear_max_absolute_error, comparison.linear_mean_signed_error, comparison.linear_reference_mean_luminance,
    comparison.linear_result_mean_luminance, comparison.linear_brightness_ratio, comparison.linear_brightness_relative_error,
    comparison.linear_percentile_95_absolute_error, comparison.linear_percentile_99_absolute_error);
  return buffer;
}

std::string format_ai_comparison_report(
  const char* kind, const char* technique_tag, const char* scene_file, const char* reference_file, const char* output_file, const ImageComparisonResult& comparison) {
  const std::string safe_kind = json_escape(kind != nullptr ? kind : "");
  const std::string safe_technique = json_escape(technique_tag != nullptr ? technique_tag : "");
  const std::string safe_scene = json_escape(scene_file != nullptr ? scene_file : "");
  const std::string safe_reference = json_escape(reference_file != nullptr ? reference_file : "");
  const std::string safe_output = json_escape(output_file != nullptr ? output_file : "");

  char buffer[3072] = {};
  std::snprintf(buffer, sizeof(buffer),
    "AI_IMAGE_COMPARISON {\"schema\":\"etx.image_comparison.v2\",\"kind\":\"%s\",\"technique\":\"%s\",\"scene\":\"%s\",\"reference\":\"%s\","
    "\"output\":\"%s\",\"compare_space\":{\"similarity_percent\":%.6f,\"rmse\":%.6f,\"mae\":%.6f,\"relative_rmse\":%.6f,\"max_abs\":%.6f,"
    "\"mean_signed\":%.6f,\"ref_avg_luma\":%.6f,\"gpu_avg_luma\":%.6f,\"brightness_ratio\":%.6f,\"brightness_rel\":%.6f,\"p95_abs\":%.6f,\"p99_abs\":%.6f},"
    "\"linear\":{\"rmse\":%.6f,\"mae\":%.6f,\"relative_rmse\":%.6f,\"max_abs\":%.6f,\"mean_signed\":%.6f,\"ref_avg_luma\":%.6f,\"gpu_avg_luma\":%.6f,"
    "\"brightness_ratio\":%.6f,\"brightness_rel\":%.6f,\"p95_abs\":%.6f,\"p99_abs\":%.6f}}\n",
    safe_kind.c_str(), safe_technique.c_str(), safe_scene.c_str(), safe_reference.c_str(), safe_output.c_str(), comparison.similarity,
    comparison.root_mean_squared_error, comparison.mean_absolute_error, comparison.relative_root_mean_squared_error, comparison.max_absolute_error,
    comparison.mean_signed_error, comparison.reference_mean_luminance, comparison.result_mean_luminance, comparison.brightness_ratio,
    comparison.brightness_relative_error, comparison.percentile_95_absolute_error, comparison.percentile_99_absolute_error,
    comparison.linear_root_mean_squared_error, comparison.linear_mean_absolute_error, comparison.linear_relative_root_mean_squared_error,
    comparison.linear_max_absolute_error, comparison.linear_mean_signed_error, comparison.linear_reference_mean_luminance,
    comparison.linear_result_mean_luminance, comparison.linear_brightness_ratio, comparison.linear_brightness_relative_error,
    comparison.linear_percentile_95_absolute_error, comparison.linear_percentile_99_absolute_error);
  return buffer;
}

void append_ai_comparison_json(std::string& json_text, const char* kind, const char* technique_tag, const char* scene_file, const char* reference_file,
  const char* output_file, const ImageComparisonResult& comparison) {
  const std::string safe_kind = json_escape(kind != nullptr ? kind : "");
  const std::string safe_technique = json_escape(technique_tag != nullptr ? technique_tag : "");
  const std::string safe_scene = json_escape(scene_file != nullptr ? scene_file : "");
  const std::string safe_reference = json_escape(reference_file != nullptr ? reference_file : "");
  const std::string safe_output = json_escape(output_file != nullptr ? output_file : "");

  json_text += "    {\n";
  json_text += "      \"kind\": \"" + safe_kind + "\",\n";
  json_text += "      \"technique\": \"" + safe_technique + "\",\n";
  json_text += "      \"scene\": \"" + safe_scene + "\",\n";
  json_text += "      \"reference\": \"" + safe_reference + "\",\n";
  json_text += "      \"output\": \"" + safe_output + "\",\n";
  json_text += "      \"compare_space\": {\n";
  json_text += "        \"similarity_percent\": " + std::to_string(comparison.similarity) + ",\n";
  json_text += "        \"rmse\": " + std::to_string(comparison.root_mean_squared_error) + ",\n";
  json_text += "        \"mae\": " + std::to_string(comparison.mean_absolute_error) + ",\n";
  json_text += "        \"relative_rmse\": " + std::to_string(comparison.relative_root_mean_squared_error) + ",\n";
  json_text += "        \"max_abs\": " + std::to_string(comparison.max_absolute_error) + ",\n";
  json_text += "        \"mean_signed\": " + std::to_string(comparison.mean_signed_error) + ",\n";
  json_text += "        \"ref_avg_luma\": " + std::to_string(comparison.reference_mean_luminance) + ",\n";
  json_text += "        \"gpu_avg_luma\": " + std::to_string(comparison.result_mean_luminance) + ",\n";
  json_text += "        \"brightness_ratio\": " + std::to_string(comparison.brightness_ratio) + ",\n";
  json_text += "        \"brightness_rel\": " + std::to_string(comparison.brightness_relative_error) + ",\n";
  json_text += "        \"p95_abs\": " + std::to_string(comparison.percentile_95_absolute_error) + ",\n";
  json_text += "        \"p99_abs\": " + std::to_string(comparison.percentile_99_absolute_error) + "\n";
  json_text += "      },\n";
  json_text += "      \"linear\": {\n";
  json_text += "        \"rmse\": " + std::to_string(comparison.linear_root_mean_squared_error) + ",\n";
  json_text += "        \"mae\": " + std::to_string(comparison.linear_mean_absolute_error) + ",\n";
  json_text += "        \"relative_rmse\": " + std::to_string(comparison.linear_relative_root_mean_squared_error) + ",\n";
  json_text += "        \"max_abs\": " + std::to_string(comparison.linear_max_absolute_error) + ",\n";
  json_text += "        \"mean_signed\": " + std::to_string(comparison.linear_mean_signed_error) + ",\n";
  json_text += "        \"ref_avg_luma\": " + std::to_string(comparison.linear_reference_mean_luminance) + ",\n";
  json_text += "        \"gpu_avg_luma\": " + std::to_string(comparison.linear_result_mean_luminance) + ",\n";
  json_text += "        \"brightness_ratio\": " + std::to_string(comparison.linear_brightness_ratio) + ",\n";
  json_text += "        \"brightness_rel\": " + std::to_string(comparison.linear_brightness_relative_error) + ",\n";
  json_text += "        \"p95_abs\": " + std::to_string(comparison.linear_percentile_95_absolute_error) + ",\n";
  json_text += "        \"p99_abs\": " + std::to_string(comparison.linear_percentile_99_absolute_error) + "\n";
  json_text += "      }\n";
  json_text += "    }";
}

void print_comparison_report(
  const char* technique_tag, const char* kind, const char* scene_file, const char* reference_file, const char* output_file, const ImageComparisonResult& comparison) {
  const std::string line = format_comparison_report(technique_tag, comparison);
  printf("%s", line.c_str());
  const std::string ai_line = format_ai_comparison_report(kind, technique_tag, scene_file, reference_file, output_file, comparison);
  printf("%s", ai_line.c_str());
}

bool save_text_to_file(const std::string& file_name, const std::string& text) {
  if (ensure_parent_directory_exists(file_name) == false) {
    return false;
  }

  FILE* file = std::fopen(file_name.c_str(), "wb");
  if (file == nullptr) {
    log::error("Failed to open text file for writing: %s", file_name.c_str());
    return false;
  }

  const size_t text_size = text.size();
  const size_t written_size = std::fwrite(text.data(), 1u, text_size, file);
  const int close_result = std::fclose(file);
  if (written_size != text_size) {
    log::error("Failed to write text file: %s", file_name.c_str());
    return false;
  }
  if (close_result != 0) {
    log::error("Failed to close text file: %s", file_name.c_str());
    return false;
  }

  return true;
}

void append_comparison_report(std::string& report_text, const char* technique_tag, const ImageComparisonResult& comparison) {
  report_text += format_comparison_report(technique_tag, comparison);
}

bool resolve_batch_output_window(
  const BatchRenderOptions& options, const uint2& image_size, uint32_t& window_x, uint32_t& window_y, uint32_t& window_width, uint32_t& window_height) {
  if ((image_size.x == 0u) || (image_size.y == 0u)) {
    return false;
  }

  if (options.override_crop == false) {
    window_x = 0u;
    window_y = 0u;
    window_width = image_size.x;
    window_height = image_size.y;
    return true;
  }

  if ((options.crop_x >= image_size.x) || (options.crop_y >= image_size.y) || (options.crop_width > (image_size.x - options.crop_x)) ||
      (options.crop_height > (image_size.y - options.crop_y))) {
    log::error("Crop window is outside image bounds: crop=%u,%u,%u,%u image=%ux%u", options.crop_x, options.crop_y, options.crop_width, options.crop_height,
      image_size.x, image_size.y);
    return false;
  }

  window_x = options.crop_x;
  window_y = options.crop_y;
  window_width = options.crop_width;
  window_height = options.crop_height;
  return true;
}

bool prepare_batch_output_buffer(
  const BatchRenderOptions& options, const float4* input, const uint2& input_size, std::vector<float4>& output, uint2& output_size) {
  if ((input == nullptr) || (input_size.x == 0u) || (input_size.y == 0u)) {
    log::error("Invalid batch output image");
    return false;
  }

  uint32_t window_x = 0u;
  uint32_t window_y = 0u;
  uint32_t window_width = 0u;
  uint32_t window_height = 0u;
  if (resolve_batch_output_window(options, input_size, window_x, window_y, window_width, window_height) == false) {
    return false;
  }

  output_size = {window_width, window_height};
  output.resize(static_cast<size_t>(window_width) * static_cast<size_t>(window_height));

  for (uint32_t y = 0u; y < window_height; ++y) {
    const uint32_t source_row = window_y + y;
    const uint32_t source_offset = source_row * input_size.x + window_x;
    const uint32_t destination_offset = y * window_width;
    std::memcpy(output.data() + destination_offset, input + source_offset, static_cast<size_t>(window_width) * sizeof(float4));
  }

  return true;
}

void apply_batch_scene_overrides(const BatchRenderOptions& options, SceneRepresentation& scene) {
  if (options.samples > 0u) {
    scene.data().options.samples = options.samples;
  }

  if (options.max_path_length > 0u) {
    scene.data().options.max_path_length = options.max_path_length;
    scene.data().options.min_path_length = std::min(scene.data().options.min_path_length, scene.data().options.max_path_length);
  }

  if (options.override_random_seed) {
    scene.data().options.random_seed = options.random_seed;
  }

  if (options.override_strategy_flags) {
    scene.data().options.strategy_flags = options.strategy_flags;
  }

  if (options.override_resolution) {
    Camera& camera = scene.mutable_camera();
    const uint2 output_dimensions = {options.resolution_width, options.resolution_height};
    if (camera.cls == Camera::Class::Perspective) {
      const float camera_fov = get_camera_fov(camera);
      build_camera(camera, camera.position, camera.direction, camera.up, output_dimensions, camera_fov);
    } else {
      camera.film_size = output_dimensions;
    }
    scene.store_active_camera();
  }
}

bool save_batch_output_file(const std::string& output_file, const float4* output, const uint2& image_size, float exposure) {
  if (ensure_parent_directory_exists(output_file) == false) {
    return false;
  }

  ImageOutputParameters output_params = {
    .mode = save_image_mode_from_file_name(output_file),
    .exposure = exposure,
  };
  return save_image_to_file(output_file, output, image_size, output_params);
}

bool save_batch_output_with_png(const std::string& output_file, const float4* output, const uint2& image_size, float exposure) {
  if (save_batch_output_file(output_file, output, image_size, exposure) == false) {
    return false;
  }

  ImageOutputParameters png_output_params = {
    .mode = SaveImageMode::TonemappedLDR,
    .exposure = exposure,
  };
  const std::string png_file_name = png_file_name_from_output_file(output_file);
  return save_image_to_file(png_file_name, output, image_size, png_output_params);
}

bool compare_output_to_reference_and_save(
  const BatchRenderOptions& options, const std::string& reference_file, const std::string& output_file, const float4* output, const uint2& image_size,
  ImageComparisonResult* out_comparison) {
  std::vector<float4> reference_image = {};
  uint2 reference_image_size = {};
  if (load_hdr_image_from_file(reference_file, reference_image, reference_image_size) == false) {
    return false;
  }

  const float4* reference_data = reference_image.data();
  std::vector<float4> prepared_reference = {};
  uint2 prepared_reference_size = reference_image_size;
  if ((reference_image_size.x != image_size.x) || (reference_image_size.y != image_size.y)) {
    if (options.override_crop == false) {
      log::error("Reference image size mismatch: reference=%ux%u, render=%ux%u", reference_image_size.x, reference_image_size.y, image_size.x, image_size.y);
      return false;
    }

    if (prepare_batch_output_buffer(options, reference_image.data(), reference_image_size, prepared_reference, prepared_reference_size) == false) {
      return false;
    }

    if ((prepared_reference_size.x != image_size.x) || (prepared_reference_size.y != image_size.y)) {
      log::error("Reference image size mismatch after crop: reference=%ux%u, render=%ux%u", prepared_reference_size.x, prepared_reference_size.y, image_size.x, image_size.y);
      return false;
    }

    reference_data = prepared_reference.data();
  }

  std::vector<float4> difference_image = {};
  ImageComparisonResult comparison = {};
  if (compare_images(reference_data, output, image_size, difference_image, comparison) == false) {
    return false;
  }

  ImageOutputParameters difference_params = {
    .mode = SaveImageMode::RGB,
    .exposure = 1.0f,
  };
  const std::string difference_file_name = comparison_file_name_from_output(output_file);
  if (save_image_to_file(difference_file_name, difference_image.data(), image_size, difference_params) == false) {
    return false;
  }

  if (out_comparison != nullptr) {
    *out_comparison = comparison;
  }

  return true;
}

void append_full_comparison_html_header(std::string& html_text, const std::string& scene_file) {
  html_text += "<!doctype html>\n";
  html_text += "<html lang=\"en\">\n";
  html_text += "<head>\n";
  html_text += "  <meta charset=\"utf-8\">\n";
  html_text += "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n";
  html_text += "  <title>ETX Full Comparison</title>\n";
  html_text += "  <style>\n";
  html_text += "    :root { color-scheme: dark; --bg: #10151d; --panel: #17202c; --panel-2: #1d2937; --text: #edf2f7; --muted: #9fb0c3; --accent: #7dd3fc; --line: #2b3a4b; }\n";
  html_text += "    * { box-sizing: border-box; }\n";
  html_text += "    body { margin: 0; font-family: \"Cascadia Mono\", \"SFMono-Regular\", Consolas, \"Liberation Mono\", Menlo, monospace; background: radial-gradient(circle at top, #1a2533 0%, #10151d 45%, #0b1016 100%); color: var(--text); }\n";
  html_text += "    .page { max-width: 1500px; margin: 0 auto; padding: 32px 24px 64px; }\n";
  html_text += "    .hero { margin-bottom: 28px; padding: 24px 28px; border: 1px solid var(--line); border-radius: 20px; background: linear-gradient(135deg, rgba(125, 211, 252, 0.08), rgba(255, 255, 255, 0.02)); }\n";
  html_text += "    .hero h1 { margin: 0 0 8px; font-size: 34px; }\n";
  html_text += "    .hero p { margin: 0; color: var(--muted); word-break: break-all; }\n";
  html_text += "    .grid { display: grid; gap: 20px; }\n";
  html_text += "    .card { border: 1px solid var(--line); border-radius: 20px; background: linear-gradient(180deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.015)); overflow: hidden; box-shadow: 0 18px 48px rgba(0, 0, 0, 0.22); }\n";
  html_text += "    .card-header { padding: 20px 24px; border-bottom: 1px solid var(--line); display: flex; justify-content: space-between; gap: 16px; align-items: baseline; }\n";
  html_text += "    .card-header h2 { margin: 0; font-size: 24px; text-transform: uppercase; letter-spacing: 0.08em; }\n";
  html_text += "    .card-header .tag { color: var(--accent); font-size: 13px; }\n";
  html_text += "    .card-body { display: grid; grid-template-columns: minmax(340px, 1.25fr) minmax(280px, 0.9fr); gap: 0; }\n";
  html_text += "    .viewer-wrap { padding: 20px 24px 24px; }\n";
  html_text += "    .viewer-toolbar { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; margin-bottom: 16px; color: var(--muted); }\n";
  html_text += "    .viewer-toolbar label { display: inline-flex; gap: 8px; align-items: center; }\n";
  html_text += "    .viewer { position: relative; width: 100%; border-radius: 14px; overflow: hidden; border: 1px solid var(--line); background: #0c1117; }\n";
  html_text += "    .viewer img { display: block; width: 100%; height: auto; }\n";
  html_text += "    .viewer .overlay { position: absolute; inset: 0; opacity: 1; transition: opacity 120ms linear; pointer-events: none; }\n";
  html_text += "    .legend { margin-top: 12px; color: var(--muted); font-size: 13px; display: flex; gap: 18px; flex-wrap: wrap; }\n";
  html_text += "    .metrics { padding: 20px 24px 24px; border-left: 1px solid var(--line); background: rgba(12, 17, 23, 0.42); }\n";
  html_text += "    .metrics h3 { margin: 0 0 12px; font-size: 16px; color: var(--accent); }\n";
  html_text += "    table { width: 100%; border-collapse: collapse; font-size: 14px; }\n";
  html_text += "    th, td { padding: 8px 0; border-bottom: 1px solid rgba(159, 176, 195, 0.14); text-align: left; }\n";
  html_text += "    th { color: var(--muted); font-weight: 600; }\n";
  html_text += "    td { font-variant-numeric: tabular-nums; }\n";
  html_text += "    .metric-label { padding-left: 10px; border-radius: 6px; cursor: help; }\n";
  html_text += "    .files { margin-top: 18px; color: var(--muted); font-size: 13px; }\n";
  html_text += "    .files div { margin-top: 6px; word-break: break-all; }\n";
  html_text += "    input[type=\"range\"] { width: 220px; }\n";
  html_text += "    @media (max-width: 1024px) { .card-body { grid-template-columns: 1fr; } .metrics { border-left: 0; border-top: 1px solid var(--line); } }\n";
  html_text += "  </style>\n";
  html_text += "</head>\n";
  html_text += "<body>\n";
  html_text += "  <div class=\"page\">\n";
  html_text += "    <section class=\"hero\">\n";
  html_text += "      <h1>ETX Full Comparison</h1>\n";
  html_text += "      <p>Scene: ";
  html_text += html_escape(scene_file);
  html_text += "</p>\n";
  html_text += "    </section>\n";
  html_text += "    <section class=\"grid\">\n";
}

void append_full_comparison_html_entry(std::string& html_text, const FullComparisonTechniqueInfo& technique, const std::string& cpu_output_file,
  const std::string& gpu_output_file, const ImageComparisonResult& comparison) {
  const std::string cpu_png_file = html_file_name_only(png_file_name_from_output_file(cpu_output_file));
  const std::string gpu_png_file = html_file_name_only(png_file_name_from_output_file(gpu_output_file));
  const std::string cpu_exr_file = html_file_name_only(cpu_output_file);
  const std::string gpu_exr_file = html_file_name_only(gpu_output_file);
  const std::string viewer_id = std::string("viewer_") + technique.file_tag;
  const float compare_similarity_score = clamp_metric_score(comparison.similarity / 100.0f);
  const float compare_rmse_score = inverse_error_metric_score(comparison.root_mean_squared_error, 1.0f);
  const float compare_mae_score = inverse_error_metric_score(comparison.mean_absolute_error, 1.0f);
  const float compare_relative_rmse_score = inverse_error_metric_score(comparison.relative_root_mean_squared_error, 1.0f);
  const float compare_max_abs_score = inverse_error_metric_score(comparison.max_absolute_error, 1.0f);
  const float compare_mean_signed_score = inverse_error_metric_score(fabsf(comparison.mean_signed_error), 0.5f);
  const float compare_reference_luminance_score = 1.0f;
  const float compare_result_luminance_score = 1.0f;
  const float compare_brightness_ratio_score = inverse_error_metric_score(fabsf(comparison.brightness_ratio - 1.0f), 0.5f);
  const float compare_brightness_relative_score = inverse_error_metric_score(fabsf(comparison.brightness_relative_error), 0.5f);
  const float compare_p95_score = inverse_error_metric_score(comparison.percentile_95_absolute_error, 1.0f);
  const float compare_p99_score = inverse_error_metric_score(comparison.percentile_99_absolute_error, 1.0f);
  const float linear_rmse_score = inverse_error_metric_score(comparison.linear_root_mean_squared_error, 1.0f);
  const float linear_mae_score = inverse_error_metric_score(comparison.linear_mean_absolute_error, 1.0f);
  const float linear_relative_rmse_score = inverse_error_metric_score(comparison.linear_relative_root_mean_squared_error, 1.0f);
  const float linear_max_abs_score = inverse_error_metric_score(comparison.linear_max_absolute_error, 1.0f);
  const float linear_mean_signed_score = inverse_error_metric_score(fabsf(comparison.linear_mean_signed_error), 0.5f);
  const float linear_reference_luminance_score = 1.0f;
  const float linear_result_luminance_score = 1.0f;
  const float linear_brightness_ratio_score = inverse_error_metric_score(fabsf(comparison.linear_brightness_ratio - 1.0f), 0.5f);
  const float linear_brightness_relative_score = inverse_error_metric_score(fabsf(comparison.linear_brightness_relative_error), 0.5f);
  const float linear_p95_score = inverse_error_metric_score(comparison.linear_percentile_95_absolute_error, 1.0f);
  const float linear_p99_score = inverse_error_metric_score(comparison.linear_percentile_99_absolute_error, 1.0f);
  std::string compare_rows = {};
  compare_rows += metric_row_percent("Similarity", comparison.similarity, compare_similarity_score, false);
  compare_rows += metric_row_value("RMSE", comparison.root_mean_squared_error, compare_rmse_score, false);
  compare_rows += metric_row_value("MAE", comparison.mean_absolute_error, compare_mae_score, false);
  compare_rows += metric_row_value("Relative RMSE", comparison.relative_root_mean_squared_error, compare_relative_rmse_score, false);
  compare_rows += metric_row_value("Max Abs", comparison.max_absolute_error, compare_max_abs_score, false);
  compare_rows += metric_row_value("Mean Signed", comparison.mean_signed_error, compare_mean_signed_score, false);
  compare_rows += metric_row_value("Ref Avg Luma", comparison.reference_mean_luminance, compare_reference_luminance_score, false);
  compare_rows += metric_row_value("GPU Avg Luma", comparison.result_mean_luminance, compare_result_luminance_score, false);
  compare_rows += metric_row_value("Brightness Ratio", comparison.brightness_ratio, compare_brightness_ratio_score, false);
  compare_rows += metric_row_value("Brightness Rel", comparison.brightness_relative_error, compare_brightness_relative_score, false);
  compare_rows += metric_row_value("P95 Abs", comparison.percentile_95_absolute_error, compare_p95_score, false);
  compare_rows += metric_row_value("P99 Abs", comparison.percentile_99_absolute_error, compare_p99_score, false);
  std::string linear_rows = {};
  linear_rows += metric_row_value("RMSE", comparison.linear_root_mean_squared_error, linear_rmse_score, true);
  linear_rows += metric_row_value("MAE", comparison.linear_mean_absolute_error, linear_mae_score, true);
  linear_rows += metric_row_value("Relative RMSE", comparison.linear_relative_root_mean_squared_error, linear_relative_rmse_score, true);
  linear_rows += metric_row_value("Max Abs", comparison.linear_max_absolute_error, linear_max_abs_score, true);
  linear_rows += metric_row_value("Mean Signed", comparison.linear_mean_signed_error, linear_mean_signed_score, true);
  linear_rows += metric_row_value("Ref Avg Luma", comparison.linear_reference_mean_luminance, linear_reference_luminance_score, true);
  linear_rows += metric_row_value("GPU Avg Luma", comparison.linear_result_mean_luminance, linear_result_luminance_score, true);
  linear_rows += metric_row_value("Brightness Ratio", comparison.linear_brightness_ratio, linear_brightness_ratio_score, true);
  linear_rows += metric_row_value("Brightness Rel", comparison.linear_brightness_relative_error, linear_brightness_relative_score, true);
  linear_rows += metric_row_value("P95 Abs", comparison.linear_percentile_95_absolute_error, linear_p95_score, true);
  linear_rows += metric_row_value("P99 Abs", comparison.linear_percentile_99_absolute_error, linear_p99_score, true);

  html_text += "      <div class=\"card\">\n";
  html_text += "        <div class=\"card-header\">\n";
  html_text += "          <h2>";
  html_text += html_escape(technique.file_tag);
  html_text += "</h2>\n";
  html_text += "          <div class=\"tag\">CPU/GPU overlay</div>\n";
  html_text += "        </div>\n";
  html_text += "        <div class=\"card-body\">\n";
  html_text += "          <div class=\"viewer-wrap\">\n";
  html_text += "            <div class=\"viewer-toolbar\">\n";
  html_text += "              <label><input type=\"checkbox\" data-target=\"";
  html_text += html_escape(viewer_id);
  html_text += "\" checked> Show GPU overlay</label>\n";
  html_text += "              <label>Overlay opacity <input type=\"range\" min=\"0\" max=\"100\" value=\"100\" data-opacity-target=\"";
  html_text += html_escape(viewer_id);
  html_text += "\"></label>\n";
  html_text += "            </div>\n";
  html_text += "            <div class=\"viewer\">\n";
  html_text += "              <img src=\"";
  html_text += html_escape(cpu_png_file);
  html_text += "\" alt=\"";
  html_text += html_escape(std::string(technique.file_tag) + " CPU reference");
  html_text += "\">\n";
  html_text += "              <img class=\"overlay\" id=\"";
  html_text += html_escape(viewer_id);
  html_text += "\" src=\"";
  html_text += html_escape(gpu_png_file);
  html_text += "\" alt=\"";
  html_text += html_escape(std::string(technique.file_tag) + " GPU result");
  html_text += "\">\n";
  html_text += "            </div>\n";
  html_text += "            <div class=\"legend\">\n";
  html_text += "              <span>Base layer: CPU</span>\n";
  html_text += "              <span>Overlay: GPU</span>\n";
  html_text += "            </div>\n";
  html_text += "          </div>\n";
  html_text += "          <div class=\"metrics\">\n";
  html_text +=
    "            <h3 title=\"Metrics measured after the comparison-space transform used by the report. Useful for perceptual closeness; lower error is better, higher similarity is better.\">Compare-Space Metrics</h3>\n";
  html_text += "            <table>\n";
  html_text += compare_rows;
  html_text += "            </table>\n";
  html_text +=
    "            <h3 title=\"Metrics measured directly on linear HDR pixel values. Useful for physical/numeric mismatch checks; lower error is better.\">Linear Metrics</h3>\n";
  html_text += "            <table>\n";
  html_text += linear_rows;
  html_text += "            </table>\n";
  html_text += "            <div class=\"files\">\n";
  html_text += "              <div>CPU EXR: ";
  html_text += html_escape(cpu_exr_file);
  html_text += "</div>\n";
  html_text += "              <div>GPU EXR: ";
  html_text += html_escape(gpu_exr_file);
  html_text += "</div>\n";
  html_text += "              <div>CPU PNG: ";
  html_text += html_escape(cpu_png_file);
  html_text += "</div>\n";
  html_text += "              <div>GPU PNG: ";
  html_text += html_escape(gpu_png_file);
  html_text += "</div>\n";
  html_text += "            </div>\n";
  html_text += "          </div>\n";
  html_text += "        </div>\n";
  html_text += "      </div>\n";
}

void append_full_comparison_html_footer(std::string& html_text) {
  html_text += "    </section>\n";
  html_text += "  </div>\n";
  html_text += "  <script>\n";
  html_text += "    const overlayToggles = document.querySelectorAll('input[data-target]');\n";
  html_text += "    for (const toggle of overlayToggles) {\n";
  html_text += "      toggle.addEventListener('change', () => {\n";
  html_text += "        const target = document.getElementById(toggle.dataset.target);\n";
  html_text += "        if (target) {\n";
  html_text += "          target.style.display = toggle.checked ? 'block' : 'none';\n";
  html_text += "        }\n";
  html_text += "      });\n";
  html_text += "    }\n";
  html_text += "    const opacitySliders = document.querySelectorAll('input[data-opacity-target]');\n";
  html_text += "    for (const slider of opacitySliders) {\n";
  html_text += "      slider.addEventListener('input', () => {\n";
  html_text += "        const target = document.getElementById(slider.dataset.opacityTarget);\n";
  html_text += "        if (target) {\n";
  html_text += "          target.style.opacity = String(Number(slider.value) / 100.0);\n";
  html_text += "        }\n";
  html_text += "      });\n";
  html_text += "    }\n";
  html_text += "  </script>\n";
  html_text += "</body>\n";
  html_text += "</html>\n";
}

struct BatchRenderSession {
  BatchRenderSession()
    : film(scheduler)
    , rt(scheduler, film)
    , scene(scheduler, ior_database)
    , cpu_renderer(rt, scene)
    , gpu_renderer(scheduler) {
  }

  ~BatchRenderSession() {
    cleanup();
  }

  bool init(bool initialize_cpu_renderer, bool initialize_gpu_renderer) {
    scene_global_init();
    render_context.init();
    if (render_context.context().valid() == false) {
      log::error("Failed to initialize headless RHI context");
      return false;
    }

    scene.set_scattering_rhi(render_context.context());

    std::string ior_folder = env().file_in_data("./spectrum/");
    ior_database.load(ior_folder.c_str());

    if (initialize_cpu_renderer) {
      cpu_renderer.init(render_context.context(), scene);
    }
    if (initialize_gpu_renderer) {
      gpu_renderer.init(render_context.context(), scene);
    }
    return true;
  }

  void cleanup() {
    if (render_context.context().valid()) {
      gpu_renderer.cleanup(render_context.context());
      cpu_renderer.cleanup(render_context.context());
    }
    scene_global_deinit();
    render_context.cleanup();
  }

  TaskScheduler scheduler;
  Film film;
  Raytracing rt;
  IORDatabase ior_database;
  SceneRepresentation scene;
  HeadlessRenderContext render_context;
  CPURaytracingRenderer cpu_renderer;
  GPURaytracingRenderer gpu_renderer;
};

bool configure_batch_render_window(const BatchRenderOptions& options, BatchRenderSession& session) {
  const uint2 image_size = session.scene.camera().film_size;
  uint32_t window_x = 0u;
  uint32_t window_y = 0u;
  uint32_t window_width = 0u;
  uint32_t window_height = 0u;
  if (resolve_batch_output_window(options, image_size, window_x, window_y, window_width, window_height) == false) {
    return false;
  }

  bool cpu_can_render_crop_window = true;
  const auto& integrator_data = session.scene.integrator_data();
  if (integrator_data.selected == Integrator::Type::Bidirectional) {
    const auto settings_it = integrator_data.settings.find(Integrator::Type::Bidirectional);
    uint32_t bdpt_mode = kBDPTModeFast;
    if (settings_it != integrator_data.settings.end()) {
      bdpt_mode = settings_it->second.get_integral("bdpt-mode", bdpt_mode);
    }
    cpu_can_render_crop_window = bdpt_mode == kBDPTModePathTracing;
  } else if ((integrator_data.selected == Integrator::Type::VCM) || (integrator_data.selected == Integrator::Type::BDPTDistilled)) {
    cpu_can_render_crop_window = false;
  }

  if (cpu_can_render_crop_window) {
    if (session.film.set_render_window({window_x, window_y}, {window_width, window_height}) == false) {
      log::error("Failed to configure CPU render window: %u,%u,%u,%u", window_x, window_y, window_width, window_height);
      return false;
    }
  } else {
    session.film.reset_render_window();
  }

  if (session.gpu_renderer.set_render_window({window_x, window_y}, {window_width, window_height}, image_size) == false) {
    log::error("Failed to configure GPU render window: %u,%u,%u,%u", window_x, window_y, window_width, window_height);
    return false;
  }

  return true;
}

Integrator* select_integrator(const BatchRenderOptions& options, BatchRenderSession& session, const SceneRepresentation::IntegratorData& integrator_data) {
  Integrator* selected_integrator = nullptr;

  if (options.integrator.empty() == false) {
    const Integrator::Type requested_type = integrator_id_to_type(options.integrator.c_str());
    selected_integrator = integrator_type_to_instance(requested_type, session.cpu_renderer.integrator_list(), session.cpu_renderer.integrator_count());
    if (selected_integrator == nullptr) {
      log::error("Unknown integrator: %s", options.integrator.c_str());
      return nullptr;
    }
  }

  if ((selected_integrator == nullptr) && (integrator_data.selected != Integrator::Type::Invalid)) {
    selected_integrator =
      integrator_type_to_instance(integrator_data.selected, session.cpu_renderer.integrator_list(), session.cpu_renderer.integrator_count());
  }

  if ((selected_integrator == nullptr) && (session.cpu_renderer.integrator_count() > 1u)) {
    selected_integrator = session.cpu_renderer.integrator_list()[1];
  }

  return selected_integrator;
}

bool save_batch_output_and_compare(const BatchRenderOptions& options, const float4* output, const uint2& image_size) {
  std::vector<float4> prepared_output = {};
  uint2 prepared_image_size = {};
  if (prepare_batch_output_buffer(options, output, image_size, prepared_output, prepared_image_size) == false) {
    return false;
  }

  if (save_batch_output_file(options.output_file, prepared_output.data(), prepared_image_size, options.exposure) == false) {
    return false;
  }

  if (options.compare_mode == "render") {
    ImageComparisonResult comparison = {};
    if (compare_output_to_reference_and_save(options, options.reference_file, options.output_file, prepared_output.data(), prepared_image_size, &comparison) == false) {
      return false;
    }

    print_comparison_report("reference", "reference_compare", options.scene_file.c_str(), options.reference_file.c_str(), options.output_file.c_str(), comparison);
  }

  return true;
}

bool read_texture_to_float4_buffer(RHIContext& ctx, RHITexture texture, const uint2& image_size, std::vector<float4>& output) {
  output.clear();
  if ((texture.valid() == false) || (image_size.x == 0u) || (image_size.y == 0u)) {
    log::error("Invalid GPU output texture for readback");
    return false;
  }

  const uint64_t pixel_count = static_cast<uint64_t>(image_size.x) * static_cast<uint64_t>(image_size.y);
  const uint64_t buffer_size = pixel_count * sizeof(float4);
  RHIBufferDesc readback_desc = {};
  readback_desc.size = buffer_size;
  readback_desc.usage = RHIBufferUsage::TransferDst;
  readback_desc.host_visible = true;

  auto readback_result = ctx.device().create_buffer(readback_desc);
  if ((readback_result.result != RHIResult::Success) || (readback_result.handle.valid() == false)) {
    log::error("Failed to create GPU readback buffer (%u)", static_cast<uint32_t>(readback_result.result));
    return false;
  }

  ctx.begin_frame();
  auto cmd = ctx.get_command_buffer();
  if (cmd.valid() == false) {
    log::error("Failed to get command buffer for GPU readback");
    ctx.device().destroy_buffer(readback_result.handle);
    ctx.end_frame();
    return false;
  }

  ctx.command_buffer_begin(cmd);
  ctx.cmd_texture_barrier(cmd, texture, RHIResourceState::ShaderReadOnly, RHIResourceState::TransferSrc);
  ctx.cmd_copy_texture_to_buffer(cmd, texture, readback_result.handle, image_size.x, image_size.y);
  ctx.cmd_texture_barrier(cmd, texture, RHIResourceState::TransferSrc, RHIResourceState::ShaderReadOnly);
  ctx.command_buffer_end(cmd);
  ctx.submit_command_buffer({cmd});
  ctx.end_frame();

  const RHIResult wait_result = ctx.wait_idle();
  if (wait_result != RHIResult::Success) {
    log::error("GPU readback wait_idle failed (%u)", static_cast<uint32_t>(wait_result));
    ctx.destroy_command_buffer(cmd);
    ctx.device().destroy_buffer(readback_result.handle);
    return false;
  }
  ctx.destroy_command_buffer(cmd);

  output.resize(static_cast<size_t>(pixel_count));
  const RHIResult read_result = ctx.device().read_buffer(readback_result.handle, output.data(), buffer_size);
  const RHIResult destroy_result = ctx.device().destroy_buffer(readback_result.handle);
  if (destroy_result != RHIResult::Success) {
    log::warning("Failed to destroy GPU readback buffer (%u)", static_cast<uint32_t>(destroy_result));
  }

  if (read_result != RHIResult::Success) {
    log::error("Failed to read GPU output buffer (%u)", static_cast<uint32_t>(read_result));
    output.clear();
    return false;
  }

  return true;
}

bool load_scene_for_batch(const BatchRenderOptions& options, BatchRenderSession& session, Integrator*& selected_integrator, bool configure_cpu_renderer) {
  SceneRepresentation::IntegratorData integrator_data = {};
  const std::string absolute_scene_path = resolve_input_path(options.scene_file);
  if (session.scene.load_from_file(absolute_scene_path.c_str(), SceneRepresentation::LoadEverything, &integrator_data) == false) {
    log::error("Failed to load scene from file: %s", absolute_scene_path.c_str());
    return false;
  }

  if (session.scene.valid() == false) {
    log::error("Scene is invalid after loading: %s", absolute_scene_path.c_str());
    return false;
  }

  if (options.integrator.empty() == false) {
    const Integrator::Type requested_type = integrator_id_to_type(options.integrator.c_str());
    if (requested_type == Integrator::Type::Invalid) {
      log::error("Unknown integrator: %s", options.integrator.c_str());
      return false;
    }
    integrator_data.selected = requested_type;
  }

  session.scene.set_integrator_data(integrator_data);
  apply_batch_scene_overrides(options, session.scene);
  if (configure_batch_render_window(options, session) == false) {
    return false;
  }

  if (configure_cpu_renderer) {
    session.cpu_renderer.set_output_dimensions(session.render_context.context(), session.scene.camera().film_size);

    for (const auto& [type, options_data] : integrator_data.settings) {
      Integrator* integrator =
        integrator_type_to_instance(type, session.cpu_renderer.integrator_list(), session.cpu_renderer.integrator_count());
      if (integrator != nullptr) {
        integrator->sync_from_options(options_data);
        integrator->update_options();
      }
    }

    selected_integrator = select_integrator(options, session, integrator_data);
    if (selected_integrator == nullptr) {
      log::error("Failed to select an integrator for batch rendering");
      return false;
    }

    session.cpu_renderer.set_integrator(selected_integrator);
  } else {
    selected_integrator = nullptr;
  }

  return true;
}

bool load_scene_for_full_comparison(
  const BatchRenderOptions& options, BatchRenderSession& session, const FullComparisonTechniqueInfo& technique, Integrator*& selected_integrator,
  bool configure_cpu_renderer) {
  SceneRepresentation::IntegratorData integrator_data = {};
  const std::string absolute_scene_path = resolve_input_path(options.scene_file);
  if (session.scene.load_from_file(absolute_scene_path.c_str(), SceneRepresentation::LoadEverything, &integrator_data) == false) {
    log::error("Failed to load scene from file: %s", absolute_scene_path.c_str());
    return false;
  }

  if (session.scene.valid() == false) {
    log::error("Scene is invalid after loading: %s", absolute_scene_path.c_str());
    return false;
  }

  integrator_data.selected = Integrator::Type::Bidirectional;
  Options& bdpt_options = integrator_data.settings[Integrator::Type::Bidirectional];
  bdpt_options.set_integral("bdpt-mode", technique.bdpt_mode, "Mode", Option::Meta::EnumValue);

  session.scene.data().options.strategy_flags = technique.strategy_flags;

  session.scene.set_integrator_data(integrator_data);
  apply_batch_scene_overrides(options, session.scene);
  if (configure_batch_render_window(options, session) == false) {
    return false;
  }

  if (configure_cpu_renderer == false) {
    selected_integrator = nullptr;
    return true;
  }

  session.cpu_renderer.set_output_dimensions(session.render_context.context(), session.scene.camera().film_size);

  for (const auto& [type, options_data] : integrator_data.settings) {
    Integrator* integrator =
      integrator_type_to_instance(type, session.cpu_renderer.integrator_list(), session.cpu_renderer.integrator_count());
    if (integrator != nullptr) {
      integrator->sync_from_options(options_data);
      integrator->update_options();
    }
  }

  selected_integrator =
    integrator_type_to_instance(Integrator::Type::Bidirectional, session.cpu_renderer.integrator_list(), session.cpu_renderer.integrator_count());
  if (selected_integrator == nullptr) {
    log::error("Failed to select bidirectional integrator for full comparison");
    return false;
  }

  session.cpu_renderer.set_integrator(selected_integrator);
  return true;
}

bool configure_preloaded_scene_for_full_comparison(const BatchRenderOptions& options, BatchRenderSession& session,
  const SceneRepresentation::IntegratorData& base_integrator_data, const Scene::Options& base_scene_options, const Camera& base_camera,
  const FullComparisonTechniqueInfo& technique, Integrator*& selected_integrator) {
  SceneRepresentation::IntegratorData integrator_data = base_integrator_data;
  integrator_data.selected = Integrator::Type::Bidirectional;
  Options& bdpt_options = integrator_data.settings[Integrator::Type::Bidirectional];
  bdpt_options.set_integral("bdpt-mode", technique.bdpt_mode, "Mode", Option::Meta::EnumValue);

  session.scene.set_integrator_data(integrator_data);
  session.scene.data().options = base_scene_options;
  session.scene.mutable_camera() = base_camera;
  session.scene.data().options.strategy_flags = technique.strategy_flags;
  apply_batch_scene_overrides(options, session.scene);
  if (configure_batch_render_window(options, session) == false) {
    return false;
  }

  session.cpu_renderer.set_output_dimensions(session.render_context.context(), session.scene.camera().film_size);
  for (const auto& [type, options_data] : integrator_data.settings) {
    Integrator* integrator =
      integrator_type_to_instance(type, session.cpu_renderer.integrator_list(), session.cpu_renderer.integrator_count());
    if (integrator != nullptr) {
      integrator->sync_from_options(options_data);
      integrator->update_options();
    }
  }

  selected_integrator =
    integrator_type_to_instance(Integrator::Type::Bidirectional, session.cpu_renderer.integrator_list(), session.cpu_renderer.integrator_count());
  if (selected_integrator == nullptr) {
    log::error("Failed to select bidirectional integrator for full comparison");
    return false;
  }

  session.cpu_renderer.set_integrator(selected_integrator);
  session.gpu_renderer.cleanup(session.render_context.context());
  session.gpu_renderer.init(session.render_context.context(), session.scene);
  session.cpu_renderer.integrator_thread().request_scene_check();
  session.gpu_renderer.on_scene_changed(session.scene);
  return true;
}

bool run_cpu_preloaded_scene_to_buffer(const BatchRenderOptions& options, BatchRenderSession& session, std::vector<float4>& output, uint2& image_size) {
  ETX_ASSERT(session.cpu_renderer.current_integrator() != nullptr);

  session.cpu_renderer.film().clear(Film::ClearEverything);
  session.cpu_renderer.start();

  uint32_t last_completed_iterations = 0u;
  const uint32_t target_iterations = session.scene.data().options.samples;

  while (true) {
    session.cpu_renderer.integrator_thread().update();

    const Integrator::Status& status = session.cpu_renderer.integrator_thread().status();
    if (status.completed_iterations != last_completed_iterations) {
      log::info("Rendering progress: %u / %u", status.completed_iterations, target_iterations);
      last_completed_iterations = status.completed_iterations;
    }

    const bool render_finished = (session.cpu_renderer.is_running() == false) && (status.completed_iterations >= target_iterations);
    if (render_finished) {
      break;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  uint32_t output_layer = ViewLayer::Result;
  if (options.denoise) {
    session.cpu_renderer.film().denoise(ViewLayer::Result, session.rt.scene().options.radiance_clamp);
    output_layer = ViewLayer::Denoised;
  }

  const float4* layer = session.cpu_renderer.film().layer(output_layer, session.rt.scene().options.radiance_clamp);
  image_size = session.scene.camera().film_size;
  const size_t pixel_count = static_cast<size_t>(image_size.x) * static_cast<size_t>(image_size.y);
  output.assign(layer, layer + pixel_count);
  return true;
}

bool run_cpu_batch_render_to_buffer(const BatchRenderOptions& options, BatchRenderSession& session, std::vector<float4>& output, uint2& image_size) {
  if (options.renderer != "cpu") {
    log::error("Unsupported batch renderer: %s", options.renderer.c_str());
    return false;
  }

  Integrator* selected_integrator = nullptr;
  if (load_scene_for_batch(options, session, selected_integrator, true) == false) {
    return false;
  }

  ETX_ASSERT(selected_integrator != nullptr);
  return run_cpu_preloaded_scene_to_buffer(options, session, output, image_size);
}

bool run_cpu_batch_render(const BatchRenderOptions& options, BatchRenderSession& session) {
  std::vector<float4> output = {};
  uint2 image_size = {};
  if (run_cpu_batch_render_to_buffer(options, session, output, image_size) == false) {
    return false;
  }

  return save_batch_output_and_compare(options, output.data(), image_size);
}

bool run_gpu_preloaded_scene_to_buffer(const BatchRenderOptions& options, BatchRenderSession& session, std::vector<float4>& output, uint2& image_size) {
  const uint32_t gpu_frame_count = max(1u, session.scene.data().options.samples);
  for (uint32_t frame_index = 0u; frame_index < gpu_frame_count; ++frame_index) {
    session.render_context.begin_frame();
    Renderer::FrameData frame_data = {};
    frame_data.dt = 0.0f;
    session.gpu_renderer.render(session.render_context.context(), session.scene, frame_data);
    session.render_context.end_frame();

    log::info("GPU rendering progress: %u / %u", frame_index + 1u, gpu_frame_count);
  }

  image_size = session.gpu_renderer.output_size();
  if (read_texture_to_float4_buffer(session.render_context.context(), session.gpu_renderer.output_texture(), image_size, output) == false) {
    return false;
  }

  return true;
}

bool run_gpu_batch_render_to_buffer(const BatchRenderOptions& options, BatchRenderSession& session, std::vector<float4>& output, uint2& image_size) {
  if (options.renderer != "gpu") {
    log::error("Unsupported batch renderer: %s", options.renderer.c_str());
    return false;
  }

  if (options.denoise) {
    log::warning("Ignoring --denoise for GPU batch rendering");
  }

  Integrator* selected_integrator = nullptr;
  if (load_scene_for_batch(options, session, selected_integrator, false) == false) {
    return false;
  }

  return run_gpu_preloaded_scene_to_buffer(options, session, output, image_size);
}

bool run_gpu_batch_render(const BatchRenderOptions& options, BatchRenderSession& session) {
  std::vector<float4> output = {};
  uint2 image_size = {};
  if (run_gpu_batch_render_to_buffer(options, session, output, image_size) == false) {
    return false;
  }

  return save_batch_output_and_compare(options, output.data(), image_size);
}

bool run_gpu_shader_compile_test(const BatchRenderOptions& options, BatchRenderSession& session) {
  if (options.renderer != "gpu") {
    log::error("Unsupported batch renderer for GPU compile test: %s", options.renderer.c_str());
    return false;
  }

  Integrator* selected_integrator = nullptr;
  if (load_scene_for_batch(options, session, selected_integrator, false) == false) {
    return false;
  }

  session.gpu_renderer.reload_shaders(session.render_context.context());
  if (session.gpu_renderer.pipelines_valid() == false) {
    log::error("GPU shader compile test failed");
    return false;
  }

  log::info("GPU shader compile test succeeded");
  return true;
}

bool run_full_comparison_batch_render(const BatchRenderOptions& options) {
  if (options.denoise) {
    log::warning("Ignoring --denoise for full comparison mode");
  }

  const std::string absolute_scene_path = resolve_input_path(options.scene_file);
  const std::string report_file_name = full_comparison_report_file_name(absolute_scene_path);
  const std::string ai_report_file_name = full_comparison_ai_report_file_name(absolute_scene_path);
  std::string report_html = {};
  std::string ai_report_json = {};
  bool first_ai_result = true;
  append_full_comparison_html_header(report_html, absolute_scene_path);
  ai_report_json += "{\n";
  ai_report_json += "  \"schema\": \"etx.full_comparison.v1\",\n";
  ai_report_json += "  \"scene\": \"" + json_escape(absolute_scene_path) + "\",\n";
  ai_report_json += "  \"results\": [\n";
  printf("Full comparison scene: %s\n", absolute_scene_path.c_str());

  for (const FullComparisonTechniqueInfo& technique : kFullComparisonTechniques) {
    const std::string cpu_output_file = full_comparison_output_file_name(absolute_scene_path, technique.file_tag, "cpu");
    const std::string gpu_output_file = full_comparison_output_file_name(absolute_scene_path, technique.file_tag, "gpu");
    log::info("Full comparison '%s': loading scene in a fresh session and running CPU/GPU renders", technique.file_tag);

    BatchRenderSession session = {};
    if (options.gpu_compile_stage.empty() == false) {
      session.gpu_renderer.set_compile_stage_filter(options.gpu_compile_stage);
    }
    if (session.init(true, true) == false) {
      return false;
    }

    Integrator* selected_integrator = nullptr;
    if (load_scene_for_full_comparison(options, session, technique, selected_integrator, true) == false) {
      return false;
    }

    BatchRenderOptions cpu_options = options;
    cpu_options.renderer = "cpu";
    BatchRenderOptions gpu_options = options;
    gpu_options.renderer = "gpu";

    std::vector<float4> cpu_output = {};
    uint2 cpu_image_size = {};
    if (run_cpu_preloaded_scene_to_buffer(cpu_options, session, cpu_output, cpu_image_size) == false) {
      return false;
    }

    std::vector<float4> gpu_output = {};
    uint2 gpu_image_size = {};
    if (run_gpu_preloaded_scene_to_buffer(gpu_options, session, gpu_output, gpu_image_size) == false) {
      return false;
    }

    std::vector<float4> prepared_cpu_output = {};
    uint2 prepared_cpu_image_size = {};
    if (prepare_batch_output_buffer(options, cpu_output.data(), cpu_image_size, prepared_cpu_output, prepared_cpu_image_size) == false) {
      return false;
    }

    std::vector<float4> prepared_gpu_output = {};
    uint2 prepared_gpu_image_size = {};
    if (prepare_batch_output_buffer(options, gpu_output.data(), gpu_image_size, prepared_gpu_output, prepared_gpu_image_size) == false) {
      return false;
    }

    if ((prepared_cpu_image_size.x != prepared_gpu_image_size.x) || (prepared_cpu_image_size.y != prepared_gpu_image_size.y)) {
      log::error("Full comparison size mismatch for '%s': cpu=%ux%u gpu=%ux%u", technique.file_tag, prepared_cpu_image_size.x, prepared_cpu_image_size.y,
        prepared_gpu_image_size.x, prepared_gpu_image_size.y);
      return false;
    }

    ImageComparisonResult comparison = {};
    if (save_batch_output_with_png(cpu_output_file, prepared_cpu_output.data(), prepared_cpu_image_size, options.exposure) == false) {
      return false;
    }
    if (save_batch_output_with_png(gpu_output_file, prepared_gpu_output.data(), prepared_gpu_image_size, options.exposure) == false) {
      return false;
    }
    if (compare_output_to_reference_and_save(options, cpu_output_file, gpu_output_file, prepared_gpu_output.data(), prepared_gpu_image_size, &comparison) == false) {
      return false;
    }

    print_comparison_report(
      technique.file_tag, "full_comparison", absolute_scene_path.c_str(), cpu_output_file.c_str(), gpu_output_file.c_str(), comparison);
    append_full_comparison_html_entry(report_html, technique, cpu_output_file, gpu_output_file, comparison);
    if (first_ai_result == false) {
      ai_report_json += ",\n";
    }
    append_ai_comparison_json(
      ai_report_json, "full_comparison", technique.file_tag, absolute_scene_path.c_str(), cpu_output_file.c_str(), gpu_output_file.c_str(), comparison);
    first_ai_result = false;
  }

  append_full_comparison_html_footer(report_html);
  ai_report_json += "\n  ]\n";
  ai_report_json += "}\n";
  if (save_text_to_file(report_file_name, report_html) == false) {
    return false;
  }
  if (save_text_to_file(ai_report_file_name, ai_report_json) == false) {
    return false;
  }

  log::info("Saved comparison report to %s", report_file_name.c_str());
  log::info("Saved AI comparison report to %s", ai_report_file_name.c_str());
  return true;
}

}  // namespace

BatchModeCommand parse_batch_command_line(int argc, char* argv[], BatchRenderOptions& options, std::string& message) {
  bool render_requested = false;
  bool full_comparison_requested = false;
  bool batch_argument_seen = false;

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];

    if ((argument == "--help") || (argument == "-h")) {
      message = batch_usage_string();
      return BatchModeCommand::Help;
    }

    if (argument == "--render") {
      render_requested = true;
      batch_argument_seen = true;
      continue;
    }

    if (argument == "--full-comparison") {
      full_comparison_requested = true;
      options.full_comparison = true;
      batch_argument_seen = true;
      continue;
    }

    if (argument == "--scene") {
      batch_argument_seen = true;
      if ((i + 1) >= argc) {
        message = "Missing value for --scene\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      options.scene_file = argv[++i];
      continue;
    }

    if (argument == "--output") {
      batch_argument_seen = true;
      if ((i + 1) >= argc) {
        message = "Missing value for --output\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      options.output_file = argv[++i];
      continue;
    }

    if (argument == "--reference") {
      batch_argument_seen = true;
      if ((i + 1) >= argc) {
        message = "Missing value for --reference\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      options.reference_file = argv[++i];
      continue;
    }

    if (argument == "--integrator") {
      batch_argument_seen = true;
      if ((i + 1) >= argc) {
        message = "Missing value for --integrator\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      options.integrator = argv[++i];
      continue;
    }

    if (argument == "--renderer") {
      batch_argument_seen = true;
      if ((i + 1) >= argc) {
        message = "Missing value for --renderer\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      options.renderer = argv[++i];
      continue;
    }

    if (argument == "--samples") {
      batch_argument_seen = true;
      if ((i + 1) >= argc) {
        message = "Missing value for --samples\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      if (parse_u32_argument(argv[i + 1], options.samples) == false) {
        message = "Invalid value for --samples\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      i += 1;
      continue;
    }

    if (argument == "--max-path-length") {
      batch_argument_seen = true;
      if ((i + 1) >= argc) {
        message = "Missing value for --max-path-length\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      if (parse_u32_argument(argv[i + 1], options.max_path_length) == false) {
        message = "Invalid value for --max-path-length\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      if (options.max_path_length == 0u) {
        message = "--max-path-length must be greater than zero\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      i += 1;
      continue;
    }

    if (argument == "--random-seed") {
      batch_argument_seen = true;
      if ((i + 1) >= argc) {
        message = "Missing value for --random-seed\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      if (parse_u32_argument(argv[i + 1], options.random_seed) == false) {
        message = "Invalid value for --random-seed\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      options.override_random_seed = true;
      i += 1;
      continue;
    }

    if (argument == "--resolution") {
      batch_argument_seen = true;
      if ((i + 1) >= argc) {
        message = "Missing value for --resolution\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      if (parse_resolution_argument(argv[i + 1], options.resolution_width, options.resolution_height) == false) {
        message = "Invalid value for --resolution. Expected <width>x<height>\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      options.override_resolution = true;
      i += 1;
      continue;
    }

    if (argument == "--crop") {
      batch_argument_seen = true;
      if ((i + 1) >= argc) {
        message = "Missing value for --crop\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      if (parse_crop_argument(argv[i + 1], options.crop_x, options.crop_y, options.crop_width, options.crop_height) == false) {
        message = "Invalid value for --crop. Expected <x>,<y>,<width>,<height>\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      options.override_crop = true;
      i += 1;
      continue;
    }

    if (argument == "--strategy-flags") {
      batch_argument_seen = true;
      if ((i + 1) >= argc) {
        message = "Missing value for --strategy-flags\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      if (parse_strategy_flags_argument(argv[i + 1], options.strategy_flags) == false) {
        message = "Invalid value for --strategy-flags\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      options.override_strategy_flags = true;
      i += 1;
      continue;
    }

    if (argument == "--gpu-compile-only") {
      batch_argument_seen = true;
      options.gpu_compile_only = true;
      continue;
    }

    if (argument == "--gpu-compile-stage") {
      batch_argument_seen = true;
      if ((i + 1) >= argc) {
        message = "Missing value for --gpu-compile-stage\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      options.gpu_compile_stage = argv[++i];
      continue;
    }

    if (argument == "--exposure") {
      batch_argument_seen = true;
      if ((i + 1) >= argc) {
        message = "Missing value for --exposure\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      if (parse_f32_argument(argv[i + 1], options.exposure) == false) {
        message = "Invalid value for --exposure\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      i += 1;
      continue;
    }

    if (argument == "--compare") {
      batch_argument_seen = true;
      if ((i + 1) >= argc) {
        message = "Missing value for --compare\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }

      options.compare_mode = argv[++i];
      if (options.compare_mode != "render") {
        message = "Unsupported value for --compare. Expected: render\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      continue;
    }

    if (argument == "--denoise") {
      batch_argument_seen = true;
      options.denoise = true;
      continue;
    }

    message = "Unknown argument: " + argument + "\n\n";
    message += batch_usage_string();
    return BatchModeCommand::Error;
  }

  if (render_requested && full_comparison_requested) {
    message = "Use either --render or --full-comparison, not both\n\n";
    message += batch_usage_string();
    return BatchModeCommand::Error;
  }

  if ((render_requested == false) && (full_comparison_requested == false)) {
    if (batch_argument_seen) {
      message = "Batch render options require --render or --full-comparison\n\n";
      message += batch_usage_string();
      return BatchModeCommand::Error;
    }
    return BatchModeCommand::None;
  }

  if (full_comparison_requested) {
    if (options.scene_file.empty()) {
      message = "Full comparison requires --scene\n\n";
      message += batch_usage_string();
      return BatchModeCommand::Error;
    }

    if ((options.output_file.empty() == false) || (options.reference_file.empty() == false) || (options.compare_mode.empty() == false) ||
        (options.renderer != "cpu") || (options.integrator.empty() == false) || options.gpu_compile_only) {
      message =
        "--full-comparison does not accept --output, --reference, --compare, --renderer, --integrator, or --gpu-compile-only\n\n";
      message += batch_usage_string();
      return BatchModeCommand::Error;
    }

    return BatchModeCommand::Run;
  }

  if ((options.scene_file.empty()) || (options.output_file.empty())) {
    message = "Batch rendering requires --scene and --output\n\n";
    message += batch_usage_string();
    return BatchModeCommand::Error;
  }

  if ((options.compare_mode.empty() == false) && (options.reference_file.empty())) {
    message = "Image comparison requires --reference\n\n";
    message += batch_usage_string();
    return BatchModeCommand::Error;
  }

  return BatchModeCommand::Run;
}

int run_batch_render(const BatchRenderOptions& options) {
  if (options.full_comparison) {
    return run_full_comparison_batch_render(options) ? 0 : 1;
  }

  BatchRenderSession session = {};
  if (options.gpu_compile_stage.empty() == false) {
    session.gpu_renderer.set_compile_stage_filter(options.gpu_compile_stage);
  }
  const bool initialize_cpu_renderer = options.renderer == "cpu";
  const bool initialize_gpu_renderer = (options.renderer == "gpu") || options.gpu_compile_only;
  if (session.init(initialize_cpu_renderer, initialize_gpu_renderer) == false) {
    return 1;
  }

  if (options.gpu_compile_only) {
    return run_gpu_shader_compile_test(options, session) ? 0 : 1;
  }

  if (options.renderer == "cpu") {
    if (run_cpu_batch_render(options, session) == false) {
      return 1;
    }
    return 0;
  }

  if (options.renderer == "gpu") {
    if (run_gpu_batch_render(options, session) == false) {
      return 1;
    }
    return 0;
  }

  log::error("Unsupported batch renderer: %s", options.renderer.c_str());
  return 1;
}

}  // namespace etx
