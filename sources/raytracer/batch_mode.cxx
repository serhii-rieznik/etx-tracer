#include "batch_mode.hxx"

#include <etx/core/environment.hxx>
#include <etx/core/log.hxx>
#include <etx/render/host/film.hxx>
#include <etx/render/host/scene_global.hxx>
#include <etx/render/host/scene_representation.hxx>
#include <etx/render/shared/ior_database.hxx>
#include <etx/rt/integrators/integrator.hxx>
#include <etx/rt/shared/bdpt_mode.hxx>
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
#include <limits>
#include <thread>

namespace etx {

namespace {

double elapsed_ms(const std::chrono::steady_clock::time_point& begin, const std::chrono::steady_clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

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
    const std::string token = (token_end == std::string::npos) ? normalized_text.substr(token_begin) : normalized_text.substr(token_begin, token_end - token_begin);
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

bool parse_bdpt_mode_argument(const char* value, uint32_t& result) {
  if ((value == nullptr) || (value[0] == 0)) {
    return false;
  }

  const std::string mode = normalize_strategy_flag_name(value);
  if ((mode == "pt") || (mode == "path_tracing") || (mode == "pathtracing")) {
    result = static_cast<uint32_t>(BDPTMode::PathTracing);
    return true;
  }
  if ((mode == "lt") || (mode == "light_tracing") || (mode == "lighttracing")) {
    result = static_cast<uint32_t>(BDPTMode::LightTracing);
    return true;
  }
  if ((mode == "bdpt_fast") || (mode == "bdptfast")) {
    result = static_cast<uint32_t>(BDPTMode::BDPTFast);
    return true;
  }
  if ((mode == "bdpt_full") || (mode == "bdptfull")) {
    result = static_cast<uint32_t>(BDPTMode::BDPTFull);
    return true;
  }

  return false;
}

struct FullComparisonTechniqueInfo {
  const char* file_tag = "";
  BDPTMode bdpt_mode = BDPTMode::PathTracing;
  uint32_t strategy_flags = 0u;
};

struct CPUComparisonTechniqueInfo {
  const char* file_tag = "";
  const char* display_name = "";
  const char* description = "";
  Integrator::Type integrator = Integrator::Type::Invalid;
  BDPTMode bdpt_mode = BDPTMode::PathTracing;
  uint32_t strategy_flags = 0u;
  bool reference = false;
};

const FullComparisonTechniqueInfo kFullComparisonTechniques[] = {
  {
    .file_tag = "pt",
    .bdpt_mode = BDPTMode::PathTracing,
    .strategy_flags = Scene::Strategy::DirectHit | Scene::Strategy::ConnectToLight,
  },
  {
    .file_tag = "pt-direct-hit",
    .bdpt_mode = BDPTMode::PathTracing,
    .strategy_flags = Scene::Strategy::DirectHit,
  },
  {
    .file_tag = "pt-connect-to-light",
    .bdpt_mode = BDPTMode::PathTracing,
    .strategy_flags = Scene::Strategy::ConnectToLight,
  },
  {
    .file_tag = "lt",
    .bdpt_mode = BDPTMode::LightTracing,
    .strategy_flags = Scene::Strategy::ConnectToCamera,
  },
  {
    .file_tag = "lt-connect-to-camera",
    .bdpt_mode = BDPTMode::LightTracing,
    .strategy_flags = Scene::Strategy::ConnectToCamera,
  },
  {
    .file_tag = "bdpt-fast",
    .bdpt_mode = BDPTMode::BDPTFast,
    .strategy_flags = Scene::Strategy::DirectHit | Scene::Strategy::ConnectToLight | Scene::Strategy::ConnectToCamera,
  },
  {
    .file_tag = "bdpt-fast-direct-hit",
    .bdpt_mode = BDPTMode::BDPTFast,
    .strategy_flags = Scene::Strategy::DirectHit,
  },
  {
    .file_tag = "bdpt-fast-connect-to-light",
    .bdpt_mode = BDPTMode::BDPTFast,
    .strategy_flags = Scene::Strategy::ConnectToLight,
  },
  {
    .file_tag = "bdpt-fast-connect-to-camera",
    .bdpt_mode = BDPTMode::BDPTFast,
    .strategy_flags = Scene::Strategy::ConnectToCamera,
  },
  {
    .file_tag = "bdpt-full",
    .bdpt_mode = BDPTMode::BDPTFull,
    .strategy_flags = Scene::Strategy::DirectHit | Scene::Strategy::ConnectToLight | Scene::Strategy::ConnectToCamera | Scene::Strategy::ConnectVertices,
  },
  {
    .file_tag = "bdpt-full-direct-hit",
    .bdpt_mode = BDPTMode::BDPTFull,
    .strategy_flags = Scene::Strategy::DirectHit,
  },
  {
    .file_tag = "bdpt-full-connect-to-light",
    .bdpt_mode = BDPTMode::BDPTFull,
    .strategy_flags = Scene::Strategy::ConnectToLight,
  },
  {
    .file_tag = "bdpt-full-connect-to-camera",
    .bdpt_mode = BDPTMode::BDPTFull,
    .strategy_flags = Scene::Strategy::ConnectToCamera,
  },
  {
    .file_tag = "bdpt-full-connect-vertices",
    .bdpt_mode = BDPTMode::BDPTFull,
    .strategy_flags = Scene::Strategy::ConnectVertices,
  },
};

const CPUComparisonTechniqueInfo kCPUComparisonTechniques[] = {
  {
    .file_tag = "pt",
    .display_name = "PT",
    .description = "Standalone CPU path tracing reference.",
    .integrator = Integrator::Type::PathTracing,
    .bdpt_mode = BDPTMode::PathTracing,
    .strategy_flags = Scene::Strategy::DirectHit | Scene::Strategy::ConnectToLight,
    .reference = true,
  },
  {
    .file_tag = "bdpt-pt",
    .display_name = "BDPT PT",
    .description = "Bidirectional integrator in path-tracing mode.",
    .integrator = Integrator::Type::Bidirectional,
    .bdpt_mode = BDPTMode::PathTracing,
    .strategy_flags = Scene::Strategy::DirectHit | Scene::Strategy::ConnectToLight,
    .reference = false,
  },
  {
    .file_tag = "bdpt-lt",
    .display_name = "BDPT LT",
    .description = "Bidirectional integrator in light-tracing mode.",
    .integrator = Integrator::Type::Bidirectional,
    .bdpt_mode = BDPTMode::LightTracing,
    .strategy_flags = Scene::Strategy::ConnectToCamera,
    .reference = false,
  },
  {
    .file_tag = "bdpt-fast",
    .display_name = "BDPT Fast",
    .description = "Bidirectional path tracing with fast MIS precomputation.",
    .integrator = Integrator::Type::Bidirectional,
    .bdpt_mode = BDPTMode::BDPTFast,
    .strategy_flags = Scene::Strategy::DirectHit | Scene::Strategy::ConnectToLight | Scene::Strategy::ConnectToCamera | Scene::Strategy::ConnectVertices,
    .reference = false,
  },
  {
    .file_tag = "bdpt-full",
    .display_name = "BDPT Full",
    .description = "Full bidirectional path tracing with complete vertex connections.",
    .integrator = Integrator::Type::Bidirectional,
    .bdpt_mode = BDPTMode::BDPTFull,
    .strategy_flags = Scene::Strategy::DirectHit | Scene::Strategy::ConnectToLight | Scene::Strategy::ConnectToCamera | Scene::Strategy::ConnectVertices,
    .reference = false,
  },
  {
    .file_tag = "vcm",
    .display_name = "VCM",
    .description = "Vertex connection and merging CPU integrator.",
    .integrator = Integrator::Type::VCM,
    .bdpt_mode = BDPTMode::BDPTFast,
    .strategy_flags = Scene::Strategy::DirectHit | Scene::Strategy::ConnectToLight | Scene::Strategy::ConnectToCamera | Scene::Strategy::ConnectVertices |
                      Scene::Strategy::MergeVertices,
    .reference = false,
  },
};

const char* full_comparison_group_name(const BDPTMode mode) {
  switch (mode) {
    case BDPTMode::PathTracing:
      return "Path Tracing";
    case BDPTMode::LightTracing:
      return "Light Tracing";
    case BDPTMode::BDPTFast:
      return "BDPT Fast";
    case BDPTMode::BDPTFull:
      return "BDPT Full";
    default:
      return "Unknown";
  }
}

const char* batch_usage_string() {
  return "Usage:\n"
         "  raytracer --render --scene <scene-file> --output <output-file> [options]\n"
         "  raytracer --full-comparison --scene <scene-file> [options]\n"
         "  raytracer --cpu-comparison --scene <scene-file> [options]\n"
         "  raytracer --generate-bsdf-luts [--output <output-directory>] [options]\n"
         "  raytracer --pregenerate-bsdf-lut-cache\n"
         "\n"
         "Options:\n"
         "  --full-comparison\n"
         "  --cpu-comparison\n"
         "  --generate-bsdf-luts\n"
         "  --pregenerate-bsdf-lut-cache\n"
         "  --integrator <debug|pt|bdpt|vcm>\n"
         "  --bdpt-mode <pt|lt|bdpt-fast|bdpt-full>\n"
         "  --renderer <cpu|gpu>\n"
         "  --samples <count>\n"
         "  --bsdf-lut-samples <count>\n"
         "  --max-path-length <count>\n"
         "  --random-seed <value>\n"
         "  --resolution <width>x<height>\n"
         "  --crop <x>,<y>,<width>,<height>\n"
         "  --strategy-flags <flag[,flag...]>\n"
         "  --strict-comparison\n"
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

std::string cpu_comparison_output_file_name(const std::string& scene_file, const char* technique_tag) {
  const std::filesystem::path scene_path(resolve_input_path(scene_file));
  const std::filesystem::path output_directory = scene_path.parent_path() / (scene_path.stem().generic_string() + ".cpu-comparison");
  std::string result = output_directory.generic_string();
  result += "/";
  result += technique_tag;
  result += ".cpu.exr";
  return result;
}

std::string cpu_comparison_report_file_name(const std::string& scene_file) {
  const std::filesystem::path scene_path(resolve_input_path(scene_file));
  const std::filesystem::path output_directory = scene_path.parent_path() / (scene_path.stem().generic_string() + ".cpu-comparison");
  return (output_directory / "comparison.html").generic_string();
}

std::string cpu_comparison_ai_report_file_name(const std::string& scene_file) {
  const std::filesystem::path scene_path(resolve_input_path(scene_file));
  const std::filesystem::path output_directory = scene_path.parent_path() / (scene_path.stem().generic_string() + ".cpu-comparison");
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

std::string comparison_png_file_name_from_output_file(const std::string& output_file) {
  std::filesystem::path path(comparison_file_name_from_output(output_file));
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
    return linear_space ? "Overall similarity score derived from linear-space RMSE. Higher is better. 100% means identical images."
                        : "Overall similarity score derived from compare-space RMSE. Higher is better. 100% means identical images.";
  }
  if (strcmp(label, "Low-Freq Similarity") == 0) {
    return linear_space
             ? "Low-frequency similarity is currently reported only for compare space."
             : "Similarity derived from compare-space RMSE after a small Gaussian blur. Higher is better. Useful when images are visually identical apart from noise pattern.";
  }
  if (strcmp(label, "RMSE") == 0) {
    return linear_space ? "Root mean squared error in linear HDR space. Penalizes larger errors more strongly than MAE. Lower is better."
                        : "Root mean squared error in compare space. Penalizes larger errors more strongly than MAE. Lower is better.";
  }
  if (strcmp(label, "Low-Freq RMSE") == 0) {
    return linear_space ? "Low-frequency RMSE is currently reported only for compare space."
                        : "Compare-space RMSE after a small Gaussian blur. Lower is better. Useful for separating structure mismatch from different noise realization.";
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
    return linear_space ? "Average luminance of the CPU reference image in linear HDR space." : "Average luminance of the CPU reference image in compare space.";
  }
  if (strcmp(label, "GPU Avg Luma") == 0) {
    return linear_space ? "Average luminance of the GPU result image in linear HDR space." : "Average luminance of the GPU result image in compare space.";
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
  char buffer[1792] = {};
  std::snprintf(buffer, sizeof(buffer),
    "[%s] compare_space{similarity=%.2f%%, low_freq_similarity=%.2f%%, rmse=%.6f, low_freq_rmse=%.6f, mae=%.6f, relative_rmse=%.6f, max_abs=%.6f, "
    "mean_signed=%.6f, ref_avg_luma=%.6f, gpu_avg_luma=%.6f, brightness_ratio=%.6f, brightness_rel=%.6f, p95_abs=%.6f, p99_abs=%.6f} linear{rmse=%.6f, mae=%.6f, relative_rmse=%.6f, max_abs=%.6f, "
    "mean_signed=%.6f, ref_avg_luma=%.6f, gpu_avg_luma=%.6f, brightness_ratio=%.6f, brightness_rel=%.6f, p95_abs=%.6f, p99_abs=%.6f}\n",
    technique_tag, comparison.similarity, comparison.low_frequency_similarity, comparison.root_mean_squared_error, comparison.low_frequency_root_mean_squared_error,
    comparison.mean_absolute_error, comparison.relative_root_mean_squared_error, comparison.max_absolute_error, comparison.mean_signed_error,
    comparison.reference_mean_luminance, comparison.result_mean_luminance, comparison.brightness_ratio, comparison.brightness_relative_error,
    comparison.percentile_95_absolute_error, comparison.percentile_99_absolute_error, comparison.linear_root_mean_squared_error, comparison.linear_mean_absolute_error,
    comparison.linear_relative_root_mean_squared_error, comparison.linear_max_absolute_error, comparison.linear_mean_signed_error, comparison.linear_reference_mean_luminance,
    comparison.linear_result_mean_luminance, comparison.linear_brightness_ratio, comparison.linear_brightness_relative_error, comparison.linear_percentile_95_absolute_error,
    comparison.linear_percentile_99_absolute_error);
  return buffer;
}

std::string format_ai_comparison_report(const char* kind, const char* technique_tag, const char* scene_file, const char* reference_file, const char* output_file,
  const ImageComparisonResult& comparison) {
  const std::string safe_kind = json_escape(kind != nullptr ? kind : "");
  const std::string safe_technique = json_escape(technique_tag != nullptr ? technique_tag : "");
  const std::string safe_scene = json_escape(scene_file != nullptr ? scene_file : "");
  const std::string safe_reference = json_escape(reference_file != nullptr ? reference_file : "");
  const std::string safe_output = json_escape(output_file != nullptr ? output_file : "");

  char buffer[3584] = {};
  std::snprintf(buffer, sizeof(buffer),
    "AI_IMAGE_COMPARISON {\"schema\":\"etx.image_comparison.v3\",\"kind\":\"%s\",\"technique\":\"%s\",\"scene\":\"%s\",\"reference\":\"%s\","
    "\"output\":\"%s\",\"compare_space\":{\"similarity_percent\":%.6f,\"low_frequency_similarity_percent\":%.6f,\"rmse\":%.6f,\"low_frequency_rmse\":%.6f,\"mae\":%.6f,\"relative_rmse\":%.6f,\"max_abs\":%.6f,"
    "\"mean_signed\":%.6f,\"ref_avg_luma\":%.6f,\"gpu_avg_luma\":%.6f,\"brightness_ratio\":%.6f,\"brightness_rel\":%.6f,\"p95_abs\":%.6f,\"p99_abs\":%.6f},"
    "\"linear\":{\"rmse\":%.6f,\"mae\":%.6f,\"relative_rmse\":%.6f,\"max_abs\":%.6f,\"mean_signed\":%.6f,\"ref_avg_luma\":%.6f,\"gpu_avg_luma\":%.6f,"
    "\"brightness_ratio\":%.6f,\"brightness_rel\":%.6f,\"p95_abs\":%.6f,\"p99_abs\":%.6f}}\n",
    safe_kind.c_str(), safe_technique.c_str(), safe_scene.c_str(), safe_reference.c_str(), safe_output.c_str(), comparison.similarity,
    comparison.low_frequency_similarity, comparison.root_mean_squared_error, comparison.low_frequency_root_mean_squared_error, comparison.mean_absolute_error,
    comparison.relative_root_mean_squared_error, comparison.max_absolute_error, comparison.mean_signed_error, comparison.reference_mean_luminance,
    comparison.result_mean_luminance, comparison.brightness_ratio, comparison.brightness_relative_error, comparison.percentile_95_absolute_error,
    comparison.percentile_99_absolute_error, comparison.linear_root_mean_squared_error, comparison.linear_mean_absolute_error, comparison.linear_relative_root_mean_squared_error,
    comparison.linear_max_absolute_error, comparison.linear_mean_signed_error, comparison.linear_reference_mean_luminance, comparison.linear_result_mean_luminance,
    comparison.linear_brightness_ratio, comparison.linear_brightness_relative_error, comparison.linear_percentile_95_absolute_error,
    comparison.linear_percentile_99_absolute_error);
  return buffer;
}

void append_ai_comparison_json(std::string& json_text, const char* kind, const char* technique_tag, const char* scene_file, const char* reference_file, const char* output_file,
  const ImageComparisonResult& comparison) {
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
  json_text += "        \"low_frequency_similarity_percent\": " + std::to_string(comparison.low_frequency_similarity) + ",\n";
  json_text += "        \"rmse\": " + std::to_string(comparison.root_mean_squared_error) + ",\n";
  json_text += "        \"low_frequency_rmse\": " + std::to_string(comparison.low_frequency_root_mean_squared_error) + ",\n";
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

void print_comparison_report(const char* technique_tag, const char* kind, const char* scene_file, const char* reference_file, const char* output_file,
  const ImageComparisonResult& comparison) {
  const std::string line = format_comparison_report(technique_tag, comparison);
  printf("%s", line.c_str());
  const std::string ai_line = format_ai_comparison_report(kind, technique_tag, scene_file, reference_file, output_file, comparison);
  printf("%s", ai_line.c_str());
}

bool full_comparison_technique_is_exact_gate(const char* technique_tag) {
  (void)technique_tag;
  return false;
}

struct FullComparisonExactGateThresholds {
  float linear_max_absolute_error = 1.0e-4f;
  float linear_root_mean_squared_error = 1.0e-5f;
  float low_frequency_root_mean_squared_error = 1.0e-5f;
};

FullComparisonExactGateThresholds full_comparison_exact_gate_thresholds(const char* technique_tag) {
  FullComparisonExactGateThresholds result = {};
  if (std::strcmp(technique_tag, "lt") == 0) {
    result.linear_max_absolute_error = 2.5e-4f;
    result.linear_root_mean_squared_error = 2.0e-5f;
  }
  return result;
}

float full_comparison_linear_brightness_error(const ImageComparisonResult& comparison) {
  constexpr float kBlackLuminanceEpsilon = 1.0e-6f;
  if ((fabsf(comparison.linear_reference_mean_luminance) <= kBlackLuminanceEpsilon) && (fabsf(comparison.linear_result_mean_luminance) <= kBlackLuminanceEpsilon)) {
    return 0.0f;
  }
  return fabsf(comparison.linear_brightness_ratio - 1.0f);
}

bool full_comparison_is_low_signal_reference(const ImageComparisonResult& comparison) {
  constexpr float kLowSignalReferenceLuminance = 1.0e-4f;
  return fabsf(comparison.linear_reference_mean_luminance) <= kLowSignalReferenceLuminance;
}

bool full_comparison_passes_strict_gate(const char* technique_tag, const ImageComparisonResult& comparison) {
  if (full_comparison_technique_is_exact_gate(technique_tag)) {
    const FullComparisonExactGateThresholds thresholds = full_comparison_exact_gate_thresholds(technique_tag);
    return (comparison.linear_max_absolute_error <= thresholds.linear_max_absolute_error) &&
           (comparison.linear_root_mean_squared_error <= thresholds.linear_root_mean_squared_error) &&
           (comparison.low_frequency_root_mean_squared_error <= thresholds.low_frequency_root_mean_squared_error);
  }

  const float brightness_error = full_comparison_linear_brightness_error(comparison);
  const bool low_signal_reference = full_comparison_is_low_signal_reference(comparison);
  const bool relative_metrics_pass = low_signal_reference || ((comparison.linear_relative_root_mean_squared_error <= 5.0e-1f) && (brightness_error <= 5.0e-2f));
  return (comparison.low_frequency_root_mean_squared_error <= 2.0e-2f) && (comparison.root_mean_squared_error <= 6.0e-2f) &&
         (comparison.percentile_95_absolute_error <= 1.5e-1f) && (comparison.percentile_99_absolute_error <= 4.0e-1f) &&
         relative_metrics_pass && (comparison.linear_percentile_95_absolute_error <= 2.0e-1f) && (comparison.linear_percentile_99_absolute_error <= 9.0e-1f) &&
         (comparison.linear_max_absolute_error <= 4.0f);
}

std::string full_comparison_strict_gate_message(const char* technique_tag, const ImageComparisonResult& comparison) {
  char buffer[768] = {};
  if (full_comparison_technique_is_exact_gate(technique_tag)) {
    const FullComparisonExactGateThresholds thresholds = full_comparison_exact_gate_thresholds(technique_tag);
    std::snprintf(buffer, sizeof(buffer),
      "%s failed exact gate: linear.max_abs=%.8f <= %.8f, linear.rmse=%.8f <= %.8f, low_freq.rmse=%.8f <= %.8f", technique_tag,
      comparison.linear_max_absolute_error, thresholds.linear_max_absolute_error, comparison.linear_root_mean_squared_error,
      thresholds.linear_root_mean_squared_error, comparison.low_frequency_root_mean_squared_error, thresholds.low_frequency_root_mean_squared_error);
  } else {
    const float brightness_error = full_comparison_linear_brightness_error(comparison);
    const bool low_signal_reference = full_comparison_is_low_signal_reference(comparison);
    std::snprintf(buffer, sizeof(buffer),
      "%s failed stochastic gate: low_freq.rmse=%.8f <= 0.02000000, compare.rmse=%.8f <= 0.06000000, linear.rel_rmse=%.8f <= 0.50000000, "
      "compare.p95=%.8f <= 0.15000000, compare.p99=%.8f <= 0.40000000, linear.p95=%.8f <= 0.20000000, linear.p99=%.8f <= 0.90000000, "
      "linear.max_abs=%.8f <= 4.00000000, linear.brightness_error=%.8f <= 0.05000000, low_signal_reference=%u",
      technique_tag, comparison.low_frequency_root_mean_squared_error, comparison.root_mean_squared_error, comparison.linear_relative_root_mean_squared_error,
      comparison.percentile_95_absolute_error, comparison.percentile_99_absolute_error, comparison.linear_percentile_95_absolute_error,
      comparison.linear_percentile_99_absolute_error, comparison.linear_max_absolute_error, brightness_error, low_signal_reference ? 1u : 0u);
  }
  return buffer;
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

bool resolve_batch_output_window(const BatchRenderOptions& options, const uint2& image_size, uint32_t& window_x, uint32_t& window_y, uint32_t& window_width,
  uint32_t& window_height) {
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
    log::error("Crop window is outside image bounds: crop=%u,%u,%u,%u image=%ux%u", options.crop_x, options.crop_y, options.crop_width, options.crop_height, image_size.x,
      image_size.y);
    return false;
  }

  window_x = options.crop_x;
  window_y = options.crop_y;
  window_width = options.crop_width;
  window_height = options.crop_height;
  return true;
}

bool prepare_batch_output_buffer(const BatchRenderOptions& options, const float4* input, const uint2& input_size, std::vector<float4>& output, uint2& output_size) {
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

bool compare_output_to_reference_and_save(const BatchRenderOptions& options, const std::string& reference_file, const std::string& output_file, const float4* output,
  const uint2& image_size, ImageComparisonResult* out_comparison) {
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

  const std::string difference_png_file_name = comparison_png_file_name_from_output_file(output_file);
  if (save_image_to_file(difference_png_file_name, difference_image.data(), image_size, difference_params) == false) {
    return false;
  }

  if (out_comparison != nullptr) {
    *out_comparison = comparison;
  }

  return true;
}

void append_full_comparison_html_header(std::string& html_text, const std::string& scene_file) {
  html_text += R"html(<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ETX Full Comparison</title>
  <style>
    :root { color-scheme: dark; --bg: #0b1016; --bg-2: #121a24; --panel: rgba(18, 27, 37, 0.86); --panel-strong: rgba(24, 35, 47, 0.94); --panel-soft: rgba(12, 18, 26, 0.76); --text: #eff5fb; --muted: #98aabd; --accent: #77d9ff; --accent-strong: #d5f5ff; --warm: #f1c978; --line: rgba(146, 167, 188, 0.18); --line-strong: rgba(146, 167, 188, 0.3); --shadow: 0 30px 80px rgba(0, 0, 0, 0.34); }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body { margin: 0; min-height: 100vh; font-family: "Cascadia Mono", "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-weight: 400; background: radial-gradient(circle at 18% 0%, rgba(119, 217, 255, 0.16), transparent 28%), radial-gradient(circle at 100% 0%, rgba(241, 201, 120, 0.08), transparent 24%), linear-gradient(180deg, #101720 0%, #0b1016 100%); color: var(--text); }
    body::before { content: ""; position: fixed; inset: 0; pointer-events: none; background-image: linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px); background-size: 32px 32px; mask-image: linear-gradient(180deg, rgba(255, 255, 255, 0.45), transparent 70%); opacity: 0.18; }
    button, input, select, textarea { font: inherit; }
    .page { position: relative; z-index: 1; max-width: 1680px; margin: 0 auto; padding: 12px 14px 28px; }
    .hero { display: grid; grid-template-columns: minmax(0, 1fr) minmax(180px, auto); gap: 12px; align-items: center; margin-bottom: 10px; padding: 10px 12px; border: 1px solid var(--line); border-radius: 16px; background: rgba(10, 16, 22, 0.72); box-shadow: 0 12px 26px rgba(0, 0, 0, 0.16); backdrop-filter: blur(14px); }
    .hero-meta { min-width: 0; }
    .hero-meta-right { text-align: right; }
    .hero-label { display: block; margin-bottom: 4px; color: var(--muted); font-size: 10px; font-weight: 500; letter-spacing: 0.14em; text-transform: uppercase; }
    .scene-path { display: block; color: var(--accent-strong); font-family: "Cascadia Mono", "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 0.79rem; font-weight: 400; line-height: 1.4; word-break: break-all; }
    .selection-label { display: block; color: var(--text); font-size: 0.95rem; font-weight: 600; letter-spacing: 0.02em; }
    .workspace { display: grid; grid-template-columns: minmax(0, 0.9fr) minmax(420px, 0.78fr); gap: 12px; align-items: start; }
    .tech-nav { position: sticky; top: 10px; z-index: 5; margin: 0; padding: 8px; border: 1px solid var(--line); border-radius: 16px; background: rgba(11, 16, 22, 0.82); box-shadow: 0 12px 24px rgba(0, 0, 0, 0.16); backdrop-filter: blur(16px); }
    .tech-nav:empty { display: none; }
    .thumb-group { display: grid; gap: 8px; margin-bottom: 12px; }
    .thumb-group:last-child { margin-bottom: 0; }
    .thumb-group-title { margin: 0; color: var(--accent); font-size: 10px; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; }
    .thumb-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
    .thumb-card { padding: 6px; border: 1px solid var(--line); border-radius: 12px; background: rgba(255, 255, 255, 0.025); transition: border-color 120ms ease, background 120ms ease, transform 120ms ease; cursor: pointer; }
    .thumb-card:hover { transform: translateY(-1px); border-color: rgba(119, 217, 255, 0.28); background: rgba(255, 255, 255, 0.035); }
    .thumb-card.is-active { border-color: rgba(119, 217, 255, 0.5); background: rgba(119, 217, 255, 0.08); box-shadow: inset 0 0 0 1px rgba(119, 217, 255, 0.12); }
    .thumb-toolbar { display: flex; justify-content: space-between; align-items: flex-start; gap: 6px; margin-bottom: 6px; }
    .thumb-head { flex: 1; min-width: 0; display: grid; gap: 4px; }
    .thumb-score-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
    .thumb-name { color: var(--text); font-size: 10px; font-weight: 600; letter-spacing: 0.03em; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .thumb-score { color: var(--accent-strong); font-size: 10px; font-weight: 600; white-space: nowrap; }
    .thumb-progress { height: 6px; border: 1px solid rgba(146, 167, 188, 0.18); border-radius: 999px; overflow: hidden; background: rgba(255, 255, 255, 0.05); }
    .thumb-progress-fill { height: 100%; border-radius: 999px; }
    .thumb-toggle { display: inline-flex; align-items: center; gap: 5px; color: var(--muted); font-size: 10px; white-space: nowrap; }
    .thumb-viewer { position: relative; border: 1px solid var(--line); border-radius: 9px; overflow: hidden; background: #0c1117; }
    .thumb-viewer img { display: block; width: 100%; height: auto; }
    .thumb-viewer .overlay { position: absolute; inset: 0; pointer-events: none; }
    .grid { position: sticky; top: 10px; display: grid; gap: 12px; align-self: start; }
    .card { display: none; border: 1px solid var(--line); border-radius: 20px; background: linear-gradient(180deg, rgba(255, 255, 255, 0.038), rgba(255, 255, 255, 0.015)); box-shadow: 0 16px 44px rgba(0, 0, 0, 0.22); overflow: hidden; }
    .grid > .card:first-child { display: block; }
    .grid.has-active > .card:first-child:not(.is-active) { display: none; }
    .card.is-active { display: block; }
    .card-header { display: flex; justify-content: space-between; gap: 12px; align-items: center; padding: 12px 18px 0; }
    .card-title { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
    .card-header h2 { margin: 0; font-size: 1.15rem; font-weight: 600; line-height: 1; letter-spacing: -0.02em; text-transform: none; }
    .tag { display: inline-flex; align-items: center; width: fit-content; padding: 5px 8px; border: 1px solid rgba(241, 201, 120, 0.22); border-radius: 999px; color: var(--warm); font-size: 10px; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; background: rgba(241, 201, 120, 0.08); }
    .group-tag { border-color: rgba(119, 217, 255, 0.24); color: var(--accent-strong); background: rgba(119, 217, 255, 0.08); }
    .card-summary { display: none; }
    .summary-pill { padding: 10px 12px; border: 1px solid var(--line); border-radius: 14px; background: rgba(8, 13, 19, 0.28); }
    .summary-label { display: block; margin-bottom: 6px; color: var(--muted); font-size: 10px; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; }
    .summary-value { display: block; color: var(--accent-strong); font-size: 1rem; font-weight: 600; line-height: 1.05; }
    .card-body { display: grid; grid-template-columns: 1fr; gap: 16px; padding: 14px 18px 18px; align-items: start; }
    .viewer-wrap { display: none; }
    .viewer-toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 12px; color: var(--muted); }
    .viewer-toolbar label { display: inline-flex; gap: 10px; align-items: center; padding: 8px 10px; border: 1px solid var(--line); border-radius: 999px; background: rgba(8, 13, 19, 0.26); font-size: 13px; }
    .viewer { position: relative; width: 100%; border-radius: 18px; overflow: hidden; border: 1px solid var(--line-strong); background: linear-gradient(45deg, rgba(255, 255, 255, 0.03) 25%, transparent 25%), linear-gradient(-45deg, rgba(255, 255, 255, 0.03) 25%, transparent 25%), linear-gradient(45deg, transparent 75%, rgba(255, 255, 255, 0.03) 75%), linear-gradient(-45deg, transparent 75%, rgba(255, 255, 255, 0.03) 75%), #0c1117; background-position: 0 0, 0 10px, 10px -10px, -10px 0; background-size: 20px 20px; box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.02); }
    .viewer::after { content: "CPU base / GPU overlay"; position: absolute; right: 14px; bottom: 14px; padding: 6px 10px; border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 999px; font-size: 11px; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; color: rgba(255, 255, 255, 0.82); background: rgba(11, 16, 22, 0.72); backdrop-filter: blur(10px); }
    .viewer img { display: block; width: 100%; height: auto; }
    .viewer .overlay { position: absolute; inset: 0; opacity: 1; transition: opacity 120ms linear; pointer-events: none; }
    .legend { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; color: var(--muted); font-size: 12px; }
    .legend span { padding: 6px 9px; border: 1px solid var(--line); border-radius: 999px; background: rgba(8, 13, 19, 0.24); }
    .metrics { display: grid; gap: 10px; align-self: start; }
    .metric-section { padding: 14px 14px 12px; border: 1px solid var(--line); border-radius: 16px; background: var(--panel-soft); }
    .metrics h3 { margin: 0 0 10px; font-size: 11px; font-weight: 500; letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent); }
    table { width: 100%; border-collapse: separate; border-spacing: 0 4px; font-size: 13px; }
    th, td { padding: 5px 0; border-bottom: 0; text-align: left; vertical-align: middle; }
    th { color: var(--muted); font-weight: 500; }
    td { font-variant-numeric: tabular-nums; }
    td:last-child { padding-left: 18px; color: var(--accent-strong); font-weight: 600; text-align: right; white-space: nowrap; }
    .metric-label { padding: 6px 10px; border-radius: 8px; cursor: help; }
    .files { display: grid; gap: 8px; color: var(--muted); font-size: 12px; }
    .file-entry { display: grid; gap: 4px; padding: 8px 10px; border: 1px solid var(--line); border-radius: 12px; background: rgba(255, 255, 255, 0.02); }
    .file-entry span { color: var(--muted); font-size: 10px; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; }
    .file-entry code { color: var(--text); font-family: "Cascadia Mono", "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 12px; word-break: break-all; }
    .has-tooltip { cursor: help; }
    .report-tooltip { position: fixed; z-index: 1000; max-width: 360px; padding: 8px 10px; border: 1px solid rgba(119, 217, 255, 0.3); border-radius: 10px; background: rgba(7, 11, 16, 0.96); color: var(--text); font-size: 12px; line-height: 1.45; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.28); pointer-events: none; opacity: 0; transform: translateY(4px); transition: opacity 90ms ease, transform 90ms ease; }
    .report-tooltip.is-visible { opacity: 1; transform: translateY(0); }
    input[type="checkbox"] { accent-color: var(--accent); }
    input[type="range"] { width: min(220px, 38vw); accent-color: var(--accent); }
    @media (max-width: 1200px) { .workspace { grid-template-columns: minmax(0, 0.86fr) minmax(360px, 0.88fr); } }
    @media (max-width: 980px) { .workspace { grid-template-columns: 1fr; } .grid { position: static; } .thumb-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
    @media (max-width: 900px) { .hero { grid-template-columns: 1fr; } .hero-meta-right { text-align: left; } .thumb-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 640px) { .page { padding-left: 10px; padding-right: 10px; } .hero, .card, .viewer-wrap { border-radius: 14px; } .thumb-grid { grid-template-columns: 1fr; } .viewer-toolbar label { width: 100%; justify-content: space-between; } .viewer::after { left: 12px; right: auto; } }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="hero-meta">
        <span class="hero-label">Scene</span>
        <code class="scene-path">)html";
  html_text += html_escape(scene_file);
  html_text += R"html(</code>
      </div>
      <div class="hero-meta hero-meta-right">
        <span class="hero-label">Selected</span>
        <span class="selection-label" id="selection_label">pt</span>
      </div>
    </section>
    <div class="workspace">
    <section class="tech-nav" id="tech_nav"></section>
    <section class="grid">
)html";
}

void append_full_comparison_html_entry(std::string& html_text, const FullComparisonTechniqueInfo& technique, const std::string& cpu_output_file, const std::string& gpu_output_file,
  const ImageComparisonResult& comparison) {
  const std::string cpu_png_file = html_file_name_only(png_file_name_from_output_file(cpu_output_file));
  const std::string gpu_png_file = html_file_name_only(png_file_name_from_output_file(gpu_output_file));
  const std::string cpu_exr_file = html_file_name_only(cpu_output_file);
  const std::string gpu_exr_file = html_file_name_only(gpu_output_file);
  const std::string viewer_id = std::string("viewer_") + technique.file_tag;
  const char* group_name = full_comparison_group_name(technique.bdpt_mode);
  const float compare_similarity_score = clamp_metric_score(comparison.similarity / 100.0f);
  const float compare_low_frequency_similarity_score = clamp_metric_score(comparison.low_frequency_similarity / 100.0f);
  const float compare_rmse_score = inverse_error_metric_score(comparison.root_mean_squared_error, 1.0f);
  const float compare_low_frequency_rmse_score = inverse_error_metric_score(comparison.low_frequency_root_mean_squared_error, 1.0f);
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
  compare_rows += metric_row_percent("Low-Freq Similarity", comparison.low_frequency_similarity, compare_low_frequency_similarity_score, false);
  compare_rows += metric_row_value("RMSE", comparison.root_mean_squared_error, compare_rmse_score, false);
  compare_rows += metric_row_value("Low-Freq RMSE", comparison.low_frequency_root_mean_squared_error, compare_low_frequency_rmse_score, false);
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

  html_text += "      <div class=\"card\" data-group=\"";
  html_text += html_escape(group_name);
  html_text += "\">\n";
  html_text += "        <div class=\"card-header\">\n";
  html_text += "          <h2>";
  html_text += html_escape(technique.file_tag);
  html_text += "</h2>\n";
  html_text += "          <div class=\"tag group-tag\">";
  html_text += html_escape(group_name);
  html_text += "</div>\n";
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
    "            <h3 title=\"Metrics measured after the comparison-space transform used by the report. Useful for perceptual closeness; lower error is better, higher similarity "
    "is better.\">Compare-Space Metrics</h3>\n";
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
  html_text += R"html(    </section>
    </div>
  </div>
  <script>
    const slugify = (value) => value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
    const readMetricValue = (table, label) => {
      if (table == null) {
        return '';
      }
      const rows = table.querySelectorAll('tr');
      for (const row of rows) {
        const cells = row.querySelectorAll('td');
        if (cells.length < 2) {
          continue;
        }
        if (cells[0].textContent.trim() === label) {
          return cells[1].textContent.trim();
        }
      }
      return '';
    };

    const clampScore = (value) => Math.max(0, Math.min(1, value));
    const inverseErrorScore = (value, scale) => clampScore(1 - (value / Math.max(scale, 1.0e-6)));
    const readMetricNumber = (table, label) => {
      const text = readMetricValue(table, label).replace('%', '').trim();
      const value = Number(text);
      return Number.isFinite(value) ? value : 0;
    };
    const metricScoreColor = (score) => {
      const clamped = clampScore(score);
      return {
        r: 138 + (92 - 138) * clamped,
        g: 82 + (126 - 82) * clamped,
        b: 82 + (104 - 82) * clamped,
      };
    };
    const compositeSimilarityTooltip = 'Noise-robust similarity score emphasizing low-frequency structural agreement, with smaller penalties for compare-space error, linear relative error, and brightness mismatch. Designed so independent noise does not dominate the score.';
    const computeCompositeSimilarity = (compareTable, linearTable) => {
      const lowFreqSimilarity = readMetricNumber(compareTable, 'Low-Freq Similarity') / 100.0;
      const lowFreqRmse = readMetricNumber(compareTable, 'Low-Freq RMSE');
      const compareSimilarity = readMetricNumber(compareTable, 'Similarity') / 100.0;
      const compareRelativeRmse = readMetricNumber(compareTable, 'Relative RMSE');
      const linearRelativeRmse = readMetricNumber(linearTable, 'Relative RMSE');
      const brightnessRel = Math.abs(readMetricNumber(compareTable, 'Brightness Rel'));

      return clampScore(
        (lowFreqSimilarity * 0.55) +
        (inverseErrorScore(lowFreqRmse, 0.02) * 0.20) +
        (compareSimilarity * 0.10) +
        (inverseErrorScore(compareRelativeRmse, 0.25) * 0.05) +
        (inverseErrorScore(linearRelativeRmse, 0.12) * 0.05) +
        (inverseErrorScore(brightnessRel, 0.03) * 0.05)
      );
    };

    const overlayToggles = document.querySelectorAll('input[data-target]');
    for (const toggle of overlayToggles) {
      toggle.addEventListener('change', () => {
        const target = document.getElementById(toggle.dataset.target);
        if (target) {
          target.style.display = toggle.checked ? 'block' : 'none';
        }
      });
    }
    const opacitySliders = document.querySelectorAll('input[data-opacity-target]');
    for (const slider of opacitySliders) {
      slider.addEventListener('input', () => {
        const target = document.getElementById(slider.dataset.opacityTarget);
        if (target) {
          target.style.opacity = String(Number(slider.value) / 100.0);
        }
      });
    }

    const grid = document.querySelector('.grid');
    const techniqueNav = document.getElementById('tech_nav');
    const selectionLabel = document.getElementById('selection_label');
    const cards = document.querySelectorAll('.card');
    const thumbsBySlug = new Map();
    const cardsBySlug = new Map();
    const thumbGroups = new Map();
    const tooltip = document.createElement('div');
    tooltip.className = 'report-tooltip';
    document.body.appendChild(tooltip);

    let tooltipElement = null;

    const positionTooltip = (event) => {
      const offset = 14;
      const maxLeft = Math.max(8, window.innerWidth - tooltip.offsetWidth - 8);
      const maxTop = Math.max(8, window.innerHeight - tooltip.offsetHeight - 8);
      const nextLeft = Math.min(event.clientX + offset, maxLeft);
      const nextTop = Math.min(event.clientY + offset, maxTop);
      tooltip.style.left = `${nextLeft}px`;
      tooltip.style.top = `${nextTop}px`;
    };

    const showTooltip = (element, event) => {
      const tooltipText = element.dataset.tooltip || '';
      if (tooltipText === '') {
        return;
      }
      tooltipElement = element;
      tooltip.textContent = tooltipText;
      tooltip.classList.add('is-visible');
      positionTooltip(event);
    };

    const hideTooltip = (element) => {
      if ((tooltipElement != null) && (tooltipElement !== element)) {
        return;
      }
      tooltipElement = null;
      tooltip.classList.remove('is-visible');
    };

    const prepareTooltips = (root) => {
      const elements = root.querySelectorAll('[title]');
      for (const element of elements) {
        const tooltipText = element.getAttribute('title');
        if ((tooltipText == null) || (tooltipText === '')) {
          continue;
        }
        element.dataset.tooltip = tooltipText;
        element.removeAttribute('title');
        element.classList.add('has-tooltip');
        element.setAttribute('tabindex', '0');
        element.addEventListener('mouseenter', (event) => {
          showTooltip(element, event);
        });
        element.addEventListener('mousemove', (event) => {
          if (tooltipElement === element) {
            positionTooltip(event);
          }
        });
        element.addEventListener('mouseleave', () => {
          hideTooltip(element);
        });
        element.addEventListener('focus', () => {
          const rect = element.getBoundingClientRect();
          showTooltip(element, {
            clientX: rect.left + Math.min(rect.width, 24),
            clientY: rect.bottom,
          });
        });
        element.addEventListener('blur', () => {
          hideTooltip(element);
        });
      }
    };

    prepareTooltips(document);

    const setCardOverlayState = (card, checked, sourceToggle = null) => {
      const cardToggle = card.querySelector('.viewer-toolbar input[data-target]');
      const overlayId = cardToggle ? cardToggle.dataset.target : '';
      const overlay = overlayId ? document.getElementById(overlayId) : null;
      if ((cardToggle != null) && (cardToggle !== sourceToggle)) {
        cardToggle.checked = checked;
      }
      if (overlay != null) {
        overlay.style.display = checked ? 'block' : 'none';
      }
      const thumbCheckbox = card.dataset.thumbCheckboxId ? document.getElementById(card.dataset.thumbCheckboxId) : null;
      if ((thumbCheckbox != null) && (thumbCheckbox !== sourceToggle)) {
        thumbCheckbox.checked = checked;
      }
      const thumbOverlay = card.dataset.thumbOverlayId ? document.getElementById(card.dataset.thumbOverlayId) : null;
      if ((thumbOverlay != null) && (thumbOverlay !== sourceToggle)) {
        thumbOverlay.style.display = checked ? 'block' : 'none';
      }
    };

    const selectTechnique = (slug) => {
      for (const [entrySlug, card] of cardsBySlug) {
        card.classList.toggle('is-active', entrySlug === slug);
      }
      for (const [entrySlug, thumb] of thumbsBySlug) {
        thumb.classList.toggle('is-active', entrySlug === slug);
      }
      const activeCard = cardsBySlug.get(slug);
      if ((selectionLabel != null) && (activeCard != null)) {
        selectionLabel.textContent = activeCard.dataset.title || slug;
      }
      if (grid != null) {
        grid.classList.add('has-active');
      }
    };

    for (const card of cards) {
      const header = card.querySelector('.card-header');
      const titleElement = header ? header.querySelector('h2') : null;
      const tagElement = header ? header.querySelector('.tag') : null;
      const metrics = card.querySelector('.metrics');
      const tables = metrics ? metrics.querySelectorAll('table') : [];
      const compareTable = tables.length > 0 ? tables[0] : null;
      const linearTable = tables.length > 1 ? tables[1] : null;
      if (titleElement == null) {
        continue;
      }

      const title = titleElement.textContent.trim();
      const slug = slugify(title);
      card.id = slug;
      card.dataset.title = title;
      cardsBySlug.set(slug, card);

      if ((header != null) && (tagElement != null) && (header.querySelector('.card-title') == null)) {
        const titleGroup = document.createElement('div');
        titleGroup.className = 'card-title';
        header.insertBefore(titleGroup, header.firstChild);
        titleGroup.appendChild(titleElement);
        titleGroup.appendChild(tagElement);
      }

      if ((card.querySelector('.card-summary') == null) && (compareTable != null) && (linearTable != null)) {
        const summary = document.createElement('div');
        summary.className = 'card-summary';

        const summaryMetrics = [
          { label: 'Similarity', value: readMetricValue(compareTable, 'Similarity') },
          { label: 'Relative RMSE', value: readMetricValue(compareTable, 'Relative RMSE') },
          { label: 'Linear RMSE', value: readMetricValue(linearTable, 'RMSE') },
        ];

        for (const metric of summaryMetrics) {
          const item = document.createElement('div');
          item.className = 'summary-pill';
          item.innerHTML = `<span class="summary-label">${metric.label}</span><span class="summary-value">${metric.value}</span>`;
          summary.appendChild(item);
        }

        const body = card.querySelector('.card-body');
        if (body != null) {
          card.insertBefore(summary, body);
        }
      }

      if (metrics != null) {
        const headings = metrics.querySelectorAll(':scope > h3');
        for (const heading of headings) {
          const table = heading.nextElementSibling;
          if ((table == null) || (table.tagName !== 'TABLE')) {
            continue;
          }
          if ((heading.parentElement != null) && heading.parentElement.classList.contains('metric-section')) {
            continue;
          }
          const section = document.createElement('section');
          section.className = 'metric-section';
          metrics.insertBefore(section, heading);
          section.appendChild(heading);
          section.appendChild(table);
        }

        const files = metrics.querySelector('.files');
        if (files != null) {
          files.classList.add('metric-section');
          const entries = files.querySelectorAll('div');
          for (const entry of entries) {
            if (entry.classList.contains('file-entry')) {
              continue;
            }
            const separatorIndex = entry.textContent.indexOf(':');
            if (separatorIndex < 0) {
              continue;
            }
            const label = entry.textContent.slice(0, separatorIndex).trim();
            const value = entry.textContent.slice(separatorIndex + 1).trim();
            entry.className = 'file-entry';
            entry.innerHTML = `<span>${label}</span><code>${value}</code>`;
          }
        }
      }

      const overlayToggle = card.querySelector('.viewer-toolbar input[data-target]');
      if (overlayToggle != null) {
        overlayToggle.addEventListener('change', () => {
          setCardOverlayState(card, overlayToggle.checked, overlayToggle);
        });
        setCardOverlayState(card, overlayToggle.checked, overlayToggle);
      }

      if (techniqueNav != null) {
        const groupName = card.dataset.group || 'Other';
        let thumbGrid = thumbGroups.get(groupName);
        if (thumbGrid == null) {
          const group = document.createElement('section');
          group.className = 'thumb-group';
          const groupTitle = document.createElement('h3');
          groupTitle.className = 'thumb-group-title';
          groupTitle.textContent = groupName;
          thumbGrid = document.createElement('div');
          thumbGrid.className = 'thumb-grid';
          group.appendChild(groupTitle);
          group.appendChild(thumbGrid);
          techniqueNav.appendChild(group);
          thumbGroups.set(groupName, thumbGrid);
        }

        const viewer = card.querySelector('.viewer');
        const baseImage = viewer ? viewer.querySelector('img:not(.overlay)') : null;
        const overlayImage = viewer ? viewer.querySelector('.overlay') : null;
        if ((baseImage == null) || (overlayImage == null)) {
          continue;
        }

        const compositeSimilarity = computeCompositeSimilarity(compareTable, linearTable);
        const compositeSimilarityPercent = `${(compositeSimilarity * 100).toFixed(1)}%`;
        const compositeColor = metricScoreColor(compositeSimilarity);

        const thumb = document.createElement('article');
        thumb.className = 'thumb-card';
        thumb.dataset.target = slug;

        const thumbOverlayId = `thumb_overlay_${slug}`;
        const thumbCheckboxId = `thumb_checkbox_${slug}`;
        card.dataset.thumbOverlayId = thumbOverlayId;
        card.dataset.thumbCheckboxId = thumbCheckboxId;

        thumb.innerHTML = `
          <div class="thumb-toolbar">
            <div class="thumb-head">
              <div class="thumb-score-row">
                <span class="thumb-name">${title}</span>
                <span class="thumb-score" title="${compositeSimilarityTooltip}">${compositeSimilarityPercent}</span>
              </div>
              <div class="thumb-progress" title="${compositeSimilarityTooltip}">
                <div class="thumb-progress-fill" style="width: ${compositeSimilarityPercent}; background: linear-gradient(90deg, rgba(${compositeColor.r.toFixed(0)}, ${compositeColor.g.toFixed(0)}, ${compositeColor.b.toFixed(0)}, 0.95), rgba(${compositeColor.r.toFixed(0)}, ${compositeColor.g.toFixed(0)}, ${compositeColor.b.toFixed(0)}, 0.68));"></div>
              </div>
            </div>
            <label class="thumb-toggle">
              <input type="checkbox" id="${thumbCheckboxId}" ${overlayToggle && overlayToggle.checked ? 'checked' : ''}>
              <span>GPU</span>
            </label>
          </div>
          <div class="thumb-viewer">
            <img src="${baseImage.getAttribute('src')}" alt="${baseImage.getAttribute('alt') || title}">
            <img class="overlay" id="${thumbOverlayId}" src="${overlayImage.getAttribute('src')}" alt="${overlayImage.getAttribute('alt') || `${title} GPU result`}">
          </div>`;

        thumb.addEventListener('click', (event) => {
          if (event.target.closest('.thumb-toggle')) {
            return;
          }
          selectTechnique(slug);
        });

        const thumbCheckbox = thumb.querySelector(`#${thumbCheckboxId}`);
        if (thumbCheckbox != null) {
          thumbCheckbox.addEventListener('click', (event) => {
            event.stopPropagation();
          });
          thumbCheckbox.addEventListener('change', () => {
            setCardOverlayState(card, thumbCheckbox.checked, thumbCheckbox);
          });
        }

        thumbsBySlug.set(slug, thumb);
        thumbGrid.appendChild(thumb);
        prepareTooltips(thumb);
      }
    }

    if (cards.length > 0) {
      const firstCard = cards[0];
      if (firstCard.id) {
        selectTechnique(firstCard.id);
      }
    }
  </script>
</body>
</html>
)html";
}

struct CPUComparisonReportEntry {
  const CPUComparisonTechniqueInfo* technique = nullptr;
  std::string output_file = {};
  ImageComparisonResult comparison = {};
  std::string failure_text = {};
  int exit_code = 0;
  bool render_succeeded = true;
};

std::string strategy_flags_text(const uint32_t strategy_flags) {
  if (strategy_flags == 0u) {
    return "none";
  }

  std::string result = {};
  auto append_flag = [&result](const char* text) {
    if (result.empty() == false) {
      result += " + ";
    }
    result += text;
  };

  if ((strategy_flags & Scene::Strategy::DirectHit) != 0u) {
    append_flag("direct-hit");
  }
  if ((strategy_flags & Scene::Strategy::ConnectToLight) != 0u) {
    append_flag("connect-light");
  }
  if ((strategy_flags & Scene::Strategy::ConnectToCamera) != 0u) {
    append_flag("connect-camera");
  }
  if ((strategy_flags & Scene::Strategy::ConnectVertices) != 0u) {
    append_flag("connect-vertices");
  }
  if ((strategy_flags & Scene::Strategy::MergeVertices) != 0u) {
    append_flag("merge-vertices");
  }

  return result;
}

std::string cpu_comparison_metric_chip(const char* label, const std::string& value, const float score) {
  const float clamped_score = clamp_metric_score(score);
  const float chip_r = lerp(148.0f, 64.0f, clamped_score);
  const float chip_g = lerp(74.0f, 146.0f, clamped_score);
  const float chip_b = lerp(70.0f, 108.0f, clamped_score);

  char buffer[512] = {};
  std::snprintf(buffer, sizeof(buffer),
    "<div class=\"metric-chip\" style=\"background: linear-gradient(135deg, rgba(%.0f, %.0f, %.0f, 0.42), rgba(255, 255, 255, 0.04));\">"
    "<span>%s</span><strong>%s</strong></div>\n",
    chip_r, chip_g, chip_b, html_escape(label).c_str(), html_escape(value).c_str());
  return buffer;
}

void append_cpu_comparison_html_header(std::string& html_text, const std::string& scene_file, const CPUComparisonTechniqueInfo& reference_technique) {
  html_text += "<!doctype html>\n";
  html_text += "<html lang=\"en\">\n";
  html_text += "<head>\n";
  html_text += "  <meta charset=\"utf-8\">\n";
  html_text += "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n";
  html_text += "  <title>ETX CPU Integrator Comparison</title>\n";
  html_text += "  <style>\n";
  html_text +=
    "    :root { color-scheme: dark; --bg: #0d1318; --panel: #121b22; --panel-2: #18232d; --panel-3: #0f171d; --text: #edf3f7; --muted: #94a8b6; --accent: #8adbb4; --line: #26333f; "
    "--warm: #f5b971; }\n";
  html_text += "    * { box-sizing: border-box; }\n";
  html_text +=
    "    body { margin: 0; font-family: \"Cascadia Mono\", \"SFMono-Regular\", Consolas, \"Liberation Mono\", Menlo, monospace; background: radial-gradient(circle at top, #17232c 0%, "
    "#0d1318 50%, #081017 100%); color: var(--text); }\n";
  html_text += "    .page { max-width: 1540px; margin: 0 auto; padding: 28px 22px 56px; }\n";
  html_text +=
    "    .hero { padding: 24px 26px; border: 1px solid var(--line); border-radius: 22px; background: linear-gradient(135deg, rgba(138, 219, 180, 0.12), rgba(245, 185, 113, 0.06)); "
    "box-shadow: 0 18px 48px rgba(0, 0, 0, 0.22); }\n";
  html_text += "    .hero h1 { margin: 0 0 10px; font-size: 32px; }\n";
  html_text += "    .hero p { margin: 0; color: var(--muted); word-break: break-all; }\n";
  html_text += "    .hero-meta { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 10px; }\n";
  html_text +=
    "    .pill { display: inline-flex; align-items: center; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 999px; padding: 7px 11px; font-size: 12px; color: var(--text); "
    "background: rgba(255, 255, 255, 0.04); }\n";
  html_text += "    .layout { display: grid; gap: 18px; margin-top: 20px; }\n";
  html_text +=
    "    .reference { display: grid; grid-template-columns: minmax(340px, 1.05fr) minmax(280px, 0.95fr); gap: 18px; padding: 20px; border: 1px solid var(--line); border-radius: 20px; "
    "background: linear-gradient(180deg, rgba(255, 255, 255, 0.035), rgba(255, 255, 255, 0.015)); }\n";
  html_text += "    .section-title { margin: 0 0 12px; font-size: 18px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--warm); }\n";
  html_text += "    .frame { border: 1px solid var(--line); border-radius: 16px; overflow: hidden; background: #090d11; }\n";
  html_text += "    .frame img { display: block; width: 100%; height: auto; }\n";
  html_text += "    .reference-copy p { margin: 0 0 14px; color: var(--muted); line-height: 1.55; }\n";
  html_text += "    .summary { border: 1px solid var(--line); border-radius: 20px; overflow: hidden; background: rgba(11, 16, 21, 0.76); }\n";
  html_text += "    .summary header { padding: 18px 20px 10px; }\n";
  html_text += "    .summary header p { margin: 6px 0 0; color: var(--muted); }\n";
  html_text += "    table { width: 100%; border-collapse: collapse; }\n";
  html_text += "    th, td { padding: 12px 20px; border-top: 1px solid rgba(148, 168, 182, 0.12); text-align: left; font-size: 14px; }\n";
  html_text += "    th { color: var(--muted); font-weight: 600; }\n";
  html_text += "    td { font-variant-numeric: tabular-nums; }\n";
  html_text += "    tbody tr:hover { background: rgba(255, 255, 255, 0.025); }\n";
  html_text += "    .name-cell { font-weight: 700; }\n";
  html_text += "    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }\n";
  html_text +=
    "    .card { border: 1px solid var(--line); border-radius: 20px; background: linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.015)); overflow: hidden; "
    "box-shadow: 0 18px 42px rgba(0, 0, 0, 0.18); }\n";
  html_text += "    .card-head { padding: 18px 18px 12px; display: flex; justify-content: space-between; gap: 12px; align-items: start; }\n";
  html_text += "    .card-head h3 { margin: 0; font-size: 20px; }\n";
  html_text += "    .card-head p { margin: 6px 0 0; color: var(--muted); font-size: 13px; line-height: 1.45; }\n";
  html_text += "    .tag { border-radius: 999px; padding: 6px 10px; font-size: 12px; border: 1px solid rgba(255, 255, 255, 0.08); color: var(--accent); white-space: nowrap; }\n";
  html_text += "    .thumbs { display: grid; grid-template-columns: 1.4fr 1fr; gap: 10px; padding: 0 18px 16px; }\n";
  html_text += "    .thumb-label { padding: 8px 10px; font-size: 12px; color: var(--muted); border-bottom: 1px solid var(--line); background: rgba(255, 255, 255, 0.03); }\n";
  html_text += "    .metrics { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; padding: 0 18px 18px; }\n";
  html_text += "    .metric-chip { border-radius: 14px; border: 1px solid rgba(255, 255, 255, 0.08); padding: 10px 12px; }\n";
  html_text += "    .metric-chip span { display: block; color: var(--muted); font-size: 11px; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.08em; }\n";
  html_text += "    .metric-chip strong { font-size: 16px; }\n";
  html_text += "    .card-meta { display: flex; flex-wrap: wrap; gap: 8px; padding: 0 18px 16px; }\n";
  html_text += "    .file-list { padding: 0 18px 18px; color: var(--muted); font-size: 12px; }\n";
  html_text += "    .file-list div { margin-top: 4px; word-break: break-all; }\n";
  html_text += "    @media (max-width: 980px) { .reference { grid-template-columns: 1fr; } }\n";
  html_text += "    @media (max-width: 720px) { .thumbs { grid-template-columns: 1fr; } .metrics { grid-template-columns: 1fr 1fr; } th, td { padding: 10px 12px; font-size: 13px; } }\n";
  html_text += "  </style>\n";
  html_text += "</head>\n";
  html_text += "<body>\n";
  html_text += "  <div class=\"page\">\n";
  html_text += "    <section class=\"hero\">\n";
  html_text += "      <h1>ETX CPU Integrator Comparison</h1>\n";
  html_text += "      <p>Scene: ";
  html_text += html_escape(scene_file);
  html_text += "</p>\n";
  html_text += "      <div class=\"hero-meta\">\n";
  html_text += "        <span class=\"pill\">Reference: ";
  html_text += html_escape(reference_technique.display_name);
  html_text += "</span>\n";
  html_text += "        <span class=\"pill\">CPU-only validation</span>\n";
  html_text += "        <span class=\"pill\">All metrics compare each technique against the PT reference</span>\n";
  html_text += "      </div>\n";
  html_text += "    </section>\n";
  html_text += "    <section class=\"layout\">\n";
}

void append_cpu_comparison_html_reference(std::string& html_text, const CPUComparisonReportEntry& reference_entry) {
  const std::string reference_png_file = html_file_name_only(png_file_name_from_output_file(reference_entry.output_file));

  html_text += "      <section class=\"reference\">\n";
  html_text += "        <div>\n";
  html_text += "          <h2 class=\"section-title\">Reference</h2>\n";
  html_text += "          <div class=\"frame\"><img src=\"";
  html_text += reference_png_file;
  html_text += "\" alt=\"Reference render\"></div>\n";
  html_text += "        </div>\n";
  html_text += "        <div class=\"reference-copy\">\n";
  html_text += "          <h2 class=\"section-title\">";
  html_text += html_escape(reference_entry.technique->display_name);
  html_text += "</h2>\n";
  html_text += "          <p>";
  html_text += html_escape(reference_entry.technique->description);
  html_text += "</p>\n";
  html_text += "          <div class=\"hero-meta\">\n";
  html_text += "            <span class=\"pill\">";
  html_text += html_escape(strategy_flags_text(reference_entry.technique->strategy_flags));
  html_text += "</span>\n";
  html_text += "            <span class=\"pill\">";
  html_text += html_escape(html_file_name_only(reference_entry.output_file));
  html_text += "</span>\n";
  html_text += "          </div>\n";
  html_text += "        </div>\n";
  html_text += "      </section>\n";
}

void append_cpu_comparison_html_summary(std::string& html_text, const std::vector<CPUComparisonReportEntry>& entries) {
  html_text += "      <section class=\"summary\">\n";
  html_text += "        <header>\n";
  html_text += "          <h2 class=\"section-title\">Summary</h2>\n";
  html_text += "          <p>Compact scan of the main compare-space and linear-space metrics versus the PT reference.</p>\n";
  html_text += "        </header>\n";
  html_text += "        <table>\n";
  html_text += "          <thead><tr><th>Technique</th><th>Compare Similarity</th><th>Low-Freq Similarity</th><th>Compare RMSE</th><th>Linear RMSE</th><th>Brightness Ratio</th></tr></thead>\n";
  html_text += "          <tbody>\n";

  for (const CPUComparisonReportEntry& entry : entries) {
    html_text += "            <tr><td class=\"name-cell\">";
    html_text += html_escape(entry.technique->display_name);
    if (entry.technique->reference) {
      html_text += " <span class=\"tag\">reference</span>";
    }
    html_text += "</td><td>";
    if (entry.render_succeeded) {
      char similarity_buffer[64] = {};
      char low_freq_buffer[64] = {};
      char rmse_buffer[64] = {};
      char linear_rmse_buffer[64] = {};
      char brightness_buffer[64] = {};
      std::snprintf(similarity_buffer, sizeof(similarity_buffer), "%.2f%%", entry.comparison.similarity);
      std::snprintf(low_freq_buffer, sizeof(low_freq_buffer), "%.2f%%", entry.comparison.low_frequency_similarity);
      std::snprintf(rmse_buffer, sizeof(rmse_buffer), "%.6f", entry.comparison.root_mean_squared_error);
      std::snprintf(linear_rmse_buffer, sizeof(linear_rmse_buffer), "%.6f", entry.comparison.linear_root_mean_squared_error);
      std::snprintf(brightness_buffer, sizeof(brightness_buffer), "%.4f", entry.comparison.brightness_ratio);
      html_text += similarity_buffer;
      html_text += "</td><td>";
      html_text += low_freq_buffer;
      html_text += "</td><td>";
      html_text += rmse_buffer;
      html_text += "</td><td>";
      html_text += linear_rmse_buffer;
      html_text += "</td><td>";
      html_text += brightness_buffer;
    } else {
      html_text += "failed</td><td>failed</td><td>failed</td><td>failed</td><td>failed";
    }
    html_text += "</td></tr>\n";
  }

  html_text += "          </tbody>\n";
  html_text += "        </table>\n";
  html_text += "      </section>\n";
}

void append_cpu_comparison_html_cards(std::string& html_text, const std::vector<CPUComparisonReportEntry>& entries) {
  html_text += "      <section>\n";
  html_text += "        <h2 class=\"section-title\">Technique Cards</h2>\n";
  html_text += "        <div class=\"cards\">\n";

  for (const CPUComparisonReportEntry& entry : entries) {
    if (entry.technique->reference) {
      continue;
    }

    html_text += "          <article class=\"card\">\n";
    html_text += "            <div class=\"card-head\">\n";
    html_text += "              <div><h3>";
    html_text += html_escape(entry.technique->display_name);
    html_text += "</h3><p>";
    html_text += html_escape(entry.technique->description);
    html_text += "</p></div>\n";
    html_text += "              <span class=\"tag\">";
    html_text += html_escape(entry.technique->file_tag);
    html_text += "</span>\n";
    html_text += "            </div>\n";
    if (entry.render_succeeded) {
      const std::string output_png_file = html_file_name_only(png_file_name_from_output_file(entry.output_file));
      const std::string diff_png_file = html_file_name_only(comparison_png_file_name_from_output_file(entry.output_file));
      const std::string diff_exr_file = html_file_name_only(comparison_file_name_from_output(entry.output_file));

      char similarity_buffer[64] = {};
      char low_freq_buffer[64] = {};
      char rmse_buffer[64] = {};
      char linear_rmse_buffer[64] = {};
      char p95_buffer[64] = {};
      char brightness_buffer[64] = {};
      std::snprintf(similarity_buffer, sizeof(similarity_buffer), "%.2f%%", entry.comparison.similarity);
      std::snprintf(low_freq_buffer, sizeof(low_freq_buffer), "%.2f%%", entry.comparison.low_frequency_similarity);
      std::snprintf(rmse_buffer, sizeof(rmse_buffer), "%.6f", entry.comparison.root_mean_squared_error);
      std::snprintf(linear_rmse_buffer, sizeof(linear_rmse_buffer), "%.6f", entry.comparison.linear_root_mean_squared_error);
      std::snprintf(p95_buffer, sizeof(p95_buffer), "%.6f", entry.comparison.percentile_95_absolute_error);
      std::snprintf(brightness_buffer, sizeof(brightness_buffer), "%.4f", entry.comparison.brightness_ratio);

      html_text += "            <div class=\"thumbs\">\n";
      html_text += "              <div class=\"frame\"><div class=\"thumb-label\">Result</div><img src=\"";
      html_text += output_png_file;
      html_text += "\" alt=\"Technique result\"></div>\n";
      html_text += "              <div class=\"frame\"><div class=\"thumb-label\">Difference To PT</div><img src=\"";
      html_text += diff_png_file;
      html_text += "\" alt=\"Technique difference\"></div>\n";
      html_text += "            </div>\n";
      html_text += "            <div class=\"metrics\">\n";
      html_text += cpu_comparison_metric_chip("Similarity", similarity_buffer, entry.comparison.similarity * 0.01f);
      html_text += cpu_comparison_metric_chip("Low-Freq Similarity", low_freq_buffer, entry.comparison.low_frequency_similarity * 0.01f);
      html_text += cpu_comparison_metric_chip("Compare RMSE", rmse_buffer, inverse_error_metric_score(entry.comparison.root_mean_squared_error, 0.1f));
      html_text += cpu_comparison_metric_chip("Linear RMSE", linear_rmse_buffer, inverse_error_metric_score(entry.comparison.linear_root_mean_squared_error, 0.1f));
      html_text += cpu_comparison_metric_chip("P95 Abs", p95_buffer, inverse_error_metric_score(entry.comparison.percentile_95_absolute_error, 0.1f));
      html_text += cpu_comparison_metric_chip("Brightness Ratio", brightness_buffer, inverse_error_metric_score(fabsf(1.0f - entry.comparison.brightness_ratio), 0.05f));
      html_text += "            </div>\n";
      html_text += "            <div class=\"file-list\">\n";
      html_text += "              <div>EXR: ";
      html_text += html_escape(html_file_name_only(entry.output_file));
      html_text += "</div>\n";
      html_text += "              <div>PNG: ";
      html_text += output_png_file;
      html_text += "</div>\n";
      html_text += "              <div>Diff EXR: ";
      html_text += diff_exr_file;
      html_text += "</div>\n";
      html_text += "              <div>Diff PNG: ";
      html_text += diff_png_file;
      html_text += "</div>\n";
      html_text += "            </div>\n";
    } else {
      html_text += "            <div class=\"metrics\">\n";
      html_text += cpu_comparison_metric_chip("Status", "Render failed", 0.0f);
      html_text += cpu_comparison_metric_chip("Exit Code", std::to_string(entry.exit_code), 0.0f);
      html_text += "            </div>\n";
      html_text += "            <div class=\"file-list\">\n";
      html_text += "              <div>";
      html_text += html_escape(entry.failure_text);
      html_text += "</div>\n";
      html_text += "            </div>\n";
    }
    html_text += "            <div class=\"card-meta\">\n";
    html_text += "              <span class=\"pill\">";
    html_text += html_escape(strategy_flags_text(entry.technique->strategy_flags));
    html_text += "</span>\n";
    html_text += "            </div>\n";
    html_text += "          </article>\n";
  }

  html_text += "        </div>\n";
  html_text += "      </section>\n";
}

void append_cpu_comparison_html_footer(std::string& html_text) {
  html_text += "    </section>\n";
  html_text += "  </div>\n";
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
    const auto total_begin = std::chrono::steady_clock::now();

    const auto scene_global_begin = std::chrono::steady_clock::now();
    scene_global_init();
    const auto scene_global_end = std::chrono::steady_clock::now();

    const auto render_context_begin = std::chrono::steady_clock::now();
    render_context.init();
    const auto render_context_end = std::chrono::steady_clock::now();
    if (render_context.context().valid() == false) {
      log::error("Failed to initialize headless RHI context");
      return false;
    }

    const auto scene_rhi_begin = std::chrono::steady_clock::now();
    scene.set_scattering_rhi(render_context.context());
    gpu_renderer_supported = render_context.context().capabilities().supports_ray_tracing;
    const auto scene_rhi_end = std::chrono::steady_clock::now();

    const auto ior_begin = std::chrono::steady_clock::now();
    std::string ior_folder = env().file_in_data("./spectrum/");
    ior_database.load(ior_folder.c_str());
    const auto ior_end = std::chrono::steady_clock::now();

    double cpu_renderer_init_ms = 0.0;
    if (initialize_cpu_renderer) {
      const auto cpu_begin = std::chrono::steady_clock::now();
      cpu_renderer.init(render_context.context(), scene);
      const auto cpu_end = std::chrono::steady_clock::now();
      cpu_renderer_init_ms = elapsed_ms(cpu_begin, cpu_end);
    }
    double gpu_renderer_init_ms = 0.0;
    if (initialize_gpu_renderer) {
      if (gpu_renderer_supported == false) {
        log::error("GPU ray tracing is not supported by the active RHI backend in this build");
        return false;
      }
      const auto gpu_begin = std::chrono::steady_clock::now();
      gpu_renderer.init(render_context.context(), scene);
      const auto gpu_end = std::chrono::steady_clock::now();
      gpu_renderer_init_ms = elapsed_ms(gpu_begin, gpu_end);
    }

    const auto total_end = std::chrono::steady_clock::now();
    log::info("Batch session init timing: total=%.2fms scene_global=%.2fms render_context=%.2fms scene_rhi=%.2fms ior=%.2fms cpu_renderer=%.2fms gpu_renderer=%.2fms",
      elapsed_ms(total_begin, total_end), elapsed_ms(scene_global_begin, scene_global_end), elapsed_ms(render_context_begin, render_context_end), elapsed_ms(scene_rhi_begin, scene_rhi_end),
      elapsed_ms(ior_begin, ior_end), cpu_renderer_init_ms, gpu_renderer_init_ms);
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
  bool gpu_renderer_supported = false;
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
    BDPTMode bdpt_mode = BDPTMode::BDPTFast;
    if (settings_it != integrator_data.settings.end()) {
      bdpt_mode = settings_it->second.get_integral("bdpt-mode", bdpt_mode);
    }
    cpu_can_render_crop_window = bdpt_mode == BDPTMode::PathTracing;
  } else if (integrator_data.selected == Integrator::Type::VCM) {
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
    selected_integrator = integrator_type_to_instance(integrator_data.selected, session.cpu_renderer.integrator_list(), session.cpu_renderer.integrator_count());
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
  const auto total_begin = std::chrono::steady_clock::now();
  SceneRepresentation::IntegratorData integrator_data = {};
  const std::string absolute_scene_path = resolve_input_path(options.scene_file);
  const auto scene_load_begin = std::chrono::steady_clock::now();
  if (session.scene.load_from_file(absolute_scene_path.c_str(), SceneRepresentation::LoadEverything, &integrator_data) == false) {
    log::error("Failed to load scene from file: %s", absolute_scene_path.c_str());
    return false;
  }
  const auto scene_load_end = std::chrono::steady_clock::now();

  if (session.scene.valid() == false) {
    log::error("Scene is invalid after loading: %s", absolute_scene_path.c_str());
    return false;
  }

  const auto configure_begin = std::chrono::steady_clock::now();
  if (options.integrator.empty() == false) {
    const Integrator::Type requested_type = integrator_id_to_type(options.integrator.c_str());
    if (requested_type == Integrator::Type::Invalid) {
      log::error("Unknown integrator: %s", options.integrator.c_str());
      return false;
    }
    integrator_data.selected = requested_type;
  }
  if (options.override_bdpt_mode) {
    if (integrator_data.selected == Integrator::Type::Invalid) {
      integrator_data.selected = Integrator::Type::Bidirectional;
    } else if (integrator_data.selected != Integrator::Type::Bidirectional) {
      log::error("--bdpt-mode requires the bidirectional integrator");
      return false;
    }

    Options& bdpt_options = integrator_data.settings[Integrator::Type::Bidirectional];
    bdpt_options.set_integral("bdpt-mode", static_cast<int32_t>(options.bdpt_mode), "Mode", Option::Meta::EnumValue);
  }

  session.scene.set_integrator_data(integrator_data);
  apply_batch_scene_overrides(options, session.scene);
  if (configure_batch_render_window(options, session) == false) {
    return false;
  }
  const auto configure_end = std::chrono::steady_clock::now();

  double cpu_setup_ms = 0.0;
  if (configure_cpu_renderer) {
    const auto cpu_setup_begin = std::chrono::steady_clock::now();
    session.cpu_renderer.set_output_dimensions(session.render_context.context(), session.scene.camera().film_size);

    for (const auto& [type, options_data] : integrator_data.settings) {
      Integrator* integrator = integrator_type_to_instance(type, session.cpu_renderer.integrator_list(), session.cpu_renderer.integrator_count());
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
    const auto cpu_setup_end = std::chrono::steady_clock::now();
    cpu_setup_ms = elapsed_ms(cpu_setup_begin, cpu_setup_end);
  } else {
    selected_integrator = nullptr;
  }

  const auto total_end = std::chrono::steady_clock::now();
  log::info("Batch scene load timing: total=%.2fms load=%.2fms configure=%.2fms cpu_setup=%.2fms path=%s", elapsed_ms(total_begin, total_end),
    elapsed_ms(scene_load_begin, scene_load_end), elapsed_ms(configure_begin, configure_end), cpu_setup_ms, absolute_scene_path.c_str());

  return true;
}

bool load_scene_for_full_comparison(const BatchRenderOptions& options, BatchRenderSession& session, const FullComparisonTechniqueInfo& technique, Integrator*& selected_integrator,
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
    Integrator* integrator = integrator_type_to_instance(type, session.cpu_renderer.integrator_list(), session.cpu_renderer.integrator_count());
    if (integrator != nullptr) {
      integrator->sync_from_options(options_data);
      integrator->update_options();
    }
  }

  selected_integrator = integrator_type_to_instance(Integrator::Type::Bidirectional, session.cpu_renderer.integrator_list(), session.cpu_renderer.integrator_count());
  if (selected_integrator == nullptr) {
    log::error("Failed to select bidirectional integrator for full comparison");
    return false;
  }

  session.cpu_renderer.set_integrator(selected_integrator);
  return true;
}

bool configure_preloaded_scene_for_full_comparison(const BatchRenderOptions& options, BatchRenderSession& session, const SceneRepresentation::IntegratorData& base_integrator_data,
  const Scene::Options& base_scene_options, const Camera& base_camera, const FullComparisonTechniqueInfo& technique, Integrator*& selected_integrator) {
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
    Integrator* integrator = integrator_type_to_instance(type, session.cpu_renderer.integrator_list(), session.cpu_renderer.integrator_count());
    if (integrator != nullptr) {
      integrator->sync_from_options(options_data);
      integrator->update_options();
    }
  }

  selected_integrator = integrator_type_to_instance(Integrator::Type::Bidirectional, session.cpu_renderer.integrator_list(), session.cpu_renderer.integrator_count());
  if (selected_integrator == nullptr) {
    log::error("Failed to select bidirectional integrator for full comparison");
    return false;
  }

  session.cpu_renderer.set_integrator(selected_integrator);
  session.gpu_renderer.cleanup(session.render_context.context());
  if (session.gpu_renderer.camera_controller() == nullptr) {
    session.gpu_renderer.init(session.render_context.context(), session.scene);
  }
  session.cpu_renderer.integrator_thread().request_scene_check();
  session.gpu_renderer.on_scene_changed(session.scene);
  return true;
}

bool load_scene_for_cpu_comparison(const BatchRenderOptions& options, BatchRenderSession& session, const CPUComparisonTechniqueInfo& technique, Integrator*& selected_integrator) {
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

  integrator_data.selected = technique.integrator;
  if (technique.integrator == Integrator::Type::Bidirectional) {
    Options& bdpt_options = integrator_data.settings[Integrator::Type::Bidirectional];
    bdpt_options.set_integral("bdpt-mode", technique.bdpt_mode, "Mode", Option::Meta::EnumValue);
  }

  session.scene.data().options.strategy_flags = technique.strategy_flags;

  session.scene.set_integrator_data(integrator_data);
  apply_batch_scene_overrides(options, session.scene);
  if (configure_batch_render_window(options, session) == false) {
    return false;
  }

  session.cpu_renderer.set_output_dimensions(session.render_context.context(), session.scene.camera().film_size);
  for (const auto& [type, options_data] : integrator_data.settings) {
    Integrator* integrator = integrator_type_to_instance(type, session.cpu_renderer.integrator_list(), session.cpu_renderer.integrator_count());
    if (integrator != nullptr) {
      integrator->sync_from_options(options_data);
      integrator->update_options();
    }
  }

  selected_integrator = integrator_type_to_instance(technique.integrator, session.cpu_renderer.integrator_list(), session.cpu_renderer.integrator_count());
  if (selected_integrator == nullptr) {
    log::error("Failed to select CPU integrator comparison technique '%s'", technique.file_tag);
    return false;
  }

  session.cpu_renderer.set_integrator(selected_integrator);
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
  if (session.gpu_renderer_supported == false) {
    log::error("GPU ray tracing is not supported by the active RHI backend");
    return false;
  }

  session.gpu_renderer.reload_shaders(session.render_context.context(), session.scene);
  if (session.gpu_renderer.finish_preparation(session.render_context.context(), session.scene) == false) {
    log::error("GPU renderer preparation failed before batch rendering");
    return false;
  }

  const uint32_t target_sample_count = max(1u, session.scene.data().options.samples);
  const uint64_t frames_per_sample_budget = std::max<uint64_t>(4096u, static_cast<uint64_t>(session.scene.data().options.max_path_length) + 2u);
  const uint64_t max_gpu_frame_count_u64 = std::max<uint64_t>(1024u, static_cast<uint64_t>(target_sample_count) * frames_per_sample_budget);
  const uint32_t max_gpu_frame_count = static_cast<uint32_t>(std::min<uint64_t>(max_gpu_frame_count_u64, std::numeric_limits<uint32_t>::max()));
  const auto render_begin = std::chrono::steady_clock::now();
  double total_frame_time_ms = 0.0;
  double first_frame_time_ms = 0.0;
  uint32_t frame_index = 0u;
  while ((session.gpu_renderer.completed_samples() < target_sample_count) && (frame_index < max_gpu_frame_count)) {
    const auto frame_begin = std::chrono::steady_clock::now();
    const auto begin_frame_begin = std::chrono::steady_clock::now();
    session.render_context.begin_frame();
    const auto begin_frame_end = std::chrono::steady_clock::now();
    Renderer::FrameData frame_data = {};
    frame_data.dt = 0.0f;
    const auto render_begin = std::chrono::steady_clock::now();
    session.gpu_renderer.render(session.render_context.context(), session.scene, frame_data);
    const auto render_end = std::chrono::steady_clock::now();
    const auto end_frame_begin = std::chrono::steady_clock::now();
    session.render_context.end_frame();
    const auto end_frame_end = std::chrono::steady_clock::now();
    const auto frame_end = std::chrono::steady_clock::now();
    const double frame_time_ms = std::chrono::duration<double, std::milli>(frame_end - frame_begin).count();
    total_frame_time_ms += frame_time_ms;
    if (frame_index == 0u) {
      first_frame_time_ms = frame_time_ms;
      log::info("GPU first-frame timing: total=%.2fms begin_frame=%.2fms render=%.2fms end_frame=%.2fms", frame_time_ms, elapsed_ms(begin_frame_begin, begin_frame_end),
        elapsed_ms(render_begin, render_end), elapsed_ms(end_frame_begin, end_frame_end));
    }

    frame_index += 1u;
    log::info("GPU rendering progress: frames=%u samples=%u / %u", frame_index, session.gpu_renderer.completed_samples(), target_sample_count);
  }
  if (session.gpu_renderer.completed_samples() < target_sample_count) {
    log::error("GPU batch render did not reach target samples (%u / %u) within %u frames", session.gpu_renderer.completed_samples(), target_sample_count, max_gpu_frame_count);
    return false;
  }
  const auto render_end = std::chrono::steady_clock::now();
  const double render_wall_time_ms = std::chrono::duration<double, std::milli>(render_end - render_begin).count();
  const double average_frame_time_ms = total_frame_time_ms / static_cast<double>(frame_index);
  const double steady_state_frame_time_ms = (frame_index > 1u) ? ((total_frame_time_ms - first_frame_time_ms) / static_cast<double>(frame_index - 1u)) : first_frame_time_ms;
  log::info("GPU batch render timing: frames=%u total=%.2fms first=%.2fms avg=%.2fms steady=%.2fms", frame_index, render_wall_time_ms, first_frame_time_ms,
    average_frame_time_ms, steady_state_frame_time_ms);

  image_size = session.gpu_renderer.output_size();
  const auto readback_begin = std::chrono::steady_clock::now();
  if (read_texture_to_float4_buffer(session.render_context.context(), session.gpu_renderer.output_texture(), image_size, output) == false) {
    return false;
  }
  const auto readback_end = std::chrono::steady_clock::now();
  log::info("GPU output readback timing: %.2fms", elapsed_ms(readback_begin, readback_end));

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
  if (session.gpu_renderer_supported == false) {
    log::error("GPU ray tracing is not supported by the active RHI backend");
    return false;
  }

  Integrator* selected_integrator = nullptr;
  if (load_scene_for_batch(options, session, selected_integrator, false) == false) {
    return false;
  }

  const auto shader_reload_begin = std::chrono::steady_clock::now();
  session.gpu_renderer.reload_shaders(session.render_context.context(), session.scene);
  const bool preparation_success = session.gpu_renderer.finish_preparation(session.render_context.context(), session.scene);
  const auto shader_reload_end = std::chrono::steady_clock::now();
  if ((preparation_success == false) || (session.gpu_renderer.pipelines_valid() == false)) {
    log::error("GPU shader compile test failed");
    return false;
  }

  log::info("GPU shader compile test timing: reload_shaders=%.2fms", elapsed_ms(shader_reload_begin, shader_reload_end));
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
  bool strict_comparison_passed = true;
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
      log::error("Full comparison size mismatch for '%s': cpu=%ux%u gpu=%ux%u", technique.file_tag, prepared_cpu_image_size.x, prepared_cpu_image_size.y, prepared_gpu_image_size.x,
        prepared_gpu_image_size.y);
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

    print_comparison_report(technique.file_tag, "full_comparison", absolute_scene_path.c_str(), cpu_output_file.c_str(), gpu_output_file.c_str(), comparison);
    if ((options.strict_comparison) && (full_comparison_passes_strict_gate(technique.file_tag, comparison) == false)) {
      strict_comparison_passed = false;
      log::error("%s", full_comparison_strict_gate_message(technique.file_tag, comparison).c_str());
    }
    append_full_comparison_html_entry(report_html, technique, cpu_output_file, gpu_output_file, comparison);
    if (first_ai_result == false) {
      ai_report_json += ",\n";
    }
    append_ai_comparison_json(ai_report_json, "full_comparison", technique.file_tag, absolute_scene_path.c_str(), cpu_output_file.c_str(), gpu_output_file.c_str(), comparison);
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
  if (strict_comparison_passed == false) {
    log::error("Strict full comparison failed");
    return false;
  }
  return true;
}

bool run_cpu_comparison_batch_render(const BatchRenderOptions& options) {
  const std::string absolute_scene_path = resolve_input_path(options.scene_file);
  const std::string report_file_name = cpu_comparison_report_file_name(absolute_scene_path);
  const std::string ai_report_file_name = cpu_comparison_ai_report_file_name(absolute_scene_path);

  std::vector<CPUComparisonReportEntry> report_entries = {};
  report_entries.reserve(sizeof(kCPUComparisonTechniques) / sizeof(kCPUComparisonTechniques[0]));

  std::string report_html = {};
  std::string ai_report_json = {};
  bool first_ai_result = true;

  const CPUComparisonTechniqueInfo* reference_technique = nullptr;
  std::string reference_output_file = {};

  for (const CPUComparisonTechniqueInfo& technique : kCPUComparisonTechniques) {
    const std::string output_file = cpu_comparison_output_file_name(absolute_scene_path, technique.file_tag);
    log::info("CPU comparison '%s': rendering CPU integrator variant", technique.file_tag);

    BatchRenderSession session = {};
    if (session.init(true, false) == false) {
      return false;
    }

    Integrator* selected_integrator = nullptr;
    if (load_scene_for_cpu_comparison(options, session, technique, selected_integrator) == false) {
      return false;
    }

    std::vector<float4> cpu_output = {};
    uint2 cpu_image_size = {};
    if (run_cpu_preloaded_scene_to_buffer(options, session, cpu_output, cpu_image_size) == false) {
      return false;
    }

    std::vector<float4> prepared_cpu_output = {};
    uint2 prepared_cpu_image_size = {};
    if (prepare_batch_output_buffer(options, cpu_output.data(), cpu_image_size, prepared_cpu_output, prepared_cpu_image_size) == false) {
      return false;
    }

    if (save_batch_output_with_png(output_file, prepared_cpu_output.data(), prepared_cpu_image_size, options.exposure) == false) {
      return false;
    }

    if (technique.reference) {
      reference_technique = &technique;
      reference_output_file = output_file;
    }

    if (reference_technique == nullptr) {
      log::error("CPU comparison reference technique was not initialized before '%s'", technique.file_tag);
      return false;
    }

    ImageComparisonResult comparison = {};
    if (compare_output_to_reference_and_save(options, reference_output_file, output_file, prepared_cpu_output.data(), prepared_cpu_image_size, &comparison) == false) {
      return false;
    }

    print_comparison_report(technique.file_tag, "cpu_comparison", absolute_scene_path.c_str(), reference_output_file.c_str(), output_file.c_str(), comparison);

    CPUComparisonReportEntry entry = {};
    entry.technique = &technique;
    entry.output_file = output_file;
    entry.comparison = comparison;
    report_entries.emplace_back(entry);

    if (first_ai_result == false) {
      ai_report_json += ",\n";
    }
    append_ai_comparison_json(ai_report_json, "cpu_comparison", technique.file_tag, absolute_scene_path.c_str(), reference_output_file.c_str(), output_file.c_str(), comparison);
    first_ai_result = false;
  }

  if ((reference_technique == nullptr) || report_entries.empty()) {
    log::error("CPU comparison did not produce any report entries");
    return false;
  }

  append_cpu_comparison_html_header(report_html, absolute_scene_path, *reference_technique);
  append_cpu_comparison_html_reference(report_html, report_entries.front());
  append_cpu_comparison_html_summary(report_html, report_entries);
  append_cpu_comparison_html_cards(report_html, report_entries);
  append_cpu_comparison_html_footer(report_html);

  ai_report_json = "{\n"
                   "  \"schema\": \"etx.cpu_comparison.v1\",\n"
                   "  \"scene\": \"" +
                   json_escape(absolute_scene_path) + "\",\n"
                                                    "  \"reference\": \"" +
                   json_escape(reference_output_file) + "\",\n"
                                                        "  \"results\": [\n" +
                   ai_report_json + "\n  ]\n}\n";

  if (save_text_to_file(report_file_name, report_html) == false) {
    return false;
  }
  if (save_text_to_file(ai_report_file_name, ai_report_json) == false) {
    return false;
  }

  log::info("Saved CPU comparison report to %s", report_file_name.c_str());
  log::info("Saved CPU comparison AI report to %s", ai_report_file_name.c_str());
  return true;
}

}  // namespace

BatchModeCommand parse_batch_command_line(int argc, char* argv[], BatchRenderOptions& options, std::string& message) {
  bool render_requested = false;
  bool full_comparison_requested = false;
  bool cpu_comparison_requested = false;
  bool generate_bsdf_luts_requested = false;
  bool pregenerate_bsdf_lut_cache_requested = false;
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

    if (argument == "--cpu-comparison") {
      cpu_comparison_requested = true;
      options.cpu_comparison = true;
      batch_argument_seen = true;
      continue;
    }

    if (argument == "--generate-bsdf-luts") {
      generate_bsdf_luts_requested = true;
      batch_argument_seen = true;
      continue;
    }

    if (argument == "--pregenerate-bsdf-lut-cache") {
      pregenerate_bsdf_lut_cache_requested = true;
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

    if (argument == "--bsdf-lut-samples") {
      batch_argument_seen = true;
      if ((i + 1) >= argc) {
        message = "Missing value for --bsdf-lut-samples\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      if (parse_u32_argument(argv[i + 1], options.bsdf_lut_samples) == false) {
        message = "Invalid value for --bsdf-lut-samples\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      if (options.bsdf_lut_samples == 0u) {
        message = "--bsdf-lut-samples must be greater than zero\n\n";
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

    if (argument == "--bdpt-mode") {
      batch_argument_seen = true;
      if ((i + 1) >= argc) {
        message = "Missing value for --bdpt-mode\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      if (parse_bdpt_mode_argument(argv[i + 1], options.bdpt_mode) == false) {
        message = "Invalid value for --bdpt-mode. Expected pt, lt, bdpt-fast, or bdpt-full\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      options.override_bdpt_mode = true;
      i += 1;
      continue;
    }

    if (argument == "--gpu-compile-only") {
      batch_argument_seen = true;
      options.gpu_compile_only = true;
      continue;
    }

    if (argument == "--strict-comparison") {
      batch_argument_seen = true;
      options.strict_comparison = true;
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

  const uint32_t selected_batch_modes =
    uint32_t(render_requested) + uint32_t(full_comparison_requested) + uint32_t(cpu_comparison_requested) + uint32_t(generate_bsdf_luts_requested) +
    uint32_t(pregenerate_bsdf_lut_cache_requested);
  if (selected_batch_modes > 1u) {
    message = "Use exactly one of --render, --full-comparison, --cpu-comparison, --generate-bsdf-luts, or --pregenerate-bsdf-lut-cache\n\n";
    message += batch_usage_string();
    return BatchModeCommand::Error;
  }

  if ((render_requested == false) && (full_comparison_requested == false) && (cpu_comparison_requested == false) && (generate_bsdf_luts_requested == false) &&
      (pregenerate_bsdf_lut_cache_requested == false)) {
    if (batch_argument_seen) {
      message = "Batch options require --render, --full-comparison, --cpu-comparison, --generate-bsdf-luts, or --pregenerate-bsdf-lut-cache\n\n";
      message += batch_usage_string();
      return BatchModeCommand::Error;
    }
    return BatchModeCommand::None;
  }

  if (generate_bsdf_luts_requested) {
    if ((options.scene_file.empty() == false) || (options.reference_file.empty() == false) || (options.compare_mode.empty() == false) ||
        (options.integrator.empty() == false) || (options.renderer != "cpu") || (options.samples > 0u) || (options.max_path_length > 0u) ||
        (options.gpu_compile_only) || (options.gpu_compile_stage.empty() == false) || (options.strict_comparison) || (options.denoise) ||
        (options.override_random_seed) || (options.override_resolution) || (options.override_crop) || (options.override_strategy_flags) || (options.override_bdpt_mode) ||
        (options.exposure != 1.0f)) {
      message = "--generate-bsdf-luts accepts only --output and --bsdf-lut-samples\n\n";
      message += batch_usage_string();
      return BatchModeCommand::Error;
    }

    return BatchModeCommand::GenerateBSDFLuts;
  }

  if (pregenerate_bsdf_lut_cache_requested) {
    if ((options.scene_file.empty() == false) || (options.output_file.empty() == false) || (options.reference_file.empty() == false) || (options.compare_mode.empty() == false) ||
        (options.integrator.empty() == false) || (options.renderer != "cpu") || (options.samples > 0u) || (options.max_path_length > 0u) ||
        (options.gpu_compile_only) || (options.gpu_compile_stage.empty() == false) || (options.strict_comparison) || (options.denoise) ||
        (options.override_random_seed) || (options.override_resolution) || (options.override_crop) || (options.override_strategy_flags) || (options.override_bdpt_mode) || (options.exposure != 1.0f) ||
        (options.bsdf_lut_samples != 512u)) {
      message = "--pregenerate-bsdf-lut-cache does not accept additional options\n\n";
      message += batch_usage_string();
      return BatchModeCommand::Error;
    }

    return BatchModeCommand::PregenerateBSDFLutCache;
  }

  if (full_comparison_requested) {
    if (options.scene_file.empty()) {
      message = "Full comparison requires --scene\n\n";
      message += batch_usage_string();
      return BatchModeCommand::Error;
    }

    if ((options.output_file.empty() == false) || (options.reference_file.empty() == false) || (options.compare_mode.empty() == false) || (options.renderer != "cpu") ||
        (options.integrator.empty() == false) || options.gpu_compile_only || (options.override_bdpt_mode)) {
      message = "--full-comparison does not accept --output, --reference, --compare, --renderer, --integrator, --bdpt-mode, or --gpu-compile-only\n\n";
      message += batch_usage_string();
      return BatchModeCommand::Error;
    }

    return BatchModeCommand::Run;
  }

  if (cpu_comparison_requested) {
    if (options.scene_file.empty()) {
      message = "CPU comparison requires --scene\n\n";
      message += batch_usage_string();
      return BatchModeCommand::Error;
    }

    if ((options.output_file.empty() == false) || (options.reference_file.empty() == false) || (options.compare_mode.empty() == false) || (options.renderer != "cpu") ||
        (options.integrator.empty() == false) || options.gpu_compile_only || (options.gpu_compile_stage.empty() == false) || (options.strict_comparison) ||
        (options.override_bdpt_mode)) {
      message =
        "--cpu-comparison does not accept --output, --reference, --compare, --renderer, --integrator, --bdpt-mode, --gpu-compile-only, --gpu-compile-stage, or --strict-comparison\n\n";
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

  if (options.strict_comparison) {
    message = "--strict-comparison requires --full-comparison\n\n";
    message += batch_usage_string();
    return BatchModeCommand::Error;
  }

  return BatchModeCommand::Run;
}

int run_batch_render(const BatchRenderOptions& options) {
  if (options.full_comparison) {
    return run_full_comparison_batch_render(options) ? 0 : 1;
  }
  if (options.cpu_comparison) {
    return run_cpu_comparison_batch_render(options) ? 0 : 1;
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
