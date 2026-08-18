#pragma once

#include <cstdint>
#include <string>

namespace etx {

struct BatchRenderOptions {
  std::string scene_file = {};
  std::string output_file = {};
  std::string reference_file = {};
  std::string integrator = {};
  std::string renderer = "cpu";
  std::string compare_mode = {};
  std::string gpu_compile_stage = {};
  uint32_t bdpt_mode = 0u;
  uint32_t samples = 0u;
  uint32_t max_path_length = 0u;
  uint32_t random_seed = 0u;
  uint32_t resolution_width = 0u;
  uint32_t resolution_height = 0u;
  uint32_t crop_x = 0u;
  uint32_t crop_y = 0u;
  uint32_t crop_width = 0u;
  uint32_t crop_height = 0u;
  uint32_t strategy_flags = 0u;
  uint32_t bsdf_lut_samples = 512u;
  uint32_t gpu_wavefront_steps_per_frame = 256u;
  bool gpu_compile_only = false;
  bool gpu_kernel_timings = false;
  bool full_comparison = false;
  bool cpu_comparison = false;
  bool strict_comparison = false;
  bool denoise = false;
  bool override_random_seed = false;
  bool override_resolution = false;
  bool override_crop = false;
  bool override_strategy_flags = false;
  bool override_bdpt_mode = false;
  float exposure = 1.0f;
};

enum class BatchModeCommand {
  None,
  Run,
  GenerateBSDFLuts,
  PregenerateBSDFLutCache,
  Help,
  Error,
};

BatchModeCommand parse_batch_command_line(int argc, char* argv[], BatchRenderOptions& options, std::string& message);
int run_batch_render(const BatchRenderOptions& options);

}  // namespace etx
