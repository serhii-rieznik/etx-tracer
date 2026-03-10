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
  uint32_t samples = 0u;
  bool denoise = false;
  float exposure = 1.0f;
};

enum class BatchModeCommand {
  None,
  Run,
  Help,
  Error,
};

BatchModeCommand parse_batch_command_line(int argc, char* argv[], BatchRenderOptions& options, std::string& message);
int run_batch_render(const BatchRenderOptions& options);

}  // namespace etx
