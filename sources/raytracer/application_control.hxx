#pragma once

#include "renderer.hxx"
#include "options.hxx"

#include <etx/rt/integrators/integrator.hxx>

#include <cstdint>
#include <string>
#include <vector>

namespace etx {

enum class ApplicationCommandType : uint32_t {
  LoadScene,
  SaveScene,
  LoadReferenceImage,
  SaveImage,
  Denoise,
  SetRenderConfiguration,
  SetRenderer,
  SetIntegrator,
  Run,
  Finish,
  Stop,
  Restart,
  ReloadScene,
  ReloadGeometry,
  ReloadShaders,
  CancelPreparation,
  SetExposure,
  SetViewLayer,
  SetOutputView,
  SetDisplayTransform,
  Quit,
};

struct ApplicationCommand {
  uint64_t id = 0u;
  ApplicationCommandType type = ApplicationCommandType::Run;
  std::string path = {};
  RendererMode renderer = RendererMode::CPURaytracing;
  Integrator::Type integrator = Integrator::Type::Invalid;
  SaveImageMode save_image_mode = SaveImageMode::RGB;
  uint32_t unsigned_value = 0u;
  float float_value = 0.0f;
};

struct ApplicationCommandResult {
  uint64_t command_id = 0u;
  bool success = false;
  std::string message = {};
};

struct ApplicationIntegratorInfo {
  uint32_t value = 0u;
  std::string id = {};
  std::string name = {};
  bool enabled = false;
};

struct ApplicationStateSnapshot {
  uint64_t revision = 0u;
  bool initialized = false;
  bool scene_loaded = false;
  bool gpu_renderer_available = false;
  bool can_denoise = false;
  bool quit_requested = false;
  std::string scene_file = {};
  std::string renderer_name = {};
  std::string integrator_name = {};
  RendererMode renderer_mode = RendererMode::CPURaytracing;
  Integrator::Type integrator_type = Integrator::Type::Invalid;
  std::vector<ApplicationIntegratorInfo> integrators = {};
  RendererPreparationStatus preparation = {};
  RendererStatus status = {};
  RendererControlState controls = {};
  ViewParameters view = {};
};

}  // namespace etx
