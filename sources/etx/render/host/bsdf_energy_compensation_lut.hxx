#pragma once

#include <etx/render/host/scene_data.hxx>
#include <etx/rhi/rhi_types.hxx>

namespace etx {

struct RHIContext;

enum class EnergyCompensationGenerationResult : uint32_t {
  Pending,
  Complete,
  Failed,
};

struct EnergyCompensationGenerationContext {
  RHIPipeline pipeline = {};
  void* pending_step = nullptr;
  uint32_t completed_steps = 0u;
  uint32_t total_steps = 0u;
  bool initialized = false;
  bool pipeline_initialization_failed = false;
};

bool ensure_energy_compensation_interfaces(SceneData& data, TaskScheduler& scheduler);
bool ensure_energy_compensation_interfaces(SceneData& data, TaskScheduler& scheduler, RHIContext& rhi);
EnergyCompensationGenerationResult generate_energy_compensation_interfaces_step(SceneData& data, TaskScheduler& scheduler, RHIContext& rhi,
  EnergyCompensationGenerationContext& context);
void cleanup_energy_compensation_generation(RHIContext& rhi, EnergyCompensationGenerationContext& context);
bool validate_energy_compensation_gpu_lut_parity(RHIContext& rhi, TaskScheduler& scheduler);

}  // namespace etx
