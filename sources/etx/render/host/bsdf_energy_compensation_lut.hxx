#pragma once

#include <etx/render/host/scene_data.hxx>

namespace etx {

struct RHIContext;

bool ensure_energy_compensation_interfaces(SceneData& data, TaskScheduler& scheduler);
bool ensure_energy_compensation_interfaces(SceneData& data, TaskScheduler& scheduler, RHIContext& rhi);
bool validate_energy_compensation_gpu_lut_parity(RHIContext& rhi, TaskScheduler& scheduler);

}  // namespace etx
