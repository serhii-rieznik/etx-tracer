#pragma once

#include <etx/render/host/scene_data.hxx>

namespace etx {

bool ensure_energy_compensation_interfaces(SceneData& data, TaskScheduler& scheduler);

}  // namespace etx
