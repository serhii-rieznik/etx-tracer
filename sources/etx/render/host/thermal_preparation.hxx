#pragma once

namespace etx {
struct SceneData;
bool thermal_medium_binding_valid(const SceneData& data, uint32_t index);
bool prepare_thermal_materials(SceneData& data, uint32_t camera_medium_index);
}  // namespace etx
