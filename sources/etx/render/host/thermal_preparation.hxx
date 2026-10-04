#pragma once

namespace etx {
struct SceneData;
inline constexpr const char* kThermalCameraBindingFailure = "Camera medium has interiors at different temperatures; select a distinct authored medium";
bool thermal_medium_binding_valid(const SceneData& data, uint32_t index);
bool prepare_thermal_materials(SceneData& data, uint32_t camera_medium_index);
}  // namespace etx
