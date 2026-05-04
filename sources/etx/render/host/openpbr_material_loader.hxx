#pragma once

#include <etx/render/host/scene_data.hxx>

#include <filesystem>

namespace etx {

bool load_openpbr_material_file(const std::filesystem::path& path, SceneData& data, Material& material);

}  // namespace etx
