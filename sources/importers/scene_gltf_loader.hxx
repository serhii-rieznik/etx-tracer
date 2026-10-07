#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/host/scene_data.hxx>
#include <etx/render/host/scene_loader_utils.hxx>
#include <etx/render/shared/math.hxx>

namespace etx {

struct IORDatabase;
struct TaskScheduler;

uint32_t load_from_gltf_file(const char* file_name, bool binary, SceneData& data, TaskScheduler& scheduler, Camera& active_camera);

}  // namespace etx
