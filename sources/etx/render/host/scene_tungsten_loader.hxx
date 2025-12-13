#pragma once

#include <etx/render/host/scene_data.hxx>
#include <etx/render/host/scene_loader_utils.hxx>
#include <etx/render/shared/camera.hxx>

namespace etx {

struct Scene;
struct IORDatabase;
struct TaskScheduler;

uint32_t load_from_tungsten_file(const char* file_name, SceneData& data, Scene& scene, const IORDatabase& database, TaskScheduler& scheduler, Camera& active_camera);

}  // namespace etx
