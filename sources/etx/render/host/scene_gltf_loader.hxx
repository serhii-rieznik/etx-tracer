#pragma once

#include <etx/render/host/scene_data.hxx>
#include <etx/render/host/scene_loader_utils.hxx>
#include <etx/render/shared/camera.hxx>

namespace etx {

struct Scene;
struct TaskScheduler;

uint32_t load_from_gltf_file(const char* file_name, bool binary, SceneData& data, Scene& scene, TaskScheduler& scheduler, Camera& active_camera);

}  // namespace etx
#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/host/scene_data.hxx>
#include <etx/render/host/scene_loader_utils.hxx>
#include <etx/render/shared/math.hxx>

namespace etx {

struct Scene;
struct IORDatabase;
struct TaskScheduler;

struct SceneGltfLoader {
  SceneGltfLoader();
  ~SceneGltfLoader();

  uint32_t load_from_file(const char* file_name, SceneData& data, Scene& scene, const IORDatabase& database, TaskScheduler& scheduler);

 private:
  ETX_DECLARE_PIMPL(SceneGltfLoader, 1024);
};

}  // namespace etx
