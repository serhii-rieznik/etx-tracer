#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/host/scene_data.hxx>
#include <etx/render/host/scene_loader_utils.hxx>
#include <etx/render/shared/math.hxx>

namespace etx {

struct IORDatabase;
struct TaskScheduler;

struct SceneGltfLoader {
  SceneGltfLoader();
  ~SceneGltfLoader();

  uint32_t load_from_file(const char* file_name, SceneData& data, const IORDatabase& database, TaskScheduler& scheduler);

 private:
  ETX_DECLARE_PIMPL(SceneGltfLoader, 1024);
};

}  // namespace etx
