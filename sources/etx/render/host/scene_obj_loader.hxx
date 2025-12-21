#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/host/scene_data.hxx>
#include <etx/render/host/scene_loader_utils.hxx>
#include <etx/render/shared/math.hxx>

#include <string>
#include <map>

namespace etx {

struct IORDatabase;
struct TaskScheduler;

uint32_t load_from_obj_file(const char* obj_file_name, const char* mtl_file_name, SceneData& data, const IORDatabase& database, TaskScheduler& scheduler);

}  // namespace etx
