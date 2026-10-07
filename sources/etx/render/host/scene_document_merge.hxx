#pragma once

#include <etx/render/host/scene_data.hxx>

namespace etx {

// The caller owns an isolated staging document and publishes it only after preparation succeeds.
bool append_scene_document(SceneData& destination, const SceneData& source, std::string& error);

}  // namespace etx
