#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/shared/scene.hxx>

namespace etx {

struct SceneRepresentation {
  SceneRepresentation();
  ~SceneRepresentation();

  bool load_from_file(const char* filename);

  const Scene& scene() const;

  operator bool() const;

  ETX_DECLARE_PIMPL(SceneRepresentation, 2048);
};

}  // namespace etx
