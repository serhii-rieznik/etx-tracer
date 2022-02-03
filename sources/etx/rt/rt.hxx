#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/shared/scene.hxx>

namespace etx {

struct Raytracing {
  Raytracing() = default;
  ~Raytracing();

  const Scene& scene() const;
  void set_scene(const Scene&);

  bool trace(const Ray&, Intersection&, Sampler& smp) const;

  ETX_DECLARE_PIMPL(Raytracing, 32);
};

}  // namespace etx