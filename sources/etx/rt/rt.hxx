#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/host/tasks.hxx>
#include <etx/render/shared/scene.hxx>

namespace etx {

struct Raytracing {
  Raytracing();
  ~Raytracing();

  TaskScheduler& scheduler();

  const Scene& scene() const;
  bool has_scene() const;
  void set_scene(const Scene&);

  bool trace(const Ray&, Intersection&, Sampler& smp) const;

 private:
  ETX_DECLARE_PIMPL(Raytracing, 512);
};

}  // namespace etx