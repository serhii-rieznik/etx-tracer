#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/host/tasks.hxx>
#include <etx/render/shared/scene.hxx>

#include <etx/gpu/gpu.hxx>

namespace etx {

struct Raytracing {
  struct ContinousTraceOptions {
    IntersectionBase* intersection_buffer = nullptr;
    uint32_t max_intersections = 0;
    uint32_t material_id = kInvalidIndex;
  };

  Raytracing();
  ~Raytracing();

  TaskScheduler& scheduler();
  GPUDevice* gpu();

  const Scene& scene() const;

  const Scene& gpu_scene() const;

  bool has_scene() const;
  void set_scene(const Scene&);

  bool trace(const Scene& scene, const Ray&, Intersection&, Sampler& smp) const;
  uint32_t continuous_trace(const Scene& scene, const Ray&, const ContinousTraceOptions& options, Sampler& smp) const;

  Intersection make_intersection(const Scene& scene, const float3& w_i, const IntersectionBase&) const;

 private:
  ETX_DECLARE_PIMPL(Raytracing, 1024);
};

}  // namespace etx
