#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/host/tasks.hxx>
#include <etx/render/shared/scene.hxx>

#include <etx/gpu/gpu.hxx>

namespace etx {

struct Film;

struct ContinousTraceOptions {
  IntersectionBase* intersection_buffer = nullptr;
  uint32_t max_intersections = 0;
  uint32_t material_id = kInvalidIndex;
};

struct Raytracing {
  Raytracing();
  ~Raytracing();

  TaskScheduler& scheduler();

  const Film& film() const;
  Film& film();

  void link_camera(const Camera& camera);
  const Camera& camera() const;

  void link_scene(const Scene&);
  const Scene& scene() const;

  void commit_changes();

  bool trace(const Scene& scene, const Ray&, Intersection&, Sampler& smp) const;
  bool trace_material(const Scene& scene, const Ray&, const uint32_t material_id, Intersection&, Sampler& smp) const;
  uint32_t continuous_trace(const Scene& scene, const Ray&, const ContinousTraceOptions& options, Sampler& smp) const;
  SpectralResponse trace_transmittance(const SpectralQuery spect, const Scene& scene, const float3& p0, const float3& p1, const Medium::Instance& medium, Sampler& smp) const;

 private:
  ETX_DECLARE_PIMPL(Raytracing, 1024);
};

}  // namespace etx
