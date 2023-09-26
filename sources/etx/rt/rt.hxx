#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/host/tasks.hxx>
#include <etx/render/shared/scene.hxx>

#include <etx/gpu/gpu.hxx>

namespace etx {

struct Raytracing {
  Raytracing();
  ~Raytracing();

  TaskScheduler& scheduler();
  GPUDevice* gpu();

  const Scene& scene() const;

  const Scene& gpu_scene() const;

  bool has_scene() const;
  void set_scene(const Scene&);

  bool trace(const Scene& scene, const Ray&, Intersection&, Sampler& smp) const;
  bool trace_material(const Scene& scene, const Ray&, const uint32_t material_id, Intersection&, Sampler& smp) const;
  uint32_t continuous_trace(const Scene& scene, const Ray&, const ContinousTraceOptions& options, Sampler& smp) const;
  SpectralResponse trace_transmittance(const SpectralQuery spect, const Scene& scene, const float3& p0, const float3& p1, const uint32_t medium, Sampler& smp) const;

 private:
  ETX_DECLARE_PIMPL(Raytracing, 1024);
};

}  // namespace etx
