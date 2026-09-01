#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/host/tasks.hxx>

namespace etx {

struct Film;
struct Scene;
struct SceneData;
struct UpdateFlags;

struct Raytracing {
  Raytracing(TaskScheduler&, Film&);
  ~Raytracing();

  TaskScheduler& scheduler();

  const Film& film() const;
  Film& film();

  const Camera& camera() const;

  const Scene& scene() const;
  float geometry_bounding_sphere_radius() const;
  uint32_t sample_limit() const;
  void set_sample_limit(uint32_t sample_limit);
  void commit(const SceneData& scene_data, const Camera& camera, const UpdateFlags& changes);

  bool trace(const Scene& scene, const Ray&, Intersection&, Sampler& smp) const;
  bool trace_material(const Scene& scene, const Ray&, const uint32_t material_id, Intersection&, Sampler& smp) const;
  SpectralResponse trace_transmittance(const SpectralQuery spect, const Scene& scene, const float3& p0, const float3& p1, const MediumInstance& medium, Sampler& smp) const;

 private:
  ETX_DECLARE_PIMPL(Raytracing, 4096);
};

}  // namespace etx
