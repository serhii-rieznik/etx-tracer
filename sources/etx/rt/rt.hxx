#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/host/tasks.hxx>

namespace etx {

struct Film;
struct Scene;
struct SceneData;
struct UpdateFlags;

// Geometry queries retain their original parameterization across null boundaries.
struct BoundaryRayTraversal {
  Ray ray = {};
  float distance = 0.0f;
  float hit_distance = 0.0f;

  void reset(const Ray& physical_ray) {
    ray = physical_ray;
    distance = 0.0f;
    hit_distance = 0.0f;
  }

  void record_hit(Intersection& intersection) {
    hit_distance = intersection.t;
    intersection.t -= distance;
  }

  void advance(Ray& physical_ray) {
    distance = hit_distance;
    physical_ray.o = ray.o + ray.d * distance;
    physical_ray.min_t = kMinNormalFloat;
    physical_ray.max_t = ray.max_t - distance;
    ray.min_t = std::nextafter(distance, kMaxFloat);
  }
};

struct Raytracing {
  static constexpr const char* kMediumEmissionFailure = "Medium emission is divergent or outside the numeric range; bound the emitting region or provide absorption";
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
  bool medium_emission_failed() const;
  void commit(const SceneData& scene_data, const Camera& camera, const UpdateFlags& changes);

  bool trace(const Scene& scene, const Ray&, Intersection&, Sampler& smp) const;
  bool trace_material(const Scene& scene, const Ray&, const uint32_t material_id, Intersection&, Sampler& smp) const;
  SpectralResponse trace_transmittance(const SpectralQuery spect, const Scene& scene, const float3& p0, const float3& p1, const MediumInstance& medium, Sampler& smp) const;
  SpectralResponse trace_transmittance(const SpectralQuery spect, const Scene& scene, const float3& p0, const float3& p1, const MediumInstance& medium, bool source_collision,
    bool target_collision, Sampler& smp) const;
  SpectralResponse trace_transmittance(const SpectralQuery spect, const Scene& scene, const float3& p0, const float3& p1, const MediumInstance& medium, bool source_collision,
    bool target_collision, Sampler& smp, float& flight_pdf_forward, float& flight_pdf_reverse) const;

 private:
  ETX_DECLARE_PIMPL(Raytracing, 4096);
};

}  // namespace etx
