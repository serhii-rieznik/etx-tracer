#pragma once

#include <vector>
#include <etx/render/shared/scene.hxx>
#include <etx/rt/shared/path_tracing_shared.hxx>

namespace etx {
struct Raytracing;
struct Sampler;
}  // namespace etx

namespace etx::mnee {

// Result of Manifold Next Event Estimation solver
struct Result {
  // Accumulated spectral weight of the specular chain including the emitter contribution
  SpectralResponse weight = {};
  // Area-domain PDF for the full connection (used for MIS)
  float pdf_area = 0.0f;
  // Direction from camera vertex to first specular surface (for BSDF evaluation)
  float3 camera_to_first_surface = {};
};

// Build a specular chain from current position towards light (for reverse MNEE)
bool build_reverse_specular_chain(const Scene& scene, const float3& start_pos, const float3& light_pos, std::vector<Intersection>& chain, const Raytracing& rt, Sampler& smp);

// Attempt to connect a camera-side vertex to a light through a sequence of specular (delta) surfaces.
//
// Parameters:
//   scene      – reference to the current scene
//   spect      – current spectral query
//   cam_vtx    – intersection information of the camera-side vertex (last diffuse / glossy event)
//   chain      – ordered list of consecutive specular intersections starting from cam_vtx and going toward the light
//   result     – filled with weights / pdf if the solver succeeds
//
// Returns true if a valid manifold solution was found.

// Refract a vector w_i through surface with normal n and eta ratio. Returns zero vector on total internal reflection.
float3 refract(const float3& w_i, const float3& n, float eta, bool& tir);

// Reflect helper (identical to math::reflect but provided here for symmetry)
float3 reflect(const float3& w_i, const float3& n);

// Return geometric surface partial derivatives dP/du, dP/dv required by the manifold Jacobian.
struct SurfaceDerivatives {
  float3 dp_du = {};
  float3 dp_dv = {};
};
SurfaceDerivatives derivatives(const Scene& scene, const Intersection& isect);

// Light endpoint for MNEE
struct LightEndpoint {
  float3 position = {};
  float3 normal = {};
  uint32_t emitter_index = kInvalidIndex;  // invalid for environment
  float pdf_area = 0.0f;                   // area-domain pdf of sampling this point
  SpectralResponse radiance = {};
};

// Ray propagation helpers
struct RaySegment {
  float3 origin = {};
  float3 direction = {};
};

// Reflect segment across a surface at point p with normal n.
ETX_GPU_CODE RaySegment propagate_reflect(const float3& p, const float3& n, const float3& w_i);

// Refract segment through surface; returns invalid segment if TIR.
ETX_GPU_CODE RaySegment propagate_refract(const float3& p, const float3& n, const float3& w_i, float eta, bool& tir);

// Propagates a path through a specular chain.
ETX_GPU_CODE RaySegment propagate_path(const Scene& scene, SpectralQuery spect, const std::vector<Intersection>& chain, const RaySegment& initial_ray, bool& is_valid);

// Computes the Jacobian determinant of the specular path transformation for MIS PDF conversion.
float compute_jacobian_determinant(const Scene& scene, SpectralQuery spect, const Intersection& cam_vtx, const std::vector<Intersection>& chain, const LightEndpoint& light);

// Placeholder for the main iterative solver.
bool solve_iterative(const Scene& scene, SpectralQuery spect, const Intersection& cam_vtx, const std::vector<Intersection>& chain, const LightEndpoint& light,
  const SpectralResponse& cam_throughput, Result& result, const etx::Raytracing& rt, etx::Sampler& smp);

// Sample a local-area emitter given a fixed outgoing direction from point "origin".
// Returns true if the ray hits the emitter; fills LightEndpoint accordingly.
bool sample_area_emitter_for_direction(SpectralQuery spect, const Scene& scene, uint32_t emitter_index, const float3& origin, const float3& direction, LightEndpoint& light_out,
  const etx::Raytracing& rt, etx::Sampler& smp);

bool solve_camera_to_light(const Scene& scene, SpectralQuery spect, const Intersection& cam_vtx, const std::vector<Intersection>& chain, const LightEndpoint& light,
  const SpectralResponse& cam_throughput, Result& result, const etx::Raytracing& rt, etx::Sampler& smp);

// Solve reverse MNEE (from diffuse surface through specular chain to light)
bool solve_reverse_camera_to_light(const Scene& scene, SpectralQuery spect, const Intersection& diffuse_vtx, const std::vector<Intersection>& chain, const LightEndpoint& light,
  const SpectralResponse& diffuse_throughput, Result& result, const etx::Raytracing& rt, etx::Sampler& smp);

}  // namespace etx::mnee
