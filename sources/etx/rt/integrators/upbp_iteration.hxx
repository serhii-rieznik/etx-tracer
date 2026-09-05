#pragma once

#include <etx/rt/integrators/upbp_density_mis.hxx>
#include <etx/rt/integrators/upbp_options.hxx>
#include <etx/rt/shared/vcm_shared.hxx>

namespace etx {

struct UPBPIterationParameters {
  SpectralQuery spect = {};
  UPBPDensityMISConfiguration mis = {};
  double surface_radius = 0.0;
  double pp3d_radius = 0.0;
  double pb2d_radius = 0.0;
  double bp2d_radius = 0.0;
  double bb1d_radius = 0.0;
  uint64_t camera_subpath_count = 0u;
  uint64_t light_subpath_count = 0u;
  uint64_t bb1d_light_subpath_count = 0u;
  uint64_t bpt_sample_count = 0u;
};

inline double upbp_initial_radius(const float configured_radius, const float bounding_sphere_radius, const uint64_t camera_subpath_count, const uint64_t light_subpath_count,
  const uint32_t dimension, const double relative_radius_scale) {
  if (configured_radius > 0.0f) {
    return configured_radius;
  }
  return upbp_automatic_initial_radius(bounding_sphere_radius, camera_subpath_count, light_subpath_count, dimension, relative_radius_scale);
}

inline UPBPIterationParameters upbp_iteration_parameters(const UPBPOptions& options, const float bounding_sphere_radius, const uint2& film_dimensions, const SpectralQuery& spect,
  const bool merge_vertices_enabled, const uint64_t path_count, const uint64_t iteration) {
  UPBPIterationParameters result = {};
  result.camera_subpath_count = path_count;
  result.light_subpath_count = path_count;
  result.bpt_sample_count = options.enabled(UPBPTechnique::BPT) ? 1u : 0u;
  result.spect = spect;
  result.mis.enabled_techniques = upbp_effective_technique_mask(options, merge_vertices_enabled);
  if (result.mis.enabled(UPBPTechnique::BB1D)) {
    result.bb1d_light_subpath_count = options.maximum_bb1d_light_path_count > 0u ? min(path_count, static_cast<uint64_t>(options.maximum_bb1d_light_path_count)) : path_count;
  }

  const double surface_radius_scale = kUPBPAutomaticSurfaceRadiusScale / static_cast<double>(max(film_dimensions.x, film_dimensions.y));
  result.surface_radius = upbp_progressive_radius(
    upbp_initial_radius(options.initial_surface_radius, bounding_sphere_radius, result.camera_subpath_count, result.light_subpath_count, 2u, surface_radius_scale),
    options.radius_alpha, 2u, iteration);
  result.pp3d_radius = upbp_progressive_radius(
    upbp_initial_radius(options.initial_pp3d_radius, bounding_sphere_radius, result.camera_subpath_count, result.light_subpath_count, 3u, kUPBPAutomaticVolumeRadiusScale),
    options.radius_alpha, 3u, iteration);
  result.pb2d_radius = upbp_progressive_radius(
    upbp_initial_radius(options.initial_pb2d_radius, bounding_sphere_radius, result.camera_subpath_count, result.light_subpath_count, 2u, kUPBPAutomaticVolumeRadiusScale),
    options.radius_alpha, 2u, iteration);
  result.bp2d_radius = upbp_progressive_radius(
    upbp_initial_radius(options.initial_bp2d_radius, bounding_sphere_radius, result.camera_subpath_count, result.light_subpath_count, 2u, kUPBPAutomaticVolumeRadiusScale),
    options.radius_alpha, 2u, iteration);
  const double bb1d_sample_fraction =
    result.light_subpath_count > 0u ? static_cast<double>(result.bb1d_light_subpath_count) * options.beam_selection_probability / static_cast<double>(result.light_subpath_count)
                                    : 0.0;
  result.bb1d_radius = upbp_progressive_radius_for_sample_fraction(
    upbp_initial_radius(options.initial_bb1d_radius, bounding_sphere_radius, result.camera_subpath_count, result.light_subpath_count, 1u, kUPBPAutomaticVolumeRadiusScale),
    options.radius_alpha, 1u, iteration, bb1d_sample_fraction);

  result.mis.technique_factors[0u] = result.bpt_sample_count;
  result.mis.technique_factors[1u] = upbp_density_mis_factor(UPBPTechnique::Surface, result.light_subpath_count, result.surface_radius, 1.0);
  result.mis.technique_factors[2u] = upbp_density_mis_factor(UPBPTechnique::PP3D, result.light_subpath_count, result.pp3d_radius, 1.0);
  result.mis.technique_factors[3u] = upbp_density_mis_factor(UPBPTechnique::PB2D, result.light_subpath_count, result.pb2d_radius, 1.0);
  result.mis.technique_factors[4u] = upbp_density_mis_factor(UPBPTechnique::BP2D, result.light_subpath_count, result.bp2d_radius, 1.0);
  result.mis.technique_factors[5u] = upbp_density_mis_factor(UPBPTechnique::BB1D, result.bb1d_light_subpath_count, result.bb1d_radius, options.beam_selection_probability);
  result.mis.photon_beams_long = false;
  result.mis.camera_beams_long = true;
  return result;
}

}  // namespace etx
