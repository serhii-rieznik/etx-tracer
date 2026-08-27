#include <etx/rt/integrators/upbp_core.hxx>
#include <etx/rt/integrators/upbp_connection.hxx>
#include <etx/rt/integrators/upbp_bpt.hxx>
#include <etx/rt/integrators/upbp_density_mis.hxx>
#include <etx/rt/integrators/upbp_recursive_mis.hxx>
#include <etx/rt/integrators/upbp_point_merge.hxx>
#include <etx/rt/integrators/upbp_beam_estimators.hxx>
#include <etx/rt/integrators/upbp_bpt_cross_mis.hxx>
#include <etx/rt/integrators/upbp_options.hxx>
#include <etx/rt/integrators/upbp_scene_path.hxx>
#include <etx/rt/integrators/upbp_spatial.hxx>
#include <etx/rt/integrators/upbp_subpath.hxx>

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

bool close_value(const double actual, const double expected, const double tolerance, const char* label) {
  if (fabs(actual - expected) <= tolerance) {
    return true;
  }

  std::printf("%s failed: actual %.12f expected %.12f tolerance %.12f\n", label, actual, expected, tolerance);
  return false;
}

bool validate_kernel(const etx::UPBPKernel kernel, const uint32_t dimension) {
  constexpr uint32_t sample_count = 1u << 19u;
  constexpr double radius = 2.3;
  const double step = radius / static_cast<double>(sample_count);
  double integral = 0.0;

  for (uint32_t sample_index = 0u; sample_index < sample_count; ++sample_index) {
    const double distance = (static_cast<double>(sample_index) + 0.5) * step;
    const double kernel_value = etx::upbp_kernel_value(kernel, dimension, radius, distance * distance);
    if (dimension == 1u) {
      integral += 2.0 * kernel_value * step;
    } else if (dimension == 2u) {
      integral += 2.0 * static_cast<double>(kPi) * distance * kernel_value * step;
    } else {
      integral += 4.0 * static_cast<double>(kPi) * distance * distance * kernel_value * step;
    }
  }

  return close_value(integral, 1.0, 2.0e-6, "kernel normalization");
}

bool validate_kernels() {
  bool valid = true;
  for (uint32_t dimension = 1u; dimension <= 3u; ++dimension) {
    valid = validate_kernel(etx::UPBPKernel::TopHat, dimension) && valid;
    valid = validate_kernel(etx::UPBPKernel::Epanechnikov, dimension) && valid;
  }

  valid = close_value(etx::upbp_kernel_value(etx::UPBPKernel::Epanechnikov, 2u, 1.0, 1.0), 0.0, 0.0, "kernel boundary") && valid;
  valid = close_value(etx::upbp_kernel_value(etx::UPBPKernel::Epanechnikov, 0u, 1.0, 0.0), 0.0, 0.0, "invalid kernel dimension") && valid;
  std::printf("kernel normalization %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_progressive_radii() {
  bool valid = true;
  valid = close_value(etx::upbp_progressive_radius(0.25, 0.75, 1u, 0u), 0.25, 0.0, "initial 1D radius") && valid;
  valid = close_value(etx::upbp_progressive_radius(0.25, 0.75, 2u, 15u), 0.25 * pow(16.0, -0.125), 1.0e-14, "progressive 2D radius") && valid;
  valid = close_value(etx::upbp_progressive_radius(0.25, 0.75, 3u, 63u), 0.25 * pow(64.0, -1.0 / 12.0), 1.0e-14, "progressive 3D radius") && valid;
  valid = close_value(etx::upbp_progressive_radius(0.25, 1.0, 3u, 999u), 0.25, 0.0, "fixed radius") && valid;
  valid = close_value(etx::upbp_progressive_radius(0.25, 1.1, 2u, 0u), 0.0, 0.0, "invalid radius alpha") && valid;

  constexpr uint64_t light_path_count = 4096u;
  constexpr double radius = 0.1;
  valid = close_value(etx::upbp_density_mis_factor(etx::UPBPTechnique::Surface, light_path_count, radius, 1.0),
            static_cast<double>(kPi) * radius * radius * static_cast<double>(light_path_count), 1.0e-12, "surface MIS factor") &&
          valid;
  valid = close_value(etx::upbp_density_mis_factor(etx::UPBPTechnique::PP3D, light_path_count, radius, 1.0),
            (4.0 / 3.0) * static_cast<double>(kPi) * radius * radius * radius * static_cast<double>(light_path_count), 1.0e-12, "PP3D MIS factor") &&
          valid;
  valid = close_value(etx::upbp_density_mis_factor(etx::UPBPTechnique::BB1D, light_path_count, radius, 0.5), 0.5 * radius * static_cast<double>(light_path_count) * 0.5, 1.0e-12,
            "BB1D MIS factor") &&
          valid;
  std::printf("progressive radii %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_light_splat_iteration_scale() {
  bool valid = true;
  valid = close_value(etx::upbp_light_splat_iteration_scale(1u, 1u), 1.0, 0.0, "single light-path iteration scale") && valid;
  valid = close_value(etx::upbp_light_splat_iteration_scale(4096u, 4096u), 1.0, 0.0, "whole-film light-path iteration scale") && valid;
  valid = close_value(etx::upbp_light_splat_iteration_scale(4096u, 1024u), 4.0, 0.0, "budget-limited light-path iteration scale") && valid;
  valid = close_value(etx::upbp_light_splat_iteration_scale(4096u, 0u), 0.0, 0.0, "invalid light-path iteration scale") && valid;
  std::printf("light splat iteration scale %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_bpt_path_length_limits() {
  etx::Scene scene = {};
  scene.options.min_path_length = 2u;
  scene.options.max_path_length = 64u;

  bool valid = (etx::upbp_path_length_enabled(scene, 1u) == false);
  valid = etx::upbp_path_length_enabled(scene, 2u) && valid;
  valid = etx::upbp_path_length_enabled(scene, 64u) && valid;
  valid = (etx::upbp_path_length_enabled(scene, 65u) == false) && valid;
  valid = (etx::upbp_path_length_enabled(scene, std::numeric_limits<uint64_t>::max()) == false) && valid;
  std::printf("BPT path length limits %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_light_path_selection() {
  bool valid = true;
  valid = (etx::upbp_select_light_path_index(17u, 3u, 5u, 0u) == kInvalidIndex) && valid;
  valid = (etx::upbp_select_light_path_index(17u, 3u, 5u, 1u) == 0u) && valid;
  constexpr uint32_t light_path_count = 37u;
  bool selected_multiple_paths = false;
  const uint32_t first = etx::upbp_select_light_path_index(17u, 3u, 0u, light_path_count);
  for (uint64_t camera_path_index = 0u; camera_path_index < 128u; ++camera_path_index) {
    const uint32_t selected = etx::upbp_select_light_path_index(17u, 3u, camera_path_index, light_path_count);
    valid = (selected < light_path_count) && valid;
    valid = (selected == etx::upbp_select_light_path_index(17u, 3u, camera_path_index, light_path_count)) && valid;
    selected_multiple_paths = (selected != first) || selected_multiple_paths;
  }
  valid = selected_multiple_paths && valid;
  std::printf("light path selection %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_population_radius_scale() {
  bool valid = true;
  valid = close_value(etx::upbp_population_radius_scale(4096u, 4096u, 3u), 1.0, 0.0, "equal-population radius scale") && valid;
  valid = close_value(etx::upbp_population_radius_scale(4096u, 1024u, 1u), 4.0, 1.0e-14, "1D population radius scale") && valid;
  valid = close_value(etx::upbp_population_radius_scale(4096u, 1024u, 2u), 2.0, 1.0e-14, "2D population radius scale") && valid;
  valid = close_value(etx::upbp_population_radius_scale(4096u, 512u, 3u), 2.0, 1.0e-14, "3D population radius scale") && valid;
  valid = close_value(etx::upbp_population_radius_scale(4096u, 0u, 3u), 0.0, 0.0, "invalid population radius scale") && valid;
  std::printf("population radius scale %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_automatic_initial_radius() {
  bool valid = true;
  valid = close_value(etx::upbp_automatic_initial_radius(12.0, 4096u, 4096u, 2u, etx::kUPBPAutomaticSurfaceRadiusScale), 0.018, 1.0e-14, "automatic surface radius") && valid;
  valid =
    close_value(etx::upbp_automatic_initial_radius(12.0, 4096u, 1024u, 3u, etx::kUPBPAutomaticVolumeRadiusScale), 0.012 * std::cbrt(4.0), 1.0e-14, "automatic volume radius") &&
    valid;
  valid = close_value(etx::upbp_automatic_initial_radius(0.0, 4096u, 4096u, 2u, etx::kUPBPAutomaticSurfaceRadiusScale), 0.0, 0.0, "invalid automatic radius") && valid;
  std::printf("automatic initial radius %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_projected_measure_domain() {
  bool valid = true;
  valid = etx::upbp_projected_measure_valid(1.0) && valid;
  valid = etx::upbp_projected_measure_valid(static_cast<double>(kEpsilon)) && valid;
  valid = (etx::upbp_projected_measure_valid(0.0) == false) && valid;
  valid = (etx::upbp_projected_measure_valid(0.5 * static_cast<double>(kEpsilon)) == false) && valid;
  valid = (etx::upbp_projected_measure_valid(std::numeric_limits<double>::infinity()) == false) && valid;
  valid = (etx::upbp_projected_measure_valid(std::numeric_limits<double>::quiet_NaN()) == false) && valid;
  std::printf("projected measure domain %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_medium_origin_trace_retry() {
  const Ray ray = {{1.0f, 2.0f, 3.0f}, {1.0f, 0.0f, 0.0f}, 0.0f, 10.0f};
  uint32_t attempt = 0u;
  uint32_t trace_state = 17u;
  bool reset = false;
  Intersection intersection = {};
  const bool found = etx::upbp_trace_with_medium_origin_retry(
    ray,
    [&attempt, &trace_state](const Ray& trace_ray, Intersection& trace_intersection) {
      ++attempt;
      ++trace_state;
      if (attempt == 1u) {
        return false;
      }
      if ((fabs(trace_ray.o.x - (1.0f - kRayEpsilon)) > 1.0e-7f) || (trace_ray.min_t != kRayEpsilon) || (trace_state != 18u)) {
        return false;
      }
      trace_intersection.t = kRayEpsilon + 0.25f;
      return true;
    },
    [&trace_state, &reset]() {
      trace_state = 17u;
      reset = true;
    },
    intersection);

  bool valid = found && (attempt == 2u) && reset;
  valid = close_value(intersection.t, 0.25, 1.0e-7, "medium-origin retry distance") && valid;
  std::printf("medium-origin trace retry %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_bpt_strategy_controls() {
  etx::Scene scene = {};
  etx::UPBPDensityMISConfiguration configuration = {};
  configuration.enabled_techniques = static_cast<uint32_t>(etx::UPBPTechnique::BPT);
  scene.options.properties[etx::Scene::Properties::MultipleImportanceSampling] = true;
  bool valid = etx::upbp_all_bpt_strategies_enabled(scene);
  valid = (etx::upbp_requires_exhaustive_bpt_weight(scene) == false) && valid;
  valid = close_value(etx::upbp_selected_bpt_weight(scene, 0.25, 0.5), 0.25, 0.0, "recursive BPT weight") && valid;
  configuration.enabled_techniques |= static_cast<uint32_t>(etx::UPBPTechnique::PP3D);
  valid = close_value(etx::upbp_selected_bpt_weight(scene, 0.25, 0.5), 0.25, 0.0, "cross-technique recursive BPT weight") && valid;
  scene.options.strategy_flags &= ~etx::Scene::Strategy::ConnectToCamera;
  valid = (etx::upbp_all_bpt_strategies_enabled(scene) == false) && valid;
  valid = etx::upbp_requires_exhaustive_bpt_weight(scene) && valid;
  valid = close_value(etx::upbp_selected_bpt_weight(scene, 0.25, 0.5), 0.5, 0.0, "partial exhaustive BPT weight") && valid;
  scene.options.properties[etx::Scene::Properties::MultipleImportanceSampling] = false;
  valid = (etx::upbp_requires_exhaustive_bpt_weight(scene) == false) && valid;
  valid = close_value(etx::upbp_selected_bpt_weight(scene, 0.25, 0.5), 1.0, 0.0, "unweighted BPT strategy") && valid;

  etx::UPBPPathRecord path = {};
  path.reset(4u);
  etx::UPBPPathVertexRecord camera = {};
  camera.source = etx::PathSource::Camera;
  valid = path.append_endpoint(camera) && valid;
  etx::UPBPTransportSegmentRecord segment = {};
  segment.reset({550.0f, 0u});
  valid = segment.append(etx::upbp_vacuum_interval({550.0f, 0u}, {}, {1.0f, 0.0f, 0.0f}, 1.0f)) && valid;
  etx::UPBPPathVertexRecord specular = {};
  specular.delta = true;
  valid = path.append_physical_vertex(segment, specular) && valid;
  etx::UPBPPathVertexRecord emitter = {};
  valid = path.append_physical_vertex(segment, emitter) && valid;
  valid = etx::upbp_camera_prefix_is_specular(path, 2u) && valid;
  path.vertices[1u].delta = false;
  valid = (etx::upbp_camera_prefix_is_specular(path, 2u) == false) && valid;
  std::printf("BPT strategy controls %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_density_technique_factors() {
  constexpr uint64_t light_path_count = 1000u;
  constexpr double radius = 0.2;
  constexpr double forward_ray_factor = 2.0;
  constexpr double reverse_ray_factor = 3.0;
  constexpr double sin_theta = 0.5;
  const double pp_factor = etx::upbp_density_mis_factor(etx::UPBPTechnique::PP3D, light_path_count, radius, 1.0);
  const double pb_factor = etx::upbp_density_mis_factor(etx::UPBPTechnique::PB2D, light_path_count, radius, 1.0);
  const double bp_factor = etx::upbp_density_mis_factor(etx::UPBPTechnique::BP2D, light_path_count, radius, 1.0);
  const double bb_factor = etx::upbp_density_mis_factor(etx::UPBPTechnique::BB1D, light_path_count, radius, 0.25);

  bool valid = true;
  valid =
    close_value(etx::upbp_density_competitor_factor({etx::UPBPTechnique::PP3D, etx::UPBPVertexClass::Medium, pp_factor, forward_ray_factor, reverse_ray_factor, sin_theta, false}),
      pp_factor, 0.0, "PP3D competitor factor") &&
    valid;
  valid =
    close_value(etx::upbp_density_competitor_factor({etx::UPBPTechnique::PB2D, etx::UPBPVertexClass::Medium, pb_factor, forward_ray_factor, reverse_ray_factor, sin_theta, false}),
      pb_factor * reverse_ray_factor, 0.0, "PB2D competitor factor") &&
    valid;
  valid =
    close_value(etx::upbp_density_competitor_factor({etx::UPBPTechnique::BP2D, etx::UPBPVertexClass::Medium, bp_factor, forward_ray_factor, reverse_ray_factor, sin_theta, false}),
      bp_factor * forward_ray_factor, 0.0, "BP2D competitor factor") &&
    valid;
  valid =
    close_value(etx::upbp_density_competitor_factor({etx::UPBPTechnique::BB1D, etx::UPBPVertexClass::Medium, bb_factor, forward_ray_factor, reverse_ray_factor, sin_theta, false}),
      bb_factor * sin_theta * forward_ray_factor * reverse_ray_factor, 0.0, "BB1D competitor factor") &&
    valid;
  valid = close_value(etx::upbp_density_competitor_factor({etx::UPBPTechnique::BB1D, etx::UPBPVertexClass::Medium, bb_factor, forward_ray_factor, reverse_ray_factor, 0.0, false}),
            0.0, 0.0, "parallel BB1D competitor") &&
          valid;
  valid =
    close_value(etx::upbp_density_competitor_factor({etx::UPBPTechnique::PP3D, etx::UPBPVertexClass::Medium, pp_factor, forward_ray_factor, reverse_ray_factor, sin_theta, true}),
      0.0, 0.0, "delta density competitor") &&
    valid;

  const double bb_kernel = etx::upbp_bb1d_kernel_value(etx::UPBPKernel::Epanechnikov, radius, 0.25 * radius * radius, sin_theta);
  const double expected_bb_kernel = 3.0 / (4.0 * radius * sin_theta) * 0.75;
  valid = close_value(bb_kernel, expected_bb_kernel, 1.0e-14, "BB1D Epanechnikov kernel") && valid;
  valid = close_value(etx::upbp_density_estimator_scale(etx::UPBPTechnique::BB1D, etx::UPBPKernel::Epanechnikov, light_path_count, radius, 0.25 * radius * radius, sin_theta, 0.25),
            expected_bb_kernel / (static_cast<double>(light_path_count) * 0.25), 1.0e-14, "BB1D estimator scale") &&
          valid;
  const etx::UPBPPreparedBB1D prepared_bb1d = etx::upbp_prepare_bb1d(etx::UPBPKernel::Epanechnikov, radius, light_path_count, 0.25);
  valid = prepared_bb1d.valid && valid;
  valid = close_value(prepared_bb1d.estimator_normalization, 1.0 / (static_cast<double>(light_path_count) * 0.25), 1.0e-14, "prepared BB1D estimator normalization") && valid;
  valid = close_value(etx::upbp_evaluate_prepared_bb1d_kernel(prepared_bb1d, 0.25 * radius * radius, sin_theta), expected_bb_kernel, 1.0e-14, "prepared BB1D kernel") && valid;
  std::printf("density technique factors %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_recursive_local_pde_affine() {
  etx::UPBPDensityMISConfiguration configuration = {};
  configuration.enabled_techniques = static_cast<uint32_t>(etx::UPBPTechnique::Surface) | static_cast<uint32_t>(etx::UPBPTechnique::PP3D) |
                                     static_cast<uint32_t>(etx::UPBPTechnique::PB2D) | static_cast<uint32_t>(etx::UPBPTechnique::BP2D) |
                                     static_cast<uint32_t>(etx::UPBPTechnique::BB1D);
  configuration.technique_factors = {0.0, 2.0, 3.0, 5.0, 7.0, 11.0};
  etx::UPBPRecursiveVertexWeights weights = {};
  weights.ray_sample_forward_pdf_inverse = 13.0;
  weights.ray_sample_forward_ratio = 17.0;
  constexpr double next_reverse_ratio = 19.0;
  constexpr double sin_theta = 0.6;
  constexpr double inverse_values[] = {1.0e-6, 0.5, 7.0, 1.0e6};

  bool valid = true;
  constexpr bool beam_modes[] = {false, true};
  constexpr bool boolean_values[] = {false, true};
  constexpr etx::UPBPVertexClass vertex_classes[] = {etx::UPBPVertexClass::Surface, etx::UPBPVertexClass::Medium};
  constexpr etx::PathSource sources[] = {etx::PathSource::Light, etx::PathSource::Camera};
  for (const bool photon_beams_long : beam_modes) {
    for (const bool camera_beams_long : beam_modes) {
      configuration.photon_beams_long = photon_beams_long;
      configuration.camera_beams_long = camera_beams_long;
      for (const etx::UPBPVertexClass vertex_class : vertex_classes) {
        for (const bool vertex_delta : boolean_values) {
          for (const bool vertex_density_connectible : boolean_values) {
            for (const etx::PathSource source : sources) {
              const etx::UPBPRecursiveLocalPDEAffine affine =
                etx::upbp_recursive_local_pde_affine(configuration, vertex_class, vertex_delta, vertex_density_connectible, weights, next_reverse_ratio, sin_theta, source);
              for (const double reverse_pdf_inverse : inverse_values) {
                const double expected = etx::upbp_recursive_local_pde_factor(configuration, vertex_class, vertex_delta, vertex_density_connectible, weights, reverse_pdf_inverse,
                  next_reverse_ratio, sin_theta, source);
                const double actual = affine.constant + affine.reverse_pdf_inverse_coefficient * reverse_pdf_inverse;
                const double tolerance = 1.0e-12 * fmax(1.0, fabs(expected));
                valid = close_value(actual, expected, tolerance, "recursive local PDE affine") && valid;
              }
            }
          }
        }
      }
    }
  }
  std::printf("recursive local PDE affine %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_prepared_medium_density() {
  etx::Medium medium = {};
  medium.cls = etx::Medium::Heterogeneous;
  medium.local_bounds.p_min = {-2.0f, -1.0f, -3.0f};
  medium.local_bounds.p_max = {2.0f, 3.0f, 1.0f};
  medium.grid.type = MediumGridType::NoiseFunction;
  medium.grid.noise_seed = 17u;
  medium.grid.noise_offset = {0.37f, -0.21f, 0.13f};
  medium.grid.noise_enable_border_fade = 1u;
  medium.grid.noise_octaves = 5u;
  medium.grid.noise_scale = 3.25f;
  medium.grid.noise_lacunarity = 1.9f;
  medium.grid.noise_persistence = 0.55f;
  medium.grid.noise_power = 1.4f;
  medium.grid.noise_sharpness = 0.8f;
  medium.grid.noise_border_fade_distance = 0.17f;

  bool valid = true;
  constexpr uint32_t noise_types[] = {MediumNoiseType::Perlin, MediumNoiseType::Billow};
  for (const uint32_t noise_type : noise_types) {
    medium.grid.noise_type = noise_type;
    const etx::UPBPPreparedMedium prepared = etx::upbp_prepare_medium(medium, etx::SpectralQuery::sample());
    valid = prepared.noise_prepared && valid;
    for (uint32_t z = 0u; z < 7u; ++z) {
      for (uint32_t y = 0u; y < 7u; ++y) {
        for (uint32_t x = 0u; x < 7u; ++x) {
          const float3 local = {
            (static_cast<float>(x) + 0.37f) / 7.0f,
            (static_cast<float>(y) + 0.23f) / 7.0f,
            (static_cast<float>(z) + 0.61f) / 7.0f,
          };
          const float3 position = bounding_box_from_local(medium.local_bounds, local);
          const float expected = medium.sample_density_world(position);
          const float actual = etx::upbp_prepared_medium_density(medium, prepared, position);
          valid = close_value(actual, expected, 2.0e-6, "prepared medium density") && valid;
        }
      }
    }
  }

  medium.grid.noise_type = MediumNoiseType::Worley;
  const etx::UPBPPreparedMedium fallback = etx::upbp_prepare_medium(medium, etx::SpectralQuery::sample());
  const float3 fallback_position = bounding_box_from_local(medium.local_bounds, float3{0.31f, 0.47f, 0.73f});
  valid = (fallback.noise_prepared == false) && valid;
  valid =
    close_value(etx::upbp_prepared_medium_density(medium, fallback, fallback_position), medium.sample_density_world(fallback_position), 0.0, "prepared medium fallback") && valid;
  std::printf("prepared medium density %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_balance_heuristic() {
  std::array<etx::UPBPTechniqueProbability, 4u> probabilities = {{
    {etx::UPBPTechnique::BPT, log(0.2), 1u, true},
    {etx::UPBPTechnique::PP3D, log(0.15), 2u, true},
    {etx::UPBPTechnique::PB2D, log(0.125), 4u, true},
    {etx::UPBPTechnique::BB1D, 0.0, 0u, false},
  }};

  bool valid = true;
  double weight_sum = 0.0;
  for (uint32_t index = 0u; index < probabilities.size(); ++index) {
    weight_sum += etx::upbp_balance_weight(probabilities.data(), static_cast<uint32_t>(probabilities.size()), index);
  }
  valid = close_value(weight_sum, 1.0, 1.0e-14, "balance sum") && valid;

  const double expected_terms[] = {0.2, 0.3, 0.5};
  for (uint32_t index = 0u; index < 3u; ++index) {
    const double weight = etx::upbp_balance_weight(probabilities.data(), static_cast<uint32_t>(probabilities.size()), index);
    valid = close_value(weight, expected_terms[index], 1.0e-14, "balance weight") && valid;
  }

  for (uint32_t index = 0u; index < 3u; ++index) {
    probabilities[index].log_density += 700.0;
  }

  double shifted_weight_sum = 0.0;
  for (uint32_t index = 0u; index < probabilities.size(); ++index) {
    shifted_weight_sum += etx::upbp_balance_weight(probabilities.data(), static_cast<uint32_t>(probabilities.size()), index);
  }
  valid = close_value(shifted_weight_sum, 1.0, 1.0e-14, "shifted balance sum") && valid;
  std::printf("balance heuristic %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_segment_record() {
  const etx::SpectralQuery spect = etx::SpectralQuery::sample();
  etx::MediumTrackingInput input = {};
  input.spect = spect;
  input.scattering = etx::SpectralResponse{spect, float3{0.05f, 0.1f, 0.6f}};
  input.absorption = etx::SpectralResponse{spect, float3{0.0f, 0.0f, 0.0f}};
  input.density_majorant = 1.0f;
  bool valid = false;
  for (uint32_t seed = 1u; seed < 4096u; ++seed) {
    etx::Sampler sampler{seed};
    etx::UPBPSegmentRecord record = {};
    record.reset(spect, 7u, float3{});

    etx::SpectralResponse direct_weight{spect, 1.0f};
    double direct_log_pdf_forward = 0.0;
    double direct_log_pdf_reverse = 0.0;
    float3 origin = {};
    float remaining_distance = 3.0f;
    uint32_t direct_null_count = 0u;

    for (uint32_t event_index = 0u; event_index < 64u; ++event_index) {
      const auto event = etx::sample_medium_tracking_event(input, sampler, origin, float3{1.0f, 0.0f, 0.0f}, remaining_distance, [](const float3&) {
        return 1.0f;
      });
      if (record.append(event) == false) {
        return false;
      }

      direct_log_pdf_forward += log(static_cast<double>(event.pdf_forward));
      direct_log_pdf_reverse += log(static_cast<double>(event.pdf_reverse));
      if ((event.type == etx::MediumTrackingEventType::Null) || (event.type == etx::MediumTrackingEventType::Scatter) || (event.type == etx::MediumTrackingEventType::Absorb)) {
        direct_weight *= event.weight;
      }

      if (event.type != etx::MediumTrackingEventType::Null) {
        break;
      }

      ++direct_null_count;
      origin = event.position;
      remaining_distance -= event.distance;
    }

    if ((record.complete == false) || (direct_null_count == 0u)) {
      continue;
    }

    valid = record.valid();
    valid = (record.medium_index == 7u) && (record.null_event_count == direct_null_count) && valid;
    valid = close_value(record.log_pdf_forward, direct_log_pdf_forward, 1.0e-12, "segment forward log PDF") && valid;
    valid = close_value(record.log_pdf_reverse, direct_log_pdf_reverse, 1.0e-12, "segment reverse log PDF") && valid;
    valid = close_value(record.weight.integrated.x, direct_weight.integrated.x, 1.0e-6, "segment weight x") && valid;
    valid = close_value(record.weight.integrated.y, direct_weight.integrated.y, 1.0e-6, "segment weight y") && valid;
    valid = close_value(record.weight.integrated.z, direct_weight.integrated.z, 1.0e-6, "segment weight z") && valid;
    valid = close_value(record.pdf_forward(), record.pdf_reverse(), 1.0e-14, "segment reciprocity") && valid;
    valid = (record.append(etx::MediumTrackingEvent{}) == false) && valid;
    break;
  }

  std::printf("segment record %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_segment_walker() {
  const etx::SpectralQuery spect = etx::SpectralQuery::sample();
  etx::MediumTrackingInput input = {};
  input.spect = spect;
  input.scattering = etx::SpectralResponse{spect, float3{0.01f, 0.01f, 1.0f}};
  input.absorption = etx::SpectralResponse{spect, float3{0.0f, 0.0f, 0.0f}};
  input.density_majorant = 1.0f;

  bool found_null_chain = false;
  for (uint32_t seed = 1u; seed < 4096u; ++seed) {
    etx::Sampler sampler{seed};
    etx::UPBPSegmentRecord segment = {};
    if (etx::upbp_track_medium_segment(
          input, sampler, {}, float3{1.0f, 0.0f, 0.0f}, 5.0f, 9u, 64u,
          [](const float3&) {
            return 1.0f;
          },
          segment) == false) {
      return false;
    }

    if (segment.null_event_count > 0u) {
      found_null_chain = segment.complete && segment.valid() && (segment.medium_index == 9u);
      break;
    }
  }

  etx::Sampler limited_sampler{3u};
  etx::UPBPSegmentRecord limited_segment = {};
  const bool limited_result = etx::upbp_track_medium_segment(
    input, limited_sampler, {}, float3{1.0f, 0.0f, 0.0f}, 100.0f, 9u, 0u,
    [](const float3&) {
      return 1.0f;
    },
    limited_segment);
  const bool limit_reported = (limited_result == false) && (limited_segment.failure == etx::MediumTrackingFailure::EventLimitExceeded);

  bool valid = found_null_chain && limit_reported;
  valid = close_value(etx::upbp_distance_to_scene_sphere_exit({-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {}, 1.0f), 2.0, 1.0e-7, "scene sphere boundary-to-exit distance") && valid;
  valid = close_value(etx::upbp_distance_to_scene_sphere_exit({}, {1.0f, 0.0f, 0.0f}, {}, 1.0f), 1.0, 1.0e-7, "scene sphere center-to-exit distance") && valid;
  std::printf("segment walker %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_path_record() {
  const etx::SpectralQuery spect = etx::SpectralQuery::sample();
  etx::UPBPPathRecord path = {};
  path.reset(8u);

  etx::UPBPPathVertexRecord camera = {};
  camera.throughput = etx::SpectralResponse{spect, 1.0f};
  camera.cls = etx::UPBPVertexClass::Camera;
  bool valid = path.append_endpoint(camera);

  etx::UPBPSegmentRecord segment = {};
  segment.reset(spect, 3u, {});
  etx::MediumTrackingEvent null_event = {};
  null_event.type = etx::MediumTrackingEventType::Null;
  null_event.weight = etx::SpectralResponse{spect, 1.0f};
  null_event.pdf_forward = 0.25f;
  null_event.pdf_reverse = 0.5f;
  valid = segment.append(null_event) && valid;

  etx::MediumTrackingEvent scatter_event = {};
  scatter_event.type = etx::MediumTrackingEventType::Scatter;
  scatter_event.weight = etx::SpectralResponse{spect, 1.0f};
  scatter_event.majorant = 2.0f;
  scatter_event.majorant_transmittance = 0.5f;
  scatter_event.event_probability = 0.5f;
  scatter_event.pdf_forward = 0.5f;
  scatter_event.pdf_reverse = 0.25f;
  valid = segment.append(scatter_event) && valid;

  etx::UPBPPathVertexRecord medium = {};
  medium.throughput = etx::SpectralResponse{spect, 1.0f};
  medium.incident_medium_index = 3u;
  medium.outgoing_medium_index = 3u;
  medium.cls = etx::UPBPVertexClass::Medium;
  medium.connectible = true;
  etx::UPBPTransportSegmentRecord transport_segment = {};
  transport_segment.reset(spect);
  valid = transport_segment.append(segment) && valid;
  valid = path.append_physical_vertex(transport_segment, medium) && valid;

  path.record_boundary();
  path.record_boundary();
  valid = path.valid() && valid;
  valid = (path.physical_length() == 1u) && valid;
  valid = (path.null_event_count == 1u) && valid;
  valid = (path.boundary_count == 2u) && valid;
  std::printf("path record %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_path_storage_reservation() {
  etx::UPBPPathRecord persistent_light_path = {};
  persistent_light_path.reset(65u);
  bool valid = persistent_light_path.vertices.capacity() == 0u;
  valid = (persistent_light_path.segments.capacity() == 0u) && valid;

  etx::UPBPPathRecord transient_camera_path = {};
  transient_camera_path.reset(65u, 8u);
  valid = (transient_camera_path.vertices.capacity() >= 8u) && valid;
  valid = (transient_camera_path.segments.capacity() >= 8u) && valid;

  etx::UPBPPathRecord short_path = {};
  short_path.reset(4u, 8u);
  valid = (short_path.vertices.capacity() >= 4u) && (short_path.segments.capacity() >= 3u) && valid;
  std::printf("path storage reservation %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_transport_segment() {
  const etx::SpectralQuery spect = etx::SpectralQuery::sample();
  etx::UPBPTransportSegmentRecord transport = {};
  transport.reset(spect);
  const etx::UPBPSegmentRecord first = etx::upbp_vacuum_interval(spect, {}, float3{1.0f, 0.0f, 0.0f}, 2.0f);
  const etx::UPBPSegmentRecord second = etx::upbp_vacuum_interval(spect, float3{2.0f, 0.0f, 0.0f}, float3{1.0f, 0.0f, 0.0f}, 3.0f);
  bool valid = transport.append(first);
  transport.record_boundary();
  valid = transport.append(second) && valid;
  valid = transport.valid() && valid;
  valid = (transport.intervals.size() == 2u) && (transport.boundary_count == 1u) && valid;
  valid = close_value(transport.distance, 5.0, 0.0, "transport segment distance") && valid;
  valid = close_value(transport.log_pdf_forward, 0.0, 0.0, "vacuum forward log PDF") && valid;
  valid = close_value(transport.log_pdf_reverse, 0.0, 0.0, "vacuum reverse log PDF") && valid;

  etx::UPBPSegmentRecord medium_interval = {};
  medium_interval.reset(spect, 3u, {});
  etx::MediumTrackingEvent scatter_event = {};
  scatter_event.type = etx::MediumTrackingEventType::Scatter;
  scatter_event.weight = etx::SpectralResponse{spect, 1.0f};
  scatter_event.majorant = 2.0f;
  scatter_event.majorant_transmittance = 0.5f;
  scatter_event.event_probability = 0.25f;
  scatter_event.pdf_forward = 0.25f;
  scatter_event.pdf_reverse = 0.25f;
  valid = medium_interval.append(scatter_event) && valid;
  valid = close_value(medium_interval.log_pdf_forward, std::log(0.25), 1.0e-14, "medium full log PDF") && valid;
  valid = close_value(medium_interval.log_transport_pdf_forward, std::log(0.5), 1.0e-14, "medium transport log PDF") && valid;
  valid = close_value(medium_interval.log_terminal_event_density, std::log(0.5), 1.0e-14, "medium event log density") && valid;
  valid = close_value(medium_interval.log_transport_pdf_forward + medium_interval.log_terminal_event_density, medium_interval.log_pdf_forward, 1.0e-14,
            "medium probability factorization") &&
          valid;
  std::printf("transport segment %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_recursive_balance_accumulator() {
  constexpr std::array<etx::UPBPTechniqueProbability, 6u> probabilities = {{
    {etx::UPBPTechnique::BPT, -820.0, 1u, true},
    {etx::UPBPTechnique::Surface, -818.0, 64u, true},
    {etx::UPBPTechnique::PP3D, -824.0, 128u, true},
    {etx::UPBPTechnique::PB2D, -819.0, 128u, true},
    {etx::UPBPTechnique::BP2D, -817.0, 128u, true},
    {etx::UPBPTechnique::BB1D, -815.0, 32u, true},
  }};

  bool valid = true;
  for (uint32_t selected = 0u; selected < probabilities.size(); ++selected) {
    etx::UPBPMISAccumulator accumulator = {};
    for (uint32_t index = 0u; index < probabilities.size(); ++index) {
      valid = accumulator.append(probabilities[index], index == selected) && valid;
    }

    const double exhaustive = etx::upbp_balance_weight(probabilities.data(), static_cast<uint32_t>(probabilities.size()), selected);
    valid = close_value(accumulator.weight(), exhaustive, 1.0e-14, "recursive/exhaustive MIS") && valid;
  }

  etx::UPBPMISAccumulator shifted = {};
  for (uint32_t index = 0u; index < probabilities.size(); ++index) {
    etx::UPBPTechniqueProbability probability = probabilities[index];
    probability.log_density += 1500.0;
    valid = shifted.append(probability, index == 4u) && valid;
  }
  valid =
    close_value(shifted.weight(), etx::upbp_balance_weight(probabilities.data(), static_cast<uint32_t>(probabilities.size()), 4u), 1.0e-14, "recursive MIS common scale") && valid;
  std::printf("recursive MIS accumulator %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_density_mis_recurrence() {
  etx::UPBPPathProbabilityRecord path = {};
  path.log_emitter_endpoint_density = -1.25;
  path.log_camera_endpoint_density = -0.75;
  path.vertices.resize(7u);
  path.edges = {
    {-1.1, -1.3},
    {-0.9, -1.7},
    {-2.2, -0.8},
    {-1.4, -1.6},
    {-0.6, -2.1},
    {-1.8, -1.2},
  };
  for (uint32_t index = 0u; index < path.vertices.size(); ++index) {
    path.vertices[index].connectible = index != 3u;
  }

  const std::vector<etx::UPBPDensityMISContext> contexts = {
    {2u, etx::UPBPVertexClass::Surface, 0.0, 0.0, 0.0, false, true},
    {3u, etx::UPBPVertexClass::Medium, 2.0, 3.0, 0.5, false, true},
    {4u, etx::UPBPVertexClass::Medium, 1.5, 0.75, 0.25, false, true},
  };
  etx::UPBPDensityMISConfiguration configuration = {};
  configuration.enabled_techniques = static_cast<uint32_t>(etx::UPBPTechnique::BPT) | static_cast<uint32_t>(etx::UPBPTechnique::Surface) |
                                     static_cast<uint32_t>(etx::UPBPTechnique::PP3D) | static_cast<uint32_t>(etx::UPBPTechnique::PB2D) |
                                     static_cast<uint32_t>(etx::UPBPTechnique::BP2D) | static_cast<uint32_t>(etx::UPBPTechnique::BB1D);
  configuration.technique_factors = {1.0, 12.0, 24.0, 6.0, 8.0, 2.0};

  std::vector<etx::UPBPDensityStrategyProbability> exhaustive = {};
  std::vector<etx::UPBPDensityStrategyProbability> recursive = {};
  bool valid = etx::upbp_enumerate_all_strategies_exhaustive(path, contexts, configuration, exhaustive);
  valid = etx::upbp_enumerate_all_strategies_recursive(path, contexts, configuration, recursive) && valid;
  valid = (exhaustive.size() == recursive.size()) && valid;
  if (exhaustive.size() == recursive.size()) {
    for (uint32_t index = 0u; index < exhaustive.size(); ++index) {
      valid = (exhaustive[index].technique == recursive[index].technique) && (exhaustive[index].light_vertex_count == recursive[index].light_vertex_count) &&
              (exhaustive[index].applicable == recursive[index].applicable) && valid;
      if (exhaustive[index].applicable) {
        valid = close_value(recursive[index].log_density, exhaustive[index].log_density, 1.0e-14, "density recursive/exhaustive log density") && valid;
        const double exhaustive_weight = etx::upbp_all_strategy_balance_weight(exhaustive, exhaustive[index].technique, exhaustive[index].light_vertex_count);
        const double recursive_weight = etx::upbp_all_strategy_balance_weight(recursive, recursive[index].technique, recursive[index].light_vertex_count);
        valid = close_value(recursive_weight, exhaustive_weight, 1.0e-14, "density recursive/exhaustive MIS") && valid;
      }
    }
  }

  std::printf("density MIS recurrence %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_augmented_recursive_weights() {
  const etx::SpectralQuery spect = etx::SpectralQuery::sample();
  etx::UPBPPathRecord path = {};
  path.reset(3u);
  etx::UPBPPathVertexRecord camera = {};
  camera.position = {};
  camera.throughput = etx::SpectralResponse{spect, 1.0f};
  camera.endpoint_pdf_direction = 0.5f;
  camera.source = etx::PathSource::Camera;
  camera.cls = etx::UPBPVertexClass::Camera;
  bool valid = path.append_endpoint(camera);

  auto append_medium_vertex = [&path, spect](const float distance, const float transport_density, const float event_density) {
    etx::UPBPSegmentRecord interval = {};
    interval.reset(spect, 2u, path.vertices.back().position);
    etx::MediumTrackingEvent scatter = {};
    scatter.type = etx::MediumTrackingEventType::Scatter;
    scatter.position = path.vertices.back().position + float3{distance, 0.0f, 0.0f};
    scatter.distance = distance;
    scatter.weight = etx::SpectralResponse{spect, 1.0f};
    scatter.majorant = 1.0f;
    scatter.majorant_transmittance = transport_density;
    scatter.event_probability = event_density;
    scatter.pdf_forward = transport_density * event_density;
    scatter.pdf_reverse = scatter.pdf_forward;
    if (interval.append(scatter) == false) {
      return false;
    }
    etx::UPBPTransportSegmentRecord segment = {};
    segment.reset(spect);
    if (segment.append(interval) == false) {
      return false;
    }
    etx::UPBPPathVertexRecord vertex = {};
    vertex.position = scatter.position;
    vertex.intersection.pos = scatter.position;
    vertex.intersection.w_i = {1.0f, 0.0f, 0.0f};
    vertex.sampled_direction = {1.0f, 0.0f, 0.0f};
    vertex.throughput = etx::SpectralResponse{spect, 1.0f};
    vertex.scatter_pdf_forward = 0.5f;
    vertex.scatter_pdf_reverse = 0.25f;
    vertex.log_medium_event_density = std::log(static_cast<double>(event_density));
    vertex.source = etx::PathSource::Camera;
    vertex.cls = etx::UPBPVertexClass::Medium;
    vertex.connectible = true;
    return path.append_physical_vertex(segment, vertex);
  };
  valid = append_medium_vertex(2.0f, 0.5f, 0.25f) && valid;
  valid = append_medium_vertex(1.0f, 0.5f, 0.5f) && valid;

  etx::UPBPDensityMISConfiguration configuration = {};
  configuration.enabled_techniques = static_cast<uint32_t>(etx::UPBPTechnique::BPT) | static_cast<uint32_t>(etx::UPBPTechnique::PP3D);
  configuration.technique_factors[0u] = 1.0;
  configuration.technique_factors[2u] = 10.0;
  std::vector<etx::UPBPRecursiveVertexWeights> weights = {};
  etx::Scene scene = {};
  valid = etx::upbp_compute_recursive_vertex_weights(scene, path, configuration, 8u, 1u, weights) && valid;
  valid = (weights.size() == 3u) && valid;
  if (weights.size() == 3u) {
    valid = close_value(weights[1u].d_shared, 64.0, 1.0e-12, "first augmented recursive shared weight") && valid;
    valid = close_value(weights[1u].ray_sample_forward_pdf_inverse, 8.0, 1.0e-12, "forward augmented segment inverse") && valid;
    valid = close_value(weights[1u].ray_sample_reverse_pdf_inverse, 2.0, 1.0e-12, "reverse augmented segment inverse") && valid;
    valid = close_value(weights[2u].d_shared, 8.0, 1.0e-12, "second augmented recursive shared weight") && valid;
    valid = close_value(weights[2u].d_bpt, 592.0, 1.0e-10, "augmented recursive BPT weight") && valid;
    valid = close_value(weights[2u].d_pde, 592.0, 1.0e-10, "augmented recursive PDE weight") && valid;
    valid = close_value(weights[2u].ray_sample_reverse_pdf_inverse, 8.0, 1.0e-12, "reverse endpoint-owned segment inverse") && valid;
  }

  std::vector<etx::UPBPRecursiveVertexWeights> resolution_independent_weights = {};
  valid = etx::upbp_compute_recursive_vertex_weights(scene, path, configuration, 8192u, 1u, resolution_independent_weights) && valid;
  valid = (resolution_independent_weights.size() == weights.size()) && valid;
  if (resolution_independent_weights.size() == weights.size()) {
    for (uint32_t index = 0u; index < weights.size(); ++index) {
      valid = close_value(resolution_independent_weights[index].d_shared, weights[index].d_shared, 1.0e-12, "whole-film camera shared weight") && valid;
      valid = close_value(resolution_independent_weights[index].d_bpt, weights[index].d_bpt, 1.0e-12, "whole-film camera BPT weight") && valid;
    }
  }
  std::printf("augmented recursive weights %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_zero_reverse_scattering_density() {
  etx::UPBPPathRecord path = {};
  path.vertices.resize(2u);
  etx::UPBPPathVertexRecord& vertex = path.vertices[1u];
  vertex.cls = etx::UPBPVertexClass::Medium;
  vertex.intersection.w_i = {0.0f, 1.0f, 0.0f};
  vertex.sampled_direction = {1.0f, 0.0f, 0.0f};
  vertex.scatter_pdf_forward = 0.5f;
  vertex.scatter_pdf_reverse = 0.0f;

  etx::UPBPRecursiveState state = {};
  state.weights.d_shared = 3.0;
  state.weights.d_bpt = 5.0;
  state.weights.d_pde = 7.0;
  state.weights.ray_sample_reverse_pdf_inverse = 2.0;
  etx::Scene scene = {};
  bool valid = etx::upbp_prepare_recursive_departure(scene, path, 1u, 4u, state);
  valid = (state.failure == etx::UPBPRecursiveWeightFailure::None) && valid;
  valid = close_value(state.d_bpt_a, 2.0, 0.0, "zero-reverse recursive BPT coefficient") && valid;
  valid = close_value(state.d_bpt_b, 6.0, 0.0, "zero-reverse recursive BPT constant") && valid;
  valid = close_value(state.d_pde_a, 2.0, 0.0, "zero-reverse recursive PDE coefficient") && valid;
  valid = close_value(state.d_pde_b, 24.0, 0.0, "zero-reverse recursive PDE constant") && valid;

  vertex.scatter_pdf_reverse = -1.0f;
  state = {};
  valid = (etx::upbp_prepare_recursive_departure(scene, path, 1u, 4u, state) == false) && valid;
  valid = (state.failure == etx::UPBPRecursiveWeightFailure::InvalidScatteringDensity) && valid;

  vertex.scatter_pdf_reverse = std::numeric_limits<float>::infinity();
  state = {};
  valid = (etx::upbp_prepare_recursive_departure(scene, path, 1u, 4u, state) == false) && valid;
  valid = (state.failure == etx::UPBPRecursiveWeightFailure::InvalidScatteringDensity) && valid;
  std::printf("zero-reverse scattering density %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_point_merge_mis() {
  etx::UPBPDensityMISConfiguration configuration = {};
  configuration.enabled_techniques = static_cast<uint32_t>(etx::UPBPTechnique::BPT) | static_cast<uint32_t>(etx::UPBPTechnique::PP3D) |
                                     static_cast<uint32_t>(etx::UPBPTechnique::PB2D) | static_cast<uint32_t>(etx::UPBPTechnique::BP2D) |
                                     static_cast<uint32_t>(etx::UPBPTechnique::BB1D);
  configuration.technique_factors = {1.0, 0.0, 10.0, 4.0, 6.0, 2.0};
  configuration.photon_beams_long = false;
  configuration.camera_beams_long = true;
  etx::UPBPRecursiveVertexWeights light = {};
  light.d_shared = 3.0;
  light.d_pde = 7.0;
  light.ray_sample_forward_pdf_inverse = 11.0;
  light.ray_sample_reverse_pdf_inverse = 2.0;
  light.ray_sample_forward_ratio = 2.0;
  etx::UPBPRecursiveVertexWeights camera = {};
  camera.d_shared = 4.0;
  camera.d_pde = 8.0;
  camera.ray_sample_forward_pdf_inverse = 5.0;
  camera.ray_sample_reverse_pdf_inverse = 4.0;
  camera.ray_sample_forward_ratio = 3.0;

  const double expected_denominator = 0.1 * (3.0 + 0.25 * 7.0 / 2.0) + 1.0 + 0.1 * (4.0 * 5.0 + 6.0 * 2.0 + 2.0 * 0.5 * 2.0 * 5.0) + 0.1 * (4.0 + 0.5 * 8.0 / 4.0);
  bool valid = close_value(etx::upbp_point_merge_mis_weight({etx::UPBPTechnique::PP3D, etx::UPBPVertexClass::Medium, light, camera, configuration, 0.25, 0.5, 0.5, 1u}),
                 1.0 / expected_denominator, 1.0e-14, "point merge MIS") &&
               true;

  etx::UPBPDensityMISConfiguration isolated = {};
  isolated.enabled_techniques = static_cast<uint32_t>(etx::UPBPTechnique::PP3D);
  isolated.technique_factors[2u] = 10.0;
  etx::UPBPRecursiveVertexWeights isolated_light = light;
  etx::UPBPRecursiveVertexWeights isolated_camera = camera;
  isolated_light.d_pde = 0.0;
  isolated_camera.d_pde = 0.0;
  valid = close_value(etx::upbp_point_merge_mis_weight({etx::UPBPTechnique::PP3D, etx::UPBPVertexClass::Medium, isolated_light, isolated_camera, isolated, 0.25, 0.5, 0.5, 0u}),
            1.0, 1.0e-14, "isolated point merge MIS") &&
          valid;
  std::printf("point merge MIS %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_medium_pre_collision_throughput() {
  const etx::SpectralQuery rgb = etx::SpectralQuery::sample();
  const etx::SpectralResponse rgb_throughput{rgb, float3{1.5f, 4.0f, 0.0f}};
  const etx::SpectralResponse rgb_scattering{rgb, float3{1.5f, 2.0f, 0.0f}};
  etx::SpectralResponse result = {};
  bool valid = etx::upbp_remove_medium_collision_weight(rgb_throughput, rgb_scattering, 2.0, result);
  valid = close_value(result.integrated.x, 2.0, 1.0e-7, "RGB pre-collision red") && valid;
  valid = close_value(result.integrated.y, 4.0, 1.0e-7, "RGB pre-collision green") && valid;
  valid = close_value(result.integrated.z, 0.0, 1.0e-7, "RGB pre-collision zero channel") && valid;

  const etx::SpectralQuery spectral = {550.0f, SpectralFlags::Spectral};
  valid = etx::upbp_remove_medium_collision_weight(etx::SpectralResponse{spectral, 1.5f}, etx::SpectralResponse{spectral, 2.5f}, 5.0, result) && valid;
  valid = close_value(result.value, 3.0, 1.0e-7, "spectral pre-collision throughput") && valid;

  valid = (etx::upbp_remove_medium_collision_weight(etx::SpectralResponse{spectral, 1.0f}, etx::SpectralResponse{spectral, 0.0f}, 1.0, result) == false) && valid;
  valid = (etx::upbp_remove_medium_collision_weight(etx::SpectralResponse{spectral, 1.0f}, etx::SpectralResponse{spectral, 1.0f}, 0.0, result) == false) && valid;
  std::printf("medium pre-collision throughput %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_bpt_cross_technique_mis() {
  const etx::SpectralQuery spect = etx::SpectralQuery::sample();
  etx::Scene scene = {};
  etx::UPBPPathVertexRecord light = {};
  light.position = {};
  light.intersection.pos = light.position;
  light.intersection.w_i = {1.0f, 0.0f, 0.0f};
  light.log_medium_event_density = std::log(0.25);
  light.cls = etx::UPBPVertexClass::Medium;
  etx::UPBPPathVertexRecord camera = light;
  camera.position = {1.0f, 0.0f, 0.0f};
  camera.intersection.pos = camera.position;
  camera.intersection.w_i = {-1.0f, 0.0f, 0.0f};
  etx::UPBPTransportSegmentRecord connection = {};
  connection.reset(spect);
  bool valid = connection.append(etx::upbp_vacuum_interval(spect, light.position, {1.0f, 0.0f, 0.0f}, 1.0f));
  etx::UPBPRecursiveVertexWeights light_weights = {};
  light_weights.d_shared = 2.0;
  light_weights.ray_sample_reverse_pdf_inverse = 1.0;
  etx::UPBPRecursiveVertexWeights camera_weights = light_weights;
  etx::UPBPScatteringEval scattering = {};
  scattering.value = etx::SpectralResponse{spect, 0.5f};
  scattering.pdf_forward = 0.5f;
  scattering.pdf_reverse = 0.5f;
  etx::UPBPDensityMISConfiguration configuration = {};
  configuration.enabled_techniques = static_cast<uint32_t>(etx::UPBPTechnique::BPT);
  configuration.technique_factors[0u] = 1.0;
  const double weight =
    etx::upbp_bpt_connection_cross_technique_weight({&scene, &light, &camera, &connection, light_weights, camera_weights, scattering, scattering, configuration});
  valid = close_value(weight, 2.0 / 3.0, 1.0e-14, "BPT cross-technique MIS") && valid;

  etx::EmitterSample emitter_sample = {};
  emitter_sample.origin = {2.0f, 0.0f, 0.0f};
  emitter_sample.normal = {-1.0f, 0.0f, 0.0f};
  emitter_sample.pdf_sample = 0.5f;
  emitter_sample.pdf_dir = 0.25f;
  emitter_sample.pdf_dir_out = 0.125f;
  emitter_sample.is_distant = true;
  const double nee_weight = etx::upbp_bpt_nee_cross_technique_weight({&scene, &camera, &connection, camera_weights, scattering, configuration, emitter_sample});
  valid = close_value(nee_weight, 1.0 / 5.25, 1.0e-14, "NEE cross-technique MIS") && valid;
  std::printf("BPT cross-technique MIS %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_point_index() {
  constexpr uint32_t point_count = 4096u;
  constexpr float radius = 0.35f;
  std::vector<etx::UPBPPointReference> points;
  points.reserve(point_count);
  etx::Sampler sampler{0x9127a1u};
  for (uint32_t index = 0u; index < point_count; ++index) {
    const float2 position_xy = sampler.next_2d();
    const float3 position = (float3{position_xy.x, position_xy.y, sampler.next()} - float3{0.5f, 0.5f, 0.5f}) * 20.0f;
    points.emplace_back(etx::UPBPPointReference{position, index / 8u, index % 8u});
  }

  etx::UPBPPointIndex point_index = {};
  etx::UPBPPointBeamIndex beam_index = {};
  uint64_t projected_beam_storage = 0u;
  bool valid = point_index.build(points.data(), static_cast<uint32_t>(points.size()), radius);
  const bool projected_storage_valid = beam_index.projected_storage_bytes(static_cast<uint32_t>(points.size()), projected_beam_storage);
  const bool beam_index_valid = beam_index.build(points.data(), static_cast<uint32_t>(points.size()));
  const bool beam_storage_valid = projected_beam_storage >= beam_index.storage_bytes();
  if ((projected_storage_valid == false) || (beam_index_valid == false) || (beam_storage_valid == false)) {
    std::printf("point-beam storage failed: projected %llu actual %llu\n", static_cast<unsigned long long>(projected_beam_storage),
      static_cast<unsigned long long>(beam_index.storage_bytes()));
  }
  valid = projected_storage_valid && beam_index_valid && beam_storage_valid && valid;
  valid = (point_index.size() == point_count) && (beam_index.size() == point_count) && valid;
  for (uint32_t probe_index = 0u; probe_index < 256u; ++probe_index) {
    const float2 probe_xy = sampler.next_2d();
    const float3 probe = (float3{probe_xy.x, probe_xy.y, sampler.next()} - float3{0.5f, 0.5f, 0.5f}) * 20.0f;
    std::vector<uint64_t> accelerated;
    double accelerated_estimate = 0.0;
    const bool query_valid = point_index.query(probe, radius, [&accelerated, &accelerated_estimate](const etx::UPBPPointReference& point, const float distance_squared) {
      accelerated.emplace_back((static_cast<uint64_t>(point.path_index) << 32u) | point.vertex_index);
      const double value = 1.0 + static_cast<double>(point.path_index) * 0.01 + static_cast<double>(point.vertex_index) * 0.001;
      accelerated_estimate += value * etx::upbp_kernel_value(etx::UPBPKernel::Epanechnikov, 3u, radius, distance_squared);
    });
    valid = query_valid && valid;

    std::vector<uint64_t> exhaustive;
    double exhaustive_estimate = 0.0;
    for (const etx::UPBPPointReference& point : points) {
      const float3 delta = point.position - probe;
      const float distance_squared = dot(delta, delta);
      if (distance_squared < radius * radius) {
        exhaustive.emplace_back((static_cast<uint64_t>(point.path_index) << 32u) | point.vertex_index);
        const double value = 1.0 + static_cast<double>(point.path_index) * 0.01 + static_cast<double>(point.vertex_index) * 0.001;
        exhaustive_estimate += value * etx::upbp_kernel_value(etx::UPBPKernel::Epanechnikov, 3u, radius, distance_squared);
      }
    }

    std::sort(accelerated.begin(), accelerated.end());
    std::sort(exhaustive.begin(), exhaustive.end());
    if (accelerated != exhaustive) {
      std::printf("point-beam mismatch at %u: accelerated %zu exhaustive %zu\n", probe_index, accelerated.size(), exhaustive.size());
    }
    valid = (accelerated == exhaustive) && valid;
    valid = close_value(accelerated_estimate, exhaustive_estimate, 1.0e-10, "point estimator accelerated parity") && valid;
  }

  for (uint32_t probe_index = 0u; probe_index < 256u; ++probe_index) {
    const float2 origin_xy = sampler.next_2d();
    const float3 origin = (float3{origin_xy.x, origin_xy.y, sampler.next()} - float3{0.5f, 0.5f, 0.5f}) * 20.0f;
    const float2 direction_xy = sampler.next_2d() * 2.0f - float2{1.0f, 1.0f};
    float3 direction = {direction_xy.x, direction_xy.y, sampler.next() * 2.0f - 1.0f};
    direction = normalize(direction);
    const etx::UPBPBeamReference beam = {origin, direction, 0.25f + 20.0f * sampler.next()};
    std::vector<uint64_t> accelerated;
    uint64_t candidate_count = 0u;
    const bool query_valid = beam_index.query_beam(
      beam, radius, candidate_count,
      [](const etx::UPBPPointReference&, const uint32_t) {
        return true;
      },
      [&accelerated](const etx::UPBPPointReference& point, const etx::UPBPPointBeamIntersection&) {
        accelerated.emplace_back((static_cast<uint64_t>(point.path_index) << 32u) | point.vertex_index);
      });
    valid = query_valid && valid;
    valid = (candidate_count >= accelerated.size()) && valid;

    std::vector<uint64_t> exhaustive;
    for (const etx::UPBPPointReference& point : points) {
      etx::UPBPPointBeamIntersection intersection = {};
      if (etx::upbp_intersect_point_beam(point.position, beam, radius, intersection)) {
        exhaustive.emplace_back((static_cast<uint64_t>(point.path_index) << 32u) | point.vertex_index);
      }
    }
    std::sort(accelerated.begin(), accelerated.end());
    std::sort(exhaustive.begin(), exhaustive.end());
    if (accelerated != exhaustive) {
      std::printf("point-beam mismatch at %u: accelerated %zu exhaustive %zu\n", probe_index, accelerated.size(), exhaustive.size());
    }
    valid = (accelerated == exhaustive) && valid;
  }

  const etx::UPBPBeamReference axis_beam = {{-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 2.0f};
  const bool near_corner = etx::upbp_capsule_intersects_aabb(axis_beam, 0.25f, {-0.1f, 0.17f, 0.17f}, {0.1f, 0.3f, 0.3f});
  const bool far_corner = etx::upbp_capsule_intersects_aabb(axis_beam, 0.25f, {-0.1f, 0.18f, 0.18f}, {0.1f, 0.3f, 0.3f});
  const bool tangent = etx::upbp_capsule_intersects_aabb(axis_beam, 0.25f, {-0.1f, 0.25f, -0.1f}, {0.1f, 0.3f, 0.1f});
  if ((near_corner == false) || far_corner || (tangent == false)) {
    std::printf("capsule bounds failed: near %u far %u tangent %u\n", static_cast<uint32_t>(near_corner), static_cast<uint32_t>(far_corner), static_cast<uint32_t>(tangent));
  }
  valid = near_corner && (far_corner == false) && tangent && valid;

  valid = (point_index.query({}, radius * 2.0f,
             [](const etx::UPBPPointReference&, float) {
             }) == false) &&
          valid;
  std::printf("point index parity %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_bpt_strategy_recurrence() {
  etx::Sampler sampler{0x72a11u};
  bool valid = true;
  for (uint32_t trial = 0u; trial < 2048u; ++trial) {
    const uint32_t vertex_count = 2u + static_cast<uint32_t>(sampler.next() * 14.0f);
    etx::UPBPPathProbabilityRecord path = {};
    path.log_emitter_endpoint_density = -20.0 + 40.0 * static_cast<double>(sampler.next());
    path.log_camera_endpoint_density = -20.0 + 40.0 * static_cast<double>(sampler.next());
    path.vertices.resize(vertex_count);
    path.edges.resize(vertex_count - 1u);
    for (uint32_t vertex_index = 0u; vertex_index < vertex_count; ++vertex_index) {
      path.vertices[vertex_index].connectible = (vertex_index == 0u) || (vertex_index + 1u == vertex_count) || (sampler.next() > 0.25f);
    }
    for (etx::UPBPPathProbabilityEdge& edge : path.edges) {
      edge.log_density_forward = -40.0 + 80.0 * static_cast<double>(sampler.next());
      edge.log_density_reverse = -40.0 + 80.0 * static_cast<double>(sampler.next());
    }

    std::vector<etx::UPBPBPTStrategyProbability> exhaustive;
    std::vector<etx::UPBPBPTStrategyProbability> recursive;
    valid = etx::upbp_enumerate_bpt_strategies_exhaustive(path, exhaustive) && valid;
    valid = etx::upbp_enumerate_bpt_strategies_recursive(path, recursive) && valid;
    valid = (exhaustive.size() == recursive.size()) && valid;
    if (exhaustive.size() != recursive.size()) {
      continue;
    }

    for (uint32_t strategy_index = 0u; strategy_index < exhaustive.size(); ++strategy_index) {
      valid = (exhaustive[strategy_index].applicable == recursive[strategy_index].applicable) && valid;
      valid = close_value(recursive[strategy_index].log_density, exhaustive[strategy_index].log_density, 1.0e-11, "BPT recursive log density") && valid;
      valid =
        close_value(etx::upbp_bpt_balance_weight(recursive, strategy_index), etx::upbp_bpt_balance_weight(exhaustive, strategy_index), 1.0e-13, "BPT recursive balance weight") &&
        valid;
    }
  }

  etx::UPBPPathProbabilityRecord zero_density_path = {};
  zero_density_path.vertices.resize(5u);
  zero_density_path.edges.resize(4u);
  zero_density_path.edges[1u].log_density_forward = -std::numeric_limits<double>::infinity();
  zero_density_path.edges[2u].log_density_reverse = -std::numeric_limits<double>::infinity();
  std::vector<etx::UPBPBPTStrategyProbability> exhaustive_zero;
  std::vector<etx::UPBPBPTStrategyProbability> recursive_zero;
  valid = etx::upbp_enumerate_bpt_strategies_exhaustive(zero_density_path, exhaustive_zero) && valid;
  valid = etx::upbp_enumerate_bpt_strategies_recursive(zero_density_path, recursive_zero) && valid;
  for (uint32_t strategy_index = 0u; strategy_index < exhaustive_zero.size(); ++strategy_index) {
    const double exhaustive_density = exhaustive_zero[strategy_index].log_density;
    const double recursive_density = recursive_zero[strategy_index].log_density;
    valid = ((exhaustive_density == recursive_density) ||
              (std::isfinite(exhaustive_density) && std::isfinite(recursive_density) && (fabs(exhaustive_density - recursive_density) <= 1.0e-14))) &&
            valid;
  }

  std::printf("BPT strategy recurrence %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_sampling_domains() {
  constexpr std::array<etx::UPBPRandomDomain, 17u> domains = {
    etx::UPBPRandomDomain::FilmSample,
    etx::UPBPRandomDomain::CameraPath,
    etx::UPBPRandomDomain::LightPath,
    etx::UPBPRandomDomain::CameraMediumTracking,
    etx::UPBPRandomDomain::LightMediumTracking,
    etx::UPBPRandomDomain::ConnectionTransmittance,
    etx::UPBPRandomDomain::PB2D,
    etx::UPBPRandomDomain::BP2D,
    etx::UPBPRandomDomain::BB1D,
    etx::UPBPRandomDomain::EmitterConnection,
    etx::UPBPRandomDomain::ScatteringEvaluation,
    etx::UPBPRandomDomain::IntersectionTraversal,
    etx::UPBPRandomDomain::FilmConnection,
    etx::UPBPRandomDomain::DirectHitEvaluation,
    etx::UPBPRandomDomain::CameraRussianRoulette,
    etx::UPBPRandomDomain::LightRussianRoulette,
    etx::UPBPRandomDomain::LightPathSelection,
  };
  std::array<uint32_t, domains.size()> seeds = {};
  for (uint32_t index = 0u; index < domains.size(); ++index) {
    seeds[index] = etx::upbp_sampler_seed(17u, 0x100000003ull, 0x200000005ull, 7u, 11u, domains[index]);
  }

  bool valid = true;
  for (uint32_t first = 0u; first < seeds.size(); ++first) {
    valid = (seeds[first] == etx::upbp_sampler_seed(17u, 0x100000003ull, 0x200000005ull, 7u, 11u, domains[first])) && valid;
    for (uint32_t second = first + 1u; second < seeds.size(); ++second) {
      valid = (seeds[first] != seeds[second]) && valid;
    }
  }

  valid = (seeds[0u] != etx::upbp_sampler_seed(17u, 0x100000004ull, 0x200000005ull, 7u, 11u, domains[0u])) && valid;
  valid = (seeds[0u] != etx::upbp_sampler_seed(17u, 0x100000003ull, 0x200000006ull, 7u, 11u, domains[0u])) && valid;
  valid = (seeds[0u] != etx::upbp_sampler_seed(17u, 0x100000003ull, 0x200000005ull, 8u, 11u, domains[0u])) && valid;
  valid = (seeds[0u] != etx::upbp_sampler_seed(17u, 0x100000003ull, 0x200000005ull, 7u, 12u, domains[0u])) && valid;

  constexpr uint32_t stream_sample_count = 65536u;
  constexpr float camera_threshold = 0.2f;
  constexpr float light_threshold = 0.27f;
  uint32_t camera_count = 0u;
  uint32_t light_count = 0u;
  uint32_t joint_count = 0u;
  for (uint32_t path_index = 0u; path_index < stream_sample_count; ++path_index) {
    etx::Sampler camera_sampler{etx::upbp_sampler_seed(17u, 3u, path_index, 0u, 0u, etx::upbp_medium_tracking_random_domain(etx::PathSource::Camera))};
    etx::Sampler light_sampler{etx::upbp_sampler_seed(17u, 3u, path_index, 0u, 0u, etx::upbp_medium_tracking_random_domain(etx::PathSource::Light))};
    const bool camera_event = camera_sampler.next() < camera_threshold;
    const bool light_event = light_sampler.next() < light_threshold;
    camera_count += uint32_t(camera_event);
    light_count += uint32_t(light_event);
    joint_count += uint32_t(camera_event && light_event);
  }
  const double camera_frequency = static_cast<double>(camera_count) / stream_sample_count;
  const double light_frequency = static_cast<double>(light_count) / stream_sample_count;
  const double joint_frequency = static_cast<double>(joint_count) / stream_sample_count;
  valid = (fabs(joint_frequency - camera_frequency * light_frequency) < 0.005) && valid;
  std::printf("sampling domains %s\n", valid ? "valid" : "failed");
  return valid;
}

bool reference_beam_intersection(const etx::UPBPBeamReference& first, const etx::UPBPBeamReference& second, const float radius, etx::UPBPBeamBeamIntersection& result) {
  const float direction_dot = dot(first.direction, second.direction);
  const float denominator = 1.0f - direction_dot * direction_dot;
  constexpr float degeneracy_threshold = 16.0f * std::numeric_limits<float>::epsilon();
  if (denominator <= degeneracy_threshold) {
    return false;
  }
  const float3 origin_delta = first.origin - second.origin;
  const float first_projection = dot(first.direction, origin_delta);
  const float second_projection = dot(second.direction, origin_delta);
  result.first_distance = (direction_dot * second_projection - first_projection) / denominator;
  result.second_distance = (second_projection - direction_dot * first_projection) / denominator;
  if ((result.first_distance < 0.0f) || (result.first_distance >= first.length) || (result.second_distance < 0.0f) || (result.second_distance >= second.length)) {
    return false;
  }
  const float3 first_point = first.origin + first.direction * result.first_distance;
  const float3 second_point = second.origin + second.direction * result.second_distance;
  const float3 delta = first_point - second_point;
  result.distance_squared = dot(delta, delta);
  result.sin_theta = sqrtf(denominator);
  return result.distance_squared < radius * radius;
}

bool validate_beam_geometry_and_index() {
  constexpr uint32_t beam_count = 1024u;
  constexpr float radius = 0.2f;
  etx::Sampler sampler{0xa7c315u};
  std::vector<etx::UPBPBeamReference> beams;
  beams.reserve(beam_count);
  for (uint32_t index = 0u; index < beam_count; ++index) {
    const float2 origin_xy = sampler.next_2d();
    const float3 origin = (float3{origin_xy.x, origin_xy.y, sampler.next()} - float3{0.5f, 0.5f, 0.5f}) * 12.0f;
    const float2 direction_xy = sampler.next_2d();
    float3 direction = float3{direction_xy.x, direction_xy.y, sampler.next()} - float3{0.5f, 0.5f, 0.5f};
    if (dot(direction, direction) < 1.0e-6f) {
      direction = {1.0f, 0.0f, 0.0f};
    } else {
      direction = normalize(direction);
    }
    beams.emplace_back(etx::UPBPBeamReference{origin, direction, 0.1f + 2.9f * sampler.next(), index % 3u, index, index % 8u});
  }

  etx::UPBPBeamGrid index = {};
  etx::UPBPBeamGrid sequential_index = {};
  etx::UPBPSpatialQueryState query_state = {};
  etx::UPBPSpatialQueryState sequential_query_state = {};
  etx::TaskScheduler scheduler = {};
  uint64_t projected_storage = 0u;
  bool valid = index.projected_storage_bytes(beams.data(), static_cast<uint32_t>(beams.size()), radius, projected_storage);
  valid = index.build(beams.data(), static_cast<uint32_t>(beams.size()), radius, scheduler) && valid;
  valid = sequential_index.build(beams.data(), static_cast<uint32_t>(beams.size()), radius) && valid;
  valid = (index.size() == beam_count) && valid;
  valid = (sequential_index.size() == beam_count) && (projected_storage >= index.storage_bytes()) && valid;
  for (uint32_t probe_index = 0u; probe_index < 128u; ++probe_index) {
    const float2 point_xy = sampler.next_2d();
    const float3 point = (float3{point_xy.x, point_xy.y, sampler.next()} - float3{0.5f, 0.5f, 0.5f}) * 12.0f;
    std::vector<uint32_t> accelerated;
    std::vector<uint32_t> sequential;
    uint64_t accelerated_candidate_count = 0u;
    uint64_t sequential_candidate_count = 0u;
    valid = index.query_point_intersections(
              point, radius, accelerated_candidate_count,
              [](const etx::UPBPBeamReference&, const uint32_t) {
                return true;
              },
              [&accelerated](const etx::UPBPBeamReference& beam, const uint32_t, const etx::UPBPPointBeamIntersection&) {
                accelerated.emplace_back(beam.path_index);
              }) &&
            valid;
    valid = sequential_index.query_point_intersections(
              point, radius, sequential_candidate_count,
              [](const etx::UPBPBeamReference&, const uint32_t) {
                return true;
              },
              [&sequential](const etx::UPBPBeamReference& beam, const uint32_t, const etx::UPBPPointBeamIntersection&) {
                sequential.emplace_back(beam.path_index);
              }) &&
            valid;
    valid = (accelerated == sequential) && (accelerated_candidate_count == sequential_candidate_count) && valid;

    std::vector<uint32_t> exhaustive;
    for (const etx::UPBPBeamReference& beam : beams) {
      etx::UPBPPointBeamIntersection intersection = {};
      if (etx::upbp_intersect_point_beam(point, beam, radius, intersection)) {
        exhaustive.emplace_back(beam.path_index);
      }
    }
    std::sort(accelerated.begin(), accelerated.end());
    std::sort(exhaustive.begin(), exhaustive.end());
    if (accelerated != exhaustive) {
      std::printf("beam-beam mismatch at %u: accelerated %zu exhaustive %zu\n", probe_index, accelerated.size(), exhaustive.size());
    }
    valid = (accelerated == exhaustive) && valid;
  }

  for (uint32_t probe_index = 0u; probe_index < 64u; ++probe_index) {
    const etx::UPBPBeamReference& probe = beams[probe_index * 7u];
    std::vector<uint32_t> accelerated;
    std::vector<uint32_t> sequential;
    uint64_t accelerated_candidate_count = 0u;
    uint64_t sequential_candidate_count = 0u;
    valid = index.query_beam_intersections(
              probe, radius, query_state, accelerated_candidate_count,
              [](const etx::UPBPBeamReference&, const uint32_t) {
                return true;
              },
              [&accelerated](const etx::UPBPBeamReference& beam, const uint32_t, const etx::UPBPBeamBeamIntersection&) {
                accelerated.emplace_back(beam.path_index);
              }) &&
            valid;
    valid = sequential_index.query_beam_intersections(
              probe, radius, sequential_query_state, sequential_candidate_count,
              [](const etx::UPBPBeamReference&, const uint32_t) {
                return true;
              },
              [&sequential](const etx::UPBPBeamReference& beam, const uint32_t, const etx::UPBPBeamBeamIntersection&) {
                sequential.emplace_back(beam.path_index);
              }) &&
            valid;
    valid = (accelerated == sequential) && (accelerated_candidate_count == sequential_candidate_count) && valid;

    std::vector<uint32_t> exhaustive;
    for (const etx::UPBPBeamReference& beam : beams) {
      etx::UPBPBeamBeamIntersection optimized = {};
      etx::UPBPBeamBeamIntersection reference = {};
      const bool optimized_hit = etx::upbp_intersect_beams(probe, beam, radius, optimized);
      const bool reference_hit = reference_beam_intersection(probe, beam, radius, reference);
      valid = (optimized_hit == reference_hit) && valid;
      if (optimized_hit && reference_hit) {
        valid = close_value(optimized.first_distance, reference.first_distance, 2.0e-4, "beam first distance") && valid;
        valid = close_value(optimized.second_distance, reference.second_distance, 2.0e-4, "beam second distance") && valid;
        valid = close_value(optimized.distance_squared, reference.distance_squared, 2.0e-4, "beam distance squared") && valid;
        valid = close_value(optimized.sin_theta, reference.sin_theta, 2.0e-5, "beam sine") && valid;
      }
      if (reference_hit) {
        exhaustive.emplace_back(beam.path_index);
      }
    }
    std::sort(accelerated.begin(), accelerated.end());
    std::sort(exhaustive.begin(), exhaustive.end());
    valid = (accelerated == exhaustive) && valid;
  }

  const etx::UPBPBeamReference parallel_a = {{}, {1.0f, 0.0f, 0.0f}, 2.0f, 0u, 0u, 0u};
  const etx::UPBPBeamReference parallel_b = {{0.0f, 0.1f, 0.0f}, {1.0f, 0.0f, 0.0f}, 2.0f, 0u, 1u, 0u};
  etx::UPBPBeamBeamIntersection parallel_intersection = {};
  valid = (etx::upbp_intersect_beams(parallel_a, parallel_b, radius, parallel_intersection) == false) && valid;
  std::printf("beam geometry/index parity %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_physical_beam_collection() {
  const etx::SpectralQuery spect = etx::SpectralQuery::sample();
  etx::UPBPPathRecord path = {};
  path.reset(4u);
  etx::UPBPPathVertexRecord endpoint = {};
  endpoint.throughput = etx::SpectralResponse{spect, 1.0f};
  endpoint.outgoing_throughput = endpoint.throughput;
  bool valid = path.append_endpoint(endpoint);

  etx::UPBPSegmentRecord interval = {};
  interval.reset(spect, 5u, {1.0f, 2.0f, 3.0f});
  etx::MediumTrackingEvent null_event = {};
  null_event.type = etx::MediumTrackingEventType::Null;
  null_event.position = {1.5f, 2.0f, 3.0f};
  null_event.distance = 0.5f;
  null_event.weight = etx::SpectralResponse{spect, 1.0f};
  null_event.pdf_forward = 0.5f;
  null_event.pdf_reverse = 0.5f;
  valid = interval.append(null_event) && valid;
  etx::MediumTrackingEvent scatter_event = {};
  scatter_event.type = etx::MediumTrackingEventType::Scatter;
  scatter_event.position = {3.0f, 2.0f, 3.0f};
  scatter_event.distance = 1.5f;
  scatter_event.weight = etx::SpectralResponse{spect, 1.0f};
  scatter_event.majorant = 2.0f;
  scatter_event.majorant_transmittance = 0.5f;
  scatter_event.event_probability = 0.25f;
  scatter_event.pdf_forward = 0.25f;
  scatter_event.pdf_reverse = 0.25f;
  valid = interval.append(scatter_event) && valid;

  etx::UPBPTransportSegmentRecord segment = {};
  segment.reset(spect);
  valid = segment.append(etx::upbp_vacuum_interval(spect, {}, {1.0f, 0.0f, 0.0f}, 1.0f)) && valid;
  valid = segment.append(interval) && valid;
  etx::UPBPPathVertexRecord vertex = {};
  vertex.throughput = etx::SpectralResponse{spect, 1.0f};
  vertex.cls = etx::UPBPVertexClass::Medium;
  valid = path.append_physical_vertex(segment, vertex) && valid;

  std::vector<etx::UPBPBeamReference> beams;
  valid = etx::upbp_collect_medium_beams(path, 17u, beams) && valid;
  valid = (beams.size() == 1u) && valid;
  if (beams.size() == 1u) {
    valid = close_value(beams[0u].length, 2.0, 1.0e-6, "physical beam length") && valid;
    valid = (beams[0u].medium_index == 5u) && (beams[0u].path_index == 17u) && valid;
    const etx::UPBPSegmentRecord* beam_interval = etx::upbp_beam_interval(path, beams[0u]);
    valid = (beam_interval != nullptr) && valid;
    if (beam_interval != nullptr) {
      etx::UPBPBeamTransportPrefix prefix = {};
      valid = etx::upbp_medium_interval_prefix(*beam_interval, 1.0f, prefix) && valid;
      valid = close_value(prefix.distance, 1.0, 1.0e-6, "physical beam prefix length") && valid;
      valid = close_value(prefix.log_transport_pdf_forward, std::log(0.5) - 1.0, 1.0e-7, "physical beam prefix density") && valid;
      etx::UPBPBeamTransportPrefix transport_prefix = {};
      valid = etx::upbp_beam_transport_prefix(path, beams[0u], 1.0f, transport_prefix) && valid;
      valid = close_value(transport_prefix.distance, 2.0, 1.0e-6, "boundary-aware beam prefix length") && valid;
      valid = close_value(transport_prefix.log_transport_pdf_forward, std::log(0.5) - 1.0, 1.0e-7, "boundary-aware beam prefix density") && valid;
    }
  }

  etx::UPBPSegmentRecord collapsed_interval = {};
  collapsed_interval.reset(spect, 5u, vertex.position);
  etx::MediumTrackingEvent collapsed_escape = {};
  collapsed_escape.type = etx::MediumTrackingEventType::Escape;
  collapsed_escape.position = vertex.position;
  collapsed_escape.distance = std::numeric_limits<float>::denorm_min();
  collapsed_escape.weight = etx::SpectralResponse{spect, 1.0f};
  valid = collapsed_interval.append(collapsed_escape) && valid;
  etx::UPBPTransportSegmentRecord collapsed_segment = {};
  collapsed_segment.reset(spect);
  valid = collapsed_segment.append(collapsed_interval) && valid;

  etx::UPBPSegmentRecord escape_interval = {};
  escape_interval.reset(spect, 5u, vertex.position);
  etx::MediumTrackingEvent escape = {};
  escape.type = etx::MediumTrackingEventType::Escape;
  escape.position = vertex.position + float3{1.0f, 0.0f, 0.0f};
  escape.distance = 1.0f;
  escape.weight = etx::SpectralResponse{spect, 1.0f};
  escape.majorant = 2.0f;
  escape.majorant_transmittance = std::exp(-2.0f);
  escape.pdf_forward = escape.majorant_transmittance;
  escape.pdf_reverse = escape.majorant_transmittance;
  valid = escape_interval.append(escape) && valid;
  valid = collapsed_segment.append(escape_interval) && valid;
  valid = path.append_terminal_segment(collapsed_segment) && valid;
  valid = etx::upbp_collect_medium_beams(path, 17u, beams) && valid;
  valid = (beams.size() == 2u) && valid;
  if (beams.size() == 2u) {
    const etx::UPBPSegmentRecord* terminal_interval = etx::upbp_beam_interval(path, beams[1u]);
    valid = (terminal_interval != nullptr) && valid;
    if (terminal_interval != nullptr) {
      valid = (terminal_interval->terminal_event == etx::MediumTrackingEventType::Escape) && valid;
      valid = close_value(beams[1u].length, 1.0, 1.0e-6, "escape-ending physical beam length") && valid;
    }
  }
  std::printf("physical beam collection %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_joined_path_topology() {
  const etx::SpectralQuery spect = etx::SpectralQuery::sample();
  auto make_path = [spect](const etx::UPBPVertexClass endpoint_class, const etx::PathSource source, const float start, const float step) {
    etx::UPBPPathRecord path = {};
    path.reset(4u);
    etx::UPBPPathVertexRecord endpoint = {};
    endpoint.position = {start, 0.0f, 0.0f};
    endpoint.throughput = etx::SpectralResponse{spect, 1.0f};
    endpoint.cls = endpoint_class;
    endpoint.source = source;
    endpoint.connectible = true;
    path.append_endpoint(endpoint);

    etx::UPBPTransportSegmentRecord segment = {};
    segment.reset(spect);
    segment.append(etx::upbp_vacuum_interval(spect, endpoint.position, {step > 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f}, fabsf(step)));
    etx::UPBPPathVertexRecord vertex = {};
    vertex.position = {start + step, 0.0f, 0.0f};
    vertex.throughput = etx::SpectralResponse{spect, 1.0f};
    vertex.cls = etx::UPBPVertexClass::Surface;
    vertex.source = source;
    vertex.connectible = true;
    path.append_physical_vertex(segment, vertex);
    return path;
  };

  const etx::UPBPPathRecord light = make_path(etx::UPBPVertexClass::Emitter, etx::PathSource::Light, 0.0f, 1.0f);
  etx::UPBPPathRecord camera = make_path(etx::UPBPVertexClass::Camera, etx::PathSource::Camera, 3.0f, -1.0f);
  etx::UPBPTransportSegmentRecord connection = {};
  connection.reset(spect);
  connection.append(etx::upbp_vacuum_interval(spect, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 1.0f));

  etx::UPBPJoinedPath joined = {};
  bool valid = etx::upbp_join_subpaths(light, 2u, camera, 2u, connection, false, joined);
  valid = joined.valid() && (joined.vertices.size() == 4u) && (joined.edges.size() == 3u) && valid;
  valid = (joined.vertices[0u]->cls == etx::UPBPVertexClass::Emitter) && (joined.vertices[1u]->source == etx::PathSource::Light) &&
          (joined.vertices[2u]->source == etx::PathSource::Camera) && (joined.vertices[3u]->cls == etx::UPBPVertexClass::Camera) && valid;
  valid = (joined.edges[0u].reversed == false) && (joined.edges[1u].reversed == false) && joined.edges[2u].reversed && valid;

  etx::UPBPPathRecord endpoint_only = {};
  endpoint_only.reset(1u);
  valid = endpoint_only.append_endpoint(light.vertices[0u]) && valid;
  etx::UPBPJoinedPath nee_path = {};
  valid = etx::upbp_join_subpaths(endpoint_only, 1u, camera, 2u, connection, true, nee_path) && valid;
  valid = (nee_path.vertices.size() == 3u) && (nee_path.edges.size() == 2u) && nee_path.edges[0u].reversed && nee_path.edges[1u].reversed && valid;

  etx::UPBPPathRecord camera_endpoint = {};
  camera_endpoint.reset(1u);
  valid = camera_endpoint.append_endpoint(camera.vertices[0u]) && valid;
  etx::UPBPJoinedPath light_trace_path = {};
  valid = etx::upbp_join_subpaths(light, 2u, camera_endpoint, 1u, connection, false, light_trace_path) && valid;
  valid = (light_trace_path.selected_light_vertex_count == 2u) && (light_trace_path.vertices.size() == 3u) && (light_trace_path.edges.size() == 2u) && valid;

  etx::UPBPPathRecord terminal_path = camera_endpoint;
  terminal_path.maximum_physical_vertices = 2u;
  valid = terminal_path.append_terminal_segment(connection) && valid;
  etx::UPBPPathVertexRecord environment_endpoint = light.vertices[0u];
  valid = terminal_path.promote_terminal_segment(environment_endpoint) && terminal_path.valid() && (terminal_path.has_terminal_segment == false) &&
          (terminal_path.vertices.size() == 2u) && (terminal_path.segments.size() == 1u) && valid;
  camera.vertices[1u].intersection.emitter_index = 0u;
  etx::UPBPJoinedPath direct_path = {};
  valid = etx::upbp_join_direct_hit(camera, 2u, direct_path) && valid;
  valid = (direct_path.selected_light_vertex_count == 0u) && (direct_path.vertices.size() == 2u) && (direct_path.edges.size() == 1u) && direct_path.edges[0u].reversed && valid;
  std::printf("joined path topology %s\n", valid ? "valid" : "failed");
  return valid;
}

}  // namespace

int main() {
  bool valid = true;
  valid = validate_kernels() && valid;
  valid = validate_progressive_radii() && valid;
  valid = validate_light_splat_iteration_scale() && valid;
  valid = validate_bpt_path_length_limits() && valid;
  valid = validate_light_path_selection() && valid;
  valid = validate_population_radius_scale() && valid;
  valid = validate_automatic_initial_radius() && valid;
  valid = validate_projected_measure_domain() && valid;
  valid = validate_medium_origin_trace_retry() && valid;
  valid = validate_bpt_strategy_controls() && valid;
  valid = validate_density_technique_factors() && valid;
  valid = validate_recursive_local_pde_affine() && valid;
  valid = validate_prepared_medium_density() && valid;
  valid = validate_balance_heuristic() && valid;
  valid = validate_segment_record() && valid;
  valid = validate_segment_walker() && valid;
  valid = validate_path_record() && valid;
  valid = validate_path_storage_reservation() && valid;
  valid = validate_transport_segment() && valid;
  valid = validate_recursive_balance_accumulator() && valid;
  valid = validate_density_mis_recurrence() && valid;
  valid = validate_augmented_recursive_weights() && valid;
  valid = validate_zero_reverse_scattering_density() && valid;
  valid = validate_point_merge_mis() && valid;
  valid = validate_medium_pre_collision_throughput() && valid;
  valid = validate_bpt_cross_technique_mis() && valid;
  valid = validate_point_index() && valid;
  valid = validate_bpt_strategy_recurrence() && valid;
  valid = validate_sampling_domains() && valid;
  valid = validate_beam_geometry_and_index() && valid;
  valid = validate_physical_beam_collection() && valid;
  valid = validate_joined_path_topology() && valid;
  return valid ? 0 : 1;
}
