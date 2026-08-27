#include <etx/core/core.hxx>

#include <etx/rt/integrators/upbp_cpu.hxx>

#include <etx/render/host/film.hxx>
#include <etx/render/shared/scene_camera.hxx>
#include <etx/rt/integrators/upbp_beam_estimators.hxx>
#include <etx/rt/integrators/upbp_bpt.hxx>
#include <etx/rt/integrators/upbp_options.hxx>
#include <etx/rt/shared/vcm_shared.hxx>

#include <array>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace etx {

namespace {

constexpr uint32_t kUPBPTechniqueMask = static_cast<uint32_t>(UPBPTechnique::BPT) | static_cast<uint32_t>(UPBPTechnique::Surface) | static_cast<uint32_t>(UPBPTechnique::PP3D) |
                                        static_cast<uint32_t>(UPBPTechnique::PB2D) | static_cast<uint32_t>(UPBPTechnique::BP2D) | static_cast<uint32_t>(UPBPTechnique::BB1D);

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

bool upbp_checked_add(const uint64_t first, const uint64_t second, uint64_t& result) {
  if (first > std::numeric_limits<uint64_t>::max() - second) {
    return false;
  }
  result = first + second;
  return true;
}

uint64_t upbp_segment_storage_bytes(const UPBPSegmentRecord& segment) {
  return static_cast<uint64_t>(segment.events.capacity()) * sizeof(UPBPMediumTrackingEventRecord);
}

uint64_t upbp_transport_storage_bytes(const UPBPTransportSegmentRecord& segment) {
  uint64_t result = static_cast<uint64_t>(segment.intervals.capacity()) * sizeof(UPBPSegmentRecord);
  for (const UPBPSegmentRecord& interval : segment.intervals) {
    result += upbp_segment_storage_bytes(interval);
  }
  return result;
}

uint64_t upbp_path_storage_bytes(const UPBPPathRecord& path) {
  uint64_t result =
    static_cast<uint64_t>(path.vertices.capacity()) * sizeof(UPBPPathVertexRecord) + static_cast<uint64_t>(path.segments.capacity()) * sizeof(UPBPTransportSegmentRecord);
  for (const UPBPTransportSegmentRecord& segment : path.segments) {
    result += upbp_transport_storage_bytes(segment);
  }
  if (path.has_terminal_segment) {
    result += upbp_transport_storage_bytes(path.terminal_segment);
  }
  return result;
}

uint64_t upbp_recursive_storage_bytes(const UPBPRecursivePathWeights& weights) {
  return static_cast<uint64_t>(weights.arrivals.capacity()) * sizeof(UPBPRecursiveVertexWeights) +
         static_cast<uint64_t>(weights.departures.capacity()) * sizeof(UPBPRecursiveState) + static_cast<uint64_t>(weights.has_departure.capacity() + 7u) / 8u;
}

bool upbp_options_valid(const UPBPOptions& options, std::string& reason) {
  if ((options.technique_mask & kUPBPTechniqueMask) == 0u) {
    reason = "UPBP requires at least one enabled technique";
    return false;
  }
  if ((options.technique_mask & ~kUPBPTechniqueMask) != 0u) {
    reason = "UPBP technique mask contains unsupported bits";
    return false;
  }
  if ((options.kernel != UPBPKernel::TopHat) && (options.kernel != UPBPKernel::Epanechnikov)) {
    reason = "UPBP kernel is invalid";
    return false;
  }
  const float radii[] = {
    options.initial_surface_radius,
    options.initial_pp3d_radius,
    options.initial_pb2d_radius,
    options.initial_bp2d_radius,
    options.initial_bb1d_radius,
  };
  for (const float radius : radii) {
    if ((radius < 0.0f) || (std::isfinite(radius) == false)) {
      reason = "UPBP initial radii must be finite and nonnegative";
      return false;
    }
  }
  if ((options.radius_alpha <= 0.0f) || (options.radius_alpha > 1.0f) || (std::isfinite(options.radius_alpha) == false)) {
    reason = "UPBP radius alpha must be finite and in (0, 1]";
    return false;
  }
  if ((options.beam_selection_probability <= 0.0f) || (options.beam_selection_probability > 1.0f) || (std::isfinite(options.beam_selection_probability) == false)) {
    reason = "UPBP beam-selection probability must be finite and in (0, 1]";
    return false;
  }
  if ((options.maximum_boundary_count == 0u) || (options.maximum_boundary_count > UPBPOptions::kMaximumBoundaryCount)) {
    reason = "UPBP maximum boundary count is outside the supported range";
    return false;
  }
  if ((options.maximum_null_events_per_interval == 0u) || (options.maximum_null_events_per_interval > UPBPOptions::kMaximumNullEventsPerInterval)) {
    reason = "UPBP maximum null-event count is outside the supported range";
    return false;
  }
  if (options.maximum_light_path_count > UPBPOptions::kMaximumLightPathCount) {
    reason = "UPBP maximum light-path count is outside the supported range";
    return false;
  }
  if (options.maximum_bb1d_light_path_count > UPBPOptions::kMaximumLightPathCount) {
    reason = "UPBP maximum BB1D light-path count is outside the supported range";
    return false;
  }
  if ((options.memory_budget_mb < UPBPOptions::kMinimumMemoryBudgetMiB) || (options.memory_budget_mb > UPBPOptions::kMaximumMemoryBudgetMiB)) {
    reason = "UPBP light-storage target is outside the supported range";
    return false;
  }
  return true;
}

double upbp_initial_radius(const float configured_radius, const Scene& scene, const uint64_t camera_subpath_count, const uint64_t light_subpath_count, const uint32_t dimension,
  const double relative_radius_scale) {
  if (configured_radius > 0.0f) {
    return configured_radius;
  }
  return upbp_automatic_initial_radius(scene.bounding_sphere_radius, camera_subpath_count, light_subpath_count, dimension, relative_radius_scale);
}

UPBPIterationParameters upbp_iteration_parameters(const UPBPOptions& options, const Scene& scene, const Film& film, const uint64_t iteration, const uint64_t light_subpath_count) {
  UPBPIterationParameters result = {};
  result.camera_subpath_count = film.current_pixel_count();
  result.light_subpath_count = light_subpath_count;
  result.bpt_sample_count = options.enabled(UPBPTechnique::BPT) ? 1u : 0u;
  VCMIteration spectral_iteration = {};
  spectral_iteration.iteration = static_cast<uint32_t>(iteration);
  result.spect = vcm_iteration_spectral_query(scene, spectral_iteration);

  result.surface_radius = upbp_progressive_radius(
    upbp_initial_radius(options.initial_surface_radius, scene, result.camera_subpath_count, result.light_subpath_count, 2u, kUPBPAutomaticSurfaceRadiusScale), options.radius_alpha,
    2u, iteration);
  result.pp3d_radius =
    upbp_progressive_radius(upbp_initial_radius(options.initial_pp3d_radius, scene, result.camera_subpath_count, result.light_subpath_count, 3u, kUPBPAutomaticVolumeRadiusScale),
      options.radius_alpha, 3u, iteration);
  result.pb2d_radius =
    upbp_progressive_radius(upbp_initial_radius(options.initial_pb2d_radius, scene, result.camera_subpath_count, result.light_subpath_count, 2u, kUPBPAutomaticVolumeRadiusScale),
      options.radius_alpha, 2u, iteration);
  result.bp2d_radius =
    upbp_progressive_radius(upbp_initial_radius(options.initial_bp2d_radius, scene, result.camera_subpath_count, result.light_subpath_count, 2u, kUPBPAutomaticVolumeRadiusScale),
      options.radius_alpha, 2u, iteration);
  result.bb1d_radius =
    upbp_progressive_radius(upbp_initial_radius(options.initial_bb1d_radius, scene, result.camera_subpath_count, result.light_subpath_count, 1u, kUPBPAutomaticVolumeRadiusScale),
      options.radius_alpha, 1u, iteration);

  result.mis.enabled_techniques = options.technique_mask;
  if (scene.strategy_enabled(Scene::Strategy::MergeVertices) == false) {
    result.mis.enabled_techniques &= static_cast<uint32_t>(UPBPTechnique::BPT);
  }
  if (result.mis.enabled(UPBPTechnique::BB1D)) {
    result.bb1d_light_subpath_count =
      options.maximum_bb1d_light_path_count > 0u ? min(light_subpath_count, static_cast<uint64_t>(options.maximum_bb1d_light_path_count)) : light_subpath_count;
  }
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

Sampler upbp_pair_sampler(const uint32_t render_seed, const uint64_t iteration, const uint32_t camera_path_index, const uint32_t light_path_index,
  const uint32_t camera_vertex_index, const uint32_t light_vertex_index, const UPBPRandomDomain domain) {
  const uint64_t pair_index = (static_cast<uint64_t>(camera_path_index) << 32u) | light_path_index;
  return Sampler{upbp_sampler_seed(render_seed, iteration, pair_index, camera_vertex_index, light_vertex_index, domain)};
}

}  // namespace

struct CPUUPBPImpl {
  enum CameraTimingStage : uint32_t {
    CameraTimingSubpath,
    CameraTimingRecursiveMIS,
    CameraTimingBPT,
    CameraTimingSurfaceMerge,
    CameraTimingPP3D,
    CameraTimingBP2D,
    CameraTimingBeamPreparation,
    CameraTimingPB2D,
    CameraTimingBB1D,
    CameraTimingFilmSubmission,
    CameraTimingStageCount,
  };
  static constexpr uint32_t camera_timing_sample_rate = 16u;
  static_assert((camera_timing_sample_rate & (camera_timing_sample_rate - 1u)) == 0u);

  enum SpatialStatisticIndex : uint32_t {
    SpatialStatisticBP2D,
    SpatialStatisticPB2D,
    SpatialStatisticBB1D,
    SpatialStatisticCount,
  };

  struct SpatialTechniqueStatistics {
    uint64_t queries = 0u;
    uint64_t candidates = 0u;
    uint64_t eligible_candidates = 0u;
    uint64_t intersections = 0u;
    uint64_t contributions = 0u;
    uint64_t published_queries = 0u;
    uint64_t published_candidates = 0u;
    uint64_t published_eligible_candidates = 0u;
    uint64_t published_intersections = 0u;
    uint64_t published_contributions = 0u;
  };

  struct CameraEvaluationStatistics {
    uint64_t camera_paths = 0u;
    uint64_t published_camera_paths = 0u;
    std::array<SpatialTechniqueStatistics, SpatialStatisticCount> spatial = {};
    std::array<uint64_t, CameraTimingStageCount> stage_times = {};
    std::array<uint64_t, CameraTimingStageCount> published_stage_times = {};
  };

  struct CameraWorkspace {
    UPBPCameraSubpathResult camera = {};
    UPBPRecursivePathWeights weights = {};
    std::vector<UPBPBeamReference> beams = {};
    std::vector<UPBPPreparedBeam> prepared_beams = {};
    UPBPSpatialQueryState bb1d_query_state = {};
  };

  struct LightTask : public Task {
    CPUUPBPImpl* owner = nullptr;

    LightTask(CPUUPBPImpl* value)
      : owner(value) {
    }

    void execute_range(const uint32_t begin, const uint32_t end, const uint32_t thread_id) override {
      owner->build_light_paths(begin, end, thread_id);
    }
  } light_task = {this};

  struct CameraTask : public Task {
    CPUUPBPImpl* owner = nullptr;

    CameraTask(CPUUPBPImpl* value)
      : owner(value) {
    }

    void execute_range(const uint32_t begin, const uint32_t end, const uint32_t thread_id) override {
      owner->evaluate_camera_paths(begin, end, thread_id);
    }
  } camera_task = {this};

  enum class Stage : uint32_t {
    Light,
    Camera,
  } stage = Stage::Light;

  enum DebugInfoIndex : uint32_t {
    DebugSurfaceRadius,
    DebugPP3DRadius,
    DebugPB2DRadius,
    DebugBP2DRadius,
    DebugBB1DRadius,
    DebugSurfacePoints,
    DebugMediumPoints,
    DebugLightBeams,
    DebugLightPaths,
    DebugBB1DLightPaths,
    DebugBB1DLightBeams,
    DebugLightStorage,
    DebugLightBudget,
    DebugIterationSetupTime,
    DebugLightPathTime,
    DebugStorageValidationTime,
    DebugPrimitiveCollectionTime,
    DebugSpatialIndexTime,
    DebugLightSplatTime,
    DebugCameraEvaluationTime,
    DebugCameraProgress,
    DebugCameraSubpathTime,
    DebugCameraRecursiveMISTime,
    DebugCameraBPTTime,
    DebugCameraSurfaceMergeTime,
    DebugCameraPP3DTime,
    DebugCameraBP2DTime,
    DebugCameraBeamPreparationTime,
    DebugCameraPB2DTime,
    DebugCameraBB1DTime,
    DebugCameraFilmSubmissionTime,
    DebugBP2DQueries,
    DebugBP2DCandidates,
    DebugBP2DEligibleCandidates,
    DebugBP2DIntersections,
    DebugBP2DContributions,
    DebugBP2DCandidatesPerQuery,
    DebugPB2DQueries,
    DebugPB2DCandidates,
    DebugPB2DEligibleCandidates,
    DebugPB2DIntersections,
    DebugPB2DContributions,
    DebugPB2DCandidatesPerQuery,
    DebugBB1DQueries,
    DebugBB1DCandidates,
    DebugBB1DEligibleCandidates,
    DebugBB1DIntersections,
    DebugBB1DContributions,
    DebugBB1DCandidatesPerQuery,
    DebugInfoCount,
  };

  Raytracing& rt;
  std::atomic<Integrator::State>* state = nullptr;
  Integrator::Status status = {};
  std::array<Integrator::Status::DebugInfo, DebugInfoCount> debug_info = {};
  TimeMeasure iteration_time = {};
  TimeMeasure stage_time = {};
  TimeMeasure live_report_time = {};
  Task::Handle task_handle = {};
  UPBPOptions options = {};
  UPBPIterationParameters iteration = {};
  UPBPPreparedBB1D prepared_bb1d = {};

  std::vector<UPBPLightSubpathResult> light_paths = {};
  std::vector<UPBPRecursivePathWeights> light_weights = {};
  std::vector<std::vector<UPBPLightSplat>> light_splats = {};
  std::vector<UPBPPointReference> surface_points = {};
  std::vector<UPBPPointReference> medium_points = {};
  std::vector<uint64_t> light_vertex_offsets = {};
  std::vector<SpectralResponse> prepared_light_vertex_throughputs = {};
  std::vector<uint8_t> prepared_light_vertex_validity = {};
  std::vector<UPBPBeamReference> light_beams = {};
  std::vector<UPBPPreparedBeam> prepared_bp2d_beams = {};
  std::vector<uint8_t> prepared_bp2d_validity = {};
  std::vector<UPBPBeamReference> selected_bb1d_beams = {};
  std::vector<UPBPPreparedBeam> prepared_bb1d_beams = {};
  std::vector<uint8_t> prepared_bb1d_validity = {};
  std::vector<UPBPMediumTrackingEventRecord> prepared_bb1d_tracking_events = {};
  std::vector<UPBPPreparedMedium> prepared_mediums = {};
  UPBPPointIndex surface_index = {};
  UPBPPointIndex pp3d_index = {};
  UPBPPointBeamIndex pb2d_index = {};
  UPBPBeamGrid bp2d_index = {};
  UPBPBeamGrid bb1d_index = {};
  std::vector<CameraWorkspace> camera_workspaces = {};

  std::atomic<bool> failure_claimed = false;
  std::atomic<bool> failed = false;
  std::mutex failure_lock = {};
  std::string failure_reason = {};
  uint64_t memory_target_bytes = 0u;
  std::atomic<uint64_t> evaluated_light_path_count = 0u;
  std::atomic<uint64_t> evaluated_camera_path_count = 0u;
  std::array<std::atomic<uint64_t>, SpatialStatisticCount> spatial_query_counts = {};
  std::array<std::atomic<uint64_t>, SpatialStatisticCount> spatial_candidate_counts = {};
  std::array<std::atomic<uint64_t>, SpatialStatisticCount> spatial_eligible_candidate_counts = {};
  std::array<std::atomic<uint64_t>, SpatialStatisticCount> spatial_intersection_counts = {};
  std::array<std::atomic<uint64_t>, SpatialStatisticCount> spatial_contribution_counts = {};
  std::array<std::atomic<uint64_t>, CameraTimingStageCount> camera_stage_times = {};
  double iteration_setup_time_ms = 0.0;
  double light_path_time_ms = 0.0;
  double storage_validation_time_ms = 0.0;
  double primitive_collection_time_ms = 0.0;
  double spatial_index_time_ms = 0.0;
  double light_splat_time_ms = 0.0;
  double camera_evaluation_time_ms = 0.0;

  CPUUPBPImpl(Raytracing& raytracing, std::atomic<Integrator::State>* integrator_state)
    : rt(raytracing)
    , state(integrator_state)
    , camera_workspaces(rt.scheduler().max_thread_count()) {
    debug_info[DebugSurfaceRadius].title = "Surface radius";
    debug_info[DebugPP3DRadius].title = "PP3D radius";
    debug_info[DebugPB2DRadius].title = "PB2D radius";
    debug_info[DebugBP2DRadius].title = "BP2D radius";
    debug_info[DebugBB1DRadius].title = "BB1D radius";
    debug_info[DebugSurfacePoints].title = "Surface points";
    debug_info[DebugMediumPoints].title = "Medium points";
    debug_info[DebugLightBeams].title = "Light beams";
    debug_info[DebugLightPaths].title = "Light paths";
    debug_info[DebugBB1DLightPaths].title = "BB1D light paths";
    debug_info[DebugBB1DLightBeams].title = "BB1D light beams";
    debug_info[DebugLightStorage].title = "Light storage (MiB)";
    debug_info[DebugLightBudget].title = "Light target (MiB)";
    debug_info[DebugIterationSetupTime].title = "Iteration setup (ms)";
    debug_info[DebugLightPathTime].title = "Light paths (ms)";
    debug_info[DebugStorageValidationTime].title = "Storage accounting (ms)";
    debug_info[DebugPrimitiveCollectionTime].title = "Primitive collection (ms)";
    debug_info[DebugSpatialIndexTime].title = "Spatial indices (ms)";
    debug_info[DebugLightSplatTime].title = "Light splats (ms)";
    debug_info[DebugCameraEvaluationTime].title = "Camera evaluation (ms)";
    debug_info[DebugCameraProgress].title = "Camera progress (%)";
    debug_info[DebugCameraSubpathTime].title = "Camera subpaths (estimated CPU ms)";
    debug_info[DebugCameraRecursiveMISTime].title = "Camera recursive MIS (estimated CPU ms)";
    debug_info[DebugCameraBPTTime].title = "Camera BPT/direct hit (estimated CPU ms)";
    debug_info[DebugCameraSurfaceMergeTime].title = "Camera surface merge (estimated CPU ms)";
    debug_info[DebugCameraPP3DTime].title = "Camera PP3D (estimated CPU ms)";
    debug_info[DebugCameraBP2DTime].title = "Camera BP2D (estimated CPU ms)";
    debug_info[DebugCameraBeamPreparationTime].title = "Camera beam preparation (estimated CPU ms)";
    debug_info[DebugCameraPB2DTime].title = "Camera PB2D (estimated CPU ms)";
    debug_info[DebugCameraBB1DTime].title = "Camera BB1D (estimated CPU ms)";
    debug_info[DebugCameraFilmSubmissionTime].title = "Camera film submission (estimated CPU ms)";
    debug_info[DebugBP2DQueries].title = "BP2D queries (M)";
    debug_info[DebugBP2DCandidates].title = "BP2D candidates (M)";
    debug_info[DebugBP2DEligibleCandidates].title = "BP2D eligible candidates (M)";
    debug_info[DebugBP2DIntersections].title = "BP2D intersections (M)";
    debug_info[DebugBP2DContributions].title = "BP2D contributions (M)";
    debug_info[DebugBP2DCandidatesPerQuery].title = "BP2D candidates/query";
    debug_info[DebugPB2DQueries].title = "PB2D queries (M)";
    debug_info[DebugPB2DCandidates].title = "PB2D candidates (M)";
    debug_info[DebugPB2DEligibleCandidates].title = "PB2D eligible candidates (M)";
    debug_info[DebugPB2DIntersections].title = "PB2D intersections (M)";
    debug_info[DebugPB2DContributions].title = "PB2D contributions (M)";
    debug_info[DebugPB2DCandidatesPerQuery].title = "PB2D candidates/query";
    debug_info[DebugBB1DQueries].title = "BB1D queries (M)";
    debug_info[DebugBB1DCandidates].title = "BB1D candidates (M)";
    debug_info[DebugBB1DEligibleCandidates].title = "BB1D eligible candidates (M)";
    debug_info[DebugBB1DIntersections].title = "BB1D intersections (M)";
    debug_info[DebugBB1DContributions].title = "BB1D contributions (M)";
    debug_info[DebugBB1DCandidatesPerQuery].title = "BB1D candidates/query";
    static_assert(DebugPB2DQueries == DebugBP2DQueries + 6u);
    static_assert(DebugBB1DQueries == DebugPB2DQueries + 6u);
    status.debug_info = debug_info.data();
    status.debug_info_count = static_cast<uint32_t>(debug_info.size());
  }

  ~CPUUPBPImpl() {
    wait_for_tasks();
  }

  bool running() const {
    return (state->load() != Integrator::State::Stopped) && (failure_claimed.load() == false);
  }

  void wait_for_tasks() {
    rt.scheduler().wait_and_release(task_handle);
    task_handle = {};
  }

  void fail(const std::string& reason) {
    bool expected = false;
    if (failure_claimed.compare_exchange_strong(expected, true)) {
      std::scoped_lock lock(failure_lock);
      failure_reason = reason;
      log::error("%s", reason.c_str());
      failed.store(true);
    }
  }

  const char* status_string() const {
    if (failed.load()) {
      return failure_reason.c_str();
    }
    if (state->load() == Integrator::State::Stopped) {
      return "UPBP stopped";
    }
    return stage == Stage::Light ? "UPBP building light paths" : "UPBP evaluating camera paths";
  }

  Integrator::PathProgress path_progress() const {
    if (stage == Stage::Light) {
      return {
        .phase = Integrator::PathProgress::Phase::Light,
        .completed_path_count = evaluated_light_path_count.load(std::memory_order_relaxed),
        .total_path_count = iteration.light_subpath_count,
      };
    }
    return {
      .phase = Integrator::PathProgress::Phase::Camera,
      .completed_path_count = evaluated_camera_path_count.load(std::memory_order_relaxed),
      .total_path_count = iteration.camera_subpath_count,
    };
  }

  void reset_iteration_diagnostics() {
    iteration_setup_time_ms = 0.0;
    light_path_time_ms = 0.0;
    storage_validation_time_ms = 0.0;
    primitive_collection_time_ms = 0.0;
    spatial_index_time_ms = 0.0;
    light_splat_time_ms = 0.0;
    camera_evaluation_time_ms = 0.0;
    evaluated_light_path_count.store(0u, std::memory_order_relaxed);
    evaluated_camera_path_count.store(0u, std::memory_order_relaxed);
    for (uint32_t technique_index = 0u; technique_index < SpatialStatisticCount; ++technique_index) {
      spatial_query_counts[technique_index].store(0u, std::memory_order_relaxed);
      spatial_candidate_counts[technique_index].store(0u, std::memory_order_relaxed);
      spatial_eligible_candidate_counts[technique_index].store(0u, std::memory_order_relaxed);
      spatial_intersection_counts[technique_index].store(0u, std::memory_order_relaxed);
      spatial_contribution_counts[technique_index].store(0u, std::memory_order_relaxed);
    }
    for (std::atomic<uint64_t>& camera_stage_time : camera_stage_times) {
      camera_stage_time.store(0u, std::memory_order_relaxed);
    }
    for (Integrator::Status::DebugInfo& info : debug_info) {
      info.value = 0.0f;
    }
    debug_info[DebugLightBudget].value = static_cast<float>(options.memory_budget_mb);
  }

  void refresh_live_diagnostics() {
    debug_info[DebugLightPathTime].value = static_cast<float>(light_path_time_ms);
    debug_info[DebugCameraEvaluationTime].value = static_cast<float>(camera_evaluation_time_ms);
    if ((state->load() != Integrator::State::Stopped) && (stage == Stage::Light)) {
      debug_info[DebugLightPathTime].value = static_cast<float>(stage_time.measure_ms());
    } else if (state->load() != Integrator::State::Stopped) {
      debug_info[DebugCameraEvaluationTime].value = static_cast<float>(stage_time.measure_ms());
    }
    debug_info[DebugIterationSetupTime].value = static_cast<float>(iteration_setup_time_ms);
    debug_info[DebugStorageValidationTime].value = static_cast<float>(storage_validation_time_ms);
    debug_info[DebugPrimitiveCollectionTime].value = static_cast<float>(primitive_collection_time_ms);
    debug_info[DebugSpatialIndexTime].value = static_cast<float>(spatial_index_time_ms);
    debug_info[DebugLightSplatTime].value = static_cast<float>(light_splat_time_ms);
    const uint64_t evaluated_camera_paths = evaluated_camera_path_count.load(std::memory_order_relaxed);
    debug_info[DebugCameraProgress].value =
      iteration.camera_subpath_count > 0u ? static_cast<float>(100.0 * static_cast<double>(evaluated_camera_paths) / static_cast<double>(iteration.camera_subpath_count)) : 0.0f;

    constexpr double one_million = 1.0e6;
    for (uint32_t stage_index = 0u; stage_index < CameraTimingStageCount; ++stage_index) {
      const uint64_t stage_time = camera_stage_times[stage_index].load(std::memory_order_relaxed);
      debug_info[DebugCameraSubpathTime + stage_index].value = static_cast<float>(static_cast<double>(stage_time) * static_cast<double>(camera_timing_sample_rate) / one_million);
    }
    for (uint32_t technique_index = 0u; technique_index < SpatialStatisticCount; ++technique_index) {
      const uint64_t queries = spatial_query_counts[technique_index].load(std::memory_order_relaxed);
      const uint64_t candidates = spatial_candidate_counts[technique_index].load(std::memory_order_relaxed);
      const uint64_t eligible_candidates = spatial_eligible_candidate_counts[technique_index].load(std::memory_order_relaxed);
      const uint64_t intersections = spatial_intersection_counts[technique_index].load(std::memory_order_relaxed);
      const uint64_t contributions = spatial_contribution_counts[technique_index].load(std::memory_order_relaxed);
      const uint32_t debug_base = DebugBP2DQueries + technique_index * 6u;
      debug_info[debug_base].value = static_cast<float>(static_cast<double>(queries) / one_million);
      debug_info[debug_base + 1u].value = static_cast<float>(static_cast<double>(candidates) / one_million);
      debug_info[debug_base + 2u].value = static_cast<float>(static_cast<double>(eligible_candidates) / one_million);
      debug_info[debug_base + 3u].value = static_cast<float>(static_cast<double>(intersections) / one_million);
      debug_info[debug_base + 4u].value = static_cast<float>(static_cast<double>(contributions) / one_million);
      debug_info[debug_base + 5u].value = queries > 0u ? static_cast<float>(static_cast<double>(candidates) / static_cast<double>(queries)) : 0.0f;
    }
  }

  void report_live_diagnostics() {
    refresh_live_diagnostics();
    constexpr double report_interval_ms = 5000.0;
    if (live_report_time.measure_ms() < report_interval_ms) {
      return;
    }
    live_report_time.reset();

    if (stage == Stage::Light) {
      log::info("UPBP iteration %u live: building light paths, %.2f s elapsed", status.current_iteration + 1u, stage_time.measure());
      return;
    }

    const uint64_t evaluated_camera_paths = evaluated_camera_path_count.load(std::memory_order_relaxed);
    const double camera_progress =
      iteration.camera_subpath_count > 0u ? 100.0 * static_cast<double>(evaluated_camera_paths) / static_cast<double>(iteration.camera_subpath_count) : 0.0;
    log::info("UPBP iteration %u live: camera %.2f s, %.1f%% paths", status.current_iteration + 1u, stage_time.measure(), camera_progress);
    report_spatial_statistics("BP2D", SpatialStatisticBP2D);
    report_spatial_statistics("PB2D", SpatialStatisticPB2D);
    report_spatial_statistics("BB1D", SpatialStatisticBB1D);
    log::info(
      "UPBP estimated camera CPU totals (ms): subpath %.1f, recursive MIS %.1f, BPT/direct hit %.1f, surface %.1f, PP3D %.1f, BP2D %.1f, beam preparation %.1f, PB2D %.1f, BB1D "
      "%.1f, film "
      "%.1f",
      static_cast<double>(debug_info[DebugCameraSubpathTime].value), static_cast<double>(debug_info[DebugCameraRecursiveMISTime].value),
      static_cast<double>(debug_info[DebugCameraBPTTime].value), static_cast<double>(debug_info[DebugCameraSurfaceMergeTime].value),
      static_cast<double>(debug_info[DebugCameraPP3DTime].value), static_cast<double>(debug_info[DebugCameraBP2DTime].value),
      static_cast<double>(debug_info[DebugCameraBeamPreparationTime].value), static_cast<double>(debug_info[DebugCameraPB2DTime].value),
      static_cast<double>(debug_info[DebugCameraBB1DTime].value), static_cast<double>(debug_info[DebugCameraFilmSubmissionTime].value));
  }

  void report_spatial_statistics(const char* name, const SpatialStatisticIndex technique) const {
    const uint64_t queries = spatial_query_counts[technique].load(std::memory_order_relaxed);
    const uint64_t candidates = spatial_candidate_counts[technique].load(std::memory_order_relaxed);
    const uint64_t eligible_candidates = spatial_eligible_candidate_counts[technique].load(std::memory_order_relaxed);
    const uint64_t intersections = spatial_intersection_counts[technique].load(std::memory_order_relaxed);
    const uint64_t contributions = spatial_contribution_counts[technique].load(std::memory_order_relaxed);
    const double candidates_per_query = queries > 0u ? static_cast<double>(candidates) / static_cast<double>(queries) : 0.0;
    log::info("UPBP %s: queries %.3f M, candidates %.3f M, eligible %.3f M, intersections %.3f M, contributions %.3f M, %.2f candidates/query", name,
      static_cast<double>(queries) / 1.0e6, static_cast<double>(candidates) / 1.0e6, static_cast<double>(eligible_candidates) / 1.0e6, static_cast<double>(intersections) / 1.0e6,
      static_cast<double>(contributions) / 1.0e6, candidates_per_query);
  }

  void publish_camera_statistics(CameraEvaluationStatistics& statistics) {
    const uint64_t camera_path_delta = statistics.camera_paths - statistics.published_camera_paths;
    if (camera_path_delta > 0u) {
      evaluated_camera_path_count.fetch_add(camera_path_delta, std::memory_order_relaxed);
      statistics.published_camera_paths = statistics.camera_paths;
    }
    for (uint32_t technique_index = 0u; technique_index < SpatialStatisticCount; ++technique_index) {
      SpatialTechniqueStatistics& spatial = statistics.spatial[technique_index];
      const uint64_t query_delta = spatial.queries - spatial.published_queries;
      const uint64_t candidate_delta = spatial.candidates - spatial.published_candidates;
      const uint64_t eligible_candidate_delta = spatial.eligible_candidates - spatial.published_eligible_candidates;
      const uint64_t intersection_delta = spatial.intersections - spatial.published_intersections;
      const uint64_t contribution_delta = spatial.contributions - spatial.published_contributions;
      if (query_delta > 0u) {
        spatial_query_counts[technique_index].fetch_add(query_delta, std::memory_order_relaxed);
        spatial.published_queries = spatial.queries;
      }
      if (candidate_delta > 0u) {
        spatial_candidate_counts[technique_index].fetch_add(candidate_delta, std::memory_order_relaxed);
        spatial.published_candidates = spatial.candidates;
      }
      if (eligible_candidate_delta > 0u) {
        spatial_eligible_candidate_counts[technique_index].fetch_add(eligible_candidate_delta, std::memory_order_relaxed);
        spatial.published_eligible_candidates = spatial.eligible_candidates;
      }
      if (intersection_delta > 0u) {
        spatial_intersection_counts[technique_index].fetch_add(intersection_delta, std::memory_order_relaxed);
        spatial.published_intersections = spatial.intersections;
      }
      if (contribution_delta > 0u) {
        spatial_contribution_counts[technique_index].fetch_add(contribution_delta, std::memory_order_relaxed);
        spatial.published_contributions = spatial.contributions;
      }
    }
    for (uint32_t stage_index = 0u; stage_index < CameraTimingStageCount; ++stage_index) {
      const uint64_t stage_time_delta = statistics.stage_times[stage_index] - statistics.published_stage_times[stage_index];
      if (stage_time_delta > 0u) {
        camera_stage_times[stage_index].fetch_add(stage_time_delta, std::memory_order_relaxed);
        statistics.published_stage_times[stage_index] = statistics.stage_times[stage_index];
      }
    }
  }

  void complete_camera_path(CameraEvaluationStatistics& statistics) {
    ++statistics.camera_paths;
    constexpr uint64_t publish_interval = 16u;
    if ((statistics.camera_paths - statistics.published_camera_paths) >= publish_interval) {
      publish_camera_statistics(statistics);
    }
  }

  void release_light_storage() {
    light_paths = {};
    light_weights = {};
    light_splats = {};
    surface_points = {};
    medium_points = {};
    light_vertex_offsets = {};
    prepared_light_vertex_throughputs = {};
    prepared_light_vertex_validity = {};
    light_beams = {};
    prepared_bp2d_beams = {};
    prepared_bp2d_validity = {};
    selected_bb1d_beams = {};
    prepared_bb1d_beams = {};
    prepared_bb1d_validity = {};
    prepared_bb1d_tracking_events = {};
    prepared_mediums = {};
    surface_index = {};
    pp3d_index = {};
    pb2d_index = {};
    bp2d_index = {};
    bb1d_index = {};
    for (Integrator::Status::DebugInfo& info : debug_info) {
      info.value = 0.0f;
    }
    debug_info[DebugLightBudget].value = static_cast<float>(options.memory_budget_mb);
  }

  void release_failed_storage() {
    if (failed.load()) {
      release_light_storage();
    }
  }

  void select_light_subpath_count(const uint64_t camera_subpath_count, uint64_t& result) {
    memory_target_bytes = static_cast<uint64_t>(options.memory_budget_mb) * 1024ull * 1024ull;
    result = camera_subpath_count;
    if (options.maximum_light_path_count > 0u) {
      result = min(result, static_cast<uint64_t>(options.maximum_light_path_count));
    }
  }

  uint64_t fixed_light_storage_bytes() const {
    return static_cast<uint64_t>(light_paths.capacity()) * sizeof(UPBPLightSubpathResult) + static_cast<uint64_t>(light_weights.capacity()) * sizeof(UPBPRecursivePathWeights) +
           static_cast<uint64_t>(light_splats.capacity()) * sizeof(std::vector<UPBPLightSplat>);
  }

  uint64_t current_light_storage_bytes() const {
    uint64_t result = fixed_light_storage_bytes();
    for (uint32_t path_index = 0u; path_index < light_paths.size(); ++path_index) {
      result += upbp_path_storage_bytes(light_paths[path_index].subpath.path);
      result += upbp_recursive_storage_bytes(light_weights[path_index]);
      result += static_cast<uint64_t>(light_splats[path_index].capacity()) * sizeof(UPBPLightSplat);
    }
    result += static_cast<uint64_t>(surface_points.capacity()) * sizeof(UPBPPointReference);
    result += static_cast<uint64_t>(medium_points.capacity()) * sizeof(UPBPPointReference);
    result += static_cast<uint64_t>(light_vertex_offsets.capacity()) * sizeof(uint64_t);
    result += static_cast<uint64_t>(prepared_light_vertex_throughputs.capacity()) * sizeof(SpectralResponse);
    result += static_cast<uint64_t>(prepared_light_vertex_validity.capacity()) * sizeof(uint8_t);
    result += static_cast<uint64_t>(light_beams.capacity() + selected_bb1d_beams.capacity()) * sizeof(UPBPBeamReference);
    result += static_cast<uint64_t>(prepared_bp2d_beams.capacity()) * sizeof(UPBPPreparedBeam);
    result += static_cast<uint64_t>(prepared_bp2d_validity.capacity()) * sizeof(uint8_t);
    result += static_cast<uint64_t>(prepared_bb1d_beams.capacity()) * sizeof(UPBPPreparedBeam);
    result += static_cast<uint64_t>(prepared_bb1d_validity.capacity()) * sizeof(uint8_t);
    result += static_cast<uint64_t>(prepared_bb1d_tracking_events.capacity()) * sizeof(UPBPMediumTrackingEventRecord);
    result += static_cast<uint64_t>(prepared_mediums.capacity()) * sizeof(UPBPPreparedMedium);
    for (const UPBPPreparedMedium& prepared_medium : prepared_mediums) {
      result += prepared_medium.storage_bytes();
    }
    result += surface_index.storage_bytes() + pp3d_index.storage_bytes() + pb2d_index.storage_bytes() + bp2d_index.storage_bytes() + bb1d_index.storage_bytes();
    return result;
  }

  void start(const Options& integrator_options) {
    wait_for_tasks();
    status = {};
    status.debug_info = debug_info.data();
    status.debug_info_count = static_cast<uint32_t>(debug_info.size());
    failure_claimed = false;
    failed = false;
    failure_reason.clear();
    options.load(integrator_options);
    std::string reason;
    if (upbp_options_valid(options, reason) == false) {
      fail(reason);
      *state = Integrator::State::Stopped;
      return;
    }
    release_light_storage();
    rt.film().clear(Film::ClearEverything);
    start_iteration(0u);
  }

  void start_iteration(const uint32_t iteration_index) {
    wait_for_tasks();
    iteration_time = {};
    reset_iteration_diagnostics();
    status.current_iteration = iteration_index;
    const uint64_t camera_subpath_count = rt.film().current_pixel_count();
    uint64_t light_subpath_count = 0u;
    select_light_subpath_count(camera_subpath_count, light_subpath_count);
    iteration = upbp_iteration_parameters(options, rt.scene(), rt.film(), iteration_index, light_subpath_count);
    prepared_bb1d = iteration.mis.enabled(UPBPTechnique::BB1D)
                      ? upbp_prepare_bb1d(options.kernel, iteration.bb1d_radius, iteration.bb1d_light_subpath_count, options.beam_selection_probability)
                      : UPBPPreparedBB1D{};
    auto radius_valid = [this](const UPBPTechnique technique, const double radius) {
      return (iteration.mis.enabled(technique) == false) || ((radius > 0.0) && std::isfinite(radius));
    };
    if ((iteration.camera_subpath_count == 0u) || (iteration.light_subpath_count == 0u) || (iteration.mis.enabled_techniques == 0u) ||
        (radius_valid(UPBPTechnique::Surface, iteration.surface_radius) == false) || (radius_valid(UPBPTechnique::PP3D, iteration.pp3d_radius) == false) ||
        (radius_valid(UPBPTechnique::PB2D, iteration.pb2d_radius) == false) || (radius_valid(UPBPTechnique::BP2D, iteration.bp2d_radius) == false) ||
        (radius_valid(UPBPTechnique::BB1D, iteration.bb1d_radius) == false) || (iteration.mis.enabled(UPBPTechnique::BB1D) && (prepared_bb1d.valid == false))) {
      fail("UPBP iteration parameters are invalid for the current film and scene");
      *state = Integrator::State::Stopped;
      return;
    }
    if (status.current_iteration == 0u) {
      log::info("UPBP selected %llu light paths for %llu camera paths; advisory memory target %u MiB; %llu assigned to BB1D",
        static_cast<unsigned long long>(iteration.light_subpath_count), static_cast<unsigned long long>(iteration.camera_subpath_count), options.memory_budget_mb,
        static_cast<unsigned long long>(iteration.bb1d_light_subpath_count));
    }
    constexpr uint32_t density_technique_mask = static_cast<uint32_t>(UPBPTechnique::Surface) | static_cast<uint32_t>(UPBPTechnique::PP3D) |
                                                static_cast<uint32_t>(UPBPTechnique::PB2D) | static_cast<uint32_t>(UPBPTechnique::BP2D) |
                                                static_cast<uint32_t>(UPBPTechnique::BB1D);
    const uint32_t enabled_techniques = iteration.mis.enabled_techniques;
    if ((rt.scene().multiple_importance_sampling() == false) && ((enabled_techniques & (enabled_techniques - 1u)) != 0u)) {
      fail("UPBP requires multiple importance sampling when more than one technique is enabled");
      *state = Integrator::State::Stopped;
      return;
    }
    if (iteration.mis.enabled(UPBPTechnique::BPT) && ((iteration.mis.enabled_techniques & density_technique_mask) != 0u) && rt.scene().multiple_importance_sampling() &&
        (upbp_all_bpt_strategies_enabled(rt.scene()) == false)) {
      fail("UPBP cross-technique MIS requires all BPT endpoint strategies when BPT and density techniques are enabled together");
      *state = Integrator::State::Stopped;
      return;
    }
    try {
      prepared_mediums.resize(rt.scene().mediums.count);
      for (uint32_t medium_index = 0u; medium_index < rt.scene().mediums.count; ++medium_index) {
        prepared_mediums[medium_index] = upbp_prepare_medium(rt.scene().mediums[medium_index], iteration.spect);
      }
      light_paths.clear();
      light_paths.resize(static_cast<size_t>(iteration.light_subpath_count));
      light_weights.clear();
      light_weights.resize(static_cast<size_t>(iteration.light_subpath_count));
      light_splats.clear();
      light_splats.resize(static_cast<size_t>(iteration.light_subpath_count));
    } catch (const std::bad_alloc&) {
      fail("UPBP failed to allocate fixed light-path storage");
      *state = Integrator::State::Stopped;
      return;
    } catch (const std::length_error&) {
      fail("UPBP fixed light-path storage exceeds the platform container limit");
      *state = Integrator::State::Stopped;
      return;
    }

    surface_points.clear();
    medium_points.clear();
    light_vertex_offsets.clear();
    prepared_light_vertex_throughputs.clear();
    prepared_light_vertex_validity.clear();
    light_beams.clear();
    prepared_bp2d_beams.clear();
    prepared_bp2d_validity.clear();
    selected_bb1d_beams.clear();
    prepared_bb1d_beams.clear();
    prepared_bb1d_validity.clear();
    prepared_bb1d_tracking_events.clear();
    surface_index.clear();
    pp3d_index.clear();
    pb2d_index.clear();
    bp2d_index.clear();
    bb1d_index.clear();
    debug_info[DebugSurfaceRadius].value = static_cast<float>(iteration.surface_radius);
    debug_info[DebugPP3DRadius].value = static_cast<float>(iteration.pp3d_radius);
    debug_info[DebugPB2DRadius].value = static_cast<float>(iteration.pb2d_radius);
    debug_info[DebugBP2DRadius].value = static_cast<float>(iteration.bp2d_radius);
    debug_info[DebugBB1DRadius].value = static_cast<float>(iteration.bb1d_radius);
    debug_info[DebugLightPaths].value = static_cast<float>(iteration.light_subpath_count);
    debug_info[DebugBB1DLightPaths].value = static_cast<float>(iteration.bb1d_light_subpath_count);
    iteration_setup_time_ms = iteration_time.measure_ms();
    stage = Stage::Light;
    stage_time.reset();
    live_report_time.reset();
    task_handle = rt.scheduler().schedule(iteration.light_subpath_count, &light_task);
  }

  void build_light_paths(const uint32_t begin, const uint32_t end, const uint32_t thread_id) {
    try {
      build_light_paths_impl(begin, end, thread_id);
    } catch (const std::bad_alloc&) {
      fail("UPBP failed to allocate light-path working storage");
    } catch (const std::length_error&) {
      fail("UPBP light-path working storage exceeds the platform container limit");
    }
  }

  void build_light_paths_impl(const uint32_t begin, const uint32_t end, const uint32_t) {
    const Scene& scene = rt.scene();
    const uint32_t maximum_vertices = scene.options.max_path_length + 1u;
    uint64_t unpublished_path_count = 0u;
    constexpr uint64_t publish_interval = 16u;
    for (uint32_t path_index = begin; running() && (path_index < end); ++path_index) {
      UPBPLightSubpathResult& light_path = light_paths[path_index];
      if (upbp_build_light_subpath(rt, scene, iteration.spect, scene.options.random_seed, status.current_iteration, path_index, maximum_vertices, options.maximum_boundary_count,
            options.maximum_null_events_per_interval, light_path) == false) {
        fail("UPBP light subpath failed at path " + std::to_string(path_index) + ", failure " + std::to_string(static_cast<uint32_t>(light_path.subpath.failure)) +
             ", segment failure " + std::to_string(static_cast<uint32_t>(light_path.subpath.segment_failure)) + ", vertices " +
             std::to_string(light_path.subpath.path.vertices.size()) + ", emitter " + std::to_string(light_path.emitter_sample.emitter_index) + ", triangle " +
             std::to_string(light_path.emitter_sample.triangle_index) + ", emitter PDFs (" + std::to_string(light_path.emitter_sample.pdf_area) + ", " +
             std::to_string(light_path.emitter_sample.pdf_dir) + ", " + std::to_string(light_path.emitter_sample.pdf_sample) + "), emitter cosine " +
             std::to_string(dot(light_path.emitter_sample.direction, light_path.emitter_sample.normal)) + ", ray origin (" + std::to_string(light_path.subpath.terminal_ray.o.x) +
             ", " + std::to_string(light_path.subpath.terminal_ray.o.y) + ", " + std::to_string(light_path.subpath.terminal_ray.o.z) + "), direction (" +
             std::to_string(light_path.subpath.terminal_ray.d.x) + ", " + std::to_string(light_path.subpath.terminal_ray.d.y) + ", " +
             std::to_string(light_path.subpath.terminal_ray.d.z) + ")");
        return;
      }
      if (light_path.subpath.path.vertices.size() > 1u) {
        if (upbp_compute_recursive_path_weights(scene, light_path.subpath.path, iteration.mis, iteration.light_subpath_count, iteration.bpt_sample_count,
              light_weights[path_index]) == false) {
          fail("UPBP recursive light-path MIS failed at path " + std::to_string(path_index) + ", vertex " + std::to_string(light_weights[path_index].failure_vertex_index) +
               ", failure " + std::to_string(static_cast<uint32_t>(light_weights[path_index].failure)));
          return;
        }
        if (iteration.mis.enabled(UPBPTechnique::BPT) && scene.strategy_enabled(Scene::Strategy::ConnectToCamera)) {
          UPBPLightSplatEvaluationFailure splat_failure = UPBPLightSplatEvaluationFailure::None;
          UPBPLightToCameraFailure connection_failure = UPBPLightToCameraFailure::None;
          UPBPSceneSegmentFailure segment_failure = UPBPSceneSegmentFailure::None;
          UPBPPathProbabilityFailure probability_failure = UPBPPathProbabilityFailure::None;
          uint32_t failure_vertex_count = 0u;
          if (upbp_evaluate_light_splats(rt, scene, iteration.spect, light_path.subpath.path, scene.options.random_seed, status.current_iteration, path_index,
                options.maximum_boundary_count, options.maximum_null_events_per_interval, light_weights[path_index], iteration.mis, iteration.camera_subpath_count,
                iteration.light_subpath_count, light_splats[path_index], splat_failure, connection_failure, segment_failure, probability_failure, failure_vertex_count) == false) {
            fail("UPBP light tracing failed at path " + std::to_string(path_index) + ", vertex count " + std::to_string(failure_vertex_count) + ", failure " +
                 std::to_string(static_cast<uint32_t>(splat_failure)) + ", connection failure " + std::to_string(static_cast<uint32_t>(connection_failure)) + ", segment failure " +
                 std::to_string(static_cast<uint32_t>(segment_failure)) + ", probability failure " + std::to_string(static_cast<uint32_t>(probability_failure)));
            return;
          }
        }
      }
      ++unpublished_path_count;
      if (unpublished_path_count >= publish_interval) {
        evaluated_light_path_count.fetch_add(unpublished_path_count, std::memory_order_relaxed);
        unpublished_path_count = 0u;
      }
    }
    if (unpublished_path_count > 0u) {
      evaluated_light_path_count.fetch_add(unpublished_path_count, std::memory_order_relaxed);
    }
  }

  bool preflight_light_primitives(uint64_t& surface_point_count, uint64_t& medium_point_count, uint64_t& light_vertex_count, uint64_t& light_beam_count,
    uint64_t& selected_beam_count) {
    surface_point_count = 0u;
    medium_point_count = 0u;
    light_vertex_count = 0u;
    light_beam_count = 0u;
    selected_beam_count = 0u;
    const bool collect_surface_points = iteration.mis.enabled(UPBPTechnique::Surface);
    const bool collect_medium_points = iteration.mis.enabled(UPBPTechnique::PP3D) || iteration.mis.enabled(UPBPTechnique::PB2D);
    const bool collect_bp2d_beams = iteration.mis.enabled(UPBPTechnique::BP2D);
    const bool collect_bb1d_beams = iteration.mis.enabled(UPBPTechnique::BB1D);

    auto increment = [this](uint64_t& value, const char* label) {
      if (value == std::numeric_limits<uint64_t>::max()) {
        fail(std::string{"UPBP "} + label + " count overflowed 64-bit size arithmetic");
        return false;
      }
      ++value;
      return true;
    };

    for (uint32_t path_index = 0u; path_index < light_paths.size(); ++path_index) {
      const UPBPPathRecord& path = light_paths[path_index].subpath.path;
      if (upbp_checked_add(light_vertex_count, static_cast<uint64_t>(path.vertices.size()), light_vertex_count) == false) {
        fail("UPBP light-vertex count overflowed 64-bit size arithmetic");
        return false;
      }
      for (uint32_t vertex_index = 1u; vertex_index < path.vertices.size(); ++vertex_index) {
        const UPBPPathVertexRecord& vertex = path.vertices[vertex_index];
        if (collect_surface_points && (vertex.cls == UPBPVertexClass::Surface) && (vertex.delta == false) && vertex.density_connectible) {
          if (increment(surface_point_count, "surface-point") == false) {
            return false;
          }
        } else if (collect_medium_points && (vertex.cls == UPBPVertexClass::Medium) && (vertex.delta == false) && vertex.density_connectible &&
                   (vertex.medium.index != kInvalidIndex)) {
          if (increment(medium_point_count, "medium-point") == false) {
            return false;
          }
        }
      }

      if ((collect_bp2d_beams == false) && ((collect_bb1d_beams == false) || (path_index >= iteration.bb1d_light_subpath_count))) {
        continue;
      }
      auto count_segment = [this, path_index, collect_bp2d_beams, collect_bb1d_beams, &light_beam_count, &selected_beam_count, &increment](
                             const UPBPTransportSegmentRecord& segment, const uint32_t transport_segment_index) {
        for (uint32_t transport_interval_index = 0u; transport_interval_index < segment.intervals.size(); ++transport_interval_index) {
          const UPBPSegmentRecord& interval = segment.intervals[transport_interval_index];
          if (interval.medium_index == kInvalidIndex) {
            continue;
          }
          const float3 delta = interval.end_position - interval.start_position;
          const float distance_squared = dot(delta, delta);
          if ((interval.distance <= 0.0f) || (std::isfinite(interval.distance) == false) || (distance_squared < 0.0f) || (std::isfinite(distance_squared) == false)) {
            fail("UPBP light-beam preflight encountered an invalid medium interval");
            return false;
          }
          if (distance_squared == 0.0f) {
            continue;
          }
          if (collect_bp2d_beams && (increment(light_beam_count, "light-beam") == false)) {
            return false;
          }
          if (collect_bb1d_beams && (path_index < iteration.bb1d_light_subpath_count)) {
            Sampler selection_sampler{
              upbp_sampler_seed(rt.scene().options.random_seed, status.current_iteration, path_index, transport_segment_index, transport_interval_index, UPBPRandomDomain::BB1D)};
            if ((selection_sampler.next() < options.beam_selection_probability) && (increment(selected_beam_count, "selected BB1D beam") == false)) {
              return false;
            }
          }
        }
        return true;
      };
      for (uint32_t segment_index = 0u; segment_index < path.segments.size(); ++segment_index) {
        if (count_segment(path.segments[segment_index], segment_index) == false) {
          return false;
        }
      }
      if (path.has_terminal_segment && (count_segment(path.terminal_segment, static_cast<uint32_t>(path.segments.size())) == false)) {
        return false;
      }
    }

    const uint64_t maximum_index_count = std::numeric_limits<uint32_t>::max();
    if ((surface_point_count > maximum_index_count) || (medium_point_count > maximum_index_count) || (light_beam_count > maximum_index_count) ||
        (selected_beam_count > maximum_index_count)) {
      fail("UPBP primitive count exceeds the 32-bit spatial-index limit");
      return false;
    }

    const bool prepare_light_medium_vertices = iteration.mis.enabled(UPBPTechnique::PB2D);
    if (prepare_light_medium_vertices && (light_vertex_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))) {
      fail("UPBP light-vertex preparation exceeds the platform container limit");
      return false;
    }
    return true;
  }

  bool collect_light_vertices() {
    try {
      uint64_t surface_point_count = 0u;
      uint64_t medium_point_count = 0u;
      uint64_t light_vertex_count = 0u;
      uint64_t light_beam_count = 0u;
      uint64_t selected_beam_count = 0u;
      if (preflight_light_primitives(surface_point_count, medium_point_count, light_vertex_count, light_beam_count, selected_beam_count) == false) {
        return false;
      }
      surface_points.reserve(static_cast<size_t>(surface_point_count));
      medium_points.reserve(static_cast<size_t>(medium_point_count));
      const bool prepare_light_medium_vertices = iteration.mis.enabled(UPBPTechnique::PB2D);
      if (prepare_light_medium_vertices) {
        light_vertex_offsets.resize(light_paths.size() + 1u);
      }
      if (prepare_light_medium_vertices) {
        prepared_light_vertex_throughputs.resize(static_cast<size_t>(light_vertex_count));
        prepared_light_vertex_validity.assign(static_cast<size_t>(light_vertex_count), 0u);
      }
      light_beams.reserve(static_cast<size_t>(light_beam_count));
      prepared_bp2d_beams.reserve(static_cast<size_t>(light_beam_count));
      prepared_bp2d_validity.reserve(static_cast<size_t>(light_beam_count));
      selected_bb1d_beams.reserve(static_cast<size_t>(selected_beam_count));
      prepared_bb1d_beams.reserve(static_cast<size_t>(selected_beam_count));
      prepared_bb1d_validity.reserve(static_cast<size_t>(selected_beam_count));
      const bool collect_surface_points = iteration.mis.enabled(UPBPTechnique::Surface);
      const bool collect_medium_points = iteration.mis.enabled(UPBPTechnique::PP3D) || iteration.mis.enabled(UPBPTechnique::PB2D);
      const bool collect_bp2d_beams = iteration.mis.enabled(UPBPTechnique::BP2D);
      const bool collect_bb1d_beams = iteration.mis.enabled(UPBPTechnique::BB1D);
      std::vector<UPBPBeamReference> path_beams = {};
      uint64_t light_vertex_offset = 0u;
      for (uint32_t path_index = 0u; path_index < light_paths.size(); ++path_index) {
        const UPBPPathRecord& path = light_paths[path_index].subpath.path;
        if (prepare_light_medium_vertices) {
          light_vertex_offsets[path_index] = light_vertex_offset;
        }
        for (uint32_t vertex_index = 1u; vertex_index < path.vertices.size(); ++vertex_index) {
          const UPBPPathVertexRecord& vertex = path.vertices[vertex_index];
          const uint64_t prepared_index = light_vertex_offset + vertex_index;
          if (prepare_light_medium_vertices && (vertex.cls == UPBPVertexClass::Medium)) {
            SpectralResponse pre_collision_throughput = {};
            const bool valid = upbp_medium_pre_collision_throughput(rt.scene(), iteration.spect, vertex, pre_collision_throughput);
            prepared_light_vertex_throughputs[static_cast<size_t>(prepared_index)] = pre_collision_throughput;
            prepared_light_vertex_validity[static_cast<size_t>(prepared_index)] = static_cast<uint8_t>(valid);
          }
          if (collect_surface_points && (vertex.cls == UPBPVertexClass::Surface) && (vertex.delta == false) && vertex.density_connectible) {
            surface_points.emplace_back(UPBPPointReference{vertex.position, path_index, vertex_index});
          } else if (collect_medium_points && (vertex.cls == UPBPVertexClass::Medium) && (vertex.delta == false) && vertex.density_connectible &&
                     (vertex.medium.index != kInvalidIndex)) {
            medium_points.emplace_back(UPBPPointReference{vertex.position, path_index, vertex_index});
          }
        }
        light_vertex_offset += path.vertices.size();

        if ((collect_bp2d_beams == false) && ((collect_bb1d_beams == false) || (path_index >= iteration.bb1d_light_subpath_count))) {
          continue;
        }
        if (upbp_collect_medium_beams(path, path_index, path_beams) == false) {
          fail("UPBP light-beam collection failed at path " + std::to_string(path_index));
          return false;
        }
        for (const UPBPBeamReference& beam : path_beams) {
          UPBPPreparedBeam prepared_beam = {};
          bool beam_prepared = false;
          if (collect_bp2d_beams) {
            beam_prepared = upbp_prepare_beam(path, light_weights[path_index], beam, iteration.mis, prepared_beam);
            light_beams.emplace_back(beam);
            prepared_bp2d_validity.emplace_back(static_cast<uint8_t>(beam_prepared));
            prepared_bp2d_beams.emplace_back(prepared_beam);
          }
          if (collect_bb1d_beams && (path_index < iteration.bb1d_light_subpath_count)) {
            Sampler selection_sampler{upbp_sampler_seed(rt.scene().options.random_seed, status.current_iteration, path_index, beam.transport_segment_index,
              beam.transport_interval_index, UPBPRandomDomain::BB1D)};
            if (selection_sampler.next() < options.beam_selection_probability) {
              if (collect_bp2d_beams == false) {
                beam_prepared = upbp_prepare_beam(path, light_weights[path_index], beam, iteration.mis, prepared_beam);
              }
              if (spectral_query_compatible(beam.throughput_at_origin.as_query(), iteration.spect) == false) {
                fail("UPBP light-beam spectrum is inconsistent at path " + std::to_string(path_index));
                return false;
              }
              selected_bb1d_beams.emplace_back(beam);
              prepared_bb1d_validity.emplace_back(static_cast<uint8_t>(beam_prepared));
              prepared_bb1d_beams.emplace_back(prepared_beam);
            }
          }
        }
      }
      if (prepare_light_medium_vertices) {
        light_vertex_offsets.back() = light_vertex_offset;
      }
    } catch (const std::bad_alloc&) {
      fail("UPBP failed to allocate deterministic point and beam storage");
      return false;
    } catch (const std::length_error&) {
      fail("UPBP deterministic point and beam storage exceeds the platform container limit");
      return false;
    }
    return true;
  }

  bool build_spatial_indices() {
    uint64_t selected_tracking_event_count = 0u;
    for (const UPBPPreparedBeam& prepared_beam : prepared_bb1d_beams) {
      if ((prepared_beam.tracking_events != nullptr) &&
          (upbp_checked_add(selected_tracking_event_count, static_cast<uint64_t>(prepared_beam.tracking_event_count), selected_tracking_event_count) == false)) {
        fail("UPBP selected BB1D tracking-event count overflowed 64-bit size arithmetic");
        return false;
      }
    }
    if (selected_tracking_event_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
      fail("UPBP selected BB1D tracking-event storage exceeds the platform container limit");
      return false;
    }

    try {
      if (iteration.mis.enabled(UPBPTechnique::Surface) && (surface_points.empty() == false) &&
          (surface_index.build(surface_points.data(), static_cast<uint32_t>(surface_points.size()), static_cast<float>(iteration.surface_radius)) == false)) {
        fail("UPBP surface-point index construction failed");
        return false;
      }
      if (iteration.mis.enabled(UPBPTechnique::PP3D) && (medium_points.empty() == false) &&
          (pp3d_index.build(medium_points.data(), static_cast<uint32_t>(medium_points.size()), static_cast<float>(iteration.pp3d_radius)) == false)) {
        fail("UPBP PP3D point index construction failed");
        return false;
      }
      if (iteration.mis.enabled(UPBPTechnique::PB2D) && (medium_points.empty() == false)) {
        if (pb2d_index.build(medium_points.data(), static_cast<uint32_t>(medium_points.size())) == false) {
          fail("UPBP PB2D point index construction failed");
          return false;
        }
      }
      if (iteration.mis.enabled(UPBPTechnique::BP2D) && (light_beams.empty() == false) &&
          (bp2d_index.build(light_beams.data(), static_cast<uint32_t>(light_beams.size()), static_cast<float>(iteration.bp2d_radius), rt.scheduler()) == false)) {
        fail("UPBP BP2D beam index construction failed");
        return false;
      }
      if (iteration.mis.enabled(UPBPTechnique::BB1D) && (selected_bb1d_beams.empty() == false) &&
          (bb1d_index.build(selected_bb1d_beams.data(), static_cast<uint32_t>(selected_bb1d_beams.size()), static_cast<float>(iteration.bb1d_radius), rt.scheduler()) == false)) {
        fail("UPBP BB1D beam index construction failed");
        return false;
      }
      if (iteration.mis.enabled(UPBPTechnique::BB1D) && (selected_bb1d_beams.empty() == false)) {
        prepared_bb1d_tracking_events.reserve(static_cast<size_t>(selected_tracking_event_count));
        for (UPBPPreparedBeam& prepared_beam : prepared_bb1d_beams) {
          if (prepared_beam.tracking_events == nullptr) {
            continue;
          }
          const size_t event_offset = prepared_bb1d_tracking_events.size();
          prepared_bb1d_tracking_events.insert(prepared_bb1d_tracking_events.end(), prepared_beam.tracking_events,
            prepared_beam.tracking_events + prepared_beam.tracking_event_count);
          prepared_beam.tracking_events = prepared_bb1d_tracking_events.data() + event_offset;
        }
      }
    } catch (const std::bad_alloc&) {
      fail("UPBP failed to allocate spatial acceleration structures");
      return false;
    } catch (const std::length_error&) {
      fail("UPBP spatial acceleration storage exceeds the platform container limit");
      return false;
    }
    return true;
  }

  void submit_light_splats() {
    Film& film = rt.film();
    for (const std::vector<UPBPLightSplat>& path_splats : light_splats) {
      for (const UPBPLightSplat& splat : path_splats) {
        const float3 value = splat.value.to_rgb_estimate();
        if (dot(value, value) > kEpsilon) {
          film.submit(value, splat.film_uv);
        }
      }
    }
  }

  bool evaluate_point_technique(const UPBPTechnique technique, const UPBPPointIndex& index, const double radius, const UPBPPathRecord& camera_path,
    const UPBPRecursivePathWeights& camera_weights, const uint32_t camera_path_index, const uint32_t camera_vertex_index, SpectralResponse& value) {
    if ((index.size() == 0u) || (camera_vertex_index >= camera_path.vertices.size())) {
      return true;
    }
    float3 camera_geometric_normal = {};
    if (technique == UPBPTechnique::Surface) {
      const UPBPPathVertexRecord& camera_vertex = camera_path.vertices[camera_vertex_index];
      const Triangle& triangle = rt.scene().triangles[camera_vertex.intersection.triangle_index];
      camera_geometric_normal = scene_triangle_world_geometric_normal(rt.scene(), triangle, camera_vertex.intersection.instance_index);
    }
    bool evaluation_valid = true;
    const bool query_valid = index.query(camera_path.vertices[camera_vertex_index].position, static_cast<float>(radius),
      [this, technique, radius, &camera_path, &camera_weights, camera_path_index, camera_vertex_index, &camera_geometric_normal, &value, &evaluation_valid](
        const UPBPPointReference& point, const float) {
        if (evaluation_valid == false) {
          return;
        }
        if ((point.path_index >= light_paths.size()) || (point.vertex_index >= light_paths[point.path_index].subpath.path.vertices.size()) ||
            (point.vertex_index >= light_weights[point.path_index].arrivals.size())) {
          evaluation_valid = false;
          return;
        }
        Sampler sampler = upbp_pair_sampler(rt.scene().options.random_seed, status.current_iteration, camera_path_index, point.path_index, camera_vertex_index, point.vertex_index,
          UPBPRandomDomain::ScatteringEvaluation);
        UPBPPointMergeContribution contribution = {};
        if (upbp_evaluate_point_merge(rt.scene(), iteration.spect, light_paths[point.path_index].subpath.path, point.vertex_index,
              light_weights[point.path_index].arrivals[point.vertex_index], camera_path, camera_vertex_index, camera_weights.arrivals[camera_vertex_index], iteration.mis,
              technique, options.kernel, radius, iteration.light_subpath_count, iteration.bpt_sample_count, camera_geometric_normal, sampler, contribution) == false) {
          evaluation_valid = false;
          return;
        }
        if (contribution.applicable) {
          value += contribution.contribution;
        }
      });
    return query_valid && evaluation_valid;
  }

  bool evaluate_pb2d(const std::vector<UPBPBeamReference>& camera_beams, const std::vector<UPBPPreparedBeam>& prepared_camera_beams, SpectralResponse& value,
    CameraEvaluationStatistics& statistics) {
    if (pb2d_index.size() == 0u) {
      return true;
    }
    if (prepared_camera_beams.size() != camera_beams.size()) {
      return false;
    }
    for (uint32_t camera_beam_index = 0u; camera_beam_index < camera_beams.size(); ++camera_beam_index) {
      const UPBPBeamReference& camera_beam = camera_beams[camera_beam_index];
      const UPBPPreparedBeam& prepared_camera_beam = prepared_camera_beams[camera_beam_index];
      if (camera_beam.medium_index >= prepared_mediums.size()) {
        return false;
      }
      if (prepared_camera_beam.valid == false) {
        continue;
      }
      const Medium& medium = rt.scene().mediums[camera_beam.medium_index];
      SpatialTechniqueStatistics& spatial = statistics.spatial[SpatialStatisticPB2D];
      ++spatial.queries;
      bool evaluation_valid = true;
      uint64_t query_candidate_count = 0u;
      const bool query_valid = pb2d_index.query_beam(
        camera_beam, static_cast<float>(iteration.pb2d_radius), query_candidate_count,
        [this, &camera_beam, &evaluation_valid, &statistics](const UPBPPointReference& point, const uint32_t) {
          if (evaluation_valid == false) {
            return false;
          }
          if ((point.path_index >= light_paths.size()) || (point.vertex_index >= light_paths[point.path_index].subpath.path.vertices.size()) ||
              (point.vertex_index >= light_weights[point.path_index].arrivals.size())) {
            return true;
          }
          const UPBPPathVertexRecord& light_vertex = light_paths[point.path_index].subpath.path.vertices[point.vertex_index];
          const uint32_t path_length = point.vertex_index + camera_beam.source_vertex_index + 1u;
          if ((light_vertex.cls != UPBPVertexClass::Medium) || (light_vertex.medium.index != camera_beam.medium_index) ||
              (spectral_query_compatible(light_vertex.throughput.as_query(), camera_beam.throughput_at_origin.as_query()) == false) ||
              (path_length < rt.scene().options.min_path_length) || (path_length > rt.scene().options.max_path_length)) {
            return false;
          }
          if (point.path_index + 1u >= light_vertex_offsets.size()) {
            return true;
          }
          const uint64_t prepared_index = light_vertex_offsets[point.path_index] + point.vertex_index;
          if ((prepared_index >= light_vertex_offsets[point.path_index + 1u]) || (prepared_index >= prepared_light_vertex_throughputs.size()) ||
              (prepared_light_vertex_validity[static_cast<size_t>(prepared_index)] == 0u)) {
            return true;
          }
          ++statistics.spatial[SpatialStatisticPB2D].eligible_candidates;
          return true;
        },
        [this, &medium, &prepared_camera_beam, &camera_beam, &value, &evaluation_valid, &statistics](const UPBPPointReference& point,
          const UPBPPointBeamIntersection& intersection) {
          SpatialTechniqueStatistics& spatial = statistics.spatial[SpatialStatisticPB2D];
          ++spatial.intersections;
          if ((point.path_index >= light_paths.size()) || (point.vertex_index >= light_paths[point.path_index].subpath.path.vertices.size()) ||
              (point.vertex_index >= light_weights[point.path_index].arrivals.size()) || (point.path_index + 1u >= light_vertex_offsets.size())) {
            evaluation_valid = false;
            return;
          }
          const UPBPPathVertexRecord& light_vertex = light_paths[point.path_index].subpath.path.vertices[point.vertex_index];
          const uint64_t prepared_index = light_vertex_offsets[point.path_index] + point.vertex_index;
          if ((prepared_index >= light_vertex_offsets[point.path_index + 1u]) || (prepared_index >= prepared_light_vertex_throughputs.size()) ||
              (prepared_light_vertex_validity[static_cast<size_t>(prepared_index)] == 0u)) {
            evaluation_valid = false;
            return;
          }
          UPBPBeamContribution contribution;
          if (upbp_evaluate_prepared_pb2d(light_vertex, light_weights[point.path_index].arrivals[point.vertex_index],
                prepared_light_vertex_throughputs[static_cast<size_t>(prepared_index)], medium, prepared_mediums[camera_beam.medium_index], prepared_camera_beam, camera_beam,
                intersection, iteration.mis, options.kernel, iteration.pb2d_radius, iteration.light_subpath_count, iteration.bpt_sample_count, contribution) == false) {
            evaluation_valid = false;
            return;
          }
          if (contribution.applicable) {
            ++spatial.contributions;
            value += contribution.contribution;
          }
        });
      spatial.candidates += query_candidate_count;
      if ((query_valid == false) || (evaluation_valid == false)) {
        return false;
      }
    }
    return true;
  }

  bool evaluate_bp2d(const UPBPPathRecord& camera_path, const UPBPRecursivePathWeights& camera_weights, const uint32_t camera_vertex_index, SpectralResponse& value,
    CameraEvaluationStatistics& statistics) {
    if (bp2d_index.size() == 0u) {
      return true;
    }
    if ((camera_vertex_index >= camera_path.vertices.size()) || (camera_vertex_index >= camera_weights.arrivals.size()) || (prepared_bp2d_beams.size() != bp2d_index.size()) ||
        (prepared_bp2d_validity.size() != bp2d_index.size())) {
      return false;
    }
    const UPBPPathVertexRecord& camera_vertex = camera_path.vertices[camera_vertex_index];
    if ((camera_vertex.cls != UPBPVertexClass::Medium) || (camera_vertex.medium.index >= prepared_mediums.size())) {
      return false;
    }
    const Medium& medium = rt.scene().mediums[camera_vertex.medium.index];
    const SpectralResponse scattering = upbp_medium_scattering_coefficient(medium, iteration.spect, camera_vertex.position);
    if (scattering.is_zero()) {
      return true;
    }
    SpectralResponse camera_throughput = {};
    if (upbp_remove_medium_collision_weight(camera_vertex.throughput, scattering, std::exp(camera_vertex.log_medium_event_density), camera_throughput) == false) {
      return false;
    }
    const UPBPPointMergeMISInput::Weights prepared_camera_weights = upbp_point_merge_weights(camera_weights.arrivals[camera_vertex_index]);
    SpatialTechniqueStatistics& spatial = statistics.spatial[SpatialStatisticBP2D];
    ++spatial.queries;
    bool evaluation_valid = true;
    uint64_t query_candidate_count = 0u;
    const bool query_valid = bp2d_index.query_point_intersections(
      camera_vertex.position, static_cast<float>(iteration.bp2d_radius), query_candidate_count,
      [this, &camera_vertex, camera_vertex_index, &evaluation_valid, &statistics](const UPBPBeamReference& beam, const uint32_t beam_index) {
        if (evaluation_valid == false) {
          return false;
        }
        const uint32_t path_length = beam.source_vertex_index + 1u + camera_vertex_index;
        const bool eligible = (prepared_bp2d_validity[beam_index] != 0u) && (beam.medium_index == camera_vertex.medium.index) &&
                              spectral_query_compatible(camera_vertex.throughput.as_query(), beam.throughput_at_origin.as_query()) &&
                              (path_length >= rt.scene().options.min_path_length) && (path_length <= rt.scene().options.max_path_length);
        statistics.spatial[SpatialStatisticBP2D].eligible_candidates += static_cast<uint64_t>(eligible);
        return eligible;
      },
      [this, &medium, &camera_vertex, &prepared_camera_weights, &camera_throughput, &scattering, &value, &evaluation_valid, &statistics](const UPBPBeamReference& beam,
        const uint32_t beam_index, const UPBPPointBeamIntersection& intersection) {
        SpatialTechniqueStatistics& spatial = statistics.spatial[SpatialStatisticBP2D];
        ++spatial.intersections;
        UPBPBeamContribution contribution;
        if (upbp_evaluate_prepared_bp2d(medium, prepared_mediums[camera_vertex.medium.index], prepared_bp2d_beams[beam_index], beam, intersection, camera_vertex.intersection.w_i,
              prepared_camera_weights, camera_throughput, scattering, iteration.mis, options.kernel, iteration.bp2d_radius, iteration.light_subpath_count,
              iteration.bpt_sample_count, contribution) == false) {
          evaluation_valid = false;
          return;
        }
        if (contribution.applicable) {
          ++spatial.contributions;
          value += contribution.contribution;
        }
      });
    spatial.candidates += query_candidate_count;
    return query_valid && evaluation_valid;
  }

  bool evaluate_bb1d(const std::vector<UPBPBeamReference>& camera_beams, const std::vector<UPBPPreparedBeam>& prepared_camera_beams, UPBPSpatialQueryState& query_state,
    SpectralResponse& value, CameraEvaluationStatistics& statistics) {
    if (bb1d_index.size() == 0u) {
      return true;
    }
    if ((prepared_bb1d_beams.size() != bb1d_index.size()) || (prepared_bb1d_validity.size() != bb1d_index.size()) || (prepared_camera_beams.size() != camera_beams.size())) {
      return false;
    }
    for (uint32_t camera_beam_index = 0u; camera_beam_index < camera_beams.size(); ++camera_beam_index) {
      const UPBPBeamReference& camera_beam = camera_beams[camera_beam_index];
      const UPBPPreparedBeam& prepared_camera_beam = prepared_camera_beams[camera_beam_index];
      if (camera_beam.medium_index >= prepared_mediums.size()) {
        return false;
      }
      const bool prepared_camera_beam_valid = prepared_camera_beam.valid;
      SpatialTechniqueStatistics& spatial = statistics.spatial[SpatialStatisticBB1D];
      ++spatial.queries;
      bool evaluation_valid = true;
      uint64_t query_candidate_count = 0u;
      const bool query_valid = bb1d_index.query_beam_intersections(
        camera_beam, static_cast<float>(iteration.bb1d_radius), query_state, query_candidate_count,
        [this, &camera_beam, prepared_camera_beam_valid, &evaluation_valid, &statistics](const UPBPBeamReference& light_beam, const uint32_t light_beam_index) {
          if (evaluation_valid == false) {
            return false;
          }
          const uint32_t path_length = light_beam.source_vertex_index + camera_beam.source_vertex_index + 2u;
          const bool eligible = (prepared_bb1d_validity[light_beam_index] != 0u) && prepared_camera_beam_valid && (light_beam.medium_index == camera_beam.medium_index) &&
                                (path_length >= rt.scene().options.min_path_length) && (path_length <= rt.scene().options.max_path_length);
          statistics.spatial[SpatialStatisticBB1D].eligible_candidates += static_cast<uint64_t>(eligible);
          return eligible;
        },
        [this, &prepared_camera_beam, &camera_beam, &value, &evaluation_valid, &statistics](const UPBPBeamReference& light_beam, const uint32_t light_beam_index,
          const UPBPBeamBeamIntersection& intersection) {
          SpatialTechniqueStatistics& spatial = statistics.spatial[SpatialStatisticBB1D];
          ++spatial.intersections;
          UPBPBeamContribution contribution;
          if (upbp_evaluate_bb1d(rt.scene().mediums[camera_beam.medium_index], prepared_mediums[camera_beam.medium_index], prepared_bb1d_beams[light_beam_index], light_beam,
                prepared_camera_beam, camera_beam, intersection, iteration.mis, prepared_bb1d, iteration.bpt_sample_count, contribution) == false) {
            evaluation_valid = false;
            return;
          }
          if (contribution.applicable) {
            ++spatial.contributions;
            value += contribution.contribution;
          }
        });
      spatial.candidates += query_candidate_count;
      if ((query_valid == false) || (evaluation_valid == false)) {
        return false;
      }
    }
    return true;
  }

  void evaluate_camera_paths(const uint32_t begin, const uint32_t end, const uint32_t thread_id) {
    CameraEvaluationStatistics statistics = {};
    try {
      evaluate_camera_paths_impl(begin, end, thread_id, statistics);
    } catch (const std::bad_alloc&) {
      fail("UPBP failed to allocate camera-path working storage");
    } catch (const std::length_error&) {
      fail("UPBP camera-path working storage exceeds the platform container limit");
    }
    publish_camera_statistics(statistics);
  }

  void evaluate_camera_paths_impl(const uint32_t begin, const uint32_t end, const uint32_t thread_id, CameraEvaluationStatistics& statistics) {
    const Scene& scene = rt.scene();
    Film& film = rt.film();
    const uint32_t maximum_vertices = scene.options.max_path_length + 1u;
    if (thread_id >= camera_workspaces.size()) {
      fail("UPBP camera task received an invalid scheduler thread index");
      return;
    }
    CameraWorkspace& workspace = camera_workspaces[thread_id];
    for (uint32_t path_index = begin; running() && (path_index < end); ++path_index) {
      uint2 pixel = {};
      if (film.active_pixel(path_index, pixel) == false) {
        complete_camera_path(statistics);
        continue;
      }
      uint32_t timing_sample_hash = path_index;
      timing_sample_hash ^= timing_sample_hash >> 16u;
      timing_sample_hash *= 0x7feb352du;
      timing_sample_hash ^= timing_sample_hash >> 15u;
      constexpr uint32_t camera_timing_sample_mask = camera_timing_sample_rate - 1u;
      const bool measure_camera_timing = (timing_sample_hash & camera_timing_sample_mask) == 0u;
      uint64_t camera_phase_start = 0u;
      auto begin_camera_phase = [&camera_phase_start, measure_camera_timing]() {
        if (measure_camera_timing) {
          camera_phase_start = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
        }
      };
      auto finish_camera_phase = [&statistics, &camera_phase_start, measure_camera_timing](const CameraTimingStage timing_stage) {
        if (measure_camera_timing) {
          const uint64_t phase_end = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
          statistics.stage_times[timing_stage] += phase_end - camera_phase_start;
        }
      };
      begin_camera_phase();
      Sampler film_sampler{upbp_sampler_seed(scene.options.random_seed, status.current_iteration, path_index, 0u, 0u, UPBPRandomDomain::FilmSample)};
      const float2 film_uv = film.sample(status.current_iteration == 0u ? PixelFilter::empty() : scene.pixel_sampler, pixel, film_sampler.next_2d());
      UPBPCameraSubpathResult& camera = workspace.camera;
      if (upbp_build_camera_subpath(rt, scene, iteration.spect, film_uv, scene.options.random_seed, status.current_iteration, path_index, maximum_vertices,
            options.maximum_boundary_count, options.maximum_null_events_per_interval, camera) == false) {
        fail("UPBP camera subpath failed at pixel path " + std::to_string(path_index) + ", failure " + std::to_string(static_cast<uint32_t>(camera.subpath.failure)) +
             ", segment failure " + std::to_string(static_cast<uint32_t>(camera.subpath.segment_failure)) + ", vertices " + std::to_string(camera.subpath.path.vertices.size()) +
             ", ray origin (" + std::to_string(camera.subpath.terminal_ray.o.x) + ", " + std::to_string(camera.subpath.terminal_ray.o.y) + ", " +
             std::to_string(camera.subpath.terminal_ray.o.z) + "), direction (" + std::to_string(camera.subpath.terminal_ray.d.x) + ", " +
             std::to_string(camera.subpath.terminal_ray.d.y) + ", " + std::to_string(camera.subpath.terminal_ray.d.z) + ")");
        return;
      }
      finish_camera_phase(CameraTimingSubpath);
      UPBPRecursivePathWeights& camera_weights = workspace.weights;
      begin_camera_phase();
      if (upbp_compute_recursive_path_weights(scene, camera.subpath.path, iteration.mis, iteration.light_subpath_count, iteration.bpt_sample_count, camera_weights) == false) {
        fail("UPBP recursive camera-path MIS failed at pixel path " + std::to_string(path_index) + ", vertex " + std::to_string(camera_weights.failure_vertex_index) +
             ", failure " + std::to_string(static_cast<uint32_t>(camera_weights.failure)));
        return;
      }
      finish_camera_phase(CameraTimingRecursiveMIS);

      SpectralResponse value{iteration.spect, 0.0f};
      if (iteration.mis.enabled(UPBPTechnique::BPT) || scene.strategy_enabled(Scene::Strategy::DirectHit)) {
        begin_camera_phase();
        const uint32_t light_path_index =
          iteration.light_subpath_count == iteration.camera_subpath_count
            ? path_index
            : upbp_select_light_path_index(scene.options.random_seed, status.current_iteration, path_index, static_cast<uint32_t>(iteration.light_subpath_count));
        UPBPBPTCameraEvaluation bpt = {};
        if ((light_path_index == kInvalidIndex) ||
            (upbp_evaluate_bpt_camera_path(rt, scene, iteration.spect, light_paths[light_path_index].subpath.path, camera.subpath, scene.options.random_seed,
               status.current_iteration, path_index, options.maximum_boundary_count, options.maximum_null_events_per_interval, light_weights[light_path_index], camera_weights,
               iteration.mis, iteration.light_subpath_count, bpt) == false)) {
          fail("UPBP BPT evaluation failed at pixel path " + std::to_string(path_index) + ", failure " + std::to_string(static_cast<uint32_t>(bpt.failure)) + ", segment failure " +
               std::to_string(static_cast<uint32_t>(bpt.segment_failure)) + ", medium failure " + std::to_string(static_cast<uint32_t>(bpt.medium_failure)) + ", triangle " +
               std::to_string(bpt.failure_intersection.triangle_index) + ", material " + std::to_string(bpt.failure_intersection.material_index) + ", instance " +
               std::to_string(bpt.failure_intersection.instance_index) + ", t " + std::to_string(bpt.failure_intersection.t));
          return;
        }
        value += bpt.value;
        finish_camera_phase(CameraTimingBPT);
      }

      for (uint32_t camera_vertex_index = 1u; camera_vertex_index < camera.subpath.path.vertices.size(); ++camera_vertex_index) {
        const UPBPPathVertexRecord& vertex = camera.subpath.path.vertices[camera_vertex_index];
        if ((vertex.cls == UPBPVertexClass::Surface) && iteration.mis.enabled(UPBPTechnique::Surface)) {
          begin_camera_phase();
          if (evaluate_point_technique(UPBPTechnique::Surface, surface_index, iteration.surface_radius, camera.subpath.path, camera_weights, path_index, camera_vertex_index,
                value) == false) {
            fail("UPBP surface merging failed at pixel path " + std::to_string(path_index));
            return;
          }
          finish_camera_phase(CameraTimingSurfaceMerge);
        }
        if ((vertex.cls == UPBPVertexClass::Medium) && iteration.mis.enabled(UPBPTechnique::PP3D)) {
          begin_camera_phase();
          if (evaluate_point_technique(UPBPTechnique::PP3D, pp3d_index, iteration.pp3d_radius, camera.subpath.path, camera_weights, path_index, camera_vertex_index, value) ==
              false) {
            fail("UPBP PP3D evaluation failed at pixel path " + std::to_string(path_index));
            return;
          }
          finish_camera_phase(CameraTimingPP3D);
        }
        if ((vertex.cls == UPBPVertexClass::Medium) && iteration.mis.enabled(UPBPTechnique::BP2D)) {
          begin_camera_phase();
          if (evaluate_bp2d(camera.subpath.path, camera_weights, camera_vertex_index, value, statistics) == false) {
            fail("UPBP BP2D evaluation failed at pixel path " + std::to_string(path_index));
            return;
          }
          finish_camera_phase(CameraTimingBP2D);
        }
      }

      std::vector<UPBPBeamReference>& camera_beams = workspace.beams;
      std::vector<UPBPPreparedBeam>& prepared_camera_beams = workspace.prepared_beams;
      if (iteration.mis.enabled(UPBPTechnique::PB2D) || iteration.mis.enabled(UPBPTechnique::BB1D)) {
        begin_camera_phase();
        if (upbp_collect_medium_beams(camera.subpath.path, path_index, camera_beams) == false) {
          fail("UPBP camera-beam collection failed at pixel path " + std::to_string(path_index));
          return;
        }
        prepared_camera_beams.resize(camera_beams.size());
        for (uint32_t beam_index = 0u; beam_index < camera_beams.size(); ++beam_index) {
          const UPBPBeamReference& beam = camera_beams[beam_index];
          upbp_prepare_beam(camera.subpath.path, camera_weights, beam, iteration.mis, prepared_camera_beams[beam_index]);
          if (spectral_query_compatible(beam.throughput_at_origin.as_query(), iteration.spect) == false) {
            fail("UPBP camera-beam spectrum is inconsistent at pixel path " + std::to_string(path_index) + ", beam " + std::to_string(beam_index));
            return;
          }
        }
        finish_camera_phase(CameraTimingBeamPreparation);
      }
      if (iteration.mis.enabled(UPBPTechnique::PB2D)) {
        begin_camera_phase();
        if (evaluate_pb2d(camera_beams, prepared_camera_beams, value, statistics) == false) {
          fail("UPBP PB2D evaluation failed at pixel path " + std::to_string(path_index));
          return;
        }
        finish_camera_phase(CameraTimingPB2D);
      }
      if (iteration.mis.enabled(UPBPTechnique::BB1D)) {
        begin_camera_phase();
        if (evaluate_bb1d(camera_beams, prepared_camera_beams, workspace.bb1d_query_state, value, statistics) == false) {
          fail("UPBP BB1D evaluation failed at pixel path " + std::to_string(path_index));
          return;
        }
        finish_camera_phase(CameraTimingBB1D);
      }

      begin_camera_phase();
      float3 normal = {};
      SpectralResponse albedo{iteration.spect, 0.0f};
      for (uint32_t vertex_index = 1u; vertex_index < camera.subpath.path.vertices.size(); ++vertex_index) {
        const UPBPPathVertexRecord& vertex = camera.subpath.path.vertices[vertex_index];
        if (vertex.cls == UPBPVertexClass::Surface) {
          normal = vertex.intersection.nrm;
          Sampler albedo_sampler = upbp_pair_sampler(scene.options.random_seed, status.current_iteration, path_index, 0u, vertex_index, 0u, UPBPRandomDomain::ScatteringEvaluation);
          const BSDFData data = {iteration.spect, vertex.incident_medium_index, PathSource::Camera, vertex.intersection, vertex.intersection.w_i};
          albedo = bsdf::albedo(data, scene.materials[vertex.intersection.material_index], albedo_sampler);
          break;
        }
      }
      film.submit(value.to_rgb_estimate(), normal, albedo.to_rgb_estimate(), pixel);
      finish_camera_phase(CameraTimingFilmSubmission);
      complete_camera_path(statistics);
    }
  }

  void complete_light_stage() {
    light_path_time_ms = stage_time.measure_ms();
    if (failed.load()) {
      *state = Integrator::State::Stopped;
      return;
    }

    TimeMeasure phase_time = {};
    phase_time.reset();
    if (collect_light_vertices() == false) {
      primitive_collection_time_ms = phase_time.measure_ms();
      *state = Integrator::State::Stopped;
      return;
    }
    primitive_collection_time_ms = phase_time.measure_ms();

    phase_time.reset();
    if (build_spatial_indices() == false) {
      spatial_index_time_ms = phase_time.measure_ms();
      *state = Integrator::State::Stopped;
      return;
    }
    spatial_index_time_ms = phase_time.measure_ms();

    phase_time.reset();
    submit_light_splats();
    light_splat_time_ms = phase_time.measure_ms();
    debug_info[DebugSurfacePoints].value = static_cast<float>(surface_points.size());
    debug_info[DebugMediumPoints].value = static_cast<float>(medium_points.size());
    debug_info[DebugLightBeams].value = static_cast<float>(light_beams.size());
    debug_info[DebugBB1DLightBeams].value = static_cast<float>(selected_bb1d_beams.size());
    phase_time.reset();
    const uint64_t light_storage_bytes = current_light_storage_bytes();
    storage_validation_time_ms = phase_time.measure_ms();
    debug_info[DebugLightStorage].value = static_cast<float>(light_storage_bytes) / (1024.0f * 1024.0f);
    refresh_live_diagnostics();
    if (status.current_iteration == 0u) {
      if (light_storage_bytes > memory_target_bytes) {
        log::warning("UPBP retained %.1f MiB for %llu light paths, exceeding the advisory %u MiB CPU memory target; rendering continues",
          static_cast<double>(debug_info[DebugLightStorage].value), static_cast<unsigned long long>(iteration.light_subpath_count), options.memory_budget_mb);
      } else {
        log::info("UPBP retained %.1f MiB for %llu light paths; advisory CPU memory target %u MiB", static_cast<double>(debug_info[DebugLightStorage].value),
          static_cast<unsigned long long>(iteration.light_subpath_count), options.memory_budget_mb);
      }
    }
    stage = Stage::Camera;
    stage_time.reset();
    live_report_time.reset();
    task_handle = rt.scheduler().schedule(rt.film().current_pixel_count(), &camera_task);
  }

  void complete_camera_stage() {
    camera_evaluation_time_ms = stage_time.measure_ms();
    refresh_live_diagnostics();
    if (failed.load()) {
      *state = Integrator::State::Stopped;
      return;
    }
    const Scene& scene = rt.scene();
    const Film::NoiseEstimationSchedule noise_estimation_schedule =
      scene.spectral() ? Film::NoiseEstimationSchedule::PowerOfTwoSampleCount : Film::NoiseEstimationSchedule::EveryOtherIteration;
    rt.film().commit_iteration(status.current_iteration, scene.options.samples, scene.options.noise_threshold, scene.options.radiance_clamp, noise_estimation_schedule);
    status.completed_iterations += 1u;
    status.last_iteration_time = iteration_time.measure();
    status.total_time += status.last_iteration_time;
    if (status.current_iteration == 0u) {
      log::info(
        "UPBP first-iteration timing: setup %.2f ms, light paths %.2f ms, storage accounting %.2f ms, primitive collection %.2f ms, spatial indices %.2f ms, light splats "
        "%.2f ms, camera evaluation %.2f ms",
        iteration_setup_time_ms, light_path_time_ms, storage_validation_time_ms, primitive_collection_time_ms, spatial_index_time_ms, light_splat_time_ms,
        camera_evaluation_time_ms);
      report_spatial_statistics("BP2D", SpatialStatisticBP2D);
      report_spatial_statistics("PB2D", SpatialStatisticPB2D);
      report_spatial_statistics("BB1D", SpatialStatisticBB1D);
      log::info(
        "UPBP first-iteration estimated camera CPU totals (ms): subpath %.1f, recursive MIS %.1f, BPT/direct hit %.1f, surface %.1f, PP3D %.1f, BP2D %.1f, beam preparation "
        "%.1f, PB2D %.1f, BB1D %.1f, film %.1f",
        static_cast<double>(debug_info[DebugCameraSubpathTime].value), static_cast<double>(debug_info[DebugCameraRecursiveMISTime].value),
        static_cast<double>(debug_info[DebugCameraBPTTime].value), static_cast<double>(debug_info[DebugCameraSurfaceMergeTime].value),
        static_cast<double>(debug_info[DebugCameraPP3DTime].value), static_cast<double>(debug_info[DebugCameraBP2DTime].value),
        static_cast<double>(debug_info[DebugCameraBeamPreparationTime].value), static_cast<double>(debug_info[DebugCameraPB2DTime].value),
        static_cast<double>(debug_info[DebugCameraBB1DTime].value), static_cast<double>(debug_info[DebugCameraFilmSubmissionTime].value));
    }
    if ((*state == Integrator::State::WaitingForCompletion) || (status.current_iteration + 1u >= scene.options.samples)) {
      *state = Integrator::State::Stopped;
      return;
    }
    start_iteration(status.current_iteration + 1u);
  }
};

CPUUPBP::CPUUPBP(Raytracing& rt)
  : Integrator(rt) {
  ETX_PIMPL_INIT(CPUUPBP, rt, &current_state);
  _private->options.store(integrator_options);
}

CPUUPBP::~CPUUPBP() {
  stop(Stop::Immediate);
  ETX_PIMPL_CLEANUP(CPUUPBP);
}

const char* CPUUPBP::status_str() const {
  return _private->status_string();
}

Integrator::PathProgress CPUUPBP::path_progress() const {
  return current_state == State::Stopped ? PathProgress{} : _private->path_progress();
}

void CPUUPBP::run() {
  stop(Stop::Immediate);
  if (can_run()) {
    current_state = State::Running;
    _private->start(integrator_options);
    _private->release_failed_storage();
  }
}

void CPUUPBP::update() {
  if (current_state != State::Stopped) {
    _private->report_live_diagnostics();
  }
  if ((current_state == State::Stopped) || (rt.scheduler().completed(_private->task_handle) == false)) {
    return;
  }
  _private->wait_for_tasks();
  if (_private->stage == CPUUPBPImpl::Stage::Light) {
    _private->complete_light_stage();
  } else {
    _private->complete_camera_stage();
  }
  _private->release_failed_storage();
}

void CPUUPBP::stop(const Stop stop_mode) {
  if (current_state == State::Stopped) {
    if (stop_mode == Stop::Immediate) {
      _private->release_light_storage();
    }
    return;
  }
  current_state = stop_mode == Stop::Immediate ? State::Stopped : State::WaitingForCompletion;
  if (current_state == State::Stopped) {
    _private->wait_for_tasks();
    _private->release_light_storage();
  }
}

void CPUUPBP::update_options() {
  if (current_state == State::Running) {
    run();
  }
}

void CPUUPBP::sync_from_options(const Options& options) {
  _private->options = {};
  _private->options.load(options);
  _private->options.store(integrator_options);
}

uint32_t CPUUPBP::supported_strategies() const {
  return Scene::Strategy::DirectHit | Scene::Strategy::ConnectToLight | Scene::Strategy::ConnectToCamera | Scene::Strategy::ConnectVertices | Scene::Strategy::MergeVertices;
}

const Integrator::Status& CPUUPBP::status() const {
  return _private->status;
}

}  // namespace etx
