#pragma once

#include "gpu_wavefront_shared.hxx"

ETX_STATIC_CONST uint32_t kGPUUPBPBB1DPartitionCount = 64u;
ETX_STATIC_CONST uint32_t kGPUUPBPBP2DPartitionCount = 32u;
ETX_STATIC_CONST uint32_t kGPUUPBPSurfacePartitionCount = 8u;
ETX_STATIC_ASSERT((kGPUUPBPBB1DPartitionCount > 0u) && (kGPUUPBPBB1DPartitionCount <= 64u), "GPU UPBP BB1D partitions must fit one shader thread group");
ETX_STATIC_ASSERT((kGPUUPBPBP2DPartitionCount > 0u) && (kGPUUPBPBP2DPartitionCount <= 64u), "GPU UPBP BP2D partitions must fit one shader thread group");

struct GPUUPBPTechnique {
  enum : uint32_t {
    BPT = 1u << 0u,
    Surface = 1u << 1u,
    PP3D = 1u << 2u,
    PB2D = 1u << 3u,
    BP2D = 1u << 4u,
    BB1D = 1u << 5u,
  };
};

struct GPUUPBPVertexFlags {
  enum : uint32_t {
    Valid = 1u << 0u,
    Camera = 1u << 1u,
    Emitter = 1u << 2u,
    Surface = 1u << 3u,
    Medium = 1u << 4u,
    Connectible = 1u << 5u,
    DensityConnectible = 1u << 6u,
    Delta = 1u << 7u,
    DistantEndpoint = 1u << 8u,
    HasDeparture = 1u << 9u,
    InlineMedium = 1u << 10u,
    FromLight = 1u << 11u,
  };
};

struct GPUUPBPIntervalFlags {
  enum : uint32_t {
    Valid = 1u << 0u,
    Vacuum = 1u << 1u,
    Escape = 1u << 2u,
    Scatter = 1u << 3u,
    Absorb = 1u << 4u,
    InlineMedium = 1u << 5u,
    RecomputeTracking = 1u << 6u,
  };
};

struct GPUUPBPSegmentFlags {
  enum : uint32_t {
    Valid = 1u << 0u,
    Terminal = 1u << 1u,
    HasTerminalEventDensity = 1u << 2u,
  };
};

struct GPUUPBPBeamFlags {
  enum : uint32_t {
    Valid = 1u << 0u,
    SelectedForBB1D = 1u << 1u,
  };
};

struct GPUUPBPIterationFlags {
  enum : uint32_t {
    EvaluateCameraIndependentTerms = 1u << 2u,
    MultipleImportanceSampling = 1u << 3u,
    CollectLightDensityRecords = 1u << 4u,
  };
};

struct GPUUPBPRecursiveWeightFlags {
  enum : uint32_t {
    PreviousInMedium = 1u << 0u,
    PreviousDelta = 1u << 1u,
  };
};

struct GPUUPBPRecursiveFailure {
  enum : uint32_t {
    None = 0u,
    InvalidPath = 1u,
    InvalidEndpointDensity = 2u,
    InvalidSegmentDensity = 3u,
    InvalidMeasureCosine = 4u,
    InvalidScatteringDensity = 5u,
    NonFiniteArrival = 6u,
    NonFiniteDeparture = 7u,
  };
};

struct GPUUPBPPathStateFlags {
  enum : uint32_t {
    Valid = 1u << 0u,
    Light = 1u << 1u,
    HasTerminalSegment = 1u << 2u,
  };
};

struct GPUUPBPCounterIndex {
  enum : uint32_t {
    LightVertex = 0u,
    CameraVertex = 1u,
    LightSegment = 2u,
    CameraSegment = 3u,
    LightInterval = 4u,
    CameraInterval = 5u,
    LightEvent = 6u,
    CameraEvent = 7u,
    Point = 8u,
    SurfacePoint = 9u,
    MediumPoint = 10u,
    Beam = 11u,
    DensityBeamInstance = 12u,
    OverflowFlags = 13u,
    FailedLightPaths = 14u,
    FailedCameraPaths = 15u,
    FailedConnections = 16u,
    FirstFailureCode = 17u,
    FirstFailureGlobalPath = 18u,
    FirstFailureDetail0 = 19u,
    FirstFailureDetail1 = 20u,
    FirstFailureDetail2 = 21u,
    FirstFailureDetail3 = 22u,
    MaximumLightPathLength = 23u,
    BeamAccelerationPrimitive = 24u,
    DensitySurfacePoint = 25u,
    DensityMediumPoint = 26u,
    DensityBeam = 27u,
    DensitySelectedBeam = 28u,
    CameraSurfaceVariousQuery = 29u,
    CameraSurfacePlasticQuery = 30u,
    CameraSurfaceConductorQuery = 31u,
    CameraSurfaceDielectricQuery = 32u,
    CameraMediumVertexQuery = 33u,
    CameraMediumIntervalQuery = 34u,
    Count = 35u,
  };
};

struct GPUUPBPSurfaceQueryFamily {
  enum : uint32_t {
    Various = 0u,
    Plastic = 1u,
    Conductor = 2u,
    Dielectric = 3u,
    Count = 4u,
  };
};

struct GPUUPBPPathFailure {
  enum : uint32_t {
    None = 0u,
    InitializeCamera = 1u,
    InitializeLight = 2u,
    BeginSegment = 3u,
    InvalidSegmentDistance = 4u,
    TrackInterval = 5u,
    BoundaryLimit = 6u,
    AppendMediumVertex = 7u,
    AppendSurfaceVertex = 8u,
    FinalizeVertex = 9u,
    ConnectionTracking = 10u,
    SubsurfaceExitNotFound = 11u,
    SubsurfaceTracking = 12u,
    SubsurfaceExitMaterialMismatch = 13u,
    InvalidSubsurfaceExitDistance = 14u,
    NonFiniteDensityContribution = 15u,
  };
};

struct GPUUPBPConnectionTrackingFailure {
  enum : uint32_t {
    None = 0u,
    InvalidIntervalDistance = 1u,
    MediumTracking = 2u,
    InvalidTerminal = 3u,
    BoundaryLimit = 4u,
  };
};

struct GPUUPBPOverflowFlags {
  enum : uint32_t {
    Vertex = 1u << 0u,
    Segment = 1u << 1u,
    Interval = 1u << 2u,
    Event = 1u << 3u,
    Point = 1u << 4u,
    Beam = 1u << 5u,
    BeamInstance = 1u << 6u,
  };
};

struct GPUUPBPClearMode {
  enum : uint32_t {
    All = 0u,
    LightBatch = 1u,
    BeamInstances = 2u,
    BeamAcceleration = 3u,
    DensityCompact = 4u,
    CameraQueries = 5u,
  };
};

struct GPUUPBPPointGridBuildMode {
  enum : uint32_t {
    Build = 0u,
    BuildAABBs = 1u,
  };
};

struct GPUUPBPDensityCompactMode {
  enum : uint32_t {
    Points = 0u,
    Beams = 1u,
    BPTVertices = 2u,
    BPTPathStates = 3u,
    CameraVertices = 4u,
    CameraIntervals = 5u,
  };
};

struct GPUUPBPDensityQueryMode {
  enum : uint32_t {
    Raw = 0u,
    Compacted = 1u,
  };
};

struct GPUUPBPBeamGridBuildMode {
  enum : uint32_t {
    Describe = 0u,
    Count = 1u,
    CellTotals = 2u,
    Prefix = 3u,
    ShardOffsets = 4u,
    Scatter = 5u,
    Validate = 6u,
  };
};

struct GPUUPBPBeamGridType {
  enum : uint32_t {
    BP2D = 0u,
    BB1D = 1u,
  };
};

struct GPUUPBPBeamGridBuildFailure {
  enum : uint32_t {
    None = 0u,
    InvalidInput = 1u,
    EntryCountOverflow = 2u,
    OutputCapacity = 4u,
    InvalidIndex = 8u,
    DuplicateIndex = 16u,
    InvalidOrder = 32u,
    CountScatterMismatch = 64u,
  };
};

struct GPUUPBPBeamIndexMode {
  enum : uint32_t {
    AccelerationStructure = 0u,
    ComputeGrid = 1u,
  };
};

struct GPUUPBPDensityBeamFlags {
  enum : uint32_t {
    Valid = 1u << 0u,
    ScaleDSharedByDistance = 1u << 1u,
    PreviousDelta = 1u << 2u,
  };
};

struct GPUUPBPRecursiveWeights {
  float log_d_shared ETX_INIT(0.0f);
  float log_d_bpt_base ETX_INIT(0.0f);
  float log_d_pde_base ETX_INIT(0.0f);
  float log_d_surface ETX_INIT(0.0f);
  float log_ray_sample_forward_pdf_inverse ETX_INIT(0.0f);
  float log_ray_sample_reverse_pdf_inverse ETX_INIT(0.0f);
  float log_ray_sample_forward_ratio ETX_INIT(0.0f);
  float log_ray_sample_reverse_ratio ETX_INIT(0.0f);
  uint32_t flags ETX_INIT(0u);
};

struct ETX_ALIGNED GPUUPBPRecursiveState {
  GPUUPBPRecursiveWeights weights ETX_INIT({});
  float last_sin_theta ETX_INIT(0.0f);
  float log_d_bpt_a ETX_INIT(0.0f);
  float log_d_bpt_b ETX_INIT(0.0f);
  float log_d_surface_b ETX_INIT(0.0f);
  float log_d_pde_b ETX_INIT(0.0f);
  uint32_t failure ETX_INIT(0u);
  uint32_t failure_vertex_index ETX_INIT(0u);
};

struct ETX_ALIGNED GPUUPBPVertex {
  GPUWavefrontCompactSpectralResponse throughput ETX_INIT({});
  GPUWavefrontCompactSpectralResponse outgoing_throughput ETX_INIT({});
  float3 position ETX_INIT({});
  uint32_t flags ETX_INIT(0u);
  float3 sampled_direction ETX_INIT({});
  uint32_t medium_index ETX_INIT(kInvalidIndex);
  float3 w_i ETX_INIT({});
  uint32_t incident_medium_index ETX_INIT(kInvalidIndex);
  float3 normal ETX_INIT({});
  uint32_t outgoing_medium_index ETX_INIT(kInvalidIndex);
  float3 geo_normal ETX_INIT({});
  uint32_t material_index ETX_INIT(kInvalidIndex);
  float2 texcoord ETX_INIT({});
  uint32_t triangle_index ETX_INIT(kInvalidIndex);
  uint32_t instance_index ETX_INIT(kInvalidIndex);
  float scatter_pdf_forward ETX_INIT(0.0f);
  float scatter_pdf_reverse ETX_INIT(0.0f);
  float endpoint_pdf_area ETX_INIT(0.0f);
  float endpoint_pdf_sample ETX_INIT(0.0f);
  float endpoint_pdf_direction ETX_INIT(0.0f);
  float log_medium_event_density ETX_INIT(0.0f);
  float eta ETX_INIT(1.0f);
  uint32_t sample_properties ETX_INIT(0u);
  GPUUPBPRecursiveWeights arrival_weights ETX_INIT({});
  uint32_t previous_vertex_index ETX_INIT(kInvalidIndex);
  uint32_t incoming_segment_index ETX_INIT(kInvalidIndex);
  uint32_t path_length ETX_INIT(0u);
  GPUUPBPRecursiveState departure_state ETX_INIT({});
  uint32_t global_path_index ETX_INIT(0u);
  float3 barycentric ETX_INIT({});
  uint32_t emitter_index ETX_INIT(kInvalidIndex);
  float inline_phase_function_g ETX_INIT(0.0f);
  uint32_t reserved0 ETX_INIT(0u);
  uint32_t reserved1 ETX_INIT(0u);
  GPUWavefrontCompactSpectralResponse inline_scattering ETX_INIT({});
  GPUWavefrontCompactSpectralResponse inline_extinction ETX_INIT({});
};

struct ETX_ALIGNED GPUUPBPPathState {
  GPUUPBPRecursiveState recursive_state ETX_INIT({});
  uint32_t first_vertex_index ETX_INIT(kInvalidIndex);
  uint32_t last_vertex_index ETX_INIT(kInvalidIndex);
  uint32_t current_segment_index ETX_INIT(kInvalidIndex);
  uint32_t current_interval_index ETX_INIT(kInvalidIndex);
  uint32_t transport_counts ETX_INIT(0u);
  uint32_t global_path_index ETX_INIT(0u);
  uint32_t path_length ETX_INIT(0u);
  uint32_t flags ETX_INIT(0u);
};

struct ETX_ALIGNED GPUUPBPBPTVertex {
  GPUWavefrontCompactSpectralResponse throughput ETX_INIT({});
  float3 position ETX_INIT({});
  uint32_t flags ETX_INIT(0u);
  float3 w_i ETX_INIT({});
  uint32_t medium_index ETX_INIT(kInvalidIndex);
  float3 normal ETX_INIT({});
  uint32_t material_index ETX_INIT(kInvalidIndex);
  float3 geo_normal ETX_INIT({});
  float log_medium_event_density ETX_INIT(0.0f);
  float2 texcoord ETX_INIT({});
  uint32_t triangle_index ETX_INIT(kInvalidIndex);
  uint32_t instance_index ETX_INIT(kInvalidIndex);
  GPUUPBPRecursiveWeights arrival_weights ETX_INIT({});
  uint32_t previous_vertex_index ETX_INIT(kInvalidIndex);
  uint32_t path_length ETX_INIT(0u);
  uint32_t global_path_index ETX_INIT(0u);
  uint32_t emitter_index ETX_INIT(kInvalidIndex);
  float2 barycentric ETX_INIT({});
  float scatter_pdf_forward ETX_INIT(0.0f);
  GPUWavefrontCompactSpectralResponse inline_extinction ETX_INIT({});
};

struct ETX_ALIGNED GPUUPBPBPTPathState {
  uint32_t last_vertex_index ETX_INIT(kInvalidIndex);
  uint32_t global_path_index ETX_INIT(0u);
  uint32_t path_length ETX_INIT(0u);
  uint32_t flags ETX_INIT(0u);
};

struct ETX_ALIGNED GPUUPBPSegment {
  GPUWavefrontCompactSpectralResponse weight ETX_INIT({});
  float log_pdf_forward ETX_INIT(0.0f);
  float log_pdf_reverse ETX_INIT(0.0f);
  float log_transport_pdf_forward ETX_INIT(0.0f);
  float log_transport_pdf_reverse ETX_INIT(0.0f);
  float log_terminal_event_density ETX_INIT(0.0f);
  float distance ETX_INIT(0.0f);
  uint32_t first_interval_index ETX_INIT(kInvalidIndex);
  uint32_t interval_count ETX_INIT(0u);
  uint32_t source_vertex_index ETX_INIT(kInvalidIndex);
  uint32_t target_vertex_index ETX_INIT(kInvalidIndex);
  uint32_t boundary_count ETX_INIT(0u);
  uint32_t flags ETX_INIT(0u);
};

struct ETX_ALIGNED GPUUPBPInterval {
  GPUWavefrontCompactSpectralResponse weight ETX_INIT({});
  float3 start_position ETX_INIT({});
  uint32_t medium_index ETX_INIT(kInvalidIndex);
  float3 end_position ETX_INIT({});
  uint32_t flags ETX_INIT(0u);
  float log_pdf_forward ETX_INIT(0.0f);
  float log_pdf_reverse ETX_INIT(0.0f);
  float log_transport_pdf_forward ETX_INIT(0.0f);
  float log_transport_pdf_reverse ETX_INIT(0.0f);
  float log_terminal_event_density ETX_INIT(0.0f);
  float distance ETX_INIT(0.0f);
  uint32_t first_event_index ETX_INIT(kInvalidIndex);
  uint32_t event_count ETX_INIT(0u);
  uint32_t segment_index ETX_INIT(kInvalidIndex);
  uint32_t next_interval_index ETX_INIT(kInvalidIndex);
  uint32_t tracking_seed ETX_INIT(0u);
  GPUWavefrontCompactSpectralResponse inline_scattering ETX_INIT({});
  GPUWavefrontCompactSpectralResponse inline_absorption ETX_INIT({});
};

struct ETX_ALIGNED GPUUPBPTrackingEvent {
  GPUWavefrontCompactSpectralResponse weight_before ETX_INIT({});
  float log_transport_pdf_forward_before ETX_INIT(0.0f);
  float log_transport_pdf_reverse_before ETX_INIT(0.0f);
  float distance_before ETX_INIT(0.0f);
  float end_distance ETX_INIT(0.0f);
  float majorant ETX_INIT(0.0f);
  uint32_t interval_index ETX_INIT(kInvalidIndex);
  uint32_t next_event_index ETX_INIT(kInvalidIndex);
  uint32_t reserved0 ETX_INIT(0u);
};

struct ETX_ALIGNED GPUUPBPPoint {
  float3 position ETX_INIT({});
  uint32_t vertex_index ETX_INIT(kInvalidIndex);
};

struct ETX_ALIGNED GPUUPBPBeam {
  float3 origin ETX_INIT({});
  float length ETX_INIT(0.0f);
  float3 direction ETX_INIT({});
  uint32_t flags ETX_INIT(0u);
  uint32_t source_vertex_index ETX_INIT(kInvalidIndex);
  uint32_t interval_index ETX_INIT(kInvalidIndex);
  uint32_t global_path_index ETX_INIT(0u);
  uint32_t path_length ETX_INIT(0u);
};

struct GPUUPBPAABB {
  float3 minimum ETX_INIT({});
  float3 maximum ETX_INIT({});
  uint32_t beam_index ETX_INIT(kInvalidIndex);
  uint32_t reserved0 ETX_INIT(0u);
};

struct ETX_ALIGNED GPUUPBPDensityPoint {
  GPUWavefrontCompactSpectralResponse throughput ETX_INIT({});
  float3 position ETX_INIT({});
  uint32_t flags ETX_INIT(0u);
  float3 w_i ETX_INIT({});
  uint32_t medium_index ETX_INIT(kInvalidIndex);
  float3 geo_normal ETX_INIT({});
  uint32_t path_length ETX_INIT(0u);
  GPUUPBPRecursiveWeights arrival_weights ETX_INIT({});
  uint32_t global_path_index ETX_INIT(0u);
  float log_medium_event_density ETX_INIT(0.0f);
  float inline_phase_function_g ETX_INIT(0.0f);
  GPUWavefrontCompactSpectralResponse inline_scattering ETX_INIT({});
  GPUWavefrontCompactSpectralResponse inline_extinction ETX_INIT({});
};

struct ETX_ALIGNED GPUUPBPDensityBeam {
  GPUUPBPBeam beam ETX_INIT({});
  GPUUPBPInterval interval ETX_INIT({});
  GPUWavefrontCompactSpectralResponse source_throughput ETX_INIT({});
  GPUWavefrontCompactSpectralResponse transport_weight ETX_INIT({});
  float transport_log_pdf_forward ETX_INIT(0.0f);
  float transport_log_pdf_reverse ETX_INIT(0.0f);
  float transport_distance ETX_INIT(0.0f);
  float log_d_shared ETX_INIT(0.0f);
  float log_d_pde_reverse_coefficient ETX_INIT(0.0f);
  float log_d_pde_constant ETX_INIT(0.0f);
  float source_event_log_density ETX_INIT(0.0f);
  float interval_distance ETX_INIT(0.0f);
  uint32_t flags ETX_INIT(0u);
  uint32_t event_buffer ETX_INIT(kInvalidIndex);
  uint32_t event_index_offset ETX_INIT(0u);
  float log_d_surface_constant ETX_INIT(0.0f);
};

struct GPUUPBPDensityBatch {
  uint32_t surface_point_buffer ETX_INIT(kInvalidIndex);
  uint32_t surface_point_count ETX_INIT(0u);
  uint32_t medium_point_buffer ETX_INIT(kInvalidIndex);
  uint32_t medium_point_count ETX_INIT(0u);
  uint32_t beam_buffer ETX_INIT(kInvalidIndex);
  uint32_t beam_count ETX_INIT(0u);
  uint32_t beam_instance_offset ETX_INIT(0u);
  uint32_t selected_beam_count ETX_INIT(0u);
  uint32_t event_index_offset ETX_INIT(0u);
  uint32_t reserved0 ETX_INIT(0u);
};

struct GPUUPBPBeamReference {
  float3 origin ETX_INIT({});
  float length ETX_INIT(0.0f);
  float3 direction ETX_INIT({});
  uint32_t path_length ETX_INIT(0u);
  uint32_t medium_index ETX_INIT(kInvalidIndex);
};

struct ETX_ALIGNED GPUUPBPBeamGridMetadata {
  float3 minimum ETX_INIT({});
  uint32_t resolution_x ETX_INIT(0u);
  float3 maximum ETX_INIT({});
  uint32_t resolution_y ETX_INIT(0u);
  float3 inverse_cell_size ETX_INIT({});
  uint32_t resolution_z ETX_INIT(0u);
  uint32_t cell_count ETX_INIT(0u);
  uint32_t beam_count ETX_INIT(0u);
  uint32_t entry_count ETX_INIT(0u);
  uint32_t reserved0 ETX_INIT(0u);
};

struct ETX_ALIGNED GPUUPBPBeamGridResources {
  uint32_t metadata_buffer ETX_INIT(kInvalidIndex);
  uint32_t cell_offsets_buffer ETX_INIT(kInvalidIndex);
  uint32_t reserved0 ETX_INIT(0u);
  uint32_t beam_indices_buffer ETX_INIT(kInvalidIndex);
  uint32_t beam_count ETX_INIT(0u);
  uint32_t beam_index_count ETX_INIT(0u);
  uint32_t reserved1 ETX_INIT(0u);
  uint32_t reserved2 ETX_INIT(0u);
};

struct ETX_ALIGNED GPUUPBPIteration {
  uint32_t technique_mask ETX_INIT(0u);
  uint32_t kernel ETX_INIT(0u);
  uint32_t flags ETX_INIT(0u);
  uint32_t sample_index ETX_INIT(0u);
  uint32_t global_camera_path_count ETX_INIT(0u);
  uint32_t global_light_path_count ETX_INIT(0u);
  uint32_t bb1d_light_path_count ETX_INIT(0u);
  uint32_t light_batch_offset ETX_INIT(0u);
  uint32_t light_batch_count ETX_INIT(0u);
  uint32_t camera_batch_offset ETX_INIT(0u);
  uint32_t camera_batch_count ETX_INIT(0u);
  uint32_t maximum_null_events_per_interval ETX_INIT(0u);
  float surface_radius ETX_INIT(0.0f);
  float pp3d_radius ETX_INIT(0.0f);
  float pb2d_radius ETX_INIT(0.0f);
  float bp2d_radius ETX_INIT(0.0f);
  float bb1d_radius ETX_INIT(0.0f);
  float bpt_sample_count ETX_INIT(0.0f);
  uint32_t maximum_boundary_count ETX_INIT(0u);
  float technique_factors[6u] ETX_INIT({});
  float reserved1[3u] ETX_INIT({});
};

struct ETX_ALIGNED GPUUPBPResources {
  GPUUPBPIteration iteration ETX_INIT({});
  uint32_t vertex_buffer ETX_INIT(kInvalidIndex);
  uint32_t segment_buffer ETX_INIT(kInvalidIndex);
  uint32_t interval_buffer ETX_INIT(kInvalidIndex);
  uint32_t event_buffer ETX_INIT(kInvalidIndex);
  uint32_t point_buffer ETX_INIT(kInvalidIndex);
  uint32_t beam_buffer ETX_INIT(kInvalidIndex);
  uint32_t density_output_beam_instance_buffer ETX_INIT(kInvalidIndex);
  uint32_t density_output_beam_reference_buffer ETX_INIT(kInvalidIndex);
  uint32_t density_output_beam_instance_capacity ETX_INIT(0u);
  uint32_t density_beam_acceleration_structure_reference_low ETX_INIT(0u);
  uint32_t counter_buffer ETX_INIT(kInvalidIndex);
  uint32_t path_state_buffer ETX_INIT(kInvalidIndex);
  uint32_t light_vertex_capacity ETX_INIT(0u);
  uint32_t camera_vertex_capacity ETX_INIT(0u);
  uint32_t light_segment_capacity ETX_INIT(0u);
  uint32_t camera_segment_capacity ETX_INIT(0u);
  uint32_t light_interval_capacity ETX_INIT(0u);
  uint32_t camera_interval_capacity ETX_INIT(0u);
  uint32_t light_event_capacity ETX_INIT(0u);
  uint32_t camera_event_capacity ETX_INIT(0u);
  uint32_t point_capacity ETX_INIT(0u);
  uint32_t beam_capacity ETX_INIT(0u);
  uint32_t density_beam_acceleration_structure_reference_high ETX_INIT(0u);
  uint32_t light_path_state_capacity ETX_INIT(0u);
  uint32_t camera_path_state_capacity ETX_INIT(0u);
  uint32_t reserved0 ETX_INIT(0u);
  uint32_t bb1d_beam_buffer ETX_INIT(kInvalidIndex);
  uint32_t beam_acceleration_structure ETX_INIT(kInvalidIndex);
  uint32_t density_output_bb1d_beam_instance_buffer ETX_INIT(kInvalidIndex);
  uint32_t density_output_bb1d_beam_instance_capacity ETX_INIT(0u);
  uint32_t point_acceleration_structure ETX_INIT(kInvalidIndex);
  uint32_t point_aabb_buffer ETX_INIT(kInvalidIndex);
  uint32_t point_aabb_capacity ETX_INIT(0u);
  uint32_t density_batch_buffer ETX_INIT(kInvalidIndex);
  uint32_t density_batch_count ETX_INIT(0u);
  uint32_t density_output_surface_point_buffer ETX_INIT(kInvalidIndex);
  uint32_t density_output_surface_point_capacity ETX_INIT(0u);
  uint32_t density_output_beam_buffer ETX_INIT(kInvalidIndex);
  uint32_t density_output_beam_capacity ETX_INIT(0u);
  uint32_t density_output_event_buffer ETX_INIT(kInvalidIndex);
  uint32_t density_output_medium_point_buffer ETX_INIT(kInvalidIndex);
  uint32_t density_output_medium_point_capacity ETX_INIT(0u);
  uint32_t medium_point_acceleration_structure ETX_INIT(kInvalidIndex);
  uint32_t density_output_medium_point_aabb_buffer ETX_INIT(kInvalidIndex);
  uint32_t density_output_medium_point_aabb_capacity ETX_INIT(0u);
  uint32_t beam_reference_buffer ETX_INIT(kInvalidIndex);
  uint32_t bpt_light_vertex_buffer ETX_INIT(kInvalidIndex);
  uint32_t bpt_light_path_state_buffer ETX_INIT(kInvalidIndex);
  uint32_t bb1d_partition_acceleration_structures[kGPUUPBPBB1DPartitionCount - 1u] ETX_INIT({});
  uint32_t bp2d_beam_acceleration_structures[kGPUUPBPBP2DPartitionCount] ETX_INIT({});
  GPUUPBPBeamGridResources bp2d_beam_grid ETX_INIT({});
  GPUUPBPBeamGridResources bb1d_beam_grid ETX_INIT({});
  uint32_t beam_index_mode ETX_INIT(GPUUPBPBeamIndexMode::AccelerationStructure);
  uint32_t reserved1[3u] ETX_INIT({});
};
