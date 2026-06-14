#pragma once

#include "geometry.hxx"
#include "ray.hxx"
#include "spectrum.hxx"

struct GPUWavefrontPathFlags {
  enum : uint32_t {
    Valid = 1u << 0u,
    Connectible = 1u << 1u,
    Delta = 1u << 2u,
    Specular_bounce = 1u << 3u,
    From_camera = 1u << 4u,
    From_light = 1u << 5u,
    Hit_emitter = 1u << 6u,
    Surface_vertex = 1u << 7u,
    Medium_vertex = 1u << 8u,
    Depth_limit_reached_while_refractive = 1u << 9u,
  };
};

struct GPUWavefrontHitFlags {
  enum : uint32_t {
    Valid = 1u << 0u,
    Miss = 1u << 1u,
    Local_emitter = 1u << 2u,
    Environment_emitter = 1u << 3u,
    Medium = 1u << 4u,
    Subsurface = 1u << 5u,
  };
};

struct GPUWavefrontVertexFlags {
  enum : uint32_t {
    Valid = 1u << 0u,
    Connectible = 1u << 1u,
    Delta = 1u << 2u,
    From_camera = 1u << 3u,
    From_light = 1u << 4u,
    Surface = 1u << 5u,
    Medium = 1u << 6u,
    Emitter = 1u << 7u,
    Mis_connectible = 1u << 8u,
    Camera = 1u << 9u,
    Subsurface = 1u << 10u,
  };
};

struct GPUWavefrontSubsurfaceFlags {
  enum : uint32_t {
    Active = 1u << 0u,
    InlineMedium = 1u << 1u,
  };
};

struct GPUWavefrontPathMetaFlags {
  enum : uint32_t {
    Camera_active = 1u << 0u,
    Light_active = 1u << 1u,
  };
};

struct GPUWavefrontDirectLightSampleFlags {
  enum : uint32_t {
    Valid = 1u << 0u,
    Delta = 1u << 1u,
    Distant = 1u << 2u,
  };
};

struct ETX_ALIGNED GPUWavefrontQueueHeader {
  uint32_t count ETX_INIT(0u);
  uint32_t pad0 ETX_INIT(0u);
  uint32_t pad1 ETX_INIT(0u);
  uint32_t pad2 ETX_INIT(0u);
};

struct ETX_ALIGNED GPUWavefrontPathState {
  Ray ray ETX_INIT({});
  SpectralResponse throughput ETX_INIT({});
  float eta ETX_INIT(1.0f);
  float eta_scale ETX_INIT(1.0f);
  float forward_pdf ETX_INIT(0.0f);
  float reverse_pdf ETX_INIT(0.0f);
  float sampled_bsdf_pdf ETX_INIT(0.0f);
  float last_emitter_pdf ETX_INIT(0.0f);
  uint32_t medium_index ETX_INIT(kInvalidIndex);
  uint32_t path_length ETX_INIT(0u);
  uint32_t pixel_index ETX_INIT(0u);
  uint32_t flags ETX_INIT(0u);
  uint32_t path_source ETX_INIT(0u);
  uint32_t sampler_seed ETX_INIT(0u);
  uint2 pixel ETX_INIT({});
  SpectralQuery spect ETX_INIT({});
  float2 film_uv ETX_INIT({});
  uint32_t last_vertex_index ETX_INIT(kInvalidIndex);
  uint32_t reserved0 ETX_INIT(0u);
};

struct ETX_ALIGNED GPUWavefrontHit {
  SpectralResponse transmittance ETX_INIT({});
  Vertex vertex ETX_INIT({});
  float3 geo_normal ETX_INIT({});
  float hit_t ETX_INIT(0.0f);
  uint32_t triangle_index ETX_INIT(kInvalidIndex);
  uint32_t material_index ETX_INIT(kInvalidIndex);
  uint32_t emitter_index ETX_INIT(kInvalidIndex);
  uint32_t medium_index ETX_INIT(kInvalidIndex);
  uint32_t flags ETX_INIT(0u);
  float2 barycentric ETX_INIT({});
};

struct ETX_ALIGNED GPUWavefrontPathVertex {
  SpectralResponse throughput ETX_INIT({});
  SpectralResponse inline_medium_extinction ETX_INIT({});
  float3 position ETX_INIT({});
  uint32_t triangle_index ETX_INIT(kInvalidIndex);
  float3 normal ETX_INIT({});
  uint32_t material_index ETX_INIT(kInvalidIndex);
  float3 geo_normal ETX_INIT({});
  uint32_t medium_index ETX_INIT(kInvalidIndex);
  float3 w_i ETX_INIT({});
  uint32_t emitter_index ETX_INIT(kInvalidIndex);
  float2 texcoord ETX_INIT({});
  float forward_pdf ETX_INIT(0.0f);
  float reverse_pdf ETX_INIT(0.0f);
  float sampled_bsdf_pdf ETX_INIT(0.0f);
  float eta_scale ETX_INIT(1.0f);
  uint32_t path_length ETX_INIT(0u);
  uint32_t pixel_index ETX_INIT(0u);
  uint32_t flags ETX_INIT(0u);
  float pdf_from_prev ETX_INIT(0.0f);
  float pdf_from_next ETX_INIT(0.0f);
  float pdf_accumulated ETX_INIT(0.0f);
  float pdf_history ETX_INIT(0.0f);
  float pdf_ratio ETX_INIT(0.0f);
  float2 barycentric ETX_INIT({});
  uint32_t inline_medium_flags ETX_INIT(0u);
  uint32_t reserved0 ETX_INIT(0u);
};

struct ETX_ALIGNED GPUWavefrontPathMeta {
  uint32_t camera_path_length ETX_INIT(0u);
  uint32_t light_path_length ETX_INIT(0u);
  uint32_t flags ETX_INIT(0u);
  uint32_t reserved0 ETX_INIT(0u);
  float camera_mis_history ETX_INIT(0.0f);
  float light_mis_history ETX_INIT(0.0f);
  uint32_t from_delta ETX_INIT(0u);
  uint32_t reserved1 ETX_INIT(0u);
};

struct ETX_ALIGNED GPUWavefrontSubsurfaceState {
  SpectralResponse extinction ETX_INIT({});
  SpectralResponse scattering ETX_INIT({});
  SpectralResponse albedo ETX_INIT({});
  uint32_t material_index ETX_INIT(kInvalidIndex);
  uint32_t medium_index ETX_INIT(kInvalidIndex);
  uint32_t scatter_material_index ETX_INIT(kInvalidIndex);
  uint32_t flags ETX_INIT(0u);
  float phase_function_g ETX_INIT(0.0f);
  uint32_t reserved0 ETX_INIT(0u);
  uint32_t reserved1 ETX_INIT(0u);
  uint32_t reserved2 ETX_INIT(0u);
};

struct ETX_ALIGNED GPUWavefrontDirectLightSample {
  SpectralResponse value ETX_INIT({});
  float3 origin ETX_INIT({});
  float pdf_sample ETX_INIT(0.0f);
  float3 direction ETX_INIT({});
  float pdf_area ETX_INIT(0.0f);
  float3 normal ETX_INIT({});
  float pdf_dir ETX_INIT(0.0f);
  float2 texcoord ETX_INIT({});
  uint32_t emitter_index ETX_INIT(kInvalidIndex);
  uint32_t triangle_index ETX_INIT(kInvalidIndex);
  uint32_t flags ETX_INIT(0u);
  uint32_t reserved0 ETX_INIT(0u);
  uint32_t reserved1 ETX_INIT(0u);
  uint32_t reserved2 ETX_INIT(0u);
};

struct ETX_ALIGNED GPUWavefrontDirectLightTask {
  Ray shadow_ray ETX_INIT({});
  float3 shadow_target ETX_INIT({});
  uint32_t reserved0 ETX_INIT(0u);
  SpectralResponse contribution ETX_INIT({});
  float mis_weight ETX_INIT(0.0f);
  uint32_t pixel_index ETX_INIT(0u);
  uint32_t medium_index ETX_INIT(kInvalidIndex);
  uint32_t flags ETX_INIT(0u);
  uint32_t path_index ETX_INIT(0u);
  uint32_t sampler_seed ETX_INIT(0u);
};

struct ETX_ALIGNED GPUWavefrontDirectLightResult {
  SpectralResponse transmittance ETX_INIT({});
  uint32_t visible ETX_INIT(0u);
  uint32_t reserved0 ETX_INIT(0u);
  uint32_t reserved1 ETX_INIT(0u);
  uint32_t reserved2 ETX_INIT(0u);
};

struct ETX_ALIGNED GPUWavefrontConnectLightTask {
  Ray shadow_ray ETX_INIT({});
  float3 shadow_target ETX_INIT({});
  uint32_t reserved0 ETX_INIT(0u);
  SpectralResponse contribution ETX_INIT({});
  float mis_weight ETX_INIT(0.0f);
  uint32_t pixel_index ETX_INIT(0u);
  uint32_t medium_index ETX_INIT(kInvalidIndex);
  uint32_t flags ETX_INIT(0u);
  uint32_t path_index ETX_INIT(0u);
  uint32_t sampler_seed ETX_INIT(0u);
  SpectralResponse inline_medium_extinction ETX_INIT({});
  uint32_t inline_medium_flags ETX_INIT(0u);
  uint32_t reserved1 ETX_INIT(0u);
  uint32_t reserved2 ETX_INIT(0u);
  uint32_t reserved3 ETX_INIT(0u);
};

struct ETX_ALIGNED GPUWavefrontConnectLightResult {
  SpectralResponse transmittance ETX_INIT({});
  uint32_t visible ETX_INIT(0u);
  uint32_t reserved0 ETX_INIT(0u);
  uint32_t reserved1 ETX_INIT(0u);
  uint32_t reserved2 ETX_INIT(0u);
};

struct ETX_ALIGNED GPUWavefrontConnectCameraTask {
  Ray shadow_ray ETX_INIT({});
  float3 shadow_target ETX_INIT({});
  uint32_t reserved0 ETX_INIT(0u);
  SpectralResponse contribution ETX_INIT({});
  float mis_weight ETX_INIT(0.0f);
  uint32_t pixel_index ETX_INIT(0u);
  uint32_t medium_index ETX_INIT(kInvalidIndex);
  uint32_t flags ETX_INIT(0u);
  uint32_t path_index ETX_INIT(0u);
  uint32_t sampler_seed ETX_INIT(0u);
};

struct ETX_ALIGNED GPUWavefrontConnectCameraResult {
  SpectralResponse transmittance ETX_INIT({});
  uint32_t visible ETX_INIT(0u);
  uint32_t reserved0 ETX_INIT(0u);
  uint32_t reserved1 ETX_INIT(0u);
  uint32_t reserved2 ETX_INIT(0u);
};

struct ETX_ALIGNED GPUWavefrontResources {
  uint32_t camera_state_buffer ETX_INIT(kInvalidIndex);
  uint32_t light_state_buffer ETX_INIT(kInvalidIndex);
  uint32_t camera_hit_buffer ETX_INIT(kInvalidIndex);
  uint32_t light_hit_buffer ETX_INIT(kInvalidIndex);

  uint32_t camera_queue_a_buffer ETX_INIT(kInvalidIndex);
  uint32_t camera_queue_b_buffer ETX_INIT(kInvalidIndex);
  uint32_t light_queue_a_buffer ETX_INIT(kInvalidIndex);
  uint32_t light_queue_b_buffer ETX_INIT(kInvalidIndex);

  uint32_t camera_vertex_buffer ETX_INIT(kInvalidIndex);
  uint32_t light_vertex_buffer ETX_INIT(kInvalidIndex);
  uint32_t film_buffer ETX_INIT(kInvalidIndex);
  uint32_t path_meta_buffer ETX_INIT(kInvalidIndex);

  uint32_t direct_light_sample_buffer ETX_INIT(kInvalidIndex);
  uint32_t direct_light_task_buffer ETX_INIT(kInvalidIndex);
  uint32_t direct_light_result_buffer ETX_INIT(kInvalidIndex);
  uint32_t connect_light_task_buffer ETX_INIT(kInvalidIndex);
  uint32_t connect_light_result_buffer ETX_INIT(kInvalidIndex);
  uint32_t connect_camera_task_buffer ETX_INIT(kInvalidIndex);
  uint32_t connect_camera_result_buffer ETX_INIT(kInvalidIndex);
  uint32_t camera_subsurface_state_buffer ETX_INIT(kInvalidIndex);
  uint32_t light_subsurface_state_buffer ETX_INIT(kInvalidIndex);
  uint32_t path_capacity ETX_INIT(0u);
  uint32_t max_path_length ETX_INIT(0u);
  uint32_t camera_vertex_capacity ETX_INIT(0u);
  uint32_t light_vertex_capacity ETX_INIT(0u);
  uint32_t camera_fixed_max_bounces ETX_INIT(0u);
  uint32_t light_fixed_max_bounces ETX_INIT(0u);
  uint32_t reserved0 ETX_INIT(0u);
};
