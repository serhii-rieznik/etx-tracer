#pragma once

#include "interop.hxx"
#include "camera.hxx"
#include "spectrum.hxx"
#include "material.hxx"
#include "gpu_abi_constants.hxx"

struct ETX_ALIGNED GPUSceneGlobals {
  uint32_t vertex_count;
  uint32_t triangle_count;
  uint32_t mesh_count;
  uint32_t emitter_profile_count;

  uint32_t emitter_instance_count;
  uint32_t environment_emitter_count;
  uint32_t pad0;
  uint32_t pad1;

  float3 bounding_sphere_center;
  float bounding_sphere_radius;

  float3 bounding_box_min;
  float pad2;

  float3 bounding_box_max;
  float pad3;

  uint32_t environment_emitters[SceneLimits::MaxEnvironmentEmitters];
  uint32_t environment_emitters_pad;

  uint32_t default_black_spectrum;
  uint32_t default_white_spectrum;
  uint32_t default_rayleigh_spectrum;
  uint32_t default_mie_spectrum;

  uint32_t default_ozone_spectrum;
  uint32_t default_subsurface_scatter_material;
  uint32_t default_subsurface_exit_material;
  uint32_t default_missing_material;

  uint32_t default_dielectric_eta;
  uint32_t default_conductor_eta;
  uint32_t default_conductor_k;
  uint32_t defaults_pad0;
};

struct ETX_ALIGNED GPUSceneOptions {
  uint32_t min_path_length;
  uint32_t max_path_length;
  uint32_t samples;
  uint32_t random_path_termination;

  float noise_threshold;
  float radiance_clamp;
  uint32_t strategy_flags;
  uint32_t light_sampling;

  uint32_t properties_flags;
  uint32_t pad0;
  uint32_t pad1;
  uint32_t pad2;
};

struct ETX_ALIGNED GPUImageBlobHeader {
  uint32_t image_count;
  uint32_t images_offset;
  uint32_t data_chunk_count;
  uint32_t data_chunk_indices_offset;
};

struct ETX_ALIGNED GPUMediumBlobHeader {
  uint32_t medium_count;
  uint32_t mediums_offset;
  uint32_t data_chunk_count;
  uint32_t data_chunk_indices_offset;
};

struct ETX_ALIGNED GPUScene {
  // Safe to upload now (linear arrays / packed globals).
  uint32_t vertex_positions;
  uint32_t vertex_normals;
  uint32_t vertex_tangents;
  uint32_t vertex_bitangents;
  uint32_t vertex_texcoords;
  uint32_t triangles;
  uint32_t meshes;
  uint32_t emitter_profiles;
  uint32_t emitter_instances;
  uint32_t scene_globals;
  uint32_t materials;
  uint32_t spectrums;

  // Packed image metadata blob: GPUImageBlobHeader + ::Image[] + uint32_t[data_chunk_count] descriptor indices.
  uint32_t images;

  // Packed medium metadata blob: GPUMediumBlobHeader + ::Medium[] + uint32_t[data_chunk_count] descriptor indices.
  uint32_t mediums;

  // Packed Distribution::Entry[] (active emitters + sentinel).
  uint32_t emitters_distribution;

  uint32_t scene_options;
};

#if defined(__cplusplus)
static_assert(std::is_standard_layout_v<GPUSceneGlobals>, "GPUSceneGlobals must stay standard layout for C++/HLSL interop");
static_assert(std::is_standard_layout_v<GPUSceneOptions>, "GPUSceneOptions must stay standard layout for C++/HLSL interop");
static_assert(std::is_standard_layout_v<GPUImageBlobHeader>, "GPUImageBlobHeader must stay standard layout for C++/HLSL interop");
static_assert(std::is_standard_layout_v<GPUMediumBlobHeader>, "GPUMediumBlobHeader must stay standard layout for C++/HLSL interop");
static_assert(std::is_standard_layout_v<GPUScene>, "GPUScene must stay standard layout for C++/HLSL interop");
static_assert(alignof(GPUSceneGlobals) == 16, "GPUSceneGlobals alignment must match HLSL packing");
static_assert(alignof(GPUSceneOptions) == 16, "GPUSceneOptions alignment must match HLSL packing");
static_assert(alignof(GPUImageBlobHeader) == 16, "GPUImageBlobHeader alignment must match HLSL packing");
static_assert(alignof(GPUMediumBlobHeader) == 16, "GPUMediumBlobHeader alignment must match HLSL packing");
static_assert(alignof(GPUScene) == 16, "GPUScene alignment must match HLSL packing");
static_assert(sizeof(GPUSceneGlobals) == 384, "GPUSceneGlobals size changed; update shared ABI");
static_assert(sizeof(GPUSceneOptions) == 48, "GPUSceneOptions size changed; update shared ABI");
static_assert(sizeof(GPUImageBlobHeader) == 16, "GPUImageBlobHeader size changed; update shared ABI");
static_assert(sizeof(GPUMediumBlobHeader) == 16, "GPUMediumBlobHeader size changed; update shared ABI");
static_assert(sizeof(GPUScene) == 64, "GPUScene size changed; update shared ABI");
#endif
