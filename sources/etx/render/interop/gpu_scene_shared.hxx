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
  uint32_t active_emitter_count;
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

  uint32_t pixel_filter_image_index;
  float pixel_filter_radius;
  uint32_t pixel_filter_pad0;
  uint32_t pixel_filter_pad1;
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
  uint32_t path_mode;
  uint32_t random_seed;
  uint32_t pad0;
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
