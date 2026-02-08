#pragma once

#include "interop.hxx"
#include "camera.hxx"
#include "spectrum.hxx"
#include "material.hxx"

struct ETX_ALIGNED GPUSceneGlobals {
  enum : uint32_t {
    MaxEnvironmentEmitters = 63u,
  };

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

  uint32_t environment_emitters[MaxEnvironmentEmitters];
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

  // TODO: add packed scene options once GPU options ABI is finalized.
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

  // TODO: Image contains pointer-based views and nested distributions. Split into packed image metadata + raw tables.
  uint32_t images;

  // TODO: Medium contains pointer-based density views. Split into packed medium metadata + density grids.
  uint32_t mediums;

  // TODO: Distribution contains ArrayView pointer; upload packed Distribution::Entry[] and compact metadata.
  uint32_t emitters_distribution;

  // TODO: Replace bool-based Scene::Options layout with packed uint flags/options for GPU ABI stability.
  uint32_t scene_options;
};

#if defined(__cplusplus)
static_assert(std::is_standard_layout_v<GPUSceneGlobals>, "GPUSceneGlobals must stay standard layout for C++/HLSL interop");
static_assert(std::is_standard_layout_v<GPUScene>, "GPUScene must stay standard layout for C++/HLSL interop");
static_assert(alignof(GPUSceneGlobals) == 16, "GPUSceneGlobals alignment must match HLSL packing");
static_assert(alignof(GPUScene) == 16, "GPUScene alignment must match HLSL packing");
static_assert(sizeof(GPUSceneGlobals) == 384, "GPUSceneGlobals size changed; update shared ABI");
static_assert(sizeof(GPUScene) == 64, "GPUScene size changed; update shared ABI");
#endif
