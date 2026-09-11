#pragma once

#include "gpu_rt_wavefront_common.hlsl"
#include <interop/image_filter_shared.hxx>
#include <access/bsdf_resource_gpu.hxx>
#include <access/material_access_gpu.hxx>
#include <interop/bsdf_core_shared.hxx>
#include <interop/bsdf_resource_shared.hxx>
#include <interop/material.hxx>
#include <interop/scene_gpu_access_shared.hxx>

struct WavefrontConnectCameraPrepareInput {
  GPUWavefrontResources resources;
  uint task_index;
  uint path_index;
  GPUWavefrontPathState state;
  GPUWavefrontPathMeta path_meta;
  GPUWavefrontHit hit;
  GPUWavefrontPathVertex current_vertex;
  GPUWavefrontPathVertex previous_vertex;
  Camera camera;
  CameraFilmSampleShared camera_sample;
  float2 splat_uv;
  Material material;
};

BSDFResourceContext wavefront_connect_camera_make_scene_bsdf_resource_gpu_context() {
  return make_bsdf_resource_gpu_context(constants.scene.images, constants.scene.spectrums, constants.scene.spectral_values, constants.scene.energy_compensation_interfaces,
    constants.scene.scene_globals);
}

bool wavefront_connect_camera_try_load_material_full(uint material_index, out Material material) {
  MaterialAccessGPUContext material_context = {constants.scene.materials};
  return material_access_try_load_full(material_context, material_index, material);
}

BSDFData wavefront_connect_camera_make_surface_bsdf_data(Vertex vertex, SpectralQuery spect, uint medium_index, float3 incoming_direction) {
  return bsdf_data_make(vertex, spect, medium_index, PathSource::Light, incoming_direction);
}

Sampler wavefront_connect_camera_make_bsdf_sampler(uint seed) {
  Sampler result = ETX_ZERO(Sampler);
  result.seed = seed;
  return result;
}

#if ETX_UPBP
uint wavefront_connect_camera_upbp_evaluation_seed(WavefrontConnectCameraPrepareInput input_value, uint domain) {
  GPUUPBPResources upbp_resources = upbp_load_resources(input_value.resources);
  GPUUPBPPathState path_state = upbp_load_path_state(upbp_resources.path_state_buffer, upbp_path_state_index(upbp_resources, false, input_value.path_index));
  return upbp_deterministic_seed(path_state.global_path_index, 1u, path_state.path_length + 1u, domain);
}
#endif

bool wavefront_connect_camera_scene_mis_enabled() {
  SceneGPUSharedOptions options = scene_gpu_load_options(constants.scene.scene_options);
  return (options.properties_flags & (1u << SceneProperty::MultipleImportanceSampling)) != 0u;
}

bool wavefront_connect_camera_valid_spectral_response(SpectralResponse value) {
  return isfinite(value.value) && all(isfinite(value.integrated));
}

float wavefront_connect_camera_weight(WavefrontConnectCameraPrepareInput input_value, inout Sampler sampler) {
  if ((wavefront_connect_camera_scene_mis_enabled() == false) || scene_path_mode_is_light_tracing()) {
    return 1.0f;
  }

  float current_from_camera_dir = camera_shared_film_pdf_out(input_value.camera, input_value.camera_sample.position, input_value.current_vertex.position);
  float current_from_camera = wavefront_convert_solid_angle_pdf_to_area(current_from_camera_dir, input_value.camera_sample.position, input_value.current_vertex.position,
    wavefront_path_vertex_is_surface(input_value.current_vertex), input_value.current_vertex.normal);

  BSDFData reverse_data =
    wavefront_connect_camera_make_surface_bsdf_data(input_value.hit.vertex, input_value.state.spect, input_value.hit.medium_index, -input_value.camera_sample.direction);
  reverse_data.path_source = PathSource::Camera;
  float3 previous_direction = normalize(input_value.previous_vertex.position - input_value.current_vertex.position);
  float previous_from_current_dir =
    wavefront_connect_camera_stage_bsdf_pdf(wavefront_connect_camera_make_scene_bsdf_resource_gpu_context(), reverse_data, previous_direction, input_value.material, sampler);
  const float surface_factor = wavefront_vcm_surface_factor();
  float vm_camera = scene_path_mode_is_vcm() && (wavefront_path_vertex_is_medium(input_value.current_vertex) == false) ? surface_factor : 0.0f;
  float adjacent_connection = input_value.current_vertex.forward_pdf;
  if (scene_path_mode_uses_bdpt_fast() && (input_value.path_meta.light_path_length != 1u)) {
    adjacent_connection = 0.0f;
  }
  float w_light = current_from_camera * (vm_camera + adjacent_connection + wavefront_connection_mis(input_value.current_vertex, surface_factor) * previous_from_current_dir);
  return 1.0f / (1.0f + w_light);
}

bool wavefront_load_connect_camera_prepare_input(uint dispatch_index, out WavefrontConnectCameraPrepareInput input_value) {
  input_value = (WavefrontConnectCameraPrepareInput)0;
  input_value.resources = wavefront_load_resources();
  if ((input_value.resources.connect_camera_task_buffer == kInvalidIndex) || (input_value.resources.connect_camera_result_buffer == kInvalidIndex) ||
      (constants.camera_buffer_index == kInvalidIndex)) {
    return false;
  }

#if ETX_ENABLE_WORK_QUEUES
  const uint material_queue_count = wavefront_material_queue_count(input_value.resources, false, constants.work_queue_index);
  if (dispatch_index >= material_queue_count) {
    return false;
  }
  dispatch_index = wavefront_material_queue_load(input_value.resources, false, constants.work_queue_index, dispatch_index);
#endif
  input_value.task_index = dispatch_index;

  uint queue_descriptor = wavefront_queue_current_descriptor(false);
  uint queue_count = wavefront_queue_count(queue_descriptor);
  if (dispatch_index >= queue_count) {
    return false;
  }

  input_value.path_index = wavefront_queue_load(queue_descriptor, dispatch_index);
  input_value.state = wavefront_load_path_state(input_value.resources.light_state_buffer, input_value.path_index);
  input_value.path_meta = wavefront_load_path_meta(input_value.resources.path_meta_buffer, input_value.path_index);
  input_value.hit = wavefront_load_hit(input_value.resources.light_hit_buffer, input_value.path_index);
  if (scene_path_mode_is_path_tracing()) {
    return false;
  }
  if ((wavefront_hit_valid(input_value.hit) == false) || wavefront_hit_is_miss(input_value.hit) || (input_value.path_meta.light_path_length == 0u) ||
      ((input_value.state.reserved0 & GPUWavefrontPendingContinuationFlags::Prepared) == 0u)) {
    return false;
  }

  input_value.current_vertex =
    wavefront_load_path_vertex(input_value.resources.light_vertex_buffer, wavefront_light_vertex_slot(input_value.path_index, input_value.path_meta.light_path_length));
  input_value.previous_vertex =
    wavefront_load_path_vertex(input_value.resources.light_vertex_buffer, wavefront_light_vertex_slot(input_value.path_index, input_value.path_meta.light_path_length - 1u));
  if ((wavefront_path_vertex_valid(input_value.current_vertex) == false) || (wavefront_path_vertex_valid(input_value.previous_vertex) == false)) {
    return false;
  }
  if (wavefront_path_vertex_connectible(input_value.current_vertex) == false) {
    return false;
  }
  uint target_path_length = input_value.path_meta.light_path_length + 1u;
  if ((scene_strategy_enabled(kSceneStrategyConnectToCamera) == false) || (target_path_length < load_scene_options_min_path_length()) ||
      (target_path_length > load_scene_options_max_path_length())) {
    return false;
  }

  if (wavefront_connect_camera_try_load_material_full(input_value.current_vertex.material_index, input_value.material) == false) {
    return false;
  }
  input_value.camera = load_camera(bindless_buffers[NonUniformResourceIndex(constants.camera_buffer_index)]);
  if (input_value.camera.cls == Camera::Class::Equirectangular) {
    return false;
  }

  uint seed = input_value.state.sampler_seed;
#if ETX_UPBP
  const bool upbp = scene_path_mode_is_upbp();
  if (upbp) {
    seed = wavefront_connect_camera_upbp_evaluation_seed(input_value, kUPBPRandomDomainFilmConnection);
  }
#else
  const bool upbp = false;
#endif
  float2 lens_rnd = float2(0.0f, 0.0f);
  if (camera_lens_sampling_enabled(input_value.camera.lens_radius, input_value.camera.focal_distance)) {
    lens_rnd = float2(rnd01(seed), rnd01(seed));
  }
  float2 sensor_sample = camera_sample_lens_uv(input_value.camera, lens_rnd) * input_value.camera.lens_radius;
  float3 lens_point = camera_film_shared_lens_point(input_value.camera, sensor_sample);
  input_value.camera_sample = camera_film_shared_evaluate(input_value.camera, input_value.current_vertex.position, lens_point);
  if ((input_value.camera_sample.pdf_dir <= 0.0f) || (input_value.camera_sample.weight <= 0.0f)) {
    return false;
  }
  input_value.splat_uv = camera_sample_splat_uv(input_value.camera, input_value.camera_sample.uv, float2(rnd01(seed), rnd01(seed)));

  if (upbp == false) {
    input_value.state.sampler_seed = seed;
    wavefront_store_path_state(input_value.resources.light_state_buffer, input_value.path_index, input_value.state);
  }
  return true;
}

void wavefront_clear_connect_camera_task(uint dispatch_index) {
  GPUWavefrontResources resources = wavefront_load_resources();
  if (resources.connect_camera_task_buffer == kInvalidIndex) {
    return;
  }

  uint queue_descriptor = wavefront_queue_current_descriptor(false);
  uint queue_count = wavefront_queue_count(queue_descriptor);
  if (dispatch_index >= queue_count) {
    return;
  }

  GPUWavefrontConnectCameraTask empty_task = (GPUWavefrontConnectCameraTask)0;
  empty_task.medium_index = kInvalidIndex;
  wavefront_store_connect_camera_task(resources.connect_camera_task_buffer, dispatch_index, empty_task);
}

void wavefront_store_connect_camera_prepare_task(uint dispatch_index, WavefrontConnectCameraPrepareInput input_value, ETX_IN(BSDFEval, bsdf_eval), inout Sampler sampler) {
#if ETX_UPBP
  const bool upbp = scene_path_mode_is_upbp();
  if (upbp == false) {
    input_value.state.sampler_seed = sampler.seed;
    wavefront_store_path_state(input_value.resources.light_state_buffer, input_value.path_index, input_value.state);
  }
#else
  const bool upbp = false;
  input_value.state.sampler_seed = sampler.seed;
  wavefront_store_path_state(input_value.resources.light_state_buffer, input_value.path_index, input_value.state);
#endif

  if ((bsdf_eval_valid(bsdf_eval) == false) || (wavefront_connect_camera_valid_spectral_response(bsdf_eval.bsdf) == false)) {
    return;
  }

  uint pixel_index = 0u;
  if (wavefront_camera_ndc_to_pixel_index(input_value.camera, input_value.splat_uv, pixel_index) == false) {
    return;
  }

  float mis_weight = 1.0f;
  float scattering_pdf_reverse = 0.0f;
  float camera_area_density = 0.0f;
#if ETX_UPBP
  if (upbp) {
    BSDFData reverse_data =
      wavefront_connect_camera_make_surface_bsdf_data(input_value.hit.vertex, input_value.state.spect, input_value.hit.medium_index, -input_value.camera_sample.direction);
    reverse_data.path_source = PathSource::Camera;
    const float3 previous_direction = normalize(input_value.previous_vertex.position - input_value.current_vertex.position);
    scattering_pdf_reverse =
      wavefront_connect_camera_stage_bsdf_pdf(wavefront_connect_camera_make_scene_bsdf_resource_gpu_context(), reverse_data, previous_direction, input_value.material, sampler);
    camera_area_density = wavefront_convert_solid_angle_pdf_to_area(input_value.camera_sample.pdf_dir_out, input_value.camera_sample.position, input_value.current_vertex.position,
      wavefront_path_vertex_is_surface(input_value.current_vertex), input_value.current_vertex.geo_normal);
  } else {
    mis_weight = wavefront_connect_camera_weight(input_value, sampler);
  }
#else
  mis_weight = wavefront_connect_camera_weight(input_value, sampler);
#endif
  if (upbp == false) {
    input_value.state.sampler_seed = sampler.seed;
    wavefront_store_path_state(input_value.resources.light_state_buffer, input_value.path_index, input_value.state);
  }
  SpectralResponse contribution =
    spectral_response_mul(input_value.current_vertex.throughput, spectral_response_mul(bsdf_eval.bsdf, input_value.camera_sample.weight * mis_weight));
  if (wavefront_connect_camera_valid_spectral_response(contribution) == false) {
    return;
  }
  float direction_scale = camera_shared_clip_direction_scale(input_value.camera, input_value.camera_sample.direction);
  float near_extent = (input_value.camera.clip_near > 0.0f) ? input_value.camera.clip_near / direction_scale : 0.0f;
  float3 shadow_origin = wavefront_surface_shading_position(input_value.hit, input_value.camera_sample.direction);
  float surface_len = length(input_value.camera_sample.position - shadow_origin);
  float3 clip_pos = shadow_origin + input_value.camera_sample.direction * max(0.0f, surface_len - near_extent);
  float3 shadow_delta = clip_pos - shadow_origin;
  float shadow_distance = length(shadow_delta);
  if (shadow_distance <= kRayEpsilon) {
    return;
  }

  GPUWavefrontConnectCameraTask task = (GPUWavefrontConnectCameraTask)0;
  task.shadow_ray.o = shadow_origin;
  task.shadow_ray.d = shadow_delta / shadow_distance;
  task.shadow_ray.min_t = kRayEpsilon;
  task.shadow_ray.max_t = shadow_distance;
  task.shadow_target = clip_pos;
  task.contribution = contribution;
  task.mis_weight = upbp ? bsdf_eval.pdf : mis_weight;
  task.upbp_scattering_pdf_reverse_bits = asuint(scattering_pdf_reverse);
  task.upbp_camera_area_density_bits = asuint(camera_area_density);
  task.pixel_index = pixel_index;
  task.medium_index = ((bsdf_eval.properties & BSDFSample::MediumChanged) != 0u) ? bsdf_eval.medium_index : input_value.hit.medium_index;
  task.flags = GPUWavefrontPointConnectionTaskFlags::Ready;
  task.flags |= (input_value.current_vertex.flags & GPUWavefrontVertexFlags::Medium) != 0u ? GPUWavefrontPointConnectionTaskFlags::SourceMedium : 0u;
  task.path_index = input_value.path_index;
#if ETX_UPBP
  if (upbp) {
    task.sampler_seed = wavefront_connect_camera_upbp_evaluation_seed(input_value, kUPBPRandomDomainIntersectionTraversal);
    task.upbp_auxiliary1_bits = wavefront_connect_camera_upbp_evaluation_seed(input_value, kUPBPRandomDomainConnectionTransmittance);
  } else
#endif
  {
    task.sampler_seed = sampler.seed;
  }
  wavefront_store_connect_camera_task(input_value.resources.connect_camera_task_buffer, dispatch_index, task);
  wavefront_shadow_queue_append(input_value.resources, kGPUWavefrontShadowQueueConnectCamera, dispatch_index);
}
