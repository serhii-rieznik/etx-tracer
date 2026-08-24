#pragma once

#include "gpu_rt_wavefront_surface_common.hlsl"

#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE)
# include "gpu_rt_wavefront_trace_common.hlsl"
#endif

#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE)
void wavefront_subsurface_remap_channel(float color, float scattering_distance, out float albedo, out float extinction, out float scattering) {
  const float a = 1.826052378200f;
  const float b = 4.985111943850f + 0.12735595943800f;
  const float c = 1.096861024240f;
  const float d = 0.496310210422f;
  const float e = 4.231902997010f + 0.00310603949088f;
  const float f = 2.406029994080f;
  const float k_min_scattering = 1.0f / 1024.0f;

  color = max(0.0f, color);
  float blend = pow(color, 0.25f);
  albedo = (1.0f - blend) * a * pow(atan(b * color), c) + blend * d * pow(atan(e * color), f);
  albedo = clamp(albedo, 0.0f, 1.0f - kEpsilon);
  extinction = 1.0f / max(scattering_distance, k_min_scattering);
  scattering = extinction * albedo;
}

void wavefront_subsurface_remap(SpectralQuery spect, SpectralResponse color, SpectralResponse distances, out SpectralResponse albedo, out SpectralResponse extinction,
  out SpectralResponse scattering) {
  if (spectral_query_is_spectral(spect)) {
    float albedo_value = 0.0f;
    float extinction_value = 0.0f;
    float scattering_value = 0.0f;
    wavefront_subsurface_remap_channel(color.value, distances.value, albedo_value, extinction_value, scattering_value);
    albedo = spectral_response_make(spect, albedo_value);
    extinction = spectral_response_make(spect, extinction_value);
    scattering = spectral_response_make(spect, scattering_value);
    return;
  }

  float3 albedo_secondary = float3(0.0f, 0.0f, 0.0f);
  float3 extinction_secondary = float3(0.0f, 0.0f, 0.0f);
  float3 scattering_secondary = float3(0.0f, 0.0f, 0.0f);
  wavefront_subsurface_remap_channel(color.integrated.x, distances.integrated.x, albedo_secondary.x, extinction_secondary.x, scattering_secondary.x);
  wavefront_subsurface_remap_channel(color.integrated.y, distances.integrated.y, albedo_secondary.y, extinction_secondary.y, scattering_secondary.y);
  wavefront_subsurface_remap_channel(color.integrated.z, distances.integrated.z, albedo_secondary.z, extinction_secondary.z, scattering_secondary.z);

  albedo = spectral_response_make(spect, albedo_secondary);
  extinction = spectral_response_make(spect, extinction_secondary);
  scattering = spectral_response_make(spect, scattering_secondary);
}

bool wavefront_subsurface_random_walk_applicable(Material material, BSDFSample bsdf_sample) {
  return (material.subsurface_cls != SubsurfaceMaterial::Disabled) && ((bsdf_sample.properties & BSDFSample::Reflection) != 0u) &&
         ((bsdf_sample.properties & BSDFSample::Diffuse) != 0u);
}
#endif

uint wavefront_load_path_vertex_flags_only(uint descriptor_index, uint vertex_index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  if (wavefront_path_vertex_descriptor_is_light(descriptor_index)) {
    const uint packed_flags = buffer.Load(vertex_index * kGPUWavefrontLightPathVertexStride + kGPUWavefrontLightPathVertexFlagsOffset);
    return packed_flags & kGPUWavefrontLightPathVertexFlagsMask;
  }

  return buffer.Load(vertex_index * kGPUWavefrontPathVertexStride + kGPUWavefrontPathVertexFlagsOffset);
}

void wavefront_store_path_vertex_pdf_from_next_only(uint descriptor_index, uint vertex_index, float pdf_from_next) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  if (wavefront_path_vertex_descriptor_is_light(descriptor_index)) {
    buffer.Store(vertex_index * kGPUWavefrontLightPathVertexStride + kGPUWavefrontLightPathVertexPdfFromNextOffset, asuint(pdf_from_next));
    return;
  }

  buffer.Store(vertex_index * kGPUWavefrontPathVertexStride + kGPUWavefrontPathVertexPdfFromNextOffset, asuint(pdf_from_next));
}

void wavefront_surface_continue_prepare_specialized(bool from_camera, uint dispatch_index) {
  if ((constants.dispatch_item_count != 0u) && (dispatch_index >= constants.dispatch_item_count)) {
    return;
  }

  dispatch_index += constants.dispatch_item_offset;
  GPUWavefrontResources resources = wavefront_load_resources();
#if ETX_ENABLE_WORK_QUEUES
  const uint material_queue_count = wavefront_material_queue_count(resources, from_camera, constants.work_queue_index);
  if (dispatch_index >= material_queue_count) {
    return;
  }
  dispatch_index = wavefront_material_queue_load(resources, from_camera, constants.work_queue_index, dispatch_index);
#endif
  uint queue_descriptor = wavefront_queue_current_descriptor(from_camera);
  uint queue_count = wavefront_queue_count(queue_descriptor);
  if (dispatch_index >= queue_count) {
    return;
  }

  uint path_index = wavefront_queue_load(queue_descriptor, dispatch_index);
  uint state_descriptor = from_camera ? resources.camera_state_buffer : resources.light_state_buffer;
  uint hit_descriptor = from_camera ? resources.camera_hit_buffer : resources.light_hit_buffer;
  GPUWavefrontPathState state = wavefront_load_path_state(state_descriptor, path_index);
  GPUWavefrontHit hit = wavefront_load_hit(hit_descriptor, path_index);
  if ((wavefront_path_state_valid(state) == false) || (wavefront_hit_valid(hit) == false) || wavefront_hit_is_miss(hit)) {
    return;
  }
  if (wavefront_hit_is_medium(hit)) {
    return;
  }

  Material material = (Material)0;
  if (try_load_material_full(hit.material_index, material) == false) {
    state.reserved0 = 0u;
    state.flags = 0u;
    wavefront_store_path_state(state_descriptor, path_index, state);
    return;
  }

  if (gpu_bsdf_sample_supported_class(material.cls) == false) {
    state.reserved0 = 0u;
    state.flags = 0u;
    wavefront_store_path_state(state_descriptor, path_index, state);
    return;
  }

  if (wavefront_surface_continue_stage_matches_material(material.cls) == false) {
    return;
  }

  Sampler bsdf_sampler = make_bsdf_sampler(state.sampler_seed);
  float2 bsdf_rnd = float2(rnd01(bsdf_sampler.seed), rnd01(bsdf_sampler.seed));
  float2 connection_rnd = float2(rnd01(bsdf_sampler.seed), rnd01(bsdf_sampler.seed));
  float2 support_rnd = float2(rnd01(bsdf_sampler.seed), rnd01(bsdf_sampler.seed));
  if (from_camera && (state.path_length == 1u)) {
    if (sample_use_blue_noise_primary(constants.sample_index, kSamplerStreamBSDF)) {
      uint bsdf_dimension = sampler_stream_dimension_base(kSamplerStreamBSDF);
      bsdf_rnd = float2(sample_blue_noise_value(state.pixel, constants.sample_index, bsdf_dimension + 0u),
        sample_blue_noise_value(state.pixel, constants.sample_index, bsdf_dimension + 1u));
    }
    if (sample_use_blue_noise_primary(constants.sample_index, kSamplerStreamConnection)) {
      uint connection_dimension = sampler_stream_dimension_base(kSamplerStreamConnection);
      connection_rnd = float2(sample_blue_noise_value(state.pixel, constants.sample_index, connection_dimension + 0u),
        sample_blue_noise_value(state.pixel, constants.sample_index, connection_dimension + 1u));
    }
    if (sample_use_blue_noise_primary(constants.sample_index, kSamplerStreamSupport)) {
      uint support_dimension = sampler_stream_dimension_base(kSamplerStreamSupport);
      support_rnd = float2(sample_blue_noise_value(state.pixel, constants.sample_index, support_dimension + 0u),
        sample_blue_noise_value(state.pixel, constants.sample_index, support_dimension + 1u));
    }
  }
  state.film_uv = connection_rnd;
  state.last_emitter_pdf = support_rnd.y;

  BSDFData bsdf_data = make_surface_bsdf_data(hit.vertex, state.spect, state.medium_index, state.ray.d);
  if (from_camera == false) {
    bsdf_data.path_source = PathSource::Light;
  }
  bsdf_sampler_push_fixed(bsdf_sampler, bsdf_rnd.x, bsdf_rnd.y, support_rnd.x);
  BSDFSample bsdf_sample = wavefront_surface_continue_stage_bsdf_sample(make_scene_bsdf_resource_gpu_context(), bsdf_data, material, bsdf_sampler);
  bsdf_sampler_pop_fixed(bsdf_sampler);
  bool sample_valid = bsdf_sample_valid(bsdf_sample);
  bool sample_direction_valid = sample_valid ? gpu_valid_direction(bsdf_sample.w_o) : true;
  bool sample_finite = sample_direction_valid && gpu_valid_spectral_response(bsdf_sample.weight) && isfinite(bsdf_sample.pdf) && isfinite(bsdf_sample.eta);
  if (sample_finite == false) {
    state.reserved0 = 0u;
    state.flags = 0u;
    wavefront_store_path_state(state_descriptor, path_index, state);
    return;
  }

  bool subsurface_medium_walk = false;
  GPUWavefrontSubsurfaceState subsurface_state = (GPUWavefrontSubsurfaceState)0;
#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE)
  if (sample_valid && wavefront_subsurface_random_walk_applicable(material, bsdf_sample)) {
    ByteAddressBuffer scene_globals = bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)];
    uint scatter_material_index = scene_gpu_load_u32(scene_globals, kSceneGlobalsDefaultSubsurfaceScatterMaterialOffset);
    Material scatter_material = (Material)0;
    if (try_load_material_full(scatter_material_index, scatter_material) == false) {
      state.reserved0 = 0u;
      state.flags = 0u;
      wavefront_store_path_state(state_descriptor, path_index, state);
      return;
    }

    subsurface_state.material_index = hit.material_index;
    subsurface_state.medium_index = material.int_medium;
    subsurface_state.scatter_material_index = scatter_material_index;
    subsurface_state.flags = GPUWavefrontSubsurfaceFlags::Active;
    subsurface_state.phase_function_g = 0.0f;

    if (material.int_medium == kInvalidIndex) {
      SpectralResponse color = apply_image(state.spect, material.scattering, hit.vertex.tex);
      SpectralResponse distances = apply_image(state.spect, material.subsurface, hit.vertex.tex);
      wavefront_subsurface_remap(state.spect, color, distances, subsurface_state.albedo, subsurface_state.extinction, subsurface_state.scattering);
      subsurface_state.flags |= GPUWavefrontSubsurfaceFlags::InlineMedium;
    } else {
      MediumAccess medium_access = (MediumAccess)0;
      if (wavefront_try_load_medium(material.int_medium, medium_access) == false) {
        state.reserved0 = 0u;
        state.flags = 0u;
        wavefront_store_path_state(state_descriptor, path_index, state);
        return;
      }
      subsurface_state.phase_function_g = medium_access.phase_function_g;
      subsurface_state.scattering = gpu_medium_scattering(medium_access, state.spect);
      SpectralResponse absorption = gpu_medium_absorption(medium_access, state.spect);
      subsurface_state.extinction = spectral_response_add(subsurface_state.scattering, absorption);
      subsurface_state.albedo = medium_sample_shared_calculate_albedo(state.spect, subsurface_state.scattering, subsurface_state.extinction);
    }

    uint subsurface_state_buffer = wavefront_subsurface_state_buffer(resources, from_camera);
    if ((subsurface_state_buffer == kInvalidIndex) || spectral_response_is_zero(subsurface_state.extinction)) {
      state.reserved0 = 0u;
      state.flags = 0u;
      wavefront_store_path_state(state_descriptor, path_index, state);
      return;
    }

    float3 subsurface_direction = (material.subsurface_path == SubsurfaceMaterial::DiffusePath)
                                    ? sample_cosine_distribution(float2(rnd01(bsdf_sampler.seed), rnd01(bsdf_sampler.seed)), -hit.vertex.nrm, 1.0f)
                                    : normalize(state.ray.d);
    float subsurface_pdf = abs(dot(subsurface_direction, hit.vertex.nrm)) * kInvPi;
    if ((gpu_valid_direction(subsurface_direction) == false) || (subsurface_pdf <= 0.0f)) {
      state.reserved0 = 0u;
      state.flags = 0u;
      wavefront_store_path_state(state_descriptor, path_index, state);
      return;
    }

    wavefront_store_subsurface_state(subsurface_state_buffer, path_index, subsurface_state);
    bsdf_sample.w_o = subsurface_direction;
    bsdf_sample.weight = spectral_response_make(state.spect, 1.0f);
    bsdf_sample.pdf = subsurface_pdf;
    bsdf_sample.eta = 1.0f;
    bsdf_sample.medium_index = material.int_medium;
    bsdf_sample.properties = BSDFSample::Transmission | BSDFSample::Diffuse | BSDFSample::MediumChanged;
    hit.material_index = scatter_material_index;
    material = scatter_material;
    subsurface_medium_walk = true;
  }
#endif

  uint vertex_descriptor = from_camera ? resources.camera_vertex_buffer : resources.light_vertex_buffer;
  uint current_vertex_index = wavefront_path_vertex_slot(from_camera, path_index, state.path_length);
  uint previous_vertex_index = wavefront_path_vertex_slot(from_camera, path_index, state.path_length - 1u);
  GPUWavefrontPathVertex current_vertex = wavefront_load_path_vertex(vertex_descriptor, current_vertex_index);
#if ETX_WAVEFRONT_PATH_TRACING_ONLY
  uint previous_vertex_flags = wavefront_load_path_vertex_flags_only(vertex_descriptor, previous_vertex_index);
  if ((wavefront_path_vertex_valid(current_vertex) == false) || ((previous_vertex_flags & GPUWavefrontVertexFlags::Valid) == 0u)) {
#else
  GPUWavefrontPathVertex previous_vertex = wavefront_load_path_vertex(vertex_descriptor, previous_vertex_index);
  if ((wavefront_path_vertex_valid(current_vertex) == false) || (wavefront_path_vertex_valid(previous_vertex) == false)) {
#endif
    state.flags = 0u;
    wavefront_store_path_state(state_descriptor, path_index, state);
    return;
  }

#if ETX_WAVEFRONT_PATH_TRACING_ONLY == 0
  GPUWavefrontPathMeta meta = wavefront_load_path_meta(resources.path_meta_buffer, path_index);
#endif
  bool current_connectible = bsdf_sample_is_delta(bsdf_sample) == false;
#if ETX_WAVEFRONT_PATH_TRACING_ONLY
  bool previous_connectible = (previous_vertex_flags & GPUWavefrontVertexFlags::Connectible) != 0u;
#else
  bool previous_connectible = wavefront_path_vertex_connectible(previous_vertex);
#endif
  uint current_medium_index = (sample_valid && ((bsdf_sample.properties & BSDFSample::MediumChanged) != 0u)) ? bsdf_sample.medium_index : state.medium_index;
  float current_d_vcm = current_vertex.forward_pdf;
  float current_d_vc = current_vertex.reverse_pdf;
  float current_d_vm = current_vertex.d_vm;

  if (subsurface_medium_walk) {
    current_vertex.material_index = hit.material_index;
    current_vertex.medium_index = bsdf_sample.medium_index;
    current_vertex.flags |= GPUWavefrontVertexFlags::Subsurface;
    if ((subsurface_state.flags & GPUWavefrontSubsurfaceFlags::InlineMedium) != 0u) {
      current_vertex.inline_medium_extinction = subsurface_state.extinction;
      current_vertex.inline_medium_flags = GPUWavefrontSubsurfaceFlags::InlineMedium;
    }
  }

  float3 selected_direction = sample_valid ? bsdf_sample.w_o : float3(0.0f, 0.0f, 0.0f);

  float selected_sample_pdf = bsdf_sample.pdf;
  current_vertex.sampled_bsdf_pdf = selected_sample_pdf;
  current_vertex.medium_index = current_medium_index;
  current_vertex.flags &= ~(GPUWavefrontVertexFlags::Connectible | GPUWavefrontVertexFlags::Mis_connectible | GPUWavefrontVertexFlags::Delta);
  if (current_connectible) {
    current_vertex.flags |= GPUWavefrontVertexFlags::Connectible;
    if (previous_connectible) {
      current_vertex.flags |= GPUWavefrontVertexFlags::Mis_connectible;
    }
  } else {
    current_vertex.flags |= GPUWavefrontVertexFlags::Delta;
  }

#if ETX_WAVEFRONT_PATH_TRACING_ONLY
  float reverse_bsdf_pdf = 0.0f;
#else
  float3 reverse_direction = selected_direction;
  float reverse_bsdf_pdf = wavefront_surface_continue_stage_reverse_bsdf_pdf(make_scene_bsdf_resource_gpu_context(), bsdf_data, reverse_direction, material, bsdf_sampler);
  previous_vertex.pdf_from_next = wavefront_vertex_to_vertex_area_pdf(reverse_bsdf_pdf, current_vertex, previous_vertex);
  if ((from_camera == false) && (state.path_length == 1u) && (previous_vertex.emitter_index != kInvalidIndex)) {
    GPUEmitterInstanceABIData emitter_instance = (GPUEmitterInstanceABIData)0;
    if (try_load_emitter_instance(previous_vertex.emitter_index, emitter_instance) && (emitter_instance.emitter_class != EmitterClass::Area)) {
      previous_vertex.pdf_from_next = reverse_bsdf_pdf;
    }
  }

  if (from_camera) {
    wavefront_surface_precompute_camera_mis(current_connectible, state.path_length, meta, previous_vertex);
  } else {
    previous_vertex.pdf_ratio = wavefront_safe_div(previous_vertex.pdf_from_next, previous_vertex.pdf_from_prev);
    previous_vertex.pdf_history = meta.light_mis_history;
    if (scene_path_mode_uses_bdpt_fast()) {
      float scale = (state.path_length > 1u) ? previous_vertex.pdf_ratio : 1.0f;
      previous_vertex.pdf_accumulated = meta.light_mis_history * scale;
    } else {
      float previous_mis_connectible = wavefront_path_vertex_mis_connectible(previous_vertex) ? 1.0f : 0.0f;
      previous_vertex.pdf_accumulated = previous_vertex.pdf_ratio * (previous_mis_connectible + meta.light_mis_history);
    }
    meta.light_mis_history = previous_vertex.pdf_accumulated;
  }
#endif

  state.sampler_seed = bsdf_sampler.seed;

#if ETX_WAVEFRONT_PATH_TRACING_ONLY
  wavefront_store_path_vertex_pdf_from_next_only(vertex_descriptor, previous_vertex_index, 0.0f);
#else
  wavefront_store_path_vertex(vertex_descriptor, previous_vertex_index, previous_vertex);
#endif
  wavefront_store_path_vertex(vertex_descriptor, current_vertex_index, current_vertex);
#if ETX_WAVEFRONT_PATH_TRACING_ONLY == 0
  if ((from_camera == false) && (state.path_length == 1u) && (resources.fast_light_endpoint_buffer != kInvalidIndex)) {
    wavefront_store_fast_light_endpoint(resources.fast_light_endpoint_buffer, path_index, previous_vertex, current_vertex);
  }
  wavefront_store_path_meta(resources.path_meta_buffer, path_index, meta);
#endif
  state.medium_index = current_medium_index;

  bool continue_path = sample_valid && ((state.path_length + 1u) <= resources.max_path_length);
  state.reserved0 = GPUWavefrontPendingContinuationFlags::Prepared;
  if (continue_path) {
    float cos_theta_bsdf = abs(dot(hit.vertex.nrm, bsdf_sample.w_o));
    if (bsdf_sample_is_delta(bsdf_sample)) {
      state.forward_pdf = 0.0f;
      state.reverse_pdf = current_d_vc * cos_theta_bsdf;
      state.d_vm = scene_path_mode_is_vcm() ? (current_d_vm * cos_theta_bsdf) : 0.0f;
    } else {
      state.forward_pdf = wavefront_safe_div(1.0f, selected_sample_pdf);
      float vcm_connection_term = scene_path_mode_is_vcm() ? constants.vcm_vm_weight : 0.0f;
      state.reverse_pdf = wavefront_safe_div(cos_theta_bsdf * ((current_d_vc * reverse_bsdf_pdf) + current_d_vcm + vcm_connection_term), selected_sample_pdf);
      state.d_vm = scene_path_mode_is_vcm()
                     ? wavefront_safe_div(cos_theta_bsdf * ((current_d_vm * reverse_bsdf_pdf) + (current_d_vcm * constants.vcm_vc_weight) + 1.0f), selected_sample_pdf)
                     : 0.0f;
    }
    SpectralResponse next_throughput = spectral_response_mul(state.throughput, bsdf_sample.weight);
    if (from_camera == false) {
      float shading_fix = bsdf_fix_shading_normal(hit.geo_normal, hit.vertex.nrm, state.ray.d, bsdf_sample.w_o);
      next_throughput = spectral_response_mul(next_throughput, shading_fix);
    }

    state.throughput = next_throughput;
    state.sampled_bsdf_pdf = selected_sample_pdf;
    if (from_camera) {
      state.eta *= bsdf_sample.eta;
    }
    state.eta_scale *= abs(bsdf_sample.eta);
    state.medium_index = current_medium_index;
    state.ray.o = wavefront_surface_shading_position(hit, bsdf_sample.w_o);
    state.ray.d = normalize(bsdf_sample.w_o);
    state.ray.min_t = kRayEpsilon;
    state.ray.max_t = kMaxFloat;
    state.reserved0 |= GPUWavefrontPendingContinuationFlags::Continue;
  }

  wavefront_store_path_state(state_descriptor, path_index, state);
}
