#pragma once

#include "gpu_rt_wavefront_surface_common.hlsl"

void wavefront_surface_continue_prepare_specialized(bool from_camera, uint dispatch_index) {
  uint queue_descriptor = wavefront_queue_current_descriptor(from_camera);
  uint queue_count = wavefront_queue_count(queue_descriptor);
  if (dispatch_index >= queue_count) {
    return;
  }

  GPUWavefrontResources resources = wavefront_load_resources();
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

  uint vertex_descriptor = from_camera ? resources.camera_vertex_buffer : resources.light_vertex_buffer;
  uint current_vertex_index = wavefront_vertex_slot(path_index, state.path_length);
  uint previous_vertex_index = wavefront_vertex_slot(path_index, state.path_length - 1u);
  GPUWavefrontPathVertex current_vertex = wavefront_load_path_vertex(vertex_descriptor, current_vertex_index);
  GPUWavefrontPathVertex previous_vertex = wavefront_load_path_vertex(vertex_descriptor, previous_vertex_index);
  if ((wavefront_path_vertex_valid(current_vertex) == false) || (wavefront_path_vertex_valid(previous_vertex) == false)) {
    state.flags = 0u;
    wavefront_store_path_state(state_descriptor, path_index, state);
    return;
  }

  GPUWavefrontPathMeta meta = wavefront_load_path_meta(resources.path_meta_buffer, path_index);
  bool current_connectible = sample_valid ? (bsdf_sample_is_delta(bsdf_sample) == false) : true;
  bool previous_connectible = wavefront_path_vertex_connectible(previous_vertex);
  uint current_medium_index = (sample_valid && ((bsdf_sample.properties & BSDFSample::MediumChanged) != 0u)) ? bsdf_sample.medium_index : state.medium_index;
  float current_d_vcm = current_vertex.forward_pdf;
  float current_d_vc = current_vertex.reverse_pdf;

  current_vertex.sampled_bsdf_pdf = bsdf_sample.pdf;
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

  float3 reverse_direction = sample_valid ? bsdf_sample.w_o : float3(0.0f, 0.0f, 0.0f);
  float reverse_bsdf_pdf =
    wavefront_surface_continue_stage_reverse_bsdf_pdf(make_scene_bsdf_resource_gpu_context(), bsdf_data, reverse_direction, material, bsdf_sampler);
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
    float scale = (state.path_length > 1u) ? previous_vertex.pdf_ratio : 1.0f;
    previous_vertex.pdf_accumulated = meta.light_mis_history * scale;
    meta.light_mis_history = previous_vertex.pdf_accumulated;
  }

  state.sampler_seed = bsdf_sampler.seed;

  wavefront_store_path_vertex(vertex_descriptor, previous_vertex_index, previous_vertex);
  wavefront_store_path_vertex(vertex_descriptor, current_vertex_index, current_vertex);
  wavefront_store_path_meta(resources.path_meta_buffer, path_index, meta);
  state.medium_index = current_medium_index;

  bool continue_path = sample_valid && ((state.path_length + 1u) <= resources.max_path_length);
  state.reserved0 = GPUWavefrontPendingContinuationFlags::Prepared;
  if (continue_path) {
    float cos_theta_bsdf = abs(dot(hit.vertex.nrm, bsdf_sample.w_o));
    if (bsdf_sample_is_delta(bsdf_sample)) {
      state.forward_pdf = 0.0f;
      state.reverse_pdf = current_d_vc * cos_theta_bsdf;
    } else {
      state.forward_pdf = wavefront_safe_div(1.0f, bsdf_sample.pdf);
      state.reverse_pdf = wavefront_safe_div(cos_theta_bsdf * ((current_d_vc * reverse_bsdf_pdf) + current_d_vcm), bsdf_sample.pdf);
    }
    SpectralResponse next_throughput = spectral_response_mul(state.throughput, bsdf_sample.weight);
    if (from_camera == false) {
      float shading_fix = bsdf_fix_shading_normal(hit.geo_normal, hit.vertex.nrm, state.ray.d, bsdf_sample.w_o);
      if (isfinite(shading_fix) && (shading_fix > 0.0f)) {
        next_throughput = spectral_response_mul(next_throughput, shading_fix);
      }
    }

    state.throughput = next_throughput;
    state.sampled_bsdf_pdf = bsdf_sample.pdf;
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
