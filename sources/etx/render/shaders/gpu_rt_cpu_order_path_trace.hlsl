#include "gpu_rt_wavefront_trace_common.hlsl"
#include "gpu_rt_wavefront_surface_common.hlsl"

[noinline] SpectralResponse cpu_order_environment_direct_hit(SpectralQuery spect, SpectralResponse throughput, float3 direction, uint path_length, float sampled_bsdf_pdf,
  bool mis_weight_enabled) {
  if (scene_strategy_enabled(kSceneStrategyDirectHit) == false) {
    return spectral_response_zero(spect);
  }

  const bool directly_visible = path_length == 1u;
  SpectralResponse accumulated = spectral_response_zero(spect);
  const uint environment_count = wavefront_environment_emitter_count();
  for (uint local_index = 0u; local_index < environment_count; ++local_index) {
    uint emitter_index = kInvalidIndex;
    if (wavefront_environment_emitter_index(local_index, emitter_index) == false) {
      continue;
    }

    float local_pdf_dir = 0.0f;
    SpectralResponse value = wavefront_environment_direct_hit_emitter_radiance(emitter_index, direction, spect, directly_visible, local_pdf_dir);
    if ((local_pdf_dir <= 0.0f) || spectral_response_is_zero(value)) {
      continue;
    }

    const bool no_weight = (scene_multiple_importance_sampling_enabled() == false) || directly_visible || (mis_weight_enabled == false);
    const float pdf_emitter_discrete = emitter_discrete_pdf(emitter_index);
    const float weight = no_weight ? 1.0f : power_heuristic(sampled_bsdf_pdf, pdf_emitter_discrete * local_pdf_dir);
    accumulated = spectral_response_add(accumulated, spectral_response_mul(value, weight));
  }

  return spectral_response_mul(throughput, accumulated);
}

[noinline] SpectralResponse cpu_order_local_direct_hit(SpectralQuery spect, SpectralResponse throughput, TraceSurfaceResult hit, float3 source_position, float3 incoming_direction,
  uint path_length, float sampled_bsdf_pdf, bool mis_weight_enabled) {
  if ((scene_strategy_enabled(kSceneStrategyDirectHit) == false) || (hit.triangle_index == kInvalidIndex) || (hit.emitter_index == kInvalidIndex)) {
    return spectral_response_zero(spect);
  }

  if (dot(hit.tri.geo_n, incoming_direction) >= 0.0f) {
    return spectral_response_zero(spect);
  }

  float pdf_area = 0.0f;
  float pdf_dir = 0.0f;
  float pdf_dir_out = 0.0f;
  const bool directly_visible = path_length == 1u;
  SpectralResponse emission = wavefront_evaluate_local_direct_hit_radiance(hit.emitter_index, spect, source_position, hit.surface_point.vertex.pos,
    hit.surface_point.vertex.tex, directly_visible, pdf_area, pdf_dir, pdf_dir_out);
  (void)pdf_area;
  (void)pdf_dir_out;
  if ((pdf_dir <= 0.0f) || spectral_response_is_zero(emission)) {
    return spectral_response_zero(spect);
  }

  const bool no_weight = (scene_multiple_importance_sampling_enabled() == false) || directly_visible || (mis_weight_enabled == false);
  const float pdf_emitter_discrete = emitter_discrete_pdf(hit.emitter_index);
  const float weight = no_weight ? 1.0f : power_heuristic(sampled_bsdf_pdf, pdf_emitter_discrete * pdf_dir);
  return spectral_response_mul(throughput, spectral_response_mul(emission, weight));
}

[noinline] void cpu_order_accumulate(uint pixel_index, SpectralQuery spect, SpectralResponse value) {
  if (gpu_valid_spectral_response(value)) {
    wavefront_film_add(pixel_index, spectral_response_to_rgb(value) * wavefront_spectral_weight(spect));
  }
}

[noinline] bool cpu_order_bsdf_supported_class(uint material_class) {
  switch (material_class) {
    case MaterialClass::Diffuse:
    case MaterialClass::Translucent:
    case MaterialClass::Mirror:
    case MaterialClass::Plastic:
    case MaterialClass::Thinfilm:
    case MaterialClass::Boundary:
    case MaterialClass::Velvet:
    case MaterialClass::Void:
      return true;
    default:
      return false;
  }
}

[noinline] BSDFSample cpu_order_bsdf_sample(BSDFData data, Material material, inout Sampler sampler) {
  BSDFResourceContext context = make_scene_bsdf_resource_gpu_context();
  switch (material.cls) {
    case MaterialClass::Diffuse:
      return bsdf_diffuse_sample(context, data, material, sampler);
    case MaterialClass::Translucent:
      return bsdf_translucent_sample(context, data, material, sampler);
    case MaterialClass::Mirror:
      return bsdf_mirror_sample(context, data, material, sampler);
    case MaterialClass::Plastic:
      return bsdf_plastic_sample(context, data, material, sampler);
    case MaterialClass::Thinfilm:
      return bsdf_thinfilm_sample(context, data, material, sampler);
    case MaterialClass::Boundary:
      return bsdf_boundary_sample(context, data, material, sampler);
    case MaterialClass::Velvet:
      return bsdf_velvet_sample(context, data, material, sampler);
    case MaterialClass::Void:
      return bsdf_void_sample(context, data, material, sampler);
    default:
      return bsdf_sample_zero(data.spectrum_sample);
  }
}

[noinline] float cpu_order_bsdf_pdf(BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  BSDFResourceContext context = make_scene_bsdf_resource_gpu_context();
  switch (material.cls) {
    case MaterialClass::Diffuse:
      return bsdf_diffuse_pdf(context, data, outgoing_direction, material, sampler);
    case MaterialClass::Translucent:
      return bsdf_translucent_pdf(context, data, outgoing_direction, material, sampler);
    case MaterialClass::Mirror:
      return bsdf_mirror_pdf(context, data, outgoing_direction, material, sampler);
    case MaterialClass::Plastic:
      return bsdf_plastic_pdf(context, data, outgoing_direction, material, sampler);
    case MaterialClass::Thinfilm:
      return bsdf_thinfilm_pdf(context, data, outgoing_direction, material, sampler);
    case MaterialClass::Boundary:
      return bsdf_boundary_pdf(context, data, outgoing_direction, material, sampler);
    case MaterialClass::Velvet:
      return bsdf_velvet_pdf(context, data, outgoing_direction, material, sampler);
    case MaterialClass::Void:
      return bsdf_void_pdf(context, data, outgoing_direction, material, sampler);
    default:
      return 0.0f;
  }
}

[noinline] float cpu_order_vertex_to_vertex_area_pdf(float pdf_dir, GPUWavefrontPathVertex from_vertex, GPUWavefrontPathVertex to_vertex) {
  if (wavefront_path_vertex_is_infinite_emitter(to_vertex)) {
    return pdf_dir;
  }
  return wavefront_convert_solid_angle_pdf_to_area(pdf_dir, from_vertex.position, to_vertex.position, wavefront_path_vertex_is_surface(to_vertex), to_vertex.normal);
}

[noinline] void cpu_order_surface_precompute_camera_mis(bool current_connectible, uint path_length, inout GPUWavefrontPathMeta meta,
  inout GPUWavefrontPathVertex previous_vertex) {
  previous_vertex.pdf_ratio = wavefront_safe_div(previous_vertex.pdf_from_next, previous_vertex.pdf_from_prev);
  previous_vertex.pdf_history = meta.camera_mis_history;
  if (path_length == 1u) {
    previous_vertex.pdf_accumulated = current_connectible ? meta.camera_mis_history : 0.0f;
  } else {
    previous_vertex.pdf_accumulated = meta.camera_mis_history * previous_vertex.pdf_ratio;
  }
  meta.camera_mis_history = previous_vertex.pdf_accumulated;
}

[noinline] float2 cpu_order_first_surface_blue_noise(uint2 pixel, uint stream, uint dimension_offset, float2 fallback_sample) {
  if (sample_use_blue_noise_primary(constants.sample_index, stream) == false) {
    return fallback_sample;
  }

  const uint dimension_base = sampler_stream_dimension_base(stream);
  return float2(sample_blue_noise_value(pixel, constants.sample_index, dimension_base + dimension_offset),
    sample_blue_noise_value(pixel, constants.sample_index, dimension_base + dimension_offset + 1u));
}

[noinline] void cpu_order_finalize_output(uint pixel_index, uint2 output_pixel) {
  float4 accumulated = wavefront_film_load(pixel_index);
  float sample_count = float(max(1u, constants.sample_index + 1u));
  bindless_storage_textures[NonUniformResourceIndex(constants.output_image_index)][output_pixel] = float4(max(accumulated.xyz / sample_count, float3(0.0f, 0.0f, 0.0f)), 1.0f);
}

[noinline] void cpu_order_clear_direct_light_slot(uint path_index) {
  GPUWavefrontResources resources = wavefront_load_resources();
  if (resources.camera_state_buffer != kInvalidIndex) {
    GPUWavefrontPathState state = (GPUWavefrontPathState)0;
    wavefront_store_path_state(resources.camera_state_buffer, path_index, state);
  }
  if (resources.camera_hit_buffer != kInvalidIndex) {
    GPUWavefrontHit hit = (GPUWavefrontHit)0;
    wavefront_store_hit(resources.camera_hit_buffer, path_index, hit);
  }
  if (resources.path_meta_buffer != kInvalidIndex) {
    GPUWavefrontPathMeta meta = (GPUWavefrontPathMeta)0;
    wavefront_store_path_meta(resources.path_meta_buffer, path_index, meta);
  }
}

[noinline] void cpu_order_store_first_bounce_direct_light_state(uint path_index, Camera camera, Ray primary_ray, uint2 camera_space_pixel, SpectralQuery spect,
  TraceSurfaceResult trace_hit, SpectralResponse throughput, uint input_medium_index, uint output_medium_index, uint sampler_seed, float2 connection_rnd,
  float support_random_y, Material material, BSDFSample bsdf_sample_value, inout Sampler bsdf_sampler) {
  GPUWavefrontResources resources = wavefront_load_resources();
  if ((resources.camera_state_buffer == kInvalidIndex) || (resources.camera_hit_buffer == kInvalidIndex) || (resources.camera_vertex_buffer == kInvalidIndex) ||
      (resources.path_meta_buffer == kInvalidIndex)) {
    return;
  }

  GPUWavefrontPathState state = (GPUWavefrontPathState)0;
  state.ray = primary_ray;
  state.throughput = throughput;
  state.eta = 1.0f;
  state.eta_scale = 1.0f;
  state.sampled_bsdf_pdf = camera_film_shared_evaluate_out(camera, primary_ray).pdf_dir;
  state.forward_pdf = wavefront_safe_div(1.0f, state.sampled_bsdf_pdf);
  state.reverse_pdf = 0.0f;
  state.medium_index = input_medium_index;
  state.path_length = 1u;
  state.pixel_index = path_index;
  state.flags = GPUWavefrontPathFlags::Valid | GPUWavefrontPathFlags::Connectible | GPUWavefrontPathFlags::From_camera;
  state.path_source = PathSource::Camera;
  state.sampler_seed = sampler_seed;
  state.pixel = camera_space_pixel;
  state.spect = spect;
  state.film_uv = connection_rnd;
  state.last_emitter_pdf = support_random_y;
  state.last_vertex_index = wavefront_camera_vertex_slot(path_index, 1u);

  GPUWavefrontHit hit = (GPUWavefrontHit)0;
  hit.transmittance = trace_hit.transmittance;
  hit.vertex = trace_hit.surface_point.vertex;
  hit.geo_normal = trace_hit.surface_point.geo_normal;
  hit.hit_t = trace_hit.hit_t;
  hit.triangle_index = trace_hit.triangle_index;
  hit.material_index = trace_hit.tri.material_index;
  hit.emitter_index = trace_hit.emitter_index;
  hit.medium_index = input_medium_index;
  hit.flags = GPUWavefrontHitFlags::Valid;
  hit.barycentric = trace_hit.surface_point.barycentrics.yz;

  GPUWavefrontPathMeta meta = (GPUWavefrontPathMeta)0;
  meta.camera_path_length = 1u;
  meta.camera_mis_history = scene_path_mode_uses_bdpt_fast() ? 1.0f : 0.0f;
  meta.flags = GPUWavefrontPathMetaFlags::Camera_active;

  wavefront_store_path_state(resources.camera_state_buffer, path_index, state);
  wavefront_store_hit(resources.camera_hit_buffer, path_index, hit);
  wavefront_write_root_camera_vertex(path_index, camera, primary_ray, spect, path_index);
  wavefront_write_vertex(true, path_index, state, hit);

  const uint current_vertex_index = wavefront_camera_vertex_slot(path_index, 1u);
  const uint previous_vertex_index = wavefront_camera_vertex_slot(path_index, 0u);
  GPUWavefrontPathVertex current_vertex = wavefront_load_path_vertex(resources.camera_vertex_buffer, current_vertex_index);
  GPUWavefrontPathVertex previous_vertex = wavefront_load_path_vertex(resources.camera_vertex_buffer, previous_vertex_index);
  if ((wavefront_path_vertex_valid(current_vertex) == false) || (wavefront_path_vertex_valid(previous_vertex) == false)) {
    state.flags = 0u;
    wavefront_store_path_state(resources.camera_state_buffer, path_index, state);
    wavefront_store_path_meta(resources.path_meta_buffer, path_index, meta);
    return;
  }

  current_vertex.pdf_from_prev = cpu_order_vertex_to_vertex_area_pdf(state.sampled_bsdf_pdf, previous_vertex, current_vertex);
  const float cos_to_prev = abs(dot(hit.vertex.nrm, -state.ray.d));
  if (cos_to_prev > 0.0f) {
    current_vertex.forward_pdf = wavefront_safe_div(state.forward_pdf * hit.hit_t * hit.hit_t, cos_to_prev);
    current_vertex.reverse_pdf = wavefront_safe_div(state.reverse_pdf, cos_to_prev);
    state.forward_pdf = current_vertex.forward_pdf;
    state.reverse_pdf = current_vertex.reverse_pdf;
  }

  const bool current_connectible = bsdf_sample_is_delta(bsdf_sample_value) == false;
  const bool previous_connectible = wavefront_path_vertex_connectible(previous_vertex);
  const float current_d_vcm = current_vertex.forward_pdf;
  const float current_d_vc = current_vertex.reverse_pdf;
  current_vertex.sampled_bsdf_pdf = bsdf_sample_value.pdf;
  current_vertex.medium_index = output_medium_index;
  current_vertex.flags &= ~(GPUWavefrontVertexFlags::Connectible | GPUWavefrontVertexFlags::Mis_connectible | GPUWavefrontVertexFlags::Delta);
  if (current_connectible) {
    current_vertex.flags |= GPUWavefrontVertexFlags::Connectible;
    if (previous_connectible) {
      current_vertex.flags |= GPUWavefrontVertexFlags::Mis_connectible;
    }
  } else {
    current_vertex.flags |= GPUWavefrontVertexFlags::Delta;
  }

  BSDFData bsdf_data = make_surface_bsdf_data(hit.vertex, state.spect, input_medium_index, state.ray.d);
  const float reverse_bsdf_pdf = cpu_order_bsdf_pdf(bsdf_data, bsdf_sample_value.w_o, material, bsdf_sampler);
  previous_vertex.pdf_from_next = cpu_order_vertex_to_vertex_area_pdf(reverse_bsdf_pdf, current_vertex, previous_vertex);
  cpu_order_surface_precompute_camera_mis(current_connectible, state.path_length, meta, previous_vertex);

  state.sampler_seed = bsdf_sampler.seed;
  state.medium_index = output_medium_index;
  state.sampled_bsdf_pdf = bsdf_sample_value.pdf;
  state.eta *= bsdf_sample_value.eta;
  state.eta_scale *= abs(bsdf_sample_value.eta);
  if (current_connectible == false) {
    state.forward_pdf = 0.0f;
    state.reverse_pdf = current_d_vc * abs(dot(hit.vertex.nrm, bsdf_sample_value.w_o));
  } else {
    state.forward_pdf = wavefront_safe_div(1.0f, bsdf_sample_value.pdf);
    state.reverse_pdf = wavefront_safe_div(abs(dot(hit.vertex.nrm, bsdf_sample_value.w_o)) * ((current_d_vc * reverse_bsdf_pdf) + current_d_vcm), bsdf_sample_value.pdf);
  }

  wavefront_store_path_vertex(resources.camera_vertex_buffer, previous_vertex_index, previous_vertex);
  wavefront_store_path_vertex(resources.camera_vertex_buffer, current_vertex_index, current_vertex);
  wavefront_store_path_meta(resources.path_meta_buffer, path_index, meta);
  wavefront_store_path_state(resources.camera_state_buffer, path_index, state);
}

[numthreads(8, 8, 1)] void gpu_cpu_order_path_trace_main(uint3 dtid : SV_DispatchThreadID) {
  if (constants.camera_buffer_index == kInvalidIndex) {
    return;
  }

  Camera camera = load_camera(bindless_buffers[NonUniformResourceIndex(constants.camera_buffer_index)]);
  if (wavefront_render_window_contains(dtid.xy) == false) {
    return;
  }

  uint2 output_pixel = wavefront_output_pixel(dtid.xy);
  uint output_pixel_index = output_pixel.x + output_pixel.y * camera.film_size.x;
  uint queue_slot = wavefront_render_window_local_index(dtid.xy);
  wavefront_queue_store(wavefront_queue_current_descriptor(true), queue_slot, output_pixel_index);
  if ((dtid.x == 0u) && (dtid.y == 0u)) {
    uint2 render_window_size = wavefront_render_window_size();
    WAVEFRONT_RW_BUFFER(wavefront_queue_current_descriptor(true)).Store(0u, render_window_size.x * render_window_size.y);
  }
  cpu_order_clear_direct_light_slot(output_pixel_index);

  if (constants.sample_index == 0u) {
    wavefront_film_store(output_pixel_index, float4(0.0f, 0.0f, 0.0f, 0.0f));
  }

  uint2 camera_space_pixel = uint2(output_pixel.x, camera.film_size.y - 1u - output_pixel.y);
  uint seed_pixel_index = camera_space_pixel.x + camera_space_pixel.y * camera.film_size.x;
  uint seed = scene_random_seed(seed_pixel_index, constants.sample_index);
  SpectralQuery spect = spectral_query_sample();
  if (scene_uses_spectral_mode()) {
    spect = spectral_query_spectral_sample(rnd01(seed));
  }

  float2 uv_sample = float2(rnd01(seed), rnd01(seed));
  float2 uv = camera_sample_film_uv(output_pixel, camera.film_size, uv_sample);
  float2 lens_rnd = float2(rnd01(seed), rnd01(seed));
  Ray ray = camera_generate_primary_ray(camera, uv, lens_rnd);
  Ray primary_ray = ray;

  GPUWavefrontResources resources = wavefront_load_resources();
  SpectralResponse throughput = spectral_response_make(spect, 1.0f);
  uint medium_index = camera.medium_index;
  uint path_length = 1u;
  float eta = 1.0f;
  float sampled_bsdf_pdf = 0.0f;
  bool mis_weight_enabled = true;
  bool depth_limit_reached_while_refractive = false;

  [loop] for (uint depth = 0u; depth < resources.max_path_length; ++depth) {
    const bool depth_exceeded = path_length > load_scene_options_max_path_length();
    const bool neutral_eta = gpu_path_tracing_neutral_eta(eta);
    const bool exceeded_refractive_depth = depth_exceeded && (neutral_eta == false);
    const bool exceeded_neutral_depth = depth_exceeded && neutral_eta;
    if (exceeded_refractive_depth) {
      depth_limit_reached_while_refractive = true;
    }
    if (exceeded_neutral_depth && (depth_limit_reached_while_refractive == false)) {
      break;
    }

    RayDesc ray_desc = (RayDesc)0;
    ray_desc.Origin = ray.o;
    ray_desc.Direction = ray.d;
    ray_desc.TMin = max(kRayEpsilon, ray.min_t);
    ray_desc.TMax = max(ray_desc.TMin + kRayEpsilon, ray.max_t);

    TraceSurfaceResult hit = (TraceSurfaceResult)0;
    bool found_hit = wavefront_trace_path_state(ray_desc, spect, throughput, medium_index, seed, hit);
    throughput = spectral_response_mul(throughput, hit.transmittance);
    if ((gpu_valid_spectral_response(throughput) == false) || spectral_response_is_zero(throughput)) {
      break;
    }

    if (found_hit == false) {
      cpu_order_accumulate(output_pixel_index, spect, cpu_order_environment_direct_hit(spect, throughput, ray.d, path_length, sampled_bsdf_pdf, mis_weight_enabled));
      break;
    }

    if (hit.triangle_index == kInvalidIndex) {
      break;
    }

    cpu_order_accumulate(output_pixel_index, spect,
      cpu_order_local_direct_hit(spect, throughput, hit, ray.o, ray.d, path_length, sampled_bsdf_pdf, mis_weight_enabled));
    if (exceeded_neutral_depth) {
      break;
    }

    Material material = hit.material;
    if ((cpu_order_bsdf_supported_class(material.cls) == false) || (material.subsurface_cls != SubsurfaceMaterial::Disabled)) {
      break;
    }

    Sampler bsdf_sampler = make_bsdf_sampler(seed);
    float2 bsdf_rnd = float2(rnd01(bsdf_sampler.seed), rnd01(bsdf_sampler.seed));
    float2 connection_rnd = float2(rnd01(bsdf_sampler.seed), rnd01(bsdf_sampler.seed));
    float2 support_rnd = float2(rnd01(bsdf_sampler.seed), rnd01(bsdf_sampler.seed));
    if (path_length == 1u) {
      bsdf_rnd = cpu_order_first_surface_blue_noise(camera_space_pixel, kSamplerStreamBSDF, 0u, bsdf_rnd);
      connection_rnd = cpu_order_first_surface_blue_noise(camera_space_pixel, kSamplerStreamConnection, 0u, connection_rnd);
      support_rnd = cpu_order_first_surface_blue_noise(camera_space_pixel, kSamplerStreamSupport, 0u, support_rnd);
    }
    (void)connection_rnd;

    BSDFData bsdf_data = make_surface_bsdf_data(hit.surface_point.vertex, spect, medium_index, ray.d);
    bsdf_sampler_push_fixed(bsdf_sampler, bsdf_rnd.x, bsdf_rnd.y, support_rnd.x);
    BSDFSample bsdf_sample_value = cpu_order_bsdf_sample(bsdf_data, material, bsdf_sampler);
    bsdf_sampler_pop_fixed(bsdf_sampler);
    seed = bsdf_sampler.seed;
    if ((bsdf_sample_valid(bsdf_sample_value) == false) || (gpu_valid_direction(bsdf_sample_value.w_o) == false) ||
        (gpu_valid_spectral_response(bsdf_sample_value.weight) == false)) {
      break;
    }

    const uint input_medium_index = medium_index;
    uint output_medium_index = medium_index;
    if ((bsdf_sample_value.properties & BSDFSample::MediumChanged) != 0u) {
      output_medium_index = bsdf_sample_value.medium_index;
    }
    if (path_length == 1u) {
      cpu_order_store_first_bounce_direct_light_state(output_pixel_index, camera, primary_ray, camera_space_pixel, spect, hit, throughput, input_medium_index,
        output_medium_index, bsdf_sampler.seed, connection_rnd, support_rnd.y, material, bsdf_sample_value, bsdf_sampler);
    }
    medium_index = output_medium_index;

    throughput = spectral_response_mul(throughput, bsdf_sample_value.weight);
    if ((gpu_valid_spectral_response(throughput) == false) || spectral_response_is_zero(throughput)) {
      break;
    }

    eta *= bsdf_sample_value.eta;
    sampled_bsdf_pdf = bsdf_sample_value.pdf;
    mis_weight_enabled = bsdf_sample_is_delta(bsdf_sample_value) == false;
    ray.o = wavefront_trace_surface_shading_position(hit, bsdf_sample_value.w_o);
    ray.d = normalize(bsdf_sample_value.w_o);
    ray.min_t = kRayEpsilon;
    ray.max_t = kMaxFloat;
    path_length += 1u;
    if (gpu_random_continue(path_length, load_scene_options_random_path_termination(), eta, seed, throughput) == false) {
      break;
    }
  }

  cpu_order_finalize_output(output_pixel_index, output_pixel);
}
