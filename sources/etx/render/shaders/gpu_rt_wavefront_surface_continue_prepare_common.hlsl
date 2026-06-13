#pragma once

#include "gpu_rt_wavefront_surface_common.hlsl"

#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE)
#include "gpu_rt_wavefront_trace_common.hlsl"
#endif

struct WavefrontSubsurfaceGather {
  GPUWavefrontHit hit;
  SpectralResponse weight;
  float3 incoming_direction;
  uint valid;
};

#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE)
float wavefront_subsurface_safe_mul_float(float a, float b) {
  if ((a == 0.0f) || (b == 0.0f)) {
    return 0.0f;
  }

  return a * b;
}

SpectralResponse wavefront_subsurface_safe_mul_spectral(SpectralQuery spect, SpectralResponse a, SpectralResponse b) {
  if (spectral_query_is_spectral(spect)) {
    return spectral_response_make(spect, wavefront_subsurface_safe_mul_float(a.value, b.value));
  }

  return spectral_response_make(spect, float3(wavefront_subsurface_safe_mul_float(a.integrated.x, b.integrated.x),
                                        wavefront_subsurface_safe_mul_float(a.integrated.y, b.integrated.y),
                                        wavefront_subsurface_safe_mul_float(a.integrated.z, b.integrated.z)));
}

float wavefront_subsurface_response_component(SpectralResponse value, uint channel) {
  if (spectral_response_is_spectral(value)) {
    return value.value;
  }

  if (channel == 0u) {
    return value.integrated.x;
  }
  if (channel == 1u) {
    return value.integrated.y;
  }
  return value.integrated.z;
}

float wavefront_subsurface_response_sum(SpectralResponse value) {
  if (spectral_response_is_spectral(value)) {
    return value.value;
  }

  return value.integrated.x + value.integrated.y + value.integrated.z;
}

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

  float3 albedo_rgb = float3(0.0f, 0.0f, 0.0f);
  float3 extinction_rgb = float3(0.0f, 0.0f, 0.0f);
  float3 scattering_rgb = float3(0.0f, 0.0f, 0.0f);
  wavefront_subsurface_remap_channel(color.integrated.x, distances.integrated.x, albedo_rgb.x, extinction_rgb.x, scattering_rgb.x);
  wavefront_subsurface_remap_channel(color.integrated.y, distances.integrated.y, albedo_rgb.y, extinction_rgb.y, scattering_rgb.y);
  wavefront_subsurface_remap_channel(color.integrated.z, distances.integrated.z, albedo_rgb.z, extinction_rgb.z, scattering_rgb.z);
  albedo = spectral_response_make(spect, albedo_rgb);
  extinction = spectral_response_make(spect, extinction_rgb);
  scattering = spectral_response_make(spect, scattering_rgb);
}

bool wavefront_trace_material(RayDesc ray, uint material_index, inout uint seed, out TraceSurfaceResult result) {
  result = (TraceSurfaceResult)0;
  result.triangle_index = kInvalidIndex;
  result.emitter_index = kInvalidIndex;
  result.hit_t = ray.TMax;

  bool has_geometry_buffers = (constants.scene.triangles != kInvalidIndex) && (constants.scene.vertex_positions != kInvalidIndex) &&
                              (constants.scene.vertex_normals != kInvalidIndex) && (constants.scene.scene_globals != kInvalidIndex);
  if (has_geometry_buffers == false) {
    return false;
  }

  SceneGPUSharedGlobals scene_globals_data = scene_gpu_load_globals(bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)]);
  uint vertex_count = scene_globals_data.vertex_count;
  uint triangle_count = scene_globals_data.triangle_count;
  bool has_texcoords = constants.scene.vertex_texcoords != kInvalidIndex;

  RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> ray_query;
  ray_query.TraceRayInline(bindless_accel_structs[NonUniformResourceIndex(constants.as_index)], RAY_FLAG_FORCE_NON_OPAQUE, 0xFF, ray);

  while (ray_query.Proceed()) {
    if (ray_query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) {
      continue;
    }

    uint candidate_triangle_index = ray_query.CandidatePrimitiveIndex();
    if (candidate_triangle_index >= triangle_count) {
      continue;
    }

    TriangleData tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], candidate_triangle_index);
    if (tri.material_index != material_index) {
      continue;
    }

    bool valid_indices = (tri.i.x < vertex_count) && (tri.i.y < vertex_count) && (tri.i.z < vertex_count);
    if (valid_indices == false) {
      continue;
    }

    Material material = (Material)0;
    if (try_load_material_full(tri.material_index, material) == false) {
      continue;
    }
    if (material.cls == MaterialClass::Void) {
      continue;
    }

    float2 candidate_bary = ray_query.CandidateTriangleBarycentrics();
    float2 candidate_uv = float2(0.0f, 0.0f);
    if (has_texcoords) {
      candidate_uv = interpolate_uv(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_texcoords)], tri, candidate_bary);
    }
    if (alpha_test_pass(tri.material_index, candidate_uv, seed)) {
      continue;
    }

    ray_query.CommitNonOpaqueTriangleHit();
  }

  if (ray_query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
    return false;
  }

  result.triangle_index = ray_query.CommittedPrimitiveIndex();
  result.hit_t = ray_query.CommittedRayT();
  result.tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], result.triangle_index);
  result.surface_point = wavefront_load_surface_point_compact(result.tri, ray_query.CommittedTriangleBarycentrics(), ray.Direction);
  result.emitter_index = result.tri.emitter_index;
  try_load_material_full(result.tri.material_index, result.material);
  result.hit = 1u;
  return true;
}

GPUWavefrontHit wavefront_subsurface_make_exit_hit(TraceSurfaceResult trace_result, uint exit_material_index, SpectralQuery spect) {
  GPUWavefrontHit result = (GPUWavefrontHit)0;
  result.transmittance = spectral_response_make(spect, 1.0f);
  result.vertex = trace_result.surface_point.vertex;
  result.geo_normal = trace_result.surface_point.geo_normal;
  result.hit_t = trace_result.hit_t;
  result.triangle_index = trace_result.triangle_index;
  result.material_index = exit_material_index;
  result.emitter_index = kInvalidIndex;
  result.medium_index = kInvalidIndex;
  result.flags = GPUWavefrontHitFlags::Valid;
  result.barycentric = trace_result.surface_point.barycentrics.yz;
  return result;
}

bool wavefront_subsurface_gather_random_walk(Material material, GPUWavefrontHit in_hit, SpectralQuery spect, float3 incoming_direction, inout uint seed,
  out WavefrontSubsurfaceGather gather_result) {
  gather_result = (WavefrontSubsurfaceGather)0;
  const uint max_iterations = 1024u;

  float anisotropy = 0.0f;
  SpectralResponse extinction = spectral_response_zero(spect);
  SpectralResponse scattering = spectral_response_zero(spect);
  SpectralResponse albedo = spectral_response_zero(spect);

  if (material.int_medium == kInvalidIndex) {
    SpectralResponse color = apply_image(spect, material.scattering, in_hit.vertex.tex);
    SpectralResponse distances = apply_image(spect, material.subsurface, in_hit.vertex.tex);
    wavefront_subsurface_remap(spect, color, distances, albedo, extinction, scattering);
  } else {
    MediumAccess medium_access = (MediumAccess)0;
    if (wavefront_try_load_medium(material.int_medium, medium_access) == false) {
      return false;
    }
    anisotropy = medium_access.phase_function_g;
    scattering = gpu_medium_scattering(medium_access, spect);
    SpectralResponse absorption = gpu_medium_absorption(medium_access, spect);
    extinction = spectral_response_add(scattering, absorption);
    albedo = medium_sample_shared_calculate_albedo(spect, scattering, extinction);
  }

  RayDesc ray = (RayDesc)0;
  ray.Direction = (material.subsurface_path == SubsurfaceMaterial::DiffusePath) ? sample_cosine_distribution(float2(rnd01(seed), rnd01(seed)), -in_hit.vertex.nrm, 1.0f)
                                                                                : normalize(incoming_direction);
  ray.Origin = wavefront_surface_shading_position(in_hit, ray.Direction);
  ray.TMin = kRayEpsilon;
  ray.TMax = kMaxFloat;

  SpectralResponse throughput = spectral_response_make(spect, 1.0f);
  for (uint i = 0u; i < max_iterations; ++i) {
    SpectralResponse pdf = spectral_response_zero(spect);
    uint channel = medium_sample_shared_sample_spectrum_component(spect, albedo, throughput, rnd01(seed), pdf);
    float scattering_distance = wavefront_subsurface_response_component(extinction, channel);
    ray.TMax = (scattering_distance > 0.0f) ? (-log(1.0f - rnd01(seed)) / scattering_distance) : kMaxFloat;

    if ((i == 0u) && (ray.TMax <= kRayEpsilon)) {
      return false;
    }

    TraceSurfaceResult trace_result = (TraceSurfaceResult)0;
    bool intersection_found = wavefront_trace_material(ray, in_hit.material_index, seed, trace_result);
    if (intersection_found) {
      ray.TMax = trace_result.hit_t;
    }

    SpectralResponse tr = spectral_response_exp(spectral_response_mul(extinction, -ray.TMax));
    if (intersection_found) {
      pdf = spectral_response_mul(pdf, tr);
    } else {
      pdf = spectral_response_mul(pdf, wavefront_subsurface_safe_mul_spectral(spect, tr, extinction));
    }
    if (spectral_response_is_zero(pdf)) {
      return false;
    }

    SpectralResponse weight = tr;
    if (intersection_found == false) {
      weight = wavefront_subsurface_safe_mul_spectral(spect, tr, scattering);
    }
    throughput = spectral_response_mul(throughput, spectral_response_div(weight, max(kEpsilon, wavefront_subsurface_response_sum(pdf))));
    if (spectral_response_maximum(throughput) <= kEpsilon) {
      return false;
    }

    if (intersection_found) {
      ByteAddressBuffer scene_globals = bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)];
      uint exit_material_index = scene_gpu_load_u32(scene_globals, kSceneGlobalsDefaultSubsurfaceExitMaterialOffset);
      gather_result.hit = wavefront_subsurface_make_exit_hit(trace_result, exit_material_index, spect);
      gather_result.weight = throughput;
      bool incoming_inside = dot(ray.Direction, trace_result.surface_point.vertex.nrm) > 0.0f;
      gather_result.incoming_direction = incoming_inside ? -ray.Direction : ray.Direction;
      gather_result.valid = 1u;
      return true;
    }

    float3 previous_direction = ray.Direction;
    ray.Origin = ray.Origin + ray.Direction * ray.TMax;
    ray.Direction = medium_phase_shared_sample_henyey_greenstein(previous_direction, anisotropy, float2(rnd01(seed), rnd01(seed)));
  }

  return false;
}

bool wavefront_subsurface_random_walk_applicable(Material material, BSDFSample bsdf_sample) {
  return (material.subsurface_cls == SubsurfaceMaterial::RandomWalk) && ((bsdf_sample.properties & BSDFSample::Reflection) != 0u) &&
         ((bsdf_sample.properties & BSDFSample::Diffuse) != 0u);
}
#endif

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

  bool subsurface_random_walk = false;
  WavefrontSubsurfaceGather subsurface_gather = (WavefrontSubsurfaceGather)0;
#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE)
  if (from_camera && scene_path_mode_is_path_tracing() && sample_valid && wavefront_subsurface_random_walk_applicable(material, bsdf_sample)) {
    uint subsurface_seed = bsdf_sampler.seed;
    subsurface_random_walk = wavefront_subsurface_gather_random_walk(material, hit, state.spect, state.ray.d, subsurface_seed, subsurface_gather);
    bsdf_sampler.seed = subsurface_seed;
    if (subsurface_random_walk == false) {
      state.reserved0 = 0u;
      state.flags = 0u;
      wavefront_store_path_state(state_descriptor, path_index, state);
      return;
    }
    hit = subsurface_gather.hit;
    if (try_load_material_full(hit.material_index, material) == false) {
      state.reserved0 = 0u;
      state.flags = 0u;
      wavefront_store_path_state(state_descriptor, path_index, state);
      return;
    }
    wavefront_store_hit(hit_descriptor, path_index, hit);
  }
#endif

  uint vertex_descriptor = from_camera ? resources.camera_vertex_buffer : resources.light_vertex_buffer;
  uint current_vertex_index = wavefront_path_vertex_slot(from_camera, path_index, state.path_length);
  uint previous_vertex_index = wavefront_path_vertex_slot(from_camera, path_index, state.path_length - 1u);
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

  if (subsurface_random_walk) {
    current_vertex.throughput = spectral_response_mul(state.throughput, subsurface_gather.weight);
    current_vertex.position = hit.vertex.pos;
    current_vertex.triangle_index = hit.triangle_index;
    current_vertex.normal = hit.vertex.nrm;
    current_vertex.material_index = hit.material_index;
    current_vertex.geo_normal = hit.geo_normal;
    current_vertex.medium_index = state.medium_index;
    current_vertex.w_i = subsurface_gather.incoming_direction;
    current_vertex.emitter_index = kInvalidIndex;
    current_vertex.texcoord = hit.vertex.tex;
    current_vertex.flags = GPUWavefrontVertexFlags::Valid | GPUWavefrontVertexFlags::Surface | GPUWavefrontVertexFlags::From_camera |
                           GPUWavefrontVertexFlags::Connectible;
    if (previous_connectible) {
      current_vertex.flags |= GPUWavefrontVertexFlags::Mis_connectible;
    }
  }

  float3 selected_direction = sample_valid ? bsdf_sample.w_o : float3(0.0f, 0.0f, 0.0f);
  if (subsurface_random_walk) {
    selected_direction = sample_cosine_distribution(bsdf_rnd, current_vertex.normal, 1.0f);
  }

  float selected_sample_pdf = bsdf_sample.pdf;
  if (subsurface_random_walk) {
    selected_sample_pdf = abs(dot(current_vertex.normal, normalize(selected_direction))) * kInvPi;
  }
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
  previous_vertex.pdf_from_next = 0.0f;
#else
  float3 reverse_direction = selected_direction;
  float reverse_bsdf_pdf =
    wavefront_surface_continue_stage_reverse_bsdf_pdf(make_scene_bsdf_resource_gpu_context(), bsdf_data, reverse_direction, material, bsdf_sampler);
  if (subsurface_random_walk) {
    reverse_bsdf_pdf = abs(dot(current_vertex.normal, normalize(previous_vertex.position - current_vertex.position))) * kInvPi;
  }
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
#endif

  state.sampler_seed = bsdf_sampler.seed;

  wavefront_store_path_vertex(vertex_descriptor, previous_vertex_index, previous_vertex);
  wavefront_store_path_vertex(vertex_descriptor, current_vertex_index, current_vertex);
  wavefront_store_path_meta(resources.path_meta_buffer, path_index, meta);
  state.medium_index = current_medium_index;

  bool continue_path = sample_valid && ((state.path_length + 1u) <= resources.max_path_length);
  state.reserved0 = GPUWavefrontPendingContinuationFlags::Prepared;
  if (continue_path) {
    if (subsurface_random_walk) {
      bsdf_sample.w_o = selected_direction;
      bsdf_sample.pdf = abs(dot(bsdf_sample.w_o, current_vertex.normal)) * kInvPi;
      bsdf_sample.weight = subsurface_gather.weight;
      bsdf_sample.eta = 1.0f;
    }
    float cos_theta_bsdf = abs(dot(hit.vertex.nrm, bsdf_sample.w_o));
    if (bsdf_sample_is_delta(bsdf_sample)) {
      state.forward_pdf = 0.0f;
      state.reverse_pdf = current_d_vc * cos_theta_bsdf;
    } else {
      state.forward_pdf = wavefront_safe_div(1.0f, selected_sample_pdf);
      state.reverse_pdf = wavefront_safe_div(cos_theta_bsdf * ((current_d_vc * reverse_bsdf_pdf) + current_d_vcm), selected_sample_pdf);
    }
    SpectralResponse next_throughput = spectral_response_mul(state.throughput, bsdf_sample.weight);
    if (subsurface_random_walk) {
      next_throughput = spectral_response_mul(state.throughput, subsurface_gather.weight);
    }
    if (from_camera == false) {
      float shading_fix = bsdf_fix_shading_normal(hit.geo_normal, hit.vertex.nrm, state.ray.d, bsdf_sample.w_o);
      if (isfinite(shading_fix) && (shading_fix > 0.0f)) {
        next_throughput = spectral_response_mul(next_throughput, shading_fix);
      }
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
