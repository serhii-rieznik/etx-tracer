#include <interop/gpu_rt_shared.hxx>

[[vk::push_constant]] GPURTConstants constants;

#include "gpu_rt_shared.hlsl"

float evaluate_ao(RaytracingAccelerationStructure as, float3 position, float3 normal, float3 tangent_hint, float3 bitangent_hint, float radius, uint2 pixel, uint seed) {
  float occluded = 0.0f;
  float3 normal_vector = float3(0.0f, 0.0f, 0.0f);
  float3 basis_tangent = float3(0.0f, 0.0f, 0.0f);
  float3 basis_bitangent = float3(0.0f, 0.0f, 0.0f);
  scene_math_shared_build_sampling_frame(normal, tangent_hint, bitangent_hint, normal_vector, basis_tangent, basis_bitangent);

  const uint dimension_base = sampler_stream_dimension_base(kSamplerStreamBSDF);
  const uint sample_count = load_scene_options_samples();

  [loop] for (uint i = 0u; i < sample_count; ++i) {
    float2 u = float2(0.0f, 0.0f);
    const uint sample_index = constants.sample_index + i;
    if (sample_use_blue_noise_primary(sample_index, kSamplerStreamBSDF)) {
      const uint blue_noise_index = sample_index & (kSamplerBlueNoiseSampleCount - 1u);
      u.x = sample_blue_noise_value(pixel, blue_noise_index, dimension_base + 0u);
      u.y = sample_blue_noise_value(pixel, blue_noise_index, dimension_base + 1u);
    } else {
      u = float2(rnd01(seed), rnd01(seed));
    }

    float3 local_dir = sample_cosine_distribution(u, 0.0f);
    float3 world_dir = scene_math_shared_local_to_world(normal_vector, basis_tangent, basis_bitangent, local_dir);

    RayDesc ray;
    ray.Origin = position + normal_vector * 0.002f;
    ray.Direction = world_dir;
    ray.TMin = 0.001f;
    ray.TMax = radius;

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(as, RAY_FLAG_NONE, 0xFF, ray);
    q.Proceed();

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
      occluded += 1.0f;
    }
  }

  return 1.0f - occluded / float(sample_count);
}

[numthreads(8, 8, 1)] void compute_main(uint3 dtid : SV_DispatchThreadID) {
  if (constants.camera_buffer_index == kInvalidIndex)
    return;

  ByteAddressBuffer camera_buffer = bindless_buffers[NonUniformResourceIndex(constants.camera_buffer_index)];
  Camera camera = load_camera(camera_buffer);
  uint2 film_size = camera.film_size;
  if (any(dtid.xy >= film_size))
    return;

  uint pixel_index = dtid.x + dtid.y * film_size.x;
  uint seed = scene_random_seed(pixel_index, constants.sample_index);
  bool spectral_mode = scene_uses_spectral_mode();
  SpectralQuery spectral_query = spectral_query_sample();
  if (spectral_mode) {
    spectral_query = spectral_query_spectral_sample(rnd01(seed));
  }
  float2 film_sample_rnd = float2(rnd01(seed), rnd01(seed));
  float2 ndc = camera_sample_film_uv(dtid.xy, film_size, film_sample_rnd);
  float spectral_pdf = spectral_query_sampling_pdf(spectral_query);
  float spectral_weight = (spectral_pdf > 0.0f) ? (1.0f / spectral_pdf) : 0.0f;

  float2 lens_rnd = float2(0.0f, 0.0f);
  if (camera_lens_sampling_enabled(camera.lens_radius, camera.focal_distance)) {
    lens_rnd = sample_primary_hybrid_2d(dtid.xy, constants.sample_index, kSamplerStreamSupport, seed);
  }

  Ray ray_data = camera_generate_primary_ray(camera, ndc, lens_rnd);
  float3 ray_dir = ray_data.d;

  RayDesc ray;
  ray.Origin = ray_data.o;
  ray.Direction = ray_data.d;
  ray.TMin = max(kRayEpsilon, ray_data.min_t);
  ray.TMax = max(ray.TMin + kRayEpsilon, ray_data.max_t);

  RaytracingAccelerationStructure as = bindless_accel_structs[NonUniformResourceIndex(constants.as_index)];
  uint current_medium = camera.medium_index;
  RayDesc path_ray = ray;
  SpectralResponse accumulated = spectral_response_zero(spectral_query);
  SpectralResponse throughput = spectral_response_make(spectral_query, 1.0f);
  TraceSurfaceResult surface_hit = (TraceSurfaceResult)0;
  bool surface_found = trace_surface_path(as, path_ray, spectral_query, current_medium, seed, surface_hit);
  throughput = spectral_response_mul(throughput, surface_hit.transmittance);
  if (gpu_valid_spectral_response(throughput) == false) {
    accumulated = spectral_response_zero(spectral_query);
  } else if (surface_found == false) {
    SpectralResponse distant_emission = gpu_evaluate_distant_emission_spectral_all(path_ray.Direction, spectral_query);
    accumulated = spectral_response_add(accumulated, spectral_response_mul(throughput, distant_emission));
  } else if (spectral_response_is_zero(throughput) == false) {
    bool local_emission_visible = dot(surface_hit.tri.geo_n, path_ray.Direction) < 0.0f;
    if ((surface_hit.emitter_index != kInvalidIndex) && local_emission_visible) {
      SpectralResponse local_emission = gpu_evaluate_local_emission_spectral(surface_hit.emitter_index, surface_hit.surface_point.vertex.tex, spectral_query);
      accumulated = spectral_response_add(accumulated, spectral_response_mul(throughput, local_emission));
    }

    if (gpu_bsdf_sample_supported_class(surface_hit.material.cls)) {
      BSDFResourceContext bsdf_context = make_scene_bsdf_resource_gpu_context();
      Sampler bsdf_sampler = make_bsdf_sampler(seed);
      BSDFData bsdf_data = make_surface_bsdf_data(surface_hit.surface_point.vertex, spectral_query, current_medium, path_ray.Direction);
      BSDFSample sample_value = gpu_sample_material_bsdf(bsdf_context, bsdf_data, surface_hit.material, bsdf_sampler);
      seed = bsdf_sampler.seed;
      if (bsdf_sample_valid(sample_value) && gpu_valid_direction(sample_value.w_o) && gpu_valid_spectral_response(sample_value.weight)) {
        SpectralResponse sampled_weight = spectral_response_mul(throughput, sample_value.weight);
        if (gpu_valid_spectral_response(sampled_weight)) {
          float3 sampled_direction = normalize(sample_value.w_o);
          if (gpu_valid_direction(sampled_direction)) {
            SpectralResponse sampled_environment = spectral_response_zero(spectral_query);
            if (spectral_query_is_spectral(spectral_query)) {
              sampled_environment = gpu_evaluate_distant_emission_spectral_all(sampled_direction, spectral_query);
            } else {
              sampled_environment = spectral_response_make(spectral_query, evaluate_distant_emission_integrated_all(sampled_direction));
            }
            accumulated = spectral_response_add(accumulated, spectral_response_mul(sampled_weight, sampled_environment));
          }
        }
      }
    } else {
      float ao = evaluate_ao(as, surface_hit.surface_point.vertex.pos, surface_hit.surface_point.geo_normal, surface_hit.surface_point.vertex.tan,
        surface_hit.surface_point.vertex.btn, 1.0f, dtid.xy, seed);
      SpectralResponse fallback_response = spectral_response_make(spectral_query, scene_math_shared_default_ao_shading(surface_hit.surface_point.vertex.nrm, ao));
      accumulated = spectral_response_add(accumulated, spectral_response_mul(throughput, fallback_response));
    }
  }

  float3 shaded = spectral_response_to_rgb(accumulated) * spectral_weight;
  float4 color = float4(max(shaded, float3(0.0f, 0.0f, 0.0f)), 1.0f);

  bindless_storage_textures[constants.output_image_index][dtid.xy] = color;
}
