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

  [loop]
  for (uint i = 0u; i < sample_count; ++i) {
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

  float2 ndc = camera_primary_uv(dtid.xy, film_size);
  uint pixel_index = dtid.x + dtid.y * film_size.x;
  uint seed = sampler_random_seed(pixel_index, constants.sample_index);
  bool spectral_mode = scene_uses_spectral_mode();
  SpectralQuery spectral_query = spectral_query_sample();
  if (spectral_mode) {
    spectral_query = spectral_query_spectral_sample(rnd01(seed));
  }
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

  RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> q;
  RaytracingAccelerationStructure as = bindless_accel_structs[NonUniformResourceIndex(constants.as_index)];
  const bool has_geometry_buffers = (constants.scene.triangles != kInvalidIndex) && (constants.scene.vertex_positions != kInvalidIndex) &&
                                    (constants.scene.vertex_normals != kInvalidIndex) && (constants.scene.scene_globals != kInvalidIndex);
  uint ray_medium_index = camera.medium_index;
  float medium_segment_start_t = ray.TMin;
  float3 ray_transmittance_integrated = float3(1.0f, 1.0f, 1.0f);
  SpectralResponse ray_transmittance_spectral = spectral_response_make(spectral_query, 1.0f);

  if (has_geometry_buffers) {
    ByteAddressBuffer triangle_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)];
    ByteAddressBuffer scene_globals = bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)];
    SceneGPUSharedGlobals scene_globals_data = scene_gpu_load_globals(scene_globals);
    const bool has_material_buffer = constants.scene.materials != kInvalidIndex;
    uint vertex_count = scene_globals_data.vertex_count;
    uint triangle_count = scene_globals_data.triangle_count;
    const bool has_texcoords = constants.scene.vertex_texcoords != kInvalidIndex;

    q.TraceRayInline(as, RAY_FLAG_FORCE_NON_OPAQUE, 0xFF, ray);
    while (q.Proceed()) {
      if (q.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) {
        continue;
      }

      uint candidate_triangle_index = q.CandidatePrimitiveIndex();
      if (candidate_triangle_index >= triangle_count) {
        continue;
      }
      float candidate_t = q.CandidateTriangleRayT();
      if (candidate_t > medium_segment_start_t) {
        float segment_distance = candidate_t - medium_segment_start_t;
        float3 segment_origin = ray.Origin + ray.Direction * medium_segment_start_t;
        if (spectral_mode) {
          ray_transmittance_spectral = spectral_response_mul(
            ray_transmittance_spectral, medium_segment_transmittance_spectral(ray_medium_index, segment_origin, ray.Direction, segment_distance, spectral_query, seed));
        } else {
          ray_transmittance_integrated *= medium_segment_transmittance_integrated(ray_medium_index, segment_origin, ray.Direction, segment_distance, seed);
        }
        medium_segment_start_t = candidate_t;
      }

      TriangleData tri = load_triangle(triangle_buffer, candidate_triangle_index);
      bool valid_indices = (tri.i.x < vertex_count) && (tri.i.y < vertex_count) && (tri.i.z < vertex_count);
      if (valid_indices == false) {
        continue;
      }

      float2 candidate_bary = q.CandidateTriangleBarycentrics();
      float2 candidate_uv = float2(0.0f, 0.0f);
      if (has_texcoords) {
        ByteAddressBuffer texcoord_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_texcoords)];
        candidate_uv = interpolate_uv(texcoord_buffer, tri, candidate_bary);
      }

      MaterialAccess material_access = ETX_ZERO(MaterialAccess);
      if (has_material_buffer) {
        MaterialAccessGPUContext material_context = {constants.scene.materials};
        material_access_try_load(material_context, tri.material_index, material_access);
      }

      bool alpha_rejected = alpha_test_pass(tri.material_index, candidate_uv, seed);
      bool entering_surface = dot(tri.geo_n, ray_dir) < 0.0f;
      HitPolicyDecision hit_policy =
        hit_policy_evaluate(HitPolicyMode::SkipBoundaryWithMediumTransition, material_access.material_class, alpha_rejected, entering_surface,
          material_access.int_medium_index, material_access.ext_medium_index);
      if (hit_policy.action == HitPolicyAction::Ignore) {
        continue;
      }

      if (hit_policy.action == HitPolicyAction::TransitionMedium) {
        ray_medium_index = hit_policy.medium_index;
        continue;
      }

      if (hit_policy.action == HitPolicyAction::CommitSurface) {
        q.CommitNonOpaqueTriangleHit();
      }
    }
  } else {
    q.TraceRayInline(as, RAY_FLAG_NONE, 0xFF, ray);
    q.Proceed();
  }

  float medium_segment_end_t = (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? q.CommittedRayT() : ray.TMax;
  if (medium_segment_end_t > medium_segment_start_t) {
    float segment_distance = medium_segment_end_t - medium_segment_start_t;
    float3 segment_origin = ray.Origin + ray.Direction * medium_segment_start_t;
    if (spectral_mode) {
      ray_transmittance_spectral = spectral_response_mul(
        ray_transmittance_spectral, medium_segment_transmittance_spectral(ray_medium_index, segment_origin, ray.Direction, segment_distance, spectral_query, seed));
    } else {
      ray_transmittance_integrated *= medium_segment_transmittance_integrated(ray_medium_index, segment_origin, ray.Direction, segment_distance, seed);
    }
  }

  float4 color = float4(0.0f, 0.0f, 0.0f, 1.0f);

  if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
    if (has_geometry_buffers) {
      ByteAddressBuffer triangle_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)];
      ByteAddressBuffer position_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_positions)];
      ByteAddressBuffer normal_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_normals)];
      const bool has_surface_frame_buffers = (constants.scene.vertex_tangents != kInvalidIndex) && (constants.scene.vertex_bitangents != kInvalidIndex);
      const bool has_texcoords = constants.scene.vertex_texcoords != kInvalidIndex;
      uint tangent_buffer_index = constants.scene.vertex_positions;
      uint bitangent_buffer_index = constants.scene.vertex_positions;
      uint texcoord_buffer_index = constants.scene.vertex_positions;
      if (has_surface_frame_buffers) {
        tangent_buffer_index = constants.scene.vertex_tangents;
        bitangent_buffer_index = constants.scene.vertex_bitangents;
      }
      if (has_texcoords) {
        texcoord_buffer_index = constants.scene.vertex_texcoords;
      }
      ByteAddressBuffer tangent_buffer = bindless_buffers[NonUniformResourceIndex(tangent_buffer_index)];
      ByteAddressBuffer bitangent_buffer = bindless_buffers[NonUniformResourceIndex(bitangent_buffer_index)];
      ByteAddressBuffer texcoord_buffer = bindless_buffers[NonUniformResourceIndex(texcoord_buffer_index)];
      ByteAddressBuffer scene_globals = bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)];
      SceneGPUSharedGlobals scene_globals_data = scene_gpu_load_globals(scene_globals);

      uint vertex_count = scene_globals_data.vertex_count;
      uint triangle_count = scene_globals_data.triangle_count;
      float bounding_sphere_radius = scene_globals_data.bounding_sphere_radius;

      uint triangle_index = q.CommittedPrimitiveIndex();
      if (triangle_index < triangle_count) {
        TriangleData tri = load_triangle(triangle_buffer, triangle_index);

        bool valid_indices = (tri.i.x < vertex_count) && (tri.i.y < vertex_count) && (tri.i.z < vertex_count);
        if (valid_indices) {
          float2 bary = q.CommittedTriangleBarycentrics();
          SurfacePoint surface_point =
            load_surface_point(position_buffer, normal_buffer, tangent_buffer, bitangent_buffer, texcoord_buffer, has_surface_frame_buffers, has_texcoords, tri, bary, ray_dir);
          float2 hit_uv = surface_point.vertex.tex;

          float ao_radius = max(0.1f, 0.05f * bounding_sphere_radius);
          float ao = evaluate_ao(as, surface_point.vertex.pos, surface_point.geo_normal, surface_point.vertex.tan, surface_point.vertex.btn, ao_radius, dtid.xy, seed);
          MaterialAccessGPUContext material_context = {constants.scene.materials};
          MaterialAccess material_access = ETX_ZERO(MaterialAccess);
          bool has_material_scattering = material_access_try_load(material_context, tri.material_index, material_access) && material_access_has_scattering(material_access);
          SpectralImage scattering_image = ETX_ZERO(SpectralImage);
          if (has_material_scattering) {
            scattering_image.spectrum_index = material_access.scattering_spectrum_index;
            scattering_image.image_index = material_access.scattering_image_index;
          }

          if (spectral_mode) {
            float3 fallback_color = scene_math_shared_default_ao_shading(surface_point.vertex.nrm, ao);
            SpectralResponse fallback_response = spectral_response_make(spectral_query, luminance(fallback_color));
            SpectralResponse shaded_spectral = fallback_response;
            if (has_material_scattering) {
              shaded_spectral = spectral_response_mul(apply_image(spectral_query, scattering_image, hit_uv), ao);
              shaded_spectral = spectral_response_clamp_non_negative(shaded_spectral);
            }
            bool local_emission_visible = dot(tri.geo_n, ray_dir) < 0.0f;
            if ((tri.emitter_index != kInvalidIndex) && local_emission_visible) {
              shaded_spectral = spectral_response_add(shaded_spectral, evaluate_local_emission_spectral(tri.emitter_index, hit_uv, spectral_query));
            }
            shaded_spectral = spectral_response_mul(shaded_spectral, ray_transmittance_spectral);

            float3 shaded = spectral_response_to_rgb(shaded_spectral) * spectral_weight;
            color = float4(max(shaded, float3(0.0f, 0.0f, 0.0f)), 1.0f);
          } else {
            float3 shaded = scene_math_shared_default_ao_shading(surface_point.vertex.nrm, ao);
            if (has_material_scattering) {
              SpectralQuery integrated_query = spectral_query_sample();
              shaded = spectral_rgb_clamp_non_negative(apply_image(integrated_query, scattering_image, hit_uv).integrated) * ao;
            }
            bool local_emission_visible = dot(tri.geo_n, ray_dir) < 0.0f;
            if ((tri.emitter_index != kInvalidIndex) && local_emission_visible) {
              shaded += evaluate_local_emission_integrated(tri.emitter_index, hit_uv);
            }
            shaded *= ray_transmittance_integrated;
            color = float4(shaded, 1.0f);
          }
        } else {
          color = float4(1.0f, 0.0f, 1.0f, 1.0f);
        }
      } else {
        color = float4(1.0f, 0.0f, 1.0f, 1.0f);
      }
    } else {
      float2 bary = q.CommittedTriangleBarycentrics();
      color = float4(bary.x, bary.y, 1.0f - bary.x - bary.y, 1.0f);
    }
  } else {
    if (spectral_mode) {
      SpectralResponse distant_emission = evaluate_distant_emission_spectral_all(ray_dir, spectral_query);
      if (spectral_response_is_zero(distant_emission) == false) {
        SpectralResponse attenuated = spectral_response_mul(distant_emission, ray_transmittance_spectral);
        color = float4(max(spectral_response_to_rgb(attenuated) * spectral_weight, float3(0.0f, 0.0f, 0.0f)), 1.0f);
      }
    } else {
      float3 distant_emission = evaluate_distant_emission_integrated_all(ray_dir);
      if (dot(distant_emission, distant_emission) > 0.0f) {
        color = float4(distant_emission * ray_transmittance_integrated, 1.0f);
      }
    }
  }

  bindless_storage_textures[constants.output_image_index][dtid.xy] = color;
}

