bool wavefront_try_load_environment_emitter_state(out uint emitter_instance_count, out uint environment_emitter_count) {
  emitter_instance_count = 0u;
  environment_emitter_count = 0u;
  return emitter_access_try_load_environment_state(make_scene_emitter_access_gpu_context(), emitter_instance_count, environment_emitter_count);
}

bool wavefront_select_environment_emitter_random(inout uint seed, out uint emitter_index, out uint emitter_count) {
  emitter_index = kInvalidIndex;
  emitter_count = 0u;

  uint emitter_instance_count = 0u;
  if (wavefront_try_load_environment_emitter_state(emitter_instance_count, emitter_count) == false) {
    return false;
  }
  (void)emitter_instance_count;

  uint local_index = min(uint(rnd01(seed) * float(emitter_count)), emitter_count - 1u);
  return emitter_access_try_load_environment_emitter(make_scene_emitter_access_gpu_context(), local_index, emitter_index);
}

bool wavefront_scene_has_only_environment_emitters() {
  uint emitter_instance_count = 0u;
  uint environment_emitter_count = 0u;
  if (wavefront_try_load_environment_emitter_state(emitter_instance_count, environment_emitter_count) == false) {
    return false;
  }

  return (emitter_instance_count > 0u) && (environment_emitter_count == emitter_instance_count);
}

bool wavefront_sample_emitter_index(uint light_sampling_mode, inout uint seed, out uint emitter_index, out float pdf_sample) {
  emitter_index = kInvalidIndex;
  pdf_sample = 0.0f;
  if ((light_sampling_mode == kSceneLightSamplingFromDistribution) || (light_sampling_mode == kSceneLightSamplingRISFromDistribution)) {
    emitter_index = sample_emitter_distribution(seed, pdf_sample);
    return emitter_index != kInvalidIndex;
  }

  ByteAddressBuffer scene_globals = bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)];
  SceneGPUSharedGlobals globals_data = scene_gpu_load_globals(scene_globals);
  if (globals_data.emitter_instance_count == 0u) {
    return false;
  }

  if (wavefront_scene_has_only_environment_emitters()) {
    uint environment_emitter_count = 0u;
    if (wavefront_select_environment_emitter_random(seed, emitter_index, environment_emitter_count) == false) {
      return false;
    }

    pdf_sample = 1.0f / float(environment_emitter_count);
    return true;
  }

  emitter_index = min(uint(rnd01(seed) * float(globals_data.emitter_instance_count)), globals_data.emitter_instance_count - 1u);
  pdf_sample = 1.0f / float(globals_data.emitter_instance_count);
  return true;
}

bool wavefront_sample_emitter_to_point_from_index(uint emitter_index, float pdf_sample, SpectralQuery spect, float3 from_point, inout uint seed,
  out WavefrontEmitterSample sample_value) {
  sample_value = (WavefrontEmitterSample)0;
  GPUEmitterInstanceABIData emitter_instance = (GPUEmitterInstanceABIData)0;
  GPUEmitterProfileABIData emitter_profile = (GPUEmitterProfileABIData)0;
  if ((try_load_emitter_instance(emitter_index, emitter_instance) == false) || (try_load_emitter_profile(emitter_instance.emitter_profile_index, emitter_profile) == false)) {
    return false;
  }

  sample_value.emitter_index = emitter_index;
  sample_value.triangle_index = emitter_instance.triangle_index;
  sample_value.instance_index = emitter_instance.instance_index;
  sample_value.medium_index = emitter_access_external_medium_index(make_scene_emitter_access_gpu_context(), emitter_index);
  sample_value.pdf_sample = pdf_sample;
  float2 emitter_sample_rnd = float2(rnd01(seed), rnd01(seed));

  if (emitter_instance.emitter_class == EmitterClass::Area) {
    TriangleData tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], emitter_instance.triangle_index);
    sample_value.barycentric = random_barycentric(emitter_sample_rnd);
    Vertex vertex = wavefront_interpolate_vertex(tri, sample_value.barycentric);
    const GPUSceneInstanceData scene_instance = load_scene_instance(emitter_instance.instance_index);
    vertex = scene_instance_transform_vertex(scene_instance, vertex);
    const float3 geo_normal = scene_instance_transform_geometric_normal(scene_instance, tri.geo_n);
    sample_value.origin = vertex.pos;
    sample_value.normal = vertex.nrm;
    sample_value.image_uv = vertex.tex;
    sample_value.pdf_area = (emitter_instance.triangle_area > 0.0f) ? (1.0f / emitter_instance.triangle_area) : 0.0f;
    float3 dp = sample_value.origin - from_point;
    float distance_squared = dot(dp, dp);
    if (distance_squared > kEpsilon) {
      sample_value.direction = normalize(dp);
    } else {
      sample_value.direction = float3(0.0f, 0.0f, 0.0f);
    }
    Material material = (Material)0;
    if (try_load_material_full(tri.material_index, material) == false) {
      return false;
    }

    sample_value.value = spectral_response_zero(spect);
    sample_value.is_delta = 0u;
    sample_value.is_distant = 0u;

    if ((sample_value.pdf_area <= 0.0f) || (distance_squared <= kEpsilon)) {
      return true;
    }

    sample_value.value = evaluate_emission_spectral_source(emitter_profile.emission_spectrum_index, emitter_profile.emission_image_index, vertex.tex, spect);
    if (dot(geo_normal, dp) >= 0.0f) {
      sample_value.value = spectral_response_zero(spect);
      return true;
    }

    float cos_t = abs(dot(dp, geo_normal)) / sqrt(distance_squared);
    float exponent = scene_math_shared_collimation_to_exponent(material.emission_collimation);
    float cos_tx = pow(cos_t, exponent);
    if (cos_tx > kEpsilon) {
      sample_value.pdf_dir = sample_value.pdf_area * distance_squared / cos_tx;
      sample_value.pdf_dir_out = sample_value.pdf_area * cos_tx * kInvPi;
    }
    return true;
  }

  EmitterAccess access = (EmitterAccess)0;
  if (emitter_access_try_load(make_scene_emitter_access_gpu_context(), emitter_index, access) == false) {
    return false;
  }

  SceneGPUSharedGlobals globals_data = scene_gpu_load_globals(bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)]);
  if (emitter_instance.emitter_class == EmitterClass::Directional) {
    float2 disk_sample = float2(0.0f, 0.0f);
    if (emitter_profile.emitter_angular_size_cosine > kEpsilon) {
      float sin_half_angle = sqrt(max(0.0f, 1.0f - (emitter_profile.emitter_angular_size_cosine * emitter_profile.emitter_angular_size_cosine)));
      float equivalent_disk_size = 2.0f * (sin_half_angle / max(kEpsilon, emitter_profile.emitter_angular_size_cosine));
      OrthonormalBasis basis = orthonormal_basis(access.emitter_direction);
      disk_sample = sample_disk(emitter_sample_rnd);
      sample_value.direction =
        normalize(access.emitter_direction + basis.u * disk_sample.x * (0.5f * equivalent_disk_size) + basis.v * disk_sample.y * (0.5f * equivalent_disk_size));
    } else {
      sample_value.direction = normalize(access.emitter_direction);
    }
    sample_value.normal = -normalize(access.emitter_direction);
    sample_value.origin =
      from_point + sample_value.direction * distance_to_sphere(from_point, sample_value.direction, globals_data.bounding_sphere_center, globals_data.bounding_sphere_radius);
    sample_value.pdf_area = 1.0f / (kPi * globals_data.bounding_sphere_radius * globals_data.bounding_sphere_radius);
    sample_value.pdf_dir = 1.0f;
    sample_value.pdf_dir_out = sample_value.pdf_area;
    sample_value.image_uv = disk_sample * 0.5f + 0.5f;
    sample_value.value = evaluate_emission_spectral_source(emitter_profile.emission_spectrum_index, emitter_profile.emission_image_index, sample_value.image_uv, spect);
    sample_value.is_delta = 1u;
    sample_value.is_distant = 1u;
    return true;
  }

  ImageSampleGPUContext image_context = make_image_sample_gpu_context(constants.scene.images);
  ImageSampleAccess image_sample = image_sample_access_default(emitter_sample_rnd);
  if (image_sample_try_sample(image_context, emitter_profile.emission_image_index, emitter_sample_rnd, image_sample) == false) {
    return false;
  }

  bool is_atmosphere = (emitter_profile.emitter_profile_meta & EmitterProfileMeta::Atmosphere) != 0u;
  uint projection = projection_environment_mode(is_atmosphere);
  sample_value.image_uv = image_sample.uv;
  float2 image_offset = float2(0.0f, 0.0f);
  float image_u_scale = 1.0f;
  emitter_access_try_load_image_params(make_scene_emitter_access_gpu_context(), emitter_profile.emission_image_index, image_offset, image_u_scale);
  float3 local_direction = uv_to_direction(image_sample.uv, image_offset, image_u_scale, projection);
  sample_value.direction = emitter_access_environment_local_to_world(emitter_profile.emitter_direction, emitter_profile.emitter_angular_size, local_direction);
  sample_value.normal = -sample_value.direction;
  sample_value.origin =
    from_point + sample_value.direction * distance_to_sphere(from_point, sample_value.direction, globals_data.bounding_sphere_center, globals_data.bounding_sphere_radius);
  sample_value.pdf_dir = projection_environment_image_pdf_to_solid_angle(image_sample.pdf, image_sample.uv, projection);
  sample_value.pdf_area = 1.0f / (kPi * globals_data.bounding_sphere_radius * globals_data.bounding_sphere_radius);
  sample_value.pdf_dir_out = sample_value.pdf_area * sample_value.pdf_dir;
  SpectralResponse spectrum_value = load_scene_spectrum_or_zero(emitter_profile.emission_spectrum_index, spect);
  sample_value.value = spectral_response_apply_rgb_scale(spect, spectrum_value, image_sample.value.xyz);
  sample_value.is_delta = 0u;
  sample_value.is_distant = 1u;
  return true;
}

bool wavefront_sample_emitter_to_point(uint light_sampling_mode, SpectralQuery spect, float3 from_point, inout uint seed, out WavefrontEmitterSample sample_value) {
  sample_value = (WavefrontEmitterSample)0;
  uint emitter_index = kInvalidIndex;
  float pdf_sample = 0.0f;
  if (wavefront_sample_emitter_index(light_sampling_mode, seed, emitter_index, pdf_sample) == false) {
    return false;
  }

  return wavefront_sample_emitter_to_point_from_index(emitter_index, pdf_sample, spect, from_point, seed, sample_value);
}

bool wavefront_sample_light_emission(SpectralQuery spect, inout uint seed, out WavefrontEmitterSample sample_value) {
  sample_value = (WavefrontEmitterSample)0;
  float pdf_sample = 0.0f;
  uint emitter_index = sample_emitter_distribution(seed, pdf_sample);
  if (emitter_index == kInvalidIndex) {
    return false;
  }

  GPUEmitterInstanceABIData emitter_instance = (GPUEmitterInstanceABIData)0;
  GPUEmitterProfileABIData emitter_profile = (GPUEmitterProfileABIData)0;
  if ((try_load_emitter_instance(emitter_index, emitter_instance) == false) || (try_load_emitter_profile(emitter_instance.emitter_profile_index, emitter_profile) == false)) {
    return false;
  }

  sample_value.emitter_index = emitter_index;
  sample_value.triangle_index = emitter_instance.triangle_index;
  sample_value.instance_index = emitter_instance.instance_index;
  sample_value.medium_index = emitter_access_external_medium_index(make_scene_emitter_access_gpu_context(), emitter_index);
  sample_value.pdf_sample = pdf_sample;

  if (emitter_instance.emitter_class == EmitterClass::Area) {
    TriangleData tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], emitter_instance.triangle_index);
    sample_value.barycentric = random_barycentric(float2(rnd01(seed), rnd01(seed)));
    Vertex vertex = wavefront_interpolate_vertex(tri, sample_value.barycentric);
    vertex = scene_instance_transform_vertex(load_scene_instance(emitter_instance.instance_index), vertex);
    Material material = (Material)0;
    if (try_load_material_full(tri.material_index, material) == false) {
      return false;
    }

    float exponent = scene_math_shared_collimation_to_exponent(material.emission_collimation);
    sample_value.origin = vertex.pos;
    sample_value.normal = vertex.nrm;
    sample_value.direction = sample_cosine_distribution(float2(rnd01(seed), rnd01(seed)), sample_value.normal, vertex.tan, vertex.btn, exponent);
    sample_value.image_uv = vertex.tex;
    sample_value.pdf_area = (emitter_instance.triangle_area > 0.0f) ? (1.0f / emitter_instance.triangle_area) : 0.0f;
    float cos_t = max(0.0f, dot(sample_value.normal, sample_value.direction));
    sample_value.pdf_dir = pow(cos_t, exponent) * kInvPi;
    sample_value.pdf_dir_out = sample_value.pdf_area * sample_value.pdf_dir;
    sample_value.value = evaluate_emission_spectral_source(emitter_profile.emission_spectrum_index, emitter_profile.emission_image_index, vertex.tex, spect);
    sample_value.is_delta = 0u;
    sample_value.is_distant = 0u;
    return sample_value.pdf_dir > 0.0f;
  }

  SceneGPUSharedGlobals globals_data = scene_gpu_load_globals(bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)]);
  if (emitter_instance.emitter_class == EmitterClass::Directional) {
    EmitterAccess access = (EmitterAccess)0;
    if (emitter_access_try_load(make_scene_emitter_access_gpu_context(), emitter_index, access) == false) {
      return false;
    }

    float3 direction_to_scene = -normalize(access.emitter_direction);
    OrthonormalBasis basis = orthonormal_basis(direction_to_scene);
    float equivalent_disk_size = 0.0f;
    if (emitter_profile.emitter_angular_size_cosine > kEpsilon) {
      float sin_half_angle = sqrt(max(0.0f, 1.0f - (emitter_profile.emitter_angular_size_cosine * emitter_profile.emitter_angular_size_cosine)));
      equivalent_disk_size = 2.0f * (sin_half_angle / emitter_profile.emitter_angular_size_cosine);
    }

    float2 position_sample = sample_disk(float2(rnd01(seed), rnd01(seed)));
    float2 direction_sample = sample_disk(float2(rnd01(seed), rnd01(seed)));
    sample_value.direction =
      normalize(direction_to_scene + basis.u * direction_sample.x * (0.5f * equivalent_disk_size) + basis.v * direction_sample.y * (0.5f * equivalent_disk_size));
    sample_value.normal = direction_to_scene;
    sample_value.origin =
      globals_data.bounding_sphere_center + globals_data.bounding_sphere_radius * (position_sample.x * basis.u + position_sample.y * basis.v - direction_to_scene);
    sample_value.origin +=
      sample_value.direction * distance_to_sphere(sample_value.origin, sample_value.direction, globals_data.bounding_sphere_center, globals_data.bounding_sphere_radius);
    sample_value.pdf_dir = 1.0f;
    sample_value.pdf_area = 1.0f / (kPi * globals_data.bounding_sphere_radius * globals_data.bounding_sphere_radius);
    sample_value.pdf_dir_out = sample_value.pdf_area;
    sample_value.value = evaluate_emission_spectral_source(emitter_profile.emission_spectrum_index, emitter_profile.emission_image_index, direction_sample * 0.5f + 0.5f, spect);
    sample_value.is_delta = 1u;
    sample_value.is_distant = 1u;
    return true;
  }

  ImageSampleGPUContext image_context = make_image_sample_gpu_context(constants.scene.images);
  float2 sample_rnd = float2(rnd01(seed), rnd01(seed));
  ImageSampleAccess image_sample = image_sample_access_default(sample_rnd);
  if (image_sample_try_sample(image_context, emitter_profile.emission_image_index, sample_rnd, image_sample) == false) {
    return false;
  }

  bool is_atmosphere = (emitter_profile.emitter_profile_meta & EmitterProfileMeta::Atmosphere) != 0u;
  uint projection = projection_environment_mode(is_atmosphere);
  sample_value.image_uv = image_sample.uv;
  float2 image_offset = float2(0.0f, 0.0f);
  float image_u_scale = 1.0f;
  emitter_access_try_load_image_params(make_scene_emitter_access_gpu_context(), emitter_profile.emission_image_index, image_offset, image_u_scale);
  float3 local_direction = -uv_to_direction(image_sample.uv, image_offset, image_u_scale, projection);
  sample_value.direction = emitter_access_environment_local_to_world(emitter_profile.emitter_direction, emitter_profile.emitter_angular_size, local_direction);
  sample_value.normal = sample_value.direction;
  OrthonormalBasis basis = orthonormal_basis(sample_value.direction);
  float2 disk_sample = sample_disk(float2(rnd01(seed), rnd01(seed)));
  sample_value.origin = globals_data.bounding_sphere_center + globals_data.bounding_sphere_radius * (disk_sample.x * basis.u + disk_sample.y * basis.v - sample_value.direction);
  sample_value.origin +=
    sample_value.direction * distance_to_sphere(sample_value.origin, sample_value.direction, globals_data.bounding_sphere_center, globals_data.bounding_sphere_radius);
  sample_value.pdf_dir = projection_environment_image_pdf_to_solid_angle(image_sample.pdf, image_sample.uv, projection);
  sample_value.pdf_area = 1.0f / (kPi * globals_data.bounding_sphere_radius * globals_data.bounding_sphere_radius);
  sample_value.pdf_dir_out = sample_value.pdf_area * sample_value.pdf_dir;
  SpectralResponse spectrum_value = load_scene_spectrum_or_zero(emitter_profile.emission_spectrum_index, spect);
  sample_value.value = spectral_response_apply_rgb_scale(spect, spectrum_value, image_sample.value.xyz);
  sample_value.is_delta = 0u;
  sample_value.is_distant = 1u;
  return sample_value.pdf_dir > 0.0f;
}
