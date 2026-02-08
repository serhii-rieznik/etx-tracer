#include "bindless.hlsl"
#include <interop/render_options.hxx>
#include <interop/gpu_rt_shared.hxx>

[[vk::push_constant]] GPURTConstants constants;

static const uint INVALID_INDEX = 0xFFFFFFFFu;
static const uint MATERIAL_STRIDE = 272u;
static const uint MATERIAL_SCATTERING_SPECTRUM_INDEX_OFFSET = 16u;
static const uint SPECTRAL_DISTRIBUTION_STRIDE = 3552u;
static const uint SPECTRAL_DISTRIBUTION_INTEGRATED_OFFSET = 0u;

uint hash_u32(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

float rnd01(inout uint state) {
  state = hash_u32(state);
  return (state & 0x00ffffffu) * (1.0f / 16777216.0f);
}

float3 load_float3(ByteAddressBuffer buffer, uint index) {
  return asfloat(buffer.Load3(index * 12u));
}

struct TriangleData {
  uint3 i;
  uint material_index;
  float3 geo_n;
  uint emitter_index;
};

TriangleData load_triangle(ByteAddressBuffer buffer, uint triangle_index) {
  const uint base_offset = triangle_index * 32u;
  uint4 a = buffer.Load4(base_offset + 0u);
  uint4 b = buffer.Load4(base_offset + 16u);

  TriangleData result;
  result.i = a.xyz;
  result.material_index = a.w;
  result.geo_n = asfloat(b.xyz);
  result.emitter_index = b.w;
  return result;
}

uint load_material_scattering_spectrum_index(ByteAddressBuffer buffer, uint material_index) {
  uint base_offset = material_index * MATERIAL_STRIDE;
  return buffer.Load(base_offset + MATERIAL_SCATTERING_SPECTRUM_INDEX_OFFSET);
}

float3 load_spectrum_integrated_value(ByteAddressBuffer buffer, uint spectrum_index) {
  uint base_offset = spectrum_index * SPECTRAL_DISTRIBUTION_STRIDE;
  return asfloat(buffer.Load3(base_offset + SPECTRAL_DISTRIBUTION_INTEGRATED_OFFSET));
}

bool has_material_spectrum_buffers() {
  return (constants.scene.materials != INVALID_INDEX) && (constants.scene.spectrums != INVALID_INDEX);
}

float3 default_ao_shading(float3 hit_normal, float ao) {
  float n_dot_up = saturate(dot(hit_normal, float3(0.0f, 1.0f, 0.0f)));
  float3 base = lerp(float3(0.35f, 0.37f, 0.42f), float3(0.85f, 0.87f, 0.9f), n_dot_up);
  return base * ao;
}

float3 material_scattering_integrated_or_fallback(uint material_index, float ao, float3 fallback_color) {
  if (has_material_spectrum_buffers() == false) {
    return fallback_color;
  }

  ByteAddressBuffer material_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.materials)];
  ByteAddressBuffer spectrum_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.spectrums)];
  uint scattering_spectrum_index = load_material_scattering_spectrum_index(material_buffer, material_index);

  if (scattering_spectrum_index == INVALID_INDEX) {
    return fallback_color;
  }

  float3 scattering_integrated = load_spectrum_integrated_value(spectrum_buffer, scattering_spectrum_index);
  return max(scattering_integrated, float3(0.0f, 0.0f, 0.0f)) * ao;
}

float3x3 basis_from_normal(float3 n) {
  float3 up = (abs(n.z) < 0.999f) ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);
  float3 t = normalize(cross(up, n));
  float3 b = cross(n, t);
  return float3x3(t, b, n);
}

float3 sample_cosine_hemisphere(float2 u) {
  float r = sqrt(u.x);
  float phi = 6.28318530718f * u.y;
  float x = r * cos(phi);
  float y = r * sin(phi);
  float z = sqrt(saturate(1.0f - x * x - y * y));
  return float3(x, y, z);
}

float evaluate_ao(RaytracingAccelerationStructure as, float3 position, float3 normal, float radius, uint seed) {
  const uint kSampleCount = 8u;
  float occluded = 0.0f;
  float3x3 basis = basis_from_normal(normal);

  [unroll]
  for (uint i = 0u; i < kSampleCount; ++i) {
    float2 u = float2(rnd01(seed), rnd01(seed));
    float3 local_dir = sample_cosine_hemisphere(u);
    float3 world_dir = normalize(mul(local_dir, basis));

    RayDesc ray;
    ray.Origin = position + normal * 0.002f;
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

  return 1.0f - occluded / float(kSampleCount);
}

[numthreads(8, 8, 1)] void compute_main(uint3 dtid : SV_DispatchThreadID) {
  if (any(dtid.xy >= constants.camera.film_size))
    return;

  float2 pixel = float2(dtid.xy) + 0.5f;
  float2 uv = pixel / float2(constants.camera.film_size);
  float2 ndc = uv * 2.0f - 1.0f;
  ndc.y = -ndc.y;

  // Generate ray from camera
  // Match etx::generate_ray logic from scene_camera.hxx
  float3 s = ndc.x * constants.camera.side;
  float3 u = ndc.y * constants.camera.up / constants.camera.aspect;
  float3 ray_dir = normalize(constants.camera.tan_half_fov * (s + u) + constants.camera.direction);

  RayDesc ray;
  ray.Origin = constants.camera.position;
  ray.Direction = ray_dir;
  ray.TMin = 0.001f;
  ray.TMax = 10000.0f;

  RayQuery<RAY_FLAG_NONE> q;
  RaytracingAccelerationStructure as = bindless_accel_structs[NonUniformResourceIndex(constants.as_index)];

  q.TraceRayInline(as, RAY_FLAG_NONE, 0xFF, ray);
  q.Proceed();

  float4 color = float4(0.0f, 0.0f, 0.0f, 1.0f);

  if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
    const bool has_geometry_buffers = (constants.scene.triangles != INVALID_INDEX) && (constants.scene.vertex_positions != INVALID_INDEX) &&
                                      (constants.scene.vertex_normals != INVALID_INDEX) && (constants.scene.scene_globals != INVALID_INDEX);

    if (has_geometry_buffers) {
      ByteAddressBuffer triangle_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)];
      ByteAddressBuffer position_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_positions)];
      ByteAddressBuffer normal_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_normals)];
      ByteAddressBuffer scene_globals = bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)];

      uint vertex_count = scene_globals.Load(0u);
      uint triangle_count = scene_globals.Load(4u);
      float bounding_sphere_radius = asfloat(scene_globals.Load(44u));

      uint triangle_index = q.CommittedPrimitiveIndex();
      if (triangle_index < triangle_count) {
        TriangleData tri = load_triangle(triangle_buffer, triangle_index);

        bool valid_indices = (tri.i.x < vertex_count) && (tri.i.y < vertex_count) && (tri.i.z < vertex_count);
        if (valid_indices) {
          float2 bary = q.CommittedTriangleBarycentrics();
          float3 bc = float3(1.0f - bary.x - bary.y, bary.x, bary.y);

          float3 p0 = load_float3(position_buffer, tri.i.x);
          float3 p1 = load_float3(position_buffer, tri.i.y);
          float3 p2 = load_float3(position_buffer, tri.i.z);
          float3 hit_position = p0 * bc.x + p1 * bc.y + p2 * bc.z;

          float3 n0 = load_float3(normal_buffer, tri.i.x);
          float3 n1 = load_float3(normal_buffer, tri.i.y);
          float3 n2 = load_float3(normal_buffer, tri.i.z);
          float3 hit_normal = normalize(n0 * bc.x + n1 * bc.y + n2 * bc.z);

          float ao_radius = max(0.1f, 0.05f * bounding_sphere_radius);
          uint seed = hash_u32((dtid.x * 73856093u) ^ (dtid.y * 19349663u) ^ (constants.frame_index * 83492791u) ^ (constants.sample_index * 2654435761u));
          float ao = evaluate_ao(as, hit_position, hit_normal, ao_radius, seed);

          float3 shaded = default_ao_shading(hit_normal, ao);
          shaded = material_scattering_integrated_or_fallback(tri.material_index, ao, shaded);
          color = float4(shaded, 1.0f);
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
    // Gradient sky
    float t = 0.5f * (ray_dir.y + 1.0f);
    color = lerp(float4(1.0f, 1.0f, 1.0f, 1.0f), float4(0.5f, 0.7f, 1.0f, 1.0f), t);
  }

  bindless_storage_textures[constants.output_image_index][dtid.xy] = color;
}
