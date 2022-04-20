#include <etx/rt/shared/optix.hxx>
#include <etx/rt/shared/path_tracing_shared.hxx>

using namespace etx;

static __constant__ PathTracingGPUData global;

RAYGEN(main) {
  uint3 idx = optixGetLaunchIndex();
  uint3 dim = optixGetLaunchDimensions();
  uint32_t index = idx.x + idx.y * dim.x;
  Sampler smp(index, 0);
  float3 rgb = float3{float(idx.x) / float(dim.x), float(idx.y) / float(dim.y), smp.next()};
  float3 xyz = spectrum::rgb_to_xyz(rgb);
  global.output[index] = {xyz.x, xyz.y, xyz.z, 1.0f};
}

CLOSEST_HIT(main_closest_hit) {
}

MISS(main_miss) {
}

CLOSEST_HIT(env_closest_hit) {
}

MISS(env_miss) {
}

/*
#include <etx/optix/gpushared/etx_optix.hpp>

#include <etx/rt/generic/generic_scene.hpp>
#include <etx/rt/generic/generic_spectral.hpp>

using namespace etx;

struct alignas(16) LaunchParams {
  CameraData camera = {};
  SceneData scene = {};
  ArrayView<float4> output = {};
  ArrayView<float4> overlay = {};
  uint32_t frame_index = {};
  uint32_t selected_material = {};
};

static __constant__ LaunchParams input;

inline __device__ void project(float2 uv) {
  uv = (uv * 0.5f + 0.5f) * float2{input.camera.film_width, input.camera.film_height};
  if ((uv.x >= 0.0f) && (uv.y >= 0.0f) && (uv.x < input.camera.film_width) && (uv.y < input.camera.film_height)) {
    uint32_t x = static_cast<uint32_t>(uv.x);
    uint32_t y = static_cast<uint32_t>(uv.y);
    input.output[x + y * uint32_t(input.camera.film_width)] = {1.0f, 0.25f, 0.25f, 1.0f};
  }
}

RAYGEN(main) {
  uint3 dim = optixGetLaunchDimensions();
  uint3 idx = optixGetLaunchIndex();

  uint32_t index = idx.x + idx.y * dim.x;
  uint32_t rnd_seed = random_seed(index, input.frame_index);

  Sampler smp(rnd_seed);
  float2 uv = get_jittered_uv(smp, {idx.x, idx.y}, {dim.x, dim.y});
  auto ray = generate_ray(smp, input.scene, input.camera, uv);

  optixTrace(input.scene.acceleration_structure, ray.o, ray.d, ray.min_t, ray.max_t, 0.0f,  //
    OptixVisibilityMask(255), OptixRayFlags(0), 0, 0, 0, rnd_seed);                         //

  if (input.selected_material != kInvalidIndex) {
    optixTrace(input.scene.acceleration_structure, ray.o, ray.d, ray.min_t, ray.max_t, 0.0f,  //
      OptixVisibilityMask(255), OptixRayFlags(0), 2, 0, 2, rnd_seed);                         //
  }
}

CLOSEST_HIT(main_closest_hit) {
  Sampler smp(optixGetPayload_0());

  const Triangle& tri = input.scene.triangles[optixGetPrimitiveIndex()];
  float3 bc = barycentrics(optixGetTriangleBarycentrics());
  Vertex v = lerp(input.scene.vertices, tri, bc);

  float3 w_o = sample_cosine_distribution(smp.next(), smp.next(), v.nrm, 1.0f);
  float3 off_p = shading_pos(input.scene.vertices, tri, bc, w_o);
  optixTrace(input.scene.acceleration_structure, off_p, w_o, 0.0f, 1.0e+38f, 0.0f,               //
    OptixVisibilityMask(255), OptixRayFlags(OPTIX_RAY_FLAG_DISABLE_ANYHIT), 1, 0, 1, smp.seed);  //

  optixSetPayload_0(smp.seed);
}

MISS(main_miss) {
  float4 result = {0.0f, 0.0f, 0.0f, 1.0f};
  uint3 dim = optixGetLaunchDimensions();
  uint3 idx = optixGetLaunchIndex();
  float4 gathered = (input.frame_index == 0.0) ? float4{} : input.output[idx.x + idx.y * dim.x];
  gathered = lerp(gathered, result, 1.0f / (1.0f + float(input.frame_index)));
  input.output[idx.x + idx.y * dim.x] = gathered;
}

ANY_HIT(sel_any_hit) {
  Sampler smp(optixGetPayload_0());

  const Triangle& tri = input.scene.triangles[optixGetPrimitiveIndex()];
  float3 bc = barycentrics(optixGetTriangleBarycentrics());
  Vertex v = lerp(input.scene.vertices, tri, bc);

  float dist = length(input.camera.position - v.pos) / 1000.0f;
  if ((tri.material_index == input.selected_material) && (min(bc.x, min(bc.y, bc.z)) < dist)) {
    auto camera_sample = sample_film(smp, input.scene, input.camera, v.pos);
    if (camera_sample.pdf_dir > 0.0f) {
      project(camera_sample.uv);
    }
  }
  optixIgnoreIntersection();
  optixSetPayload_0(smp.seed);
}

MISS(sel_miss) {
}

CLOSEST_HIT(env_closest_hit) {
  float t_max = 1.0f - exp(-0.0625 * optixGetRayTmax());
  float4 result = spectrum::rgb_to_xyz4({t_max, t_max, t_max});
  uint3 dim = optixGetLaunchDimensions();
  uint3 idx = optixGetLaunchIndex();
  float4 gathered = (input.frame_index == 0.0) ? float4{} : input.output[idx.x + idx.y * dim.x];
  gathered = lerp(gathered, result, 1.0f / (1.0f + float(input.frame_index)));
  input.output[idx.x + idx.y * dim.x] = gathered;
}

MISS(env_miss) {
  float4 result = spectrum::rgb_to_xyz4({1.0f, 1.0f, 1.0});

  uint3 dim = optixGetLaunchDimensions();
  uint3 idx = optixGetLaunchIndex();
  float4 gathered = (input.frame_index == 0.0) ? float4{} : input.output[idx.x + idx.y * dim.x];
  gathered = lerp(gathered, result, 1.0f / (1.0f + float(input.frame_index)));
  input.output[idx.x + idx.y * dim.x] = gathered;
}

// */
