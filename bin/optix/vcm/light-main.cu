#include <etx/rt/shared/optix.hxx>
#include <etx/rt/shared/vcm_shared.hxx>

using namespace etx;

static __constant__ VCMGlobal global;

ETX_GPU_CODE void continue_tracing(VCMIteration& iteration, const VCMPathState& state) {
  int i = atomicAdd(&iteration.active_light_paths, 1);
  global.output_state[i] = state;
}

ETX_GPU_CODE void push_light_vertex(VCMIteration& iteration, const VCMLightVertex& v) {
  int i = atomicAdd(&iteration.light_vertices, 1);
  global.light_vertices[i] = v;
}

ETX_GPU_CODE void project(const float2& ndc_coord, const float3& value) {
  float2 uv = ndc_coord * 0.5f + 0.5f;
  uint2 dim = global.scene.camera.image_size;
  uint32_t ax = static_cast<uint32_t>(uv.x * float(dim.x));
  uint32_t ay = static_cast<uint32_t>(uv.y * float(dim.y));
  if ((ax < dim.x) && (ay < dim.y)) {
    uint32_t i = ax + ay * dim.x;
    float4& current = global.light_iteration_image[i];
    atomicAdd(&current.x, value.x);
    atomicAdd(&current.y, value.y);
    atomicAdd(&current.z, value.z);
  }
}

RAYGEN(main) {
  const uint32_t opt_max_depth = 16u;
  const uint32_t opt_rr_start = 5u;

  uint3 idx = optixGetLaunchIndex();
  uint3 dim = optixGetLaunchDimensions();
  uint32_t index = idx.x + idx.y * dim.x;
  auto& state = global.input_state[index];
  auto& iteration = *global.iteration;

  Raytracing rt;

  Intersection intersection;
  bool found_intersection = rt.trace(global.scene, state.ray, intersection, state.sampler);

  Medium::Sample medium_sample = vcm_try_sampling_medium(global.scene, state, found_intersection ? intersection.t : kMaxFloat);

  if (medium_sample.sampled_medium()) {
    vcm_handle_sampled_medium(global.scene, medium_sample, state);
    continue_tracing(iteration, state);
    return;
  }

  if (found_intersection == false) {
    // TODO : finish iteration
    return;
  }

  state.path_distance += intersection.t;
  state.path_length += 1;
  const auto& tri = global.scene.triangles[intersection.triangle_index];
  const auto& mat = global.scene.materials[tri.material_index];

  if (vcm_handle_boundary_bsdf(global.scene, mat, intersection, state)) {
    continue_tracing(iteration, state);
    return;
  }

  vcm_update_light_vcm(intersection, state);

  if (bsdf::is_delta(mat, intersection.tex, global.scene, state.sampler) == false) {
    push_light_vertex(iteration, {state, intersection.pos, intersection.barycentric, intersection.triangle_index, index});

    if (state.path_length + 1 <= opt_max_depth) {
      float2 uv = {};
      auto value = vcm_connect_to_camera(rt, global.scene, intersection, mat, tri, iteration, state, uv);
      if (dot(value, value) > 0.0f) {
        project(uv, value);
      }
    }
  }

  if (vcm_next_ray(global.scene, PathSource::Light, intersection, opt_rr_start, state, iteration)) {
    continue_tracing(iteration, state);
  }
}
