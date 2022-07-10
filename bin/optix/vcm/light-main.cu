#include <etx/rt/shared/optix.hxx>
#include <etx/rt/shared/vcm_shared.hxx>

using namespace etx;

static __constant__ VCMGlobal global;

ETX_GPU_CODE void continue_tracing(VCMIteration& iteration, const VCMPathState& state) {
  int i = atomicAdd(&iteration.active_paths, 1u);
  global.output_state[i] = state;
}

ETX_GPU_CODE void push_light_vertex(VCMIteration& iteration, const VCMLightVertex& v) {
  uint32_t i = atomicAdd(&iteration.light_vertices, 1u);
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
  uint3 idx = optixGetLaunchIndex();
  uint3 dim = optixGetLaunchDimensions();
  auto& state = global.input_state[idx.x + idx.y * dim.x];
  if (state.path_length + 1 > global.options.max_depth) {
    return;
  }

  auto& iteration = *global.iteration;

  Raytracing rt = {};
  Intersection intersection;
  bool found_intersection = rt.trace(global.scene, state.ray, intersection, state.sampler);

  Medium::Sample medium_sample = vcm_try_sampling_medium(global.scene, state, found_intersection ? intersection.t : kMaxFloat);

  if (medium_sample.sampled_medium()) {
    vcm_handle_sampled_medium(global.scene, medium_sample, state);
    continue_tracing(iteration, state);
    return;
  }

  if (found_intersection == false) {
    return;
  }

  state.path_distance += intersection.t;
  const auto& tri = global.scene.triangles[intersection.triangle_index];
  const auto& mat = global.scene.materials[tri.material_index];

  if (vcm_handle_boundary_bsdf(global.scene, mat, intersection, PathSource::Light, state)) {
    continue_tracing(iteration, state);
    return;
  }

  state.path_length += 1;
  vcm_update_light_vcm(intersection, state);

  if (bsdf::is_delta(mat, intersection.tex, global.scene, state.sampler) == false) {
    push_light_vertex(iteration, {state, intersection.pos, intersection.barycentric, intersection.triangle_index, state.index});

    float2 uv = {};
    auto value = vcm_connect_to_camera(rt, global.scene, intersection, mat, tri, iteration, global.options, state, uv);
    if (dot(value, value) > 0.0f) {
      project(uv, value);
    }
  }

  if (vcm_next_ray(global.scene, PathSource::Light, intersection, global.options.rr_start, state, iteration)) {
    continue_tracing(iteration, state);
    return;
  }

  atomicAdd(&iteration.terminated_paths, 1u);
}
