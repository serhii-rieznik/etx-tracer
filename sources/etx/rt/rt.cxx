#include <etx/core/core.hxx>
#include <etx/render/host/scene_global.hxx>
#include <etx/rt/rt.hxx>

#include <etx/render/host/film.hxx>
#include <etx/render/host/emitter_packing.hxx>
#include <etx/render/host/scene_data.hxx>
#include <etx/render/shared/sampler.hxx>

#include <embree4/rtcore.h>

namespace etx {
namespace {

uint32_t embree_hit_instance_index(const RTCFilterFunctionNArguments* args) {
  return RTCHitN_instID(args->hit, args->N, 0u, 0u);
}

uint32_t embree_hit_triangle_index(const Scene& scene, const RTCFilterFunctionNArguments* args) {
  const uint32_t primitive_index = RTCHitN_primID(args->hit, args->N, 0u);
  const uint32_t instance_index = embree_hit_instance_index(args);
  if ((instance_index == RTC_INVALID_GEOMETRY_ID) || (instance_index >= scene.instances.count)) {
    return primitive_index;
  }
  const SceneInstance& instance = scene.instances[instance_index];
  if (instance.mesh_index >= scene.meshes.count) {
    return kInvalidIndex;
  }
  return scene.meshes[instance.mesh_index].triangle_offset + primitive_index;
}

}  // namespace

struct RaytracingImpl {
  TaskScheduler& scheduler;
  Film& film;

  Scene scene = {};
  RTCDevice rt_device = {};
  RTCScene rt_scene = {};
  std::vector<RTCScene> mesh_scenes = {};
  std::vector<RTCGeometry> instance_geometries = {};

  struct InternalSceneData {
    Camera camera = {};
    Distribution emitters_distribution = {};
    std::vector<Distribution::Entry> emitters_distribution_storage = {};
    std::vector<EmitterProfile> emitter_profiles = {};
    std::vector<Emitter> emitter_instances = {};
    std::vector<Triangle> triangles = {};
    std::vector<SceneInstance> instances = {};
    PackedEmitterTopology emitter_topology = {};
  } internal_data;

  RaytracingImpl(TaskScheduler& s, Film& f)
    : scheduler(s)
    , film(f) {
    rt_device = rtcNewDevice(nullptr);
    const auto version_major = rtcGetDeviceProperty(rt_device, RTC_DEVICE_PROPERTY_VERSION_MAJOR);
    const auto version_minor = rtcGetDeviceProperty(rt_device, RTC_DEVICE_PROPERTY_VERSION_MINOR);
    const auto version_patch = rtcGetDeviceProperty(rt_device, RTC_DEVICE_PROPERTY_VERSION_PATCH);
    log::warning("Embree version: %u.%u.%u", version_major, version_minor, version_patch);
  }

  ~RaytracingImpl() {
    scene_global_clear(this);
    release_host_scene();

    if (rt_device) {
      rtcReleaseDevice(rt_device);
      rt_device = {};
    }
  }

  void commit(const SceneData& scene_data, const Camera& camera, const UpdateFlags& update_flags) {
    ETX_PROFILER_SCOPE();
    internal_data.camera = camera;

    if (update_flags[UpdateFlags::AnyGeometry] || update_flags[UpdateFlags::AnyMaterials] || update_flags[UpdateFlags::Emitters] || update_flags[UpdateFlags::Images] ||
        update_flags[UpdateFlags::Mediums]) {
      update_scene_data(scene_data, update_flags);
    }

    if (update_flags[UpdateFlags::Options]) {
      update_scene_options(scene_data);
    }

    if (update_flags[UpdateFlags::AnyGeometryStructure]) {
      release_host_scene();
      build_host_scene(scene);
    } else if (update_flags[UpdateFlags::Transforms]) {
      update_host_scene_transforms(scene);
    }
    film.allocate(internal_data.camera.film_size);
    scene.options.properties[Scene::Properties::Committed] = true;
    scene_global_publish(this, &scene);
  }

  void compute_scene_bounding_volumes(const SceneData& scene_data) {
    auto bbox = scene_data.compute_bounding_volumes();
    scene.bounding_box_min = bbox.p_min;
    scene.bounding_box_max = bbox.p_max;
    scene.bounding_sphere_center = 0.5f * (scene.bounding_box_min + scene.bounding_box_max);
    scene.bounding_sphere_radius = length(scene.bounding_box_max - scene.bounding_sphere_center);
  }

  void update_scene_options(const SceneData& scene_data) {
    scene.options = scene_data.options;
  }

  void update_all_scene_views(const SceneData& scene_data) {
    scene.spectrums = {scene_data.spectrum_values.data(), scene_data.spectrum_values.size()};
    scene.images = {scene_data.images.as_array(), scene_data.images.array_size()};
    scene.vertices.pos = {scene_data.vertices.pos.data(), scene_data.vertices.pos.size()};
    scene.vertices.nrm = {scene_data.vertices.nrm.data(), scene_data.vertices.nrm.size()};
    scene.vertices.tan = {scene_data.vertices.tan.data(), scene_data.vertices.tan.size()};
    scene.vertices.btn = {scene_data.vertices.btn.data(), scene_data.vertices.btn.size()};
    scene.vertices.tex = {scene_data.vertices.tex.data(), scene_data.vertices.tex.size()};
    scene.triangles = {internal_data.triangles.data(), internal_data.triangles.size()};
    scene.meshes = {scene_data.meshes.data(), scene_data.meshes.size()};
    scene.instances = {internal_data.instances.data(), internal_data.instances.size()};
    scene.materials = {scene_data.materials.data(), scene_data.materials.size()};
    scene.mediums = {scene_data.mediums.as_array(), scene_data.mediums.array_size()};
    scene.energy_compensation_interfaces = {scene_data.energy_compensation_interfaces.data(), scene_data.energy_compensation_interfaces.size()};
    scene.emitter_profiles = {internal_data.emitter_profiles.data(), internal_data.emitter_profiles.size()};
    scene.emitter_instances = {internal_data.emitter_instances.data(), internal_data.emitter_instances.size()};
    scene.pixel_sampler = scene_data.pixel_filter;
    scene.defaults = scene_data.defaults;
  }

  void update_scene_data(const SceneData& scene_data, const UpdateFlags& update_flags) {
    ETX_PROFILER_SCOPE();
    const bool packed_emitters_changed = update_flags[UpdateFlags::VerticesPos] || update_flags[UpdateFlags::Triangles] || update_flags[UpdateFlags::Meshes] ||
                                         update_flags[UpdateFlags::Hierarchy] || update_flags[UpdateFlags::Transforms] || update_flags[UpdateFlags::Attachments] ||
                                         update_flags[UpdateFlags::Materials] || update_flags[UpdateFlags::Spectra] || update_flags[UpdateFlags::Emitters];
    if (packed_emitters_changed) {
      ETX_PROFILER_NAMED_SCOPE("update_instances_triangles_and_emitters");
      const bool transform_only = update_flags[UpdateFlags::Transforms] && (update_flags[UpdateFlags::VerticesPos] == false) && (update_flags[UpdateFlags::Triangles] == false) &&
                                  (update_flags[UpdateFlags::Meshes] == false) && (update_flags[UpdateFlags::Hierarchy] == false) &&
                                  (update_flags[UpdateFlags::Attachments] == false) && (update_flags[UpdateFlags::Materials] == false) &&
                                  (update_flags[UpdateFlags::Spectra] == false) && (update_flags[UpdateFlags::Emitters] == false);
      PackedEmitterData packed_emitters =
        transform_only ? build_packed_emitters_for_transforms(scene_data, internal_data.emitter_topology) : build_packed_emitters(scene_data, internal_data.emitter_topology);
      internal_data.emitters_distribution_storage = build_packed_emitter_distribution(packed_emitters);
      internal_data.emitter_profiles = std::move(packed_emitters.emitter_profiles);
      internal_data.emitter_instances = std::move(packed_emitters.emitter_instances);
      if (transform_only == false) {
        internal_data.triangles = std::move(packed_emitters.triangles);
      }
      internal_data.instances = std::move(packed_emitters.instances);
      scene.environment_emitters = packed_emitters.environment_emitters;

      if (internal_data.emitters_distribution_storage.empty()) {
        internal_data.emitters_distribution = {};
      } else {
        const uint32_t distribution_count = static_cast<uint32_t>(internal_data.emitters_distribution_storage.size() - 1u);
        internal_data.emitters_distribution = Distribution::build(internal_data.emitters_distribution_storage.data(), distribution_count);
      }
      scene.emitters_distribution = internal_data.emitters_distribution;
    }

    if (update_flags[UpdateFlags::VerticesPos] || update_flags[UpdateFlags::Triangles] || update_flags[UpdateFlags::Transforms] || update_flags[UpdateFlags::Hierarchy] ||
        update_flags[UpdateFlags::Attachments] || scene.bounding_sphere_radius == 0.0f) {
      compute_scene_bounding_volumes(scene_data);
    }

    update_all_scene_views(scene_data);
  }

  void build_host_scene(const Scene& s) {
    ETX_PROFILER_SCOPE();
    rtcSetDeviceErrorFunction(
      rt_device,
      [](void* userPtr, enum RTCError code, const char* str) {
        log::error("Embree error: %u (%s)", code, str);
      },
      nullptr);

    rt_scene = rtcNewScene(rt_device);
    mesh_scenes.assign(s.meshes.count, nullptr);
    instance_geometries.assign(s.instances.count, nullptr);
    for (uint32_t mesh_index = 0u; mesh_index < s.meshes.count; ++mesh_index) {
      const Mesh& mesh = s.meshes[mesh_index];
      if ((mesh.triangle_count == 0u) || ((mesh.triangle_offset + mesh.triangle_count) > s.triangles.count)) {
        continue;
      }

      RTCScene mesh_scene = rtcNewScene(rt_device);
      RTCGeometry geometry = rtcNewGeometry(rt_device, RTCGeometryType::RTC_GEOMETRY_TYPE_TRIANGLE);
      rtcSetSharedGeometryBuffer(geometry, RTCBufferType::RTC_BUFFER_TYPE_VERTEX, 0, RTCFormat::RTC_FORMAT_FLOAT3, s.vertices.pos.a, 0, sizeof(float3), s.vertices.pos.count);
      rtcSetSharedGeometryBuffer(geometry, RTCBufferType::RTC_BUFFER_TYPE_INDEX, 0, RTCFormat::RTC_FORMAT_UINT3, s.triangles.a, mesh.triangle_offset * sizeof(Triangle),
        sizeof(Triangle), mesh.triangle_count);
      rtcCommitGeometry(geometry);
      rtcAttachGeometry(mesh_scene, geometry);
      rtcReleaseGeometry(geometry);
      rtcCommitScene(mesh_scene);
      mesh_scenes[mesh_index] = mesh_scene;
    }

    for (uint32_t instance_index = 0u; instance_index < s.instances.count; ++instance_index) {
      const SceneInstance& instance = s.instances[instance_index];
      if ((instance.mesh_index >= mesh_scenes.size()) || (mesh_scenes[instance.mesh_index] == nullptr)) {
        continue;
      }
      RTCGeometry geometry = rtcNewGeometry(rt_device, RTCGeometryType::RTC_GEOMETRY_TYPE_INSTANCE);
      rtcSetGeometryInstancedScene(geometry, mesh_scenes[instance.mesh_index]);
      rtcSetGeometryTransform(geometry, 0u, RTCFormat::RTC_FORMAT_FLOAT3X4_ROW_MAJOR, instance.object_to_world.rows);
      rtcSetGeometryMask(geometry, (instance.flags & SceneInstance::Enabled) != 0u ? 0xffffffffu : 0u);
      rtcCommitGeometry(geometry);
      rtcAttachGeometryByID(rt_scene, geometry, instance_index);
      instance_geometries[instance_index] = geometry;
    }
    rtcCommitScene(rt_scene);
  }

  void update_host_scene_transforms(const Scene& s) {
    ETX_PROFILER_SCOPE();
    if ((rt_scene == nullptr) || (instance_geometries.size() != s.instances.count)) {
      release_host_scene();
      build_host_scene(s);
      return;
    }

    for (uint32_t instance_index = 0u; instance_index < s.instances.count; ++instance_index) {
      RTCGeometry geometry = instance_geometries[instance_index];
      if (geometry == nullptr) {
        release_host_scene();
        build_host_scene(s);
        return;
      }
      rtcSetGeometryTransform(geometry, 0u, RTCFormat::RTC_FORMAT_FLOAT3X4_ROW_MAJOR, s.instances[instance_index].object_to_world.rows);
      rtcSetGeometryMask(geometry, (s.instances[instance_index].flags & SceneInstance::Enabled) != 0u ? 0xffffffffu : 0u);
      rtcCommitGeometry(geometry);
    }
    rtcCommitScene(rt_scene);
  }

  void release_host_scene() {
    for (RTCGeometry geometry : instance_geometries) {
      if (geometry != nullptr) {
        rtcReleaseGeometry(geometry);
      }
    }
    instance_geometries.clear();
    if (rt_scene) {
      rtcReleaseScene(rt_scene);
      rt_scene = {};
    }
    for (RTCScene mesh_scene : mesh_scenes) {
      if (mesh_scene != nullptr) {
        rtcReleaseScene(mesh_scene);
      }
    }
    mesh_scenes.clear();
  }

  template <class T>
  constexpr static uint64_t array_size(const ArrayView<T>& a) {
    return align_up(a.count * sizeof(T), 16llu);
  }

  void trace_with_function(const Ray& r, RTCRayQueryContext* context, RTCFilterFunctionN filter_funtion) const {
    ETX_CHECK_FINITE(r.o);
    ETX_CHECK_FINITE(r.d);

    rtcInitRayQueryContext(context);

    RTCIntersectArguments args = {};
    rtcInitIntersectArguments(&args);

    args.context = context;
    args.feature_mask = static_cast<RTCFeatureFlags>(RTC_FEATURE_FLAG_TRIANGLE | RTC_FEATURE_FLAG_INSTANCE | RTC_FEATURE_FLAG_FILTER_FUNCTION_IN_ARGUMENTS);
    args.flags = RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER;
    args.filter = filter_funtion;

    RTCRayHit ray_hit = {};
    ray_hit.ray.dir_x = r.d.x;
    ray_hit.ray.dir_y = r.d.y;
    ray_hit.ray.dir_z = r.d.z;
    ray_hit.ray.org_x = r.o.x;
    ray_hit.ray.org_y = r.o.y;
    ray_hit.ray.org_z = r.o.z;
    ray_hit.ray.tnear = r.min_t;
    ray_hit.ray.tfar = r.max_t;
    ray_hit.ray.mask = kInvalidIndex;
    ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rtcIntersect1(rt_scene, &ray_hit, &args);
  }
};

ETX_PIMPL_IMPLEMENT(Raytracing, Impl);

Raytracing::Raytracing(TaskScheduler& s, Film& f) {
  ETX_PIMPL_INIT(Raytracing, s, f);
}

Raytracing::~Raytracing() {
  ETX_PIMPL_CLEANUP(Raytracing);
}

TaskScheduler& Raytracing::scheduler() {
  return _private->scheduler;
}

const Camera& Raytracing::camera() const {
  return _private->internal_data.camera;
}

const Scene& Raytracing::scene() const {
  return _private->scene;
}

void Raytracing::commit(const SceneData& scene_data, const Camera& camera, const UpdateFlags& changes) {
  _private->commit(scene_data, camera, changes);
}

bool Raytracing::trace_material(const Scene& scene, const Ray& r, const uint32_t material_id, Intersection& result_intersection, Sampler& smp) const {
  struct IntersectionContextExt {
    RTCRayQueryContext context;
    IntersectionBase i;
    const Scene* scene;
    Sampler* smp;
    uint32_t m_id;
  } context = {{}, {{}, kInvalidIndex, 0.0f, kInvalidIndex}, &scene, &smp, material_id};

  auto filter_funtion = [](const struct RTCFilterFunctionNArguments* args) {
    auto ctx = reinterpret_cast<IntersectionContextExt*>(args->context);

    const uint32_t triangle_index = embree_hit_triangle_index(*ctx->scene, args);
    const uint32_t instance_index = embree_hit_instance_index(args);
    if (triangle_index >= ctx->scene->triangles.count) {
      *args->valid = 0;
      return;
    }
    const auto& tri = ctx->scene->triangles[triangle_index];

    if ((ctx->m_id != kInvalidIndex) && (tri.material_index != ctx->m_id)) {
      *args->valid = 0;
      return;
    }

    const auto& mat = ctx->scene->materials[tri.material_index];
    if (mat.cls == MaterialClass::Void) {
      *args->valid = 0;
      return;
    }

    float u = RTCHitN_u(args->hit, args->N, 0);
    float v = RTCHitN_v(args->hit, args->N, 0);
    float2 uv = lerp_uv(*ctx->scene, tri, barycentrics({u, v}));
    if (alpha_test_pass(mat, uv, *ctx->smp)) {
      *args->valid = 0;
      return;
    }

    ctx->i = {{u, v}, triangle_index, RTCRayN_tfar(args->ray, args->N, 0), instance_index};
  };

  ETX_ASSERT(_private != nullptr);
  _private->trace_with_function(r, &context.context, filter_funtion);

  if (context.i.triangle_index == kInvalidIndex)
    return false;

  result_intersection = make_intersection(scene, r.d, context.i);
  return true;
}

bool Raytracing::trace(const Scene& scene, const Ray& r, Intersection& result_intersection, Sampler& smp) const {
  struct IntersectionContextExt {
    RTCRayQueryContext context;
    IntersectionBase i;
    const Scene* scene;
    Sampler* smp;
  } context = {{}, {{}, kInvalidIndex, 0.0f, kInvalidIndex}, &scene, &smp};

  auto filter_funtion = [](const struct RTCFilterFunctionNArguments* args) {
    auto ctx = reinterpret_cast<IntersectionContextExt*>(args->context);

    const uint32_t triangle_index = embree_hit_triangle_index(*ctx->scene, args);
    const uint32_t instance_index = embree_hit_instance_index(args);
    if (triangle_index >= ctx->scene->triangles.count) {
      *args->valid = 0;
      return;
    }
    const auto& tri = ctx->scene->triangles[triangle_index];
    const auto& mat = ctx->scene->materials[tri.material_index];
    if (mat.cls == MaterialClass::Void) {
      *args->valid = 0;
      return;
    }
    const auto& scene = *ctx->scene;
    float u = RTCHitN_u(args->hit, args->N, 0);
    float v = RTCHitN_v(args->hit, args->N, 0);
    float2 uv = lerp_uv(scene, tri, barycentrics({u, v}));
    if (alpha_test_pass(mat, uv, *ctx->smp)) {
      *args->valid = 0;
      return;
    }

    ctx->i = {{u, v}, triangle_index, RTCRayN_tfar(args->ray, args->N, 0), instance_index};
  };

  ETX_ASSERT(_private != nullptr);
  _private->trace_with_function(r, &context.context, filter_funtion);

  if (context.i.triangle_index == kInvalidIndex)
    return false;

  result_intersection = make_intersection(scene, r.d, context.i);
  return true;
}

SpectralResponse Raytracing::trace_transmittance(const SpectralQuery spect, const Scene& scene, const float3& p0, const float3& p1, const MediumInstance& medium,
  Sampler& smp) const {
  ETX_ASSERT(_private != nullptr);

  constexpr uint32_t kIntersectionBufferSize = 63;
  struct IntermediateIntersection {
    uint32_t primitive_id;
    uint32_t instance_index;
    float u;
    float v;
    float t;
  };

  struct IntersectionContextExt : public RTCRayQueryContext {
    uint32_t intersection_count;
    uint32_t occlusion_found;
    const Scene& scene;
    Sampler& smp;
    IntermediateIntersection intersections[kIntersectionBufferSize + 1u];
  } context = {{}, 0u, 0u, scene, smp};

  auto filter_function = [](const struct RTCFilterFunctionNArguments* args) {
    auto ctx = reinterpret_cast<IntersectionContextExt*>(args->context);
    uint32_t triangle_index = embree_hit_triangle_index(ctx->scene, args);
    if (triangle_index >= ctx->scene.triangles.count) {
      ctx->occlusion_found = 1u;
      *args->valid = -1;
      return;
    }
    const auto u = RTCHitN_u(args->hit, args->N, 0);
    const auto v = RTCHitN_v(args->hit, args->N, 0);
    const auto& tri = ctx->scene.triangles[triangle_index];
    const auto& mat = ctx->scene.materials[tri.material_index];
    if (mat.cls == MaterialClass::Void) {
      *args->valid = 0;
      return;
    }
    float2 uv = lerp_uv(ctx->scene, tri, barycentrics({u, v}));
    if (alpha_test_pass(mat, uv, ctx->smp)) {
      *args->valid = 0;
      return;
    }
    if ((mat.cls != MaterialClass::Boundary) || ((ctx->intersection_count + 1u) >= kIntersectionBufferSize)) {
      ctx->occlusion_found = 1u;
      *args->valid = -1;
      return;
    }

    ctx->intersections[ctx->intersection_count++] = {
      .primitive_id = triangle_index,
      .instance_index = embree_hit_instance_index(args),
      .u = u,
      .v = v,
      .t = RTCRayN_tfar(args->ray, args->N, 0),
    };
    *args->valid = 0;
  };

  auto direction = p1 - p0;
  ETX_CHECK_FINITE(direction);

  float t_max = dot(direction, direction);
  if (t_max <= kRayEpsilon) {
    return {spect, 1.0f};
  }

  t_max = sqrtf(t_max);
  direction /= t_max;
  t_max -= fmaxf(kRayEpsilon, t_max * kRayEpsilon);
  ETX_VALIDATE(t_max);

  _private->trace_with_function({p0, direction, kRayEpsilon, t_max}, &context, filter_function);

  if (context.occlusion_found) {
    return {spect, 0.0f};
  }

  for (uint32_t i = 0; i < context.intersection_count; ++i) {
    for (uint32_t j = i + 1; j < context.intersection_count; ++j) {
      if (context.intersections[i].t > context.intersections[j].t) {
        std::swap(context.intersections[i], context.intersections[j]);
      }
    }
  }
  context.intersections[context.intersection_count++] = {kInvalidIndex, kInvalidIndex, 0.0f, 0.0f, t_max};

  float current_t = 0.0f;
  float3 origin = p0;
  SpectralResponse result = {spect, 1.0f};
  MediumInstance current_medium = medium;
  for (uint32_t i = 0; i < context.intersection_count; ++i) {
    const auto& intersection = context.intersections[i];
    if (medium_instance_valid(current_medium)) {
      float dt = fmaxf(0.0f, intersection.t - current_t);

      if (current_medium.index != kInvalidIndex) {
        const auto& m = scene.mediums[current_medium.index];
        result *= medium_transmittance(m, spect, smp, origin, direction, dt);
        ETX_VALIDATE(result);
      } else {
        spectral_response_mul_assign(result, medium_transmittance(current_medium, dt));
        ETX_VALIDATE(result);
      }
    }

    if (intersection.primitive_id == kInvalidIndex)
      break;

    const auto& tri = scene.triangles[intersection.primitive_id];
    const auto& mat = scene.materials[tri.material_index];
    const float3 geo_normal = scene_triangle_world_geometric_normal(scene, tri, intersection.instance_index);
    const bool entering_surface = dot(geo_normal, direction) < 0.0f;
    current_medium = {
      .index = entering_surface ? mat.int_medium : mat.ext_medium,
    };
    current_t = intersection.t;
    const float3 bc = barycentrics({intersection.u, intersection.v});
    origin = scene_triangle_world_position(scene, tri, 0u, intersection.instance_index) * bc.x + scene_triangle_world_position(scene, tri, 1u, intersection.instance_index) * bc.y +
             scene_triangle_world_position(scene, tri, 2u, intersection.instance_index) * bc.z;
  }

  return result;
}

const Film& Raytracing::film() const {
  return _private->film;
}

Film& Raytracing::film() {
  return _private->film;
}

}  // namespace etx
