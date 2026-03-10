#include <etx/core/core.hxx>
#include <etx/rt/rt.hxx>
#include <etx/rt/scene_global.hxx>

#include <etx/render/host/film.hxx>
#include <etx/render/host/scene_data.hxx>
#include <etx/render/shared/sampler.hxx>

#include <embree4/rtcore.h>

namespace etx {

struct RaytracingImpl {
  TaskScheduler& scheduler;
  Film& film;

  Scene scene = {};
  RTCDevice rt_device = {};
  RTCScene rt_scene = {};

  struct InternalSceneData {
    Camera camera = {};
    Distribution emitters_distribution = {};
    std::vector<Distribution::Entry> emitters_distribution_storage = {};
    std::vector<EmitterProfile> emitter_profiles = {};
    std::vector<Emitter> emitter_instances = {};
    std::vector<Triangle> triangles = {};
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

    if (update_flags[UpdateFlags::AnyGeometryStructure] || update_flags[UpdateFlags::EmbreeScene]) {
      release_host_scene();
      build_host_scene(scene);
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
    scene.materials = {scene_data.materials.data(), scene_data.materials.size()};
    scene.mediums = {scene_data.mediums.as_array(), scene_data.mediums.array_size()};
    scene.emitter_profiles = {internal_data.emitter_profiles.data(), internal_data.emitter_profiles.size()};
    scene.emitter_instances = {internal_data.emitter_instances.data(), internal_data.emitter_instances.size()};
    scene.pixel_sampler = scene_data.pixel_filter;
    scene.defaults = scene_data.defaults;
  }

  void update_scene_data(const SceneData& scene_data, const UpdateFlags& update_flags) {
    ETX_PROFILER_SCOPE();
    if (update_flags[UpdateFlags::Triangles] || update_flags[UpdateFlags::Emitters]) {
      ETX_PROFILER_NAMED_SCOPE("update_triangles_and_emitters");
      update_triangles_internal(scene_data);
      update_emitters_internal(scene_data);
    }

    if (update_flags[UpdateFlags::VerticesPos] || update_flags[UpdateFlags::Triangles] || scene.bounding_sphere_radius == 0.0f) {
      compute_scene_bounding_volumes(scene_data);
    }

    if (update_flags[UpdateFlags::Emitters] || update_flags[UpdateFlags::AnyMaterials] || update_flags[UpdateFlags::Triangles]) {
      ETX_PROFILER_NAMED_SCOPE("build_emitters_distribution");
      build_emitters_distribution(scene_data);
    }

    update_all_scene_views(scene_data);
  }

  void build_emitters_distribution(const SceneData& scene_data) {
    ETX_PROFILER_SCOPE();
    const auto bbox = scene_data.compute_bounding_volumes();
    const float3 bounding_sphere_center = 0.5f * (bbox.p_min + bbox.p_max);
    const float bounding_sphere_radius = length(bbox.p_max - bounding_sphere_center);

    scene.environment_emitters.count = 0u;

    for (uint32_t i = 0u; i < static_cast<uint32_t>(internal_data.emitter_profiles.size()); ++i) {
      auto& profile = internal_data.emitter_profiles[i];
      if (profile.cls == EmitterProfile::Class::Directional) {
        profile.directional.equivalent_disk_size = 2.0f * std::tan(profile.directional.angular_size * 0.5f);
        profile.directional.angular_size_cosine = std::cos(profile.directional.angular_size * 0.5f);
      }
    }

    std::vector<uint32_t> active_emitter_indices = {};
    active_emitter_indices.reserve(internal_data.emitter_instances.size());

    for (uint32_t i = 0u; i < static_cast<uint32_t>(internal_data.emitter_instances.size()); ++i) {
      auto& emitter = internal_data.emitter_instances[i];
      const auto& profile = internal_data.emitter_profiles[emitter.profile];

      const float spectrum_weight = (profile.emission.spectrum_index != kInvalidIndex) ? scene_data.spectrum_values[profile.emission.spectrum_index].luminance() : 0.0f;
      emitter.spectrum_weight = spectrum_weight;

      if ((profile.cls == EmitterProfile::Class::Directional) || (profile.cls == EmitterProfile::Class::Environment)) {
        const float additional_weight = kPi * bounding_sphere_radius * bounding_sphere_radius;
        emitter.additional_weight = additional_weight;
      }

      const float total_weight = emitter.spectrum_weight * emitter.additional_weight;
      if (total_weight > 0.0f) {
        active_emitter_indices.push_back(i);
      }
    }

    for (uint32_t emitter_idx : active_emitter_indices) {
      const auto& emitter = internal_data.emitter_instances[emitter_idx];
      if ((emitter.cls == EmitterProfile::Class::Directional) || (emitter.cls == EmitterProfile::Class::Environment)) {
        if (scene.environment_emitters.count < SceneLimits::MaxEnvironmentEmitters) {
          scene.environment_emitters.emitters[scene.environment_emitters.count++] = emitter_idx;
        }
      }
    }

    const uint32_t active_count = static_cast<uint32_t>(active_emitter_indices.size());
    internal_data.emitters_distribution_storage.resize(static_cast<size_t>(active_count) + 1u);

    auto* entries = internal_data.emitters_distribution_storage.data();
    for (uint32_t i = 0u; i < active_count; ++i) {
      const uint32_t emitter_idx = active_emitter_indices[i];
      const auto& emitter = internal_data.emitter_instances[emitter_idx];
      const float total_weight = emitter.spectrum_weight * emitter.additional_weight;
      entries[i] = {total_weight, 0.0f, 0.0f, emitter_idx};
    }

    internal_data.emitters_distribution = Distribution::build(entries, active_count);

    scene.emitters_distribution = internal_data.emitters_distribution;

    log::info("Built emitters distribution for %u emitters (%u active)", static_cast<uint32_t>(internal_data.emitter_instances.size()), active_count);
  }

  void update_triangles_internal(const SceneData& scene_data) {
    internal_data.triangles = scene_data.triangles;
    for (auto& tri : internal_data.triangles) {
      tri.emitter_index = kInvalidIndex;
    }
  }

  void update_emitters_internal(const SceneData& scene_data) {
    internal_data.emitter_instances.clear();
    internal_data.emitter_profiles.clear();

    internal_data.emitter_profiles = scene_data.emitter_profiles;

    for (uint32_t i = 0u; i < static_cast<uint32_t>(scene_data.emitter_profiles.size()); ++i) {
      const auto& profile = scene_data.emitter_profiles[i];
      if (profile.cls != EmitterProfile::Class::Area) {
        Emitter& emitter = internal_data.emitter_instances.emplace_back(profile.cls);
        emitter.profile = i;
        emitter.triangle_index = kInvalidIndex;
        emitter.spectrum_weight = (profile.emission.spectrum_index != kInvalidIndex) ? scene_data.spectrum_values[profile.emission.spectrum_index].luminance() : 0.0f;
        emitter.additional_weight = (profile.cls == EmitterProfile::Class::Directional) ? kPi : (4.0f * kPi);
      }
    }

    for (size_t tri_index = 0u; tri_index < scene_data.triangles.size(); ++tri_index) {
      const Triangle& tri = scene_data.triangles[tri_index];
      if ((tri.emitter_index != kInvalidIndex) && (tri.emitter_index < scene_data.emitter_profiles.size())) {
        const auto& profile = scene_data.emitter_profiles[tri.emitter_index];
        if (profile.cls == EmitterProfile::Class::Area) {
          Emitter& emitter = internal_data.emitter_instances.emplace_back(EmitterProfile::Class::Area);
          emitter.profile = tri.emitter_index;
          emitter.triangle_index = static_cast<uint32_t>(tri_index);

          if (tri.material_index < scene_data.materials.size()) {
            const Material& mtl = scene_data.materials[tri.material_index];
            const float spectrum_weight = (profile.emission.spectrum_index != kInvalidIndex) ? scene_data.spectrum_values[profile.emission.spectrum_index].luminance() : 0.0f;

            const float3& v0 = scene_data.vertices.pos[tri.i[0]];
            const float3& v1 = scene_data.vertices.pos[tri.i[1]];
            const float3& v2 = scene_data.vertices.pos[tri.i[2]];
            const float triangle_area = 0.5f * length(cross(v1 - v0, v2 - v0));

            emitter.triangle_area = triangle_area;
            emitter.spectrum_weight = spectrum_weight;
            emitter.additional_weight = (mtl.two_sided ? 2.0f : 1.0f) * triangle_area * kPi;
          }

          internal_data.triangles[tri_index].emitter_index = static_cast<uint32_t>(internal_data.emitter_instances.size() - 1u);
        }
      }
    }
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

    auto geometry = rtcNewGeometry(rt_device, RTCGeometryType::RTC_GEOMETRY_TYPE_TRIANGLE);

    rtcSetSharedGeometryBuffer(geometry, RTCBufferType::RTC_BUFFER_TYPE_VERTEX, 0, RTCFormat::RTC_FORMAT_FLOAT3,  //
      s.vertices.pos.a, 0, sizeof(float3), s.vertices.pos.count);

    rtcSetSharedGeometryBuffer(geometry, RTCBufferType::RTC_BUFFER_TYPE_INDEX, 0, RTCFormat::RTC_FORMAT_UINT3,  //
      s.triangles.a, 0, sizeof(Triangle), s.triangles.count);

    rtcCommitGeometry(geometry);
    rtcAttachGeometry(rt_scene, geometry);
    rtcReleaseGeometry(geometry);
    rtcCommitScene(rt_scene);
  }

  void release_host_scene() {
    if (rt_scene) {
      rtcReleaseScene(rt_scene);
      rt_scene = {};
    }
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
    args.feature_mask = static_cast<RTCFeatureFlags>(RTC_FEATURE_FLAG_TRIANGLE | RTC_FEATURE_FLAG_FILTER_FUNCTION_IN_ARGUMENTS);
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
  } context = {{}, {{}, kInvalidIndex, 0.0f}, &scene, &smp, material_id};

  auto filter_funtion = [](const struct RTCFilterFunctionNArguments* args) {
    auto ctx = reinterpret_cast<IntersectionContextExt*>(args->context);

    const uint32_t triangle_index = RTCHitN_primID(args->hit, args->N, 0);
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
    if (alpha_test_pass(mat, tri, barycentrics({u, v}), *ctx->scene, *ctx->smp)) {
      *args->valid = 0;
      return;
    }

    ctx->i = {{u, v}, triangle_index, RTCRayN_tfar(args->ray, args->N, 0)};
  };

  ETX_ASSERT(_private != nullptr);
  _private->trace_with_function(r, &context.context, filter_funtion);

  if (context.i.triangle_index == kInvalidIndex)
    return false;

  result_intersection = make_intersection(scene, r.d, context.i);
  return true;
}

uint32_t Raytracing::continuous_trace(const Scene& scene, const Ray& r, const ContinousTraceOptions& options, Sampler& smp) const {
  struct IntersectionContextExt {
    RTCRayQueryContext context;
    const Scene* scene;
    Sampler* smp;
    IntersectionBase* buffer;
    uint32_t mat_id;
    uint32_t count;
    uint32_t max_count;
  } context = {{}, &scene, &smp, options.intersection_buffer, options.material_id, 0u, options.max_intersections};

  auto filter_funtion = [](const struct RTCFilterFunctionNArguments* args) {
    auto ctx = reinterpret_cast<IntersectionContextExt*>(args->context);
    uint32_t triangle_index = RTCHitN_primID(args->hit, args->N, 0);
    const auto& tri = ctx->scene->triangles[triangle_index];

    if ((ctx->mat_id != kInvalidIndex) && (ctx->mat_id != tri.material_index)) {
      *args->valid = 0;
      return;
    }

    float u = RTCHitN_u(args->hit, args->N, 0);
    float v = RTCHitN_v(args->hit, args->N, 0);
    float3 bc = barycentrics({u, v});
    const auto& scene = *ctx->scene;
    const auto& mat = ctx->scene->materials[tri.material_index];

    if (mat.cls == MaterialClass::Void) {
      *args->valid = 0;
      return;
    }

    if (alpha_test_pass(mat, tri, bc, scene, *ctx->smp)) {
      *args->valid = 0;
      return;
    }

    if (ctx->count < ctx->max_count) {
      ctx->buffer[ctx->count] = {
        .barycentric = {u, v},
        .triangle_index = triangle_index,
        .t = RTCRayN_tfar(args->ray, args->N, 0),
      };
      ctx->count += 1u;
    }

    *args->valid = (ctx->count < ctx->max_count) ? 0 : -1;
  };

  ETX_ASSERT(_private != nullptr);
  _private->trace_with_function(r, &context.context, filter_funtion);

  return context.count;
}

bool Raytracing::trace(const Scene& scene, const Ray& r, Intersection& result_intersection, Sampler& smp) const {
  struct IntersectionContextExt {
    RTCRayQueryContext context;
    IntersectionBase i;
    const Scene* scene;
    Sampler* smp;
  } context = {{}, {{}, kInvalidIndex, 0.0f}, &scene, &smp};

  auto filter_funtion = [](const struct RTCFilterFunctionNArguments* args) {
    auto ctx = reinterpret_cast<IntersectionContextExt*>(args->context);

    const uint32_t triangle_index = RTCHitN_primID(args->hit, args->N, 0);
    const auto& tri = ctx->scene->triangles[triangle_index];
    const auto& mat = ctx->scene->materials[tri.material_index];
    if (mat.cls == MaterialClass::Void) {
      *args->valid = 0;
      return;
    }
    const auto& scene = *ctx->scene;
    float u = RTCHitN_u(args->hit, args->N, 0);
    float v = RTCHitN_v(args->hit, args->N, 0);
    if (alpha_test_pass(mat, tri, barycentrics({u, v}), scene, *ctx->smp)) {
      *args->valid = 0;
      return;
    }

    ctx->i = {{u, v}, triangle_index, RTCRayN_tfar(args->ray, args->N, 0)};
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
    uint32_t triangle_index = RTCHitN_primID(args->hit, args->N, 0);
    const auto u = RTCHitN_u(args->hit, args->N, 0);
    const auto v = RTCHitN_v(args->hit, args->N, 0);
    const auto& tri = ctx->scene.triangles[triangle_index];
    const auto& mat = ctx->scene.materials[tri.material_index];
    if (mat.cls == MaterialClass::Void) {
      *args->valid = 0;
      return;
    }
    if (alpha_test_pass(mat, tri, barycentrics({u, v}), ctx->scene, ctx->smp)) {
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
  context.intersections[context.intersection_count++] = {kInvalidIndex, 0.0f, 0.0f, t_max};

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
        result *= medium_transmittance(scene, m, spect, smp, origin, direction, dt);
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
    const bool entering_surface = dot(tri.geo_n, direction) < 0.0f;
    current_medium = {
      .index = entering_surface ? mat.int_medium : mat.ext_medium,
    };
    current_t = intersection.t;
    origin = lerp_pos(scene, tri, barycentrics({intersection.u, intersection.v}));
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
