#include <etx/core/core.hxx>
#include <etx/rt/rt.hxx>

#include <etx/render/host/film.hxx>

#include <embree4/rtcore.h>

namespace etx {

struct RaytracingImpl {
  TaskScheduler scheduler;
  Film film;

  const Scene* source_scene = nullptr;
  const Camera* source_camera = nullptr;
  RTCDevice rt_device = {};
  RTCScene rt_scene = {};

  RaytracingImpl()
    : film(scheduler) {
    rt_device = rtcNewDevice(nullptr);
    const auto version_major = rtcGetDeviceProperty(rt_device, RTC_DEVICE_PROPERTY_VERSION_MAJOR);
    const auto version_minor = rtcGetDeviceProperty(rt_device, RTC_DEVICE_PROPERTY_VERSION_MINOR);
    const auto version_patch = rtcGetDeviceProperty(rt_device, RTC_DEVICE_PROPERTY_VERSION_PATCH);
    log::warning("Embree version: %u.%u.%u", version_major, version_minor, version_patch);
  }

  ~RaytracingImpl() {
    release_host_scene();

    if (rt_device) {
      rtcReleaseDevice(rt_device);
      rt_device = {};
    }
  }

  void set_scene(const Scene& s) {
    source_scene = &s;
  }

  void set_camera(const Camera& c) {
    source_camera = &c;
  }

  void commit() {
    release_host_scene();
    film.allocate(source_camera->film_size);
    build_host_scene(*source_scene);
    // release_device_scene();
    // build_device_scene();
  }

  void build_host_scene(const Scene& s) {
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

Raytracing::Raytracing() {
  ETX_PIMPL_INIT(Raytracing);
}

Raytracing::~Raytracing() {
  ETX_PIMPL_CLEANUP(Raytracing);
}

TaskScheduler& Raytracing::scheduler() {
  return _private->scheduler;
}

void Raytracing::link_scene(const Scene& scene) {
  _private->set_scene(scene);
}

void Raytracing::link_camera(const Camera& camera) {
  _private->set_camera(camera);
}

const Camera& Raytracing::camera() const {
  return *_private->source_camera;
}

const Scene& Raytracing::scene() const {
  return *_private->source_scene;
}

void Raytracing::commit_changes() {
  _private->commit();
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
    if (mat.cls == Material::Class::Void) {
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

    if (mat.cls == Material::Class::Void) {
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
    if (mat.cls == Material::Class::Void) {
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

SpectralResponse Raytracing::trace_transmittance(const SpectralQuery spect, const Scene& scene, const float3& p0, const float3& p1, const Medium::Instance& medium,
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
    if (mat.cls == Material::Class::Void) {
      *args->valid = 0;
      return;
    }
    if (alpha_test_pass(mat, tri, barycentrics({u, v}), ctx->scene, ctx->smp)) {
      *args->valid = 0;
      return;
    }
    if ((mat.cls != Material::Class::Boundary) || (ctx->intersection_count + 1u >= kIntersectionBufferSize)) {
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
  Medium::Instance current_medium = medium;
  for (uint32_t i = 0; i < context.intersection_count; ++i) {
    const auto& intersection = context.intersections[i];
    if (current_medium.valid()) {
      float dt = fmaxf(0.0f, intersection.t - current_t);

      if (current_medium.index != kInvalidIndex) {
        const auto& m = scene.mediums[current_medium.index];
        result *= medium_transmittance(scene, m, spect, smp, origin, direction, dt);
        ETX_VALIDATE(result);
      } else {
        result *= medium_transmittance(current_medium, dt);
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
