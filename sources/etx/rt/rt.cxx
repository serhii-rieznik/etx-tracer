#include <etx/rt/rt.hxx>

#define ETX_RT_API_BVH 1
#define ETX_RT_API_NANORT 2
#define ETX_RT_API_EMBREE 3

#define ETX_RT_API ETX_RT_API_EMBREE

#if (ETX_RT_API == ETX_RT_API_NANORT)

#define NANORT_USE_CPP11_FEATURE 1
#include <external/nanort/nanort.h>

#elif (ETX_RT_API == ETX_RT_API_BVH)

#include <external/bvh/bvh.hpp>
#include <external/bvh/triangle.hpp>
#include <external/bvh/sweep_sah_builder.hpp>
#include <external/bvh/single_ray_traverser.hpp>
#include <external/bvh/primitive_intersectors.hpp>

#elif (ETX_RT_API == ETX_RT_API_EMBREE)

#include <embree3/rtcore.h>

#else

#error No raytracing API defined

#endif

namespace etx {

struct RaytracingImpl {
  TaskScheduler scheduler;
  GPUDevice* gpu = nullptr;
  const Scene* scene = nullptr;

#if (ETX_RT_API == ETX_RT_API_NANORT)

  using Ray = nanort::Ray<float>;
  std::vector<uint32_t> linear_indices;
  std::vector<float> linear_vertex_data;
  float* v_ptr = nullptr;
  uint32_t* i_ptr = nullptr;

  nanort::BVHAccel<float> bvh = {};

#elif (ETX_RT_API == ETX_RT_API_BVH)

  using Bvh = bvh::Bvh<float>;
  using Builder = bvh::SweepSahBuilder<Bvh>;
  using Traverser = bvh::SingleRayTraverser<Bvh, 64, bvh::RobustNodeIntersector<Bvh> >;
  using Triangle = bvh::Triangle<Bvh::ScalarType, false>;
  using BVHVector = bvh::Vector3<Bvh::ScalarType>;
  using Intersection = bvh::AlphaTestClosestPrimitiveIntersector<Bvh, Triangle>;
  using Ray = bvh::Ray<Bvh::ScalarType>;

  std::vector<Triangle> triangles;
  bvh::Bvh<float> bvh = {};

#elif (ETX_RT_API == ETX_RT_API_EMBREE)

  RTCDevice rt_device = {};
  RTCScene rt_scene = {};

#endif

  RaytracingImpl() {
    gpu = GPUDevice::create_optix_device();
  }

  ~RaytracingImpl() {
    release_scene();
    GPUDevice::free_device(gpu);
  }

  void set_scene(const Scene& s) {
    scene = &s;
    release_scene();

#if (ETX_RT_API == ETX_RT_API_NANORT)
    linear_vertex_data.reserve(scene.vertices.count);
    for (uint32_t i = 0; i < scene.vertices.count; ++i) {
      linear_vertex_data.emplace_back(scene.vertices[i].pos.x);
      linear_vertex_data.emplace_back(scene.vertices[i].pos.y);
      linear_vertex_data.emplace_back(scene.vertices[i].pos.z);
    }
    v_ptr = linear_vertex_data.data();

    linear_indices.reserve(3llu * scene.triangles.count);
    for (uint32_t i = 0; i < scene.triangles.count; ++i) {
      const auto& tri = scene.triangles[i];
      linear_indices.emplace_back(tri.i[0]);
      linear_indices.emplace_back(tri.i[1]);
      linear_indices.emplace_back(tri.i[2]);
    }
    i_ptr = linear_indices.data();

    auto mesh = nanort::TriangleMesh<float>(v_ptr, i_ptr, sizeof(float) * 3llu);
    auto sah = nanort::TriangleSAHPred<float>(v_ptr, i_ptr, sizeof(float) * 3llu);
    auto succeed = bvh.Build(uint32_t(scene.triangles.count), mesh, sah, {});
    ETX_ASSERT(succeed);

#elif (ETX_RT_API == ETX_RT_API_BVH)

    triangles.clear();
    triangles.reserve(scene.triangles.count);
    for (uint64_t i = 0, e = scene.triangles.count; i < e; ++i) {
      const auto& tri = scene.triangles[i];
      const auto& v0 = scene.vertices[tri.i[0]];
      const auto& v1 = scene.vertices[tri.i[1]];
      const auto& v2 = scene.vertices[tri.i[2]];
      triangles.emplace_back(BVHVector{v0.pos.x, v0.pos.y, v0.pos.z}, BVHVector{v1.pos.x, v1.pos.y, v1.pos.z}, BVHVector{v2.pos.x, v2.pos.y, v2.pos.z});
    }

    auto [boxes, centers] = bvh::compute_bounding_boxes_and_centers(triangles.data(), triangles.size());
    auto global_bbox = bvh::compute_bounding_boxes_union(boxes.get(), triangles.size());

    Builder bvh_builder(bvh);
    bvh_builder.build(global_bbox, boxes.get(), centers.get(), triangles.size());

#elif (ETX_RT_API == ETX_RT_API_EMBREE)

    rt_device = rtcNewDevice(nullptr);
    rtcSetDeviceErrorFunction(
      rt_device,
      [](void* userPtr, enum RTCError code, const char* str) {
        printf("Embree error: %u (%s)\n", code, str);
      },
      nullptr);

    rt_scene = rtcNewScene(rt_device);
    auto geometry = rtcNewGeometry(rt_device, RTCGeometryType::RTC_GEOMETRY_TYPE_TRIANGLE);

    rtcSetSharedGeometryBuffer(geometry, RTCBufferType::RTC_BUFFER_TYPE_VERTEX, 0, RTCFormat::RTC_FORMAT_FLOAT3,  //
      scene->vertices.a, 0, sizeof(Vertex), scene->vertices.count);

    rtcSetSharedGeometryBuffer(geometry, RTCBufferType::RTC_BUFFER_TYPE_INDEX, 0, RTCFormat::RTC_FORMAT_UINT3,  //
      scene->triangles.a, 0, sizeof(Triangle), scene->triangles.count);

    rtcCommitGeometry(geometry);
    rtcAttachGeometry(rt_scene, geometry);
    rtcReleaseGeometry(geometry);
    rtcCommitScene(rt_scene);

#endif
  }

  void release_scene() {
#if (ETX_RT_API == ETX_RT_API_EMBREE)
    if (rt_scene) {
      rtcReleaseScene(rt_scene);
      rt_scene = {};
    }
    if (rt_device) {
      rtcReleaseDevice(rt_device);
      rt_device = {};
    }
#endif
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

GPUDevice* Raytracing::gpu() {
  return _private->gpu;
}

void Raytracing::set_scene(const Scene& scene) {
  _private->set_scene(scene);
}

bool Raytracing::has_scene() const {
  return (_private->scene != nullptr);
}

const Scene& Raytracing::scene() const {
  ETX_ASSERT(has_scene());
  return *(_private->scene);
}

bool Raytracing::trace(const Ray& r, Intersection& result_intersection, Sampler& smp) const {
  ETX_ASSERT(_private != nullptr);

  bool intersection_found = false;
  float2 barycentric = {};
  uint32_t triangle_index = kInvalidIndex;
  float t = -kMaxFloat;

#if (ETX_RT_API == ETX_RT_API_NANORT)

  RaytracerPrivate::Ray rr = RaytracerPrivate::Ray{{r.o.x, r.o.y, r.o.z}, {r.d.x, r.d.y, r.d.z}, r.min_t, r.max_t};
  nanort::TriangleIntersector<float> traverser(_private->v_ptr, _private->i_ptr, 3llu * sizeof(float));
  nanort::TriangleIntersection<float> isect;
  if (_private->bvh.Traverse(rr, traverser, &isect, {})) {
    intersection_found = true;
    barycentric = {isect.u, isect.v};
    triangle_index = isect.prim_id;
    t = isect.t;
  }

#elif (ETX_RT_API == ETX_RT_API_BVH)

  RaytracerPrivate::Ray rr = {{r.o.x, r.o.y, r.o.z}, {r.d.x, r.d.y, r.d.z}, r.min_t, r.max_t};
  RaytracerPrivate::Traverser traverser(_private->bvh);
  RaytracerPrivate::Intersection isect(_private->bvh, _private->triangles.data(), _private->scene, smp);
  if (result_intersection = traverser.traverse(rr, isect)) {
    intersection_found = true;
    barycentric = {result_intersection.barycentric.y, result_intersection.barycentric.z};
    triangle_index = result_intersection.triangle_index;
    t = result_intersection.t;
  }

#elif (ETX_RT_API == ETX_RT_API_EMBREE)

  RTCIntersectContext context = {};
  rtcInitIntersectContext(&context);

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

  for (;;) {
    rtcIntersect1(_private->rt_scene, &context, &ray_hit);
    if ((ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID)) {
      intersection_found = false;
      break;
    }

    const auto& tri = _private->scene->triangles[ray_hit.hit.primID];
    const auto& mat = _private->scene->materials[tri.material_index];
    float3 bc = {1.0f - ray_hit.hit.u - ray_hit.hit.v, ray_hit.hit.u, ray_hit.hit.v};
    if (bsdf::continue_tracing(mat, lerp_uv(_private->scene->vertices, tri, bc), *_private->scene, smp)) {
      auto p = lerp_pos(_private->scene->vertices, tri, bc);
      ray_hit.ray.org_x = p.x + r.d.x * kRayEpsilon;
      ray_hit.ray.org_y = p.y + r.d.y * kRayEpsilon;
      ray_hit.ray.org_z = p.z + r.d.z * kRayEpsilon;
      ray_hit.ray.tfar = r.max_t - ray_hit.ray.tfar;
      ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
      ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
      ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    } else {
      intersection_found = true;
      barycentric = {ray_hit.hit.u, ray_hit.hit.v};
      triangle_index = ray_hit.hit.primID;
      t = ray_hit.ray.tfar;
      break;
    }
  }

#endif

  if (intersection_found) {
    float3 bc = {1.0f - barycentric.x - barycentric.y, barycentric.x, barycentric.y};
    const auto& tri = _private->scene->triangles[triangle_index];
    result_intersection = lerp_vertex(_private->scene->vertices, tri, bc);
    result_intersection.barycentric = bc;
    result_intersection.triangle_index = static_cast<uint32_t>(triangle_index);
    result_intersection.w_i = r.d;
    result_intersection.t = t;

    const auto& mat = _private->scene->materials[tri.material_index];
    if ((mat.normal_image_index != kInvalidIndex) && (mat.normal_scale > 0.0f)) {
      auto sampled_normal = _private->scene->images[mat.normal_image_index].evaluate_normal(result_intersection.tex, mat.normal_scale);
      float3x3 from_local = {
        float3{result_intersection.tan.x, result_intersection.tan.y, result_intersection.tan.z},
        float3{result_intersection.btn.x, result_intersection.btn.y, result_intersection.btn.z},
        float3{result_intersection.nrm.x, result_intersection.nrm.y, result_intersection.nrm.z},
      };
      result_intersection.nrm = normalize(from_local * sampled_normal);
      result_intersection.tan = normalize(result_intersection.tan - result_intersection.nrm * dot(result_intersection.tan, result_intersection.nrm));
      result_intersection.btn = normalize(cross(result_intersection.nrm, result_intersection.tan));
    }
  }

  return intersection_found;
}

}  // namespace etx
