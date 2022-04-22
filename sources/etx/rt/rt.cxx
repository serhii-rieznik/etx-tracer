#include <etx/rt/rt.hxx>

#include <embree3/rtcore.h>

namespace etx {

struct RaytracingImpl {
  TaskScheduler scheduler;

  const Scene* scene = nullptr;
  RTCDevice rt_device = {};
  RTCScene rt_scene = {};

  GPUDevice* gpu_device = nullptr;
  struct {
    Scene scene = {};
    GPUAccelerationStructure accel = {};
    std::vector<GPUBuffer> buffers = {};
  } gpu = {};

  RaytracingImpl() {
    gpu_device = GPUDevice::create_optix_device();
  }

  ~RaytracingImpl() {
    release_host_scene();
    release_device_scene();
    GPUDevice::free_device(gpu_device);
  }

  void set_scene(const Scene& s) {
    scene = &s;
    release_host_scene();
    build_host_scene();

    release_device_scene();
    build_device_scene();
  }

  void build_host_scene() {
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
  }

  void release_host_scene() {
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

  template <class T>
  void upload_array_view_to_gpu(ArrayView<T>& a, GPUBuffer* out_buffer) {
    GPUBuffer buffer = gpu.buffers.emplace_back(gpu_device->create_buffer({sizeof(T) * a.count, a.a}));

    auto device_ptr = gpu_device->get_buffer_device_pointer(buffer);
    a.a = reinterpret_cast<T*>(device_ptr);

    if (out_buffer != nullptr) {
      *out_buffer = buffer;
    }
  }

  template <class T>
  void upload_array_view_to_gpu(ArrayView<T>& a) {
    upload_array_view_to_gpu(a, nullptr);
  }

  void build_device_scene() {
    GPUBuffer vertex_buffer = {};
    GPUBuffer index_buffer = {};

    gpu.scene = *scene;
    upload_array_view_to_gpu(gpu.scene.vertices, &vertex_buffer);
    upload_array_view_to_gpu(gpu.scene.triangles, &index_buffer);
    upload_array_view_to_gpu(gpu.scene.materials);
    upload_array_view_to_gpu(gpu.scene.emitters);

    /*/ TODO : update other data:
    upload_array_view_to_gpu(gpu.scene.emitters_distribution.values);
    upload_array_view_to_gpu(gpu.scene.images);
    upload_array_view_to_gpu(gpu.scene.mediums);
    // TODO : update gpu.scene.spectrums
    // */

    GPUAccelerationStructure::Descriptor desc = {};
    desc.vertex_buffer = vertex_buffer;
    desc.vertex_buffer_stride = sizeof(Vertex);
    desc.vertex_count = static_cast<uint32_t>(scene->vertices.count);
    desc.index_buffer = index_buffer;
    desc.index_buffer_stride = sizeof(Triangle);
    desc.triangle_count = static_cast<uint32_t>(scene->triangles.count);
    gpu.accel = gpu_device->create_acceleration_structure(desc);
  }

  void release_device_scene() {
    gpu_device->destroy_acceleration_structure(gpu.accel);
    for (auto& buffer : gpu.buffers) {
      gpu_device->destroy_buffer(buffer);
    }
    gpu.buffers.clear();
    gpu = {};
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
  return _private->gpu_device;
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

const Scene& Raytracing::gpu_scene() const {
  ETX_ASSERT(has_scene());
  return _private->gpu.scene;
}

const GPUAccelerationStructure Raytracing::gpu_acceleration_structure() const {
  ETX_ASSERT(has_scene());
  return _private->gpu.accel;
}

bool Raytracing::trace(const Ray& r, Intersection& result_intersection, Sampler& smp) const {
  ETX_ASSERT(_private != nullptr);

  bool intersection_found = false;
  float2 barycentric = {};
  uint32_t triangle_index = kInvalidIndex;
  float t = -kMaxFloat;

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
