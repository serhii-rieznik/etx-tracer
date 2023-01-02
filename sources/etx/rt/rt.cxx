#include <etx/core/core.hxx>
#include <etx/rt/rt.hxx>

#include <embree3/rtcore.h>

namespace etx {

struct RaytracingImpl {
  TaskScheduler scheduler;

  const Scene* source_scene = nullptr;
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
    source_scene = &s;
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
      source_scene->vertices.a, 0, sizeof(Vertex), source_scene->vertices.count);

    rtcSetSharedGeometryBuffer(geometry, RTCBufferType::RTC_BUFFER_TYPE_INDEX, 0, RTCFormat::RTC_FORMAT_UINT3,  //
      source_scene->triangles.a, 0, sizeof(Triangle), source_scene->triangles.count);

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
  inline static uint64_t array_size(const ArrayView<T>& a) {
    return align_up(a.count * sizeof(T), 16llu);
  }

  template <class T>
  inline void upload_array_view_to_gpu(ArrayView<T>& a, GPUBuffer* out_buffer) {
    GPUBuffer buffer = gpu.buffers.emplace_back(gpu_device->create_buffer({array_size(a), a.a}));

    auto device_ptr = gpu_device->get_buffer_device_pointer(buffer);
    a.a = reinterpret_cast<T*>(device_ptr);

    if (out_buffer != nullptr) {
      *out_buffer = buffer;
    }
  }

  template <class T>
  inline void upload_array_view_to_gpu(ArrayView<T>& a) {
    upload_array_view_to_gpu(a, nullptr);
  }

  template <class T>
  inline T* push_to_generic_buffer(GPUBuffer buffer, T* ptr, uint64_t size_to_copy, uint64_t& copy_offset) {
    if ((ptr == nullptr) || (size_to_copy == 0)) {
      return nullptr;
    }

    auto device_ptr = gpu_device->copy_to_buffer(buffer, ptr, copy_offset, size_to_copy);
    copy_offset = align_up(copy_offset + size_to_copy, 16llu);
    return reinterpret_cast<T*>(device_ptr);
  }

  template <class T>
  inline void push_to_generic_buffer(GPUBuffer buffer, ArrayView<T>& a, uint64_t& copy_offset) {
    if ((a.a == nullptr) || (a.count == 0)) {
      return;
    }

    auto size_to_copy = array_size(a);
    auto ptr = gpu_device->copy_to_buffer(buffer, a.a, copy_offset, size_to_copy);
    copy_offset = align_up(copy_offset + size_to_copy, 16llu);
    a.a = reinterpret_cast<T*>(ptr);
  }

  void build_device_scene() {
    GPUBuffer vertex_buffer = {};
    GPUBuffer index_buffer = {};
    GPUBuffer scene_buffer = {};

    gpu.scene = *source_scene;
    upload_array_view_to_gpu(gpu.scene.vertices, &vertex_buffer);
    upload_array_view_to_gpu(gpu.scene.triangles, &index_buffer);
    upload_array_view_to_gpu(gpu.scene.materials);
    upload_array_view_to_gpu(gpu.scene.emitters);

    uint64_t scene_buffer_size = 0;
    scene_buffer_size = align_up(scene_buffer_size + array_size(gpu.scene.emitters_distribution.values), 16llu);
    scene_buffer_size = align_up(scene_buffer_size + align_up(sizeof(Spectrums), 16llu), 16llu);

    // images
    for (uint32_t i = 0; i < gpu.scene.images.count; ++i) {
      auto& image = gpu.scene.images[i];
      if (image.format == Image::Format::RGBA32F) {
        scene_buffer_size = align_up(scene_buffer_size + array_size(image.pixels.f32), 16llu);
      } else {
        scene_buffer_size = align_up(scene_buffer_size + array_size(image.pixels.u8), 16llu);
      }
      scene_buffer_size = align_up(scene_buffer_size + array_size(image.y_distribution.values), 16llu);
      scene_buffer_size = align_up(scene_buffer_size + array_size(image.x_distributions), 16llu);
      for (uint32_t y = 0; y < image.y_distribution.values.count; ++y) {
        scene_buffer_size = align_up(scene_buffer_size + array_size(image.x_distributions[y].values), 16llu);
      }
    }
    scene_buffer_size = align_up(scene_buffer_size + array_size(gpu.scene.images), 16llu);
    for (uint32_t i = 0; i < gpu.scene.mediums.count; ++i) {
      scene_buffer_size = align_up(scene_buffer_size + array_size(gpu.scene.mediums[i].density), 16llu);
    }
    scene_buffer_size = align_up(scene_buffer_size + array_size(gpu.scene.mediums), 16llu);

    scene_buffer = gpu.buffers.emplace_back(gpu_device->create_buffer({scene_buffer_size, nullptr}));

    uint64_t copy_offset = 0;
    push_to_generic_buffer(scene_buffer, gpu.scene.emitters_distribution.values, copy_offset);
    gpu.scene.spectrums = push_to_generic_buffer(scene_buffer, gpu.scene.spectrums.ptr, sizeof(Spectrums), copy_offset);

    if (gpu.scene.images.count > 0) {
      auto images_ptr = reinterpret_cast<Image*>(calloc(gpu.scene.images.count, sizeof(Image)));

      for (uint32_t i = 0; (images_ptr != nullptr) && (i < gpu.scene.images.count); ++i) {
        Image image = gpu.scene.images[i];
        if (image.format == Image::Format::RGBA32F) {
          push_to_generic_buffer(scene_buffer, image.pixels.f32, copy_offset);
        } else {
          push_to_generic_buffer(scene_buffer, image.pixels.u8, copy_offset);
        }
        push_to_generic_buffer(scene_buffer, image.y_distribution.values, copy_offset);

        auto x_dist_ptr = calloc(image.y_distribution.values.count, sizeof(Distribution));

        ArrayView<Distribution> x_distributions = {
          reinterpret_cast<Distribution*>(x_dist_ptr),
          image.y_distribution.values.count,
        };

        for (uint32_t y = 0; y < image.y_distribution.values.count; ++y) {
          x_distributions[y] = image.x_distributions[y];
          push_to_generic_buffer(scene_buffer, x_distributions[y].values, copy_offset);
        }
        push_to_generic_buffer(scene_buffer, x_distributions, copy_offset);

        image.x_distributions = x_distributions;
        images_ptr[i] = image;

        free(x_dist_ptr);
      }
      gpu.scene.images = make_array_view<Image>(images_ptr, gpu.scene.images.count);
      push_to_generic_buffer(scene_buffer, gpu.scene.images, copy_offset);
      free(images_ptr);
    }

    if (gpu.scene.mediums.count > 0) {
      auto medium_ptr = reinterpret_cast<Medium*>(calloc(sizeof(Medium), gpu.scene.mediums.count));
      for (uint32_t i = 0; (medium_ptr != nullptr) && (i < gpu.scene.mediums.count); ++i) {
        auto medium = gpu.scene.mediums[i];
        push_to_generic_buffer(scene_buffer, medium.density, copy_offset);
        medium_ptr[i] = medium;
      }
      gpu.scene.mediums = make_array_view<Medium>(medium_ptr, gpu.scene.mediums.count);
      upload_array_view_to_gpu(gpu.scene.mediums);
      free(medium_ptr);
    }

    GPUAccelerationStructure::Descriptor desc = {};
    desc.vertex_buffer = vertex_buffer;
    desc.vertex_buffer_stride = sizeof(Vertex);
    desc.vertex_count = static_cast<uint32_t>(gpu.scene.vertices.count);
    desc.index_buffer = index_buffer;
    desc.index_buffer_stride = sizeof(Triangle);
    desc.triangle_count = static_cast<uint32_t>(gpu.scene.triangles.count);
    gpu.accel = gpu_device->create_acceleration_structure(desc);

    gpu.scene.acceleration_structure = gpu_device->get_acceleration_structure_device_pointer(gpu.accel);
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
  return (_private->source_scene != nullptr);
}

const Scene& Raytracing::scene() const {
  ETX_ASSERT(has_scene());
  return *(_private->source_scene);
}

const Scene& Raytracing::gpu_scene() const {
  ETX_ASSERT(has_scene());
  _private->gpu.scene.camera = _private->source_scene->camera;
  return _private->gpu.scene;
}

uint32_t Raytracing::continuous_trace(const Scene& scene, const Ray& r, const ContinousTraceOptions& options, Sampler& smp) const {
  ETX_ASSERT(_private != nullptr);
  ETX_CHECK_FINITE(r.d);

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

  uint32_t intersection_count = 0;

  for (;;) {
    rtcIntersect1(_private->rt_scene, &context, &ray_hit);
    if ((ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID)) {
      break;
    }

    const auto& tri = scene.triangles[ray_hit.hit.primID];
    const auto& mat = scene.materials[tri.material_index];
    float3 bc = {1.0f - ray_hit.hit.u - ray_hit.hit.v, ray_hit.hit.u, ray_hit.hit.v};

    bool add_intersection = (intersection_count < options.max_intersections)                                             //
                            && ((options.material_id == kInvalidIndex) || (tri.material_index == options.material_id));  //

    if (add_intersection && (bsdf::continue_tracing(mat, lerp_uv(scene.vertices, tri, bc), scene, smp) == false)) {
      options.intersection_buffer[intersection_count] = {
        .barycentric = {ray_hit.hit.u, ray_hit.hit.v},
        .triangle_index = ray_hit.hit.primID,
        .t = ray_hit.ray.tfar,
      };
      intersection_count += 1u;
      if (intersection_count >= options.max_intersections)
        break;
    }

    auto p = lerp_pos(scene.vertices, tri, bc);
    auto p_start = offset_ray(p, r.d);
    ray_hit.ray.org_x = p_start.x;
    ray_hit.ray.org_y = p_start.y;
    ray_hit.ray.org_z = p_start.z;
    ray_hit.ray.tfar = r.max_t - ray_hit.ray.tfar;
    ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
  }

  return intersection_count;
}

bool Raytracing::trace(const Scene& scene, const Ray& r, Intersection& result_intersection, Sampler& smp) const {
  ETX_ASSERT(_private != nullptr);
  ETX_CHECK_FINITE(r.d);

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

  IntersectionBase base = {};
  bool intersection_found = false;
  for (;;) {
    rtcIntersect1(_private->rt_scene, &context, &ray_hit);
    if ((ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID)) {
      intersection_found = false;
      break;
    }

    const auto& tri = scene.triangles[ray_hit.hit.primID];
    const auto& mat = scene.materials[tri.material_index];
    float3 bc = {1.0f - ray_hit.hit.u - ray_hit.hit.v, ray_hit.hit.u, ray_hit.hit.v};
    if (bsdf::continue_tracing(mat, lerp_uv(scene.vertices, tri, bc), scene, smp)) {
      auto p = lerp_pos(scene.vertices, tri, bc);
      ray_hit.ray.org_x = p.x + r.d.x * kRayEpsilon;
      ray_hit.ray.org_y = p.y + r.d.y * kRayEpsilon;
      ray_hit.ray.org_z = p.z + r.d.z * kRayEpsilon;
      ray_hit.ray.tfar = r.max_t - ray_hit.ray.tfar;
      ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
      ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
      ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    } else {
      intersection_found = true;
      base.barycentric = {ray_hit.hit.u, ray_hit.hit.v};
      base.triangle_index = ray_hit.hit.primID;
      base.t = ray_hit.ray.tfar;
      break;
    }
  }

  if (intersection_found) {
    result_intersection = make_intersection(scene, r.d, base);
  }

  return intersection_found;
}

Intersection Raytracing::make_intersection(const Scene& scene, const float3& w_i, const IntersectionBase& base) const {
  Intersection result_intersection;
  float3 bc = {1.0f - base.barycentric.x - base.barycentric.y, base.barycentric.x, base.barycentric.y};
  const auto& tri = scene.triangles[base.triangle_index];
  result_intersection = lerp_vertex(scene.vertices, tri, bc);
  result_intersection.barycentric = bc;
  result_intersection.triangle_index = static_cast<uint32_t>(base.triangle_index);
  result_intersection.w_i = w_i;
  result_intersection.t = base.t;

  const auto& mat = scene.materials[tri.material_index];
  if ((mat.normal_image_index != kInvalidIndex) && (mat.normal_scale > 0.0f)) {
    auto sampled_normal = scene.images[mat.normal_image_index].evaluate_normal(result_intersection.tex, mat.normal_scale);
    float3x3 from_local = {
      float3{result_intersection.tan.x, result_intersection.tan.y, result_intersection.tan.z},
      float3{result_intersection.btn.x, result_intersection.btn.y, result_intersection.btn.z},
      float3{result_intersection.nrm.x, result_intersection.nrm.y, result_intersection.nrm.z},
    };
    result_intersection.nrm = normalize(from_local * sampled_normal);
    result_intersection.tan = normalize(result_intersection.tan - result_intersection.nrm * dot(result_intersection.tan, result_intersection.nrm));
    result_intersection.btn = normalize(cross(result_intersection.nrm, result_intersection.tan));
  }
  return result_intersection;
}

}  // namespace etx
