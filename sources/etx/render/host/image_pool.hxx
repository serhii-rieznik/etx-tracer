#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/host/tasks.hxx>

#include <etx/render/shared/image.hxx>
namespace etx {

struct BufferPool;

struct ImagePool {
  ImagePool(std::vector<Image>&, BufferPool&);
  ~ImagePool();

  void init(uint32_t capacity);
  void cleanup();

  uint32_t add_copy(const Image& img);
  uint32_t add_from_file(const std::string& path, uint32_t image_options, const float2& offset, const float2& scale);
  uint32_t add_from_data(const float4* data, const uint2& dimensions, uint32_t image_options, const float2& offset, const float2& scale);
  uint32_t add_from_spherical_harmonics(TaskScheduler& scheduler, const float3 sh_coeffs[9], const uint2& dimensions, uint32_t image_options, const float2& offset,
    const float2& scale);
  uint32_t add_from_cubemap(TaskScheduler& scheduler, const uint32_t cube_face_images[6], const uint2& dimensions, uint32_t image_options, const float2& offset,
    const float2& scale);

  void remove(uint32_t handle);
  void remove_all();

  void add_options(uint32_t, uint32_t);
  void load_images(TaskScheduler& scheduler);
  void rebuild_sampling_table(uint32_t index, TaskScheduler& scheduler);

  const Image& get(uint32_t) const;
  std::string path(uint32_t) const;

  void free_image(Image&);

  const Image* as_array() const;
  const uint64_t array_size() const;

  ETX_DECLARE_PIMPL(ImagePool, 448);
};

}  // namespace etx
