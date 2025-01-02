#pragma once

#define TINYGLTF_NO_STB_IMAGE       1
#define TINYGLTF_NO_STB_IMAGE_WRITE 1
#include <tiny_gltf.hxx>

namespace etx {

template <class T>
inline T gltf_ptr_as(const uint8_t* ptr, uint32_t comp_type, uint32_t gltf_type);

#define GLTF_PTR_AS(T, COMP_TYPE, TYPE)                                              \
  template <>                                                                        \
  inline T gltf_ptr_as(const uint8_t* ptr, uint32_t comp_type, uint32_t gltf_type) { \
    ETX_ASSERT(comp_type == COMP_TYPE);                                              \
    ETX_ASSERT(gltf_type == TYPE);                                                   \
    return *reinterpret_cast<const T*>(ptr);                                         \
  }

GLTF_PTR_AS(float, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_SCALAR)
GLTF_PTR_AS(float2, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC2)
GLTF_PTR_AS(float3, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC3)
GLTF_PTR_AS(float4, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC4)

GLTF_PTR_AS(uint8_t, TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE, TINYGLTF_TYPE_SCALAR)
GLTF_PTR_AS(uchar2, TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE, TINYGLTF_TYPE_VEC2)
GLTF_PTR_AS(uchar3, TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE, TINYGLTF_TYPE_VEC3)
GLTF_PTR_AS(uchar4, TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE, TINYGLTF_TYPE_VEC4)

GLTF_PTR_AS(uint16_t, TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT, TINYGLTF_TYPE_SCALAR)
GLTF_PTR_AS(ushort2, TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT, TINYGLTF_TYPE_VEC2)
GLTF_PTR_AS(ushort3, TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT, TINYGLTF_TYPE_VEC3)
GLTF_PTR_AS(ushort4, TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT, TINYGLTF_TYPE_VEC4)

GLTF_PTR_AS(uint32_t, TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT, TINYGLTF_TYPE_SCALAR)
GLTF_PTR_AS(uint2, TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT, TINYGLTF_TYPE_VEC2)
GLTF_PTR_AS(uint3, TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT, TINYGLTF_TYPE_VEC3)
GLTF_PTR_AS(uint4, TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT, TINYGLTF_TYPE_VEC4)

#undef GLTF_PTR_AS

template <class T>
inline T gltf_read_buffer(const tinygltf::Buffer& buffer, const tinygltf::Accessor& acc, const tinygltf::BufferView& view, uint32_t index) {
  ETX_ASSERT(buffer.data.empty() == false);
  auto begin = view.byteOffset + acc.byteOffset;
  auto stride = acc.ByteStride(view);
  auto location = begin + index * stride;
  ETX_ASSERT(location + stride <= buffer.data.size());
  return gltf_ptr_as<T>(buffer.data.data() + location, acc.componentType, acc.type);
}

inline uint3 gltf_read_buffer_as_uint3(const tinygltf::Buffer& buffer, const tinygltf::Accessor& acc, const tinygltf::BufferView& view, uint32_t index) {
  ETX_ASSERT(acc.type == TINYGLTF_TYPE_VEC3);
  ETX_ASSERT(buffer.data.empty() == false);
  auto begin = view.byteOffset + acc.byteOffset;
  auto stride = acc.ByteStride(view);
  auto location = begin + index * stride;
  ETX_ASSERT(location + stride <= buffer.data.size());
  switch (acc.componentType) {
    case TINYGLTF_COMPONENT_TYPE_INT:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
      return gltf_ptr_as<uint3>(buffer.data.data() + location, acc.componentType, acc.type);
    }
    case TINYGLTF_COMPONENT_TYPE_SHORT:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
      auto value = gltf_ptr_as<ushort3>(buffer.data.data() + location, acc.componentType, acc.type);
      return {value.x, value.y, value.z};
    }
    case TINYGLTF_COMPONENT_TYPE_BYTE:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
      auto value = gltf_ptr_as<ubyte3>(buffer.data.data() + location, acc.componentType, acc.type);
      return {value.x, value.y, value.z};
    }
    default:
      ETX_FAIL("Invalid component type for read_gltf_buffer_as_uint3");
  }
}

inline uint32_t gltf_read_buffer_as_uint(const tinygltf::Buffer& buffer, const tinygltf::Accessor& acc, const tinygltf::BufferView& view, uint32_t index) {
  ETX_ASSERT(acc.type == TINYGLTF_TYPE_SCALAR);
  ETX_ASSERT(buffer.data.empty() == false);
  auto begin = view.byteOffset + acc.byteOffset;
  auto stride = acc.ByteStride(view);
  auto location = begin + index * stride;
  ETX_ASSERT(location + stride <= buffer.data.size());
  switch (acc.componentType) {
    case TINYGLTF_COMPONENT_TYPE_INT:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
      return gltf_ptr_as<uint32_t>(buffer.data.data() + location, acc.componentType, acc.type);
    }
    case TINYGLTF_COMPONENT_TYPE_SHORT:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
      return gltf_ptr_as<uint16_t>(buffer.data.data() + location, acc.componentType, acc.type);
    }
    case TINYGLTF_COMPONENT_TYPE_BYTE:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
      return gltf_ptr_as<uint8_t>(buffer.data.data() + location, acc.componentType, acc.type);
    }
    default:
      ETX_FAIL("Invalid component type for read_gltf_buffer_as_uint3");
  }
}

}  // namespace etx