#pragma once

#define TINYGLTF_NO_STB_IMAGE       1
#define TINYGLTF_NO_STB_IMAGE_WRITE 1
#include <tiny_gltf.hxx>
#include <cstring>
#include <stdexcept>

namespace etx {

template <class T>
inline T gltf_ptr_as(const uint8_t* ptr, uint32_t comp_type, uint32_t gltf_type);

#define GLTF_PTR_AS(T, COMP_TYPE, TYPE)                                                 \
  template <>                                                                           \
  inline T gltf_ptr_as(const uint8_t* ptr, uint32_t comp_type, uint32_t gltf_type) {    \
    if ((comp_type != COMP_TYPE) || (gltf_type != TYPE))                                \
      throw std::runtime_error("Unsupported glTF accessor component or element type."); \
    T value;                                                                            \
    std::memcpy(&value, ptr, sizeof(value));                                            \
    return value;                                                                       \
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

inline const uint8_t* gltf_buffer_location(const tinygltf::Buffer& buffer, const tinygltf::Accessor& acc, const tinygltf::BufferView& view, uint32_t index) {
  if (acc.sparse.isSparse)
    throw std::runtime_error("Sparse glTF geometry accessors are not supported.");
  const int stride = acc.ByteStride(view);
  const int component_size = tinygltf::GetComponentSizeInBytes(acc.componentType);
  const int component_count = tinygltf::GetNumComponentsInType(acc.type);
  if ((stride <= 0) || (component_size <= 0) || (component_count <= 0) || (index >= acc.count) || (view.byteOffset > buffer.data.size()) ||
      (view.byteLength > (buffer.data.size() - view.byteOffset)) || (acc.byteOffset > view.byteLength))
    throw std::runtime_error("Invalid glTF accessor or buffer view.");
  const size_t element_size = static_cast<size_t>(component_size) * static_cast<size_t>(component_count);
  const size_t available = view.byteLength - acc.byteOffset;
  if ((static_cast<size_t>(stride) < element_size) || (available < element_size) || ((acc.count - 1u) > ((available - element_size) / static_cast<size_t>(stride))))
    throw std::runtime_error("glTF accessor extends beyond its buffer view.");
  return buffer.data.data() + view.byteOffset + acc.byteOffset + static_cast<size_t>(index) * static_cast<size_t>(stride);
}

template <class T>
inline T gltf_read_buffer(const tinygltf::Buffer& buffer, const tinygltf::Accessor& acc, const tinygltf::BufferView& view, uint32_t index) {
  return gltf_ptr_as<T>(gltf_buffer_location(buffer, acc, view, index), acc.componentType, acc.type);
}

inline uint3 gltf_read_buffer_as_uint3(const tinygltf::Buffer& buffer, const tinygltf::Accessor& acc, const tinygltf::BufferView& view, uint32_t index) {
  const auto* location = gltf_buffer_location(buffer, acc, view, index);
  switch (acc.componentType) {
    case TINYGLTF_COMPONENT_TYPE_INT:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
      return gltf_ptr_as<uint3>(location, acc.componentType, acc.type);
    }
    case TINYGLTF_COMPONENT_TYPE_SHORT:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
      auto value = gltf_ptr_as<ushort3>(location, acc.componentType, acc.type);
      return {value.x, value.y, value.z};
    }
    case TINYGLTF_COMPONENT_TYPE_BYTE:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
      auto value = gltf_ptr_as<ubyte3>(location, acc.componentType, acc.type);
      return {value.x, value.y, value.z};
    }
    default:
      throw std::runtime_error("Unsupported glTF integer accessor component type.");
  }
}

inline uint32_t gltf_read_buffer_as_uint(const tinygltf::Buffer& buffer, const tinygltf::Accessor& acc, const tinygltf::BufferView& view, uint32_t index) {
  const auto* location = gltf_buffer_location(buffer, acc, view, index);
  switch (acc.componentType) {
    case TINYGLTF_COMPONENT_TYPE_INT:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
      return gltf_ptr_as<uint32_t>(location, acc.componentType, acc.type);
    }
    case TINYGLTF_COMPONENT_TYPE_SHORT:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
      return gltf_ptr_as<uint16_t>(location, acc.componentType, acc.type);
    }
    case TINYGLTF_COMPONENT_TYPE_BYTE:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
      return gltf_ptr_as<uint8_t>(location, acc.componentType, acc.type);
    }
    default:
      throw std::runtime_error("Unsupported glTF index component type.");
  }
}

}  // namespace etx
