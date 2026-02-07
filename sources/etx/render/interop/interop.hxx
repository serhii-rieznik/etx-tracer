#pragma once

#define ETX_GPU_CODE inline

#if defined(__cplusplus)
# define ETX_SHARED_INLINE     inline
# define ETX_IN(type, name)    const type& name
# define ETX_OUT(type, name)   type& name
# define ETX_INOUT(type, name) type& name
# define ETX_ALIGNED           alignas(16)
# define ETX_INIT(expr)        = expr
# include <etx/render/shared/base.hxx>

namespace etx {

template <class T>
inline void print_value(const char* name, const T& v, const char* filename, uint32_t line);

template <>
inline void print_value<bool>(const char* name, const bool& v, const char* filename, uint32_t line) {
  printf("Validation failed: %s (%s) at %s [%u]\n", name, v ? "true" : "false", filename, line);
}

template <>
inline void print_value<float>(const char* name, const float& v, const char* filename, uint32_t line) {
  printf("Validation failed: %s (%f) at %s [%u]\n", name, v, filename, line);
}

template <>
inline void print_value<float2>(const char* name, const float2& v, const char* filename, uint32_t line) {
  printf("Validation failed: %s (%f %f) at %s [%u]\n", name, v.x, v.y, filename, line);
}

template <>
inline void print_value<float3>(const char* name, const float3& v, const char* filename, uint32_t line) {
  printf("Validation failed: %s (%f %f %f) at %s [%u]\n", name, v.x, v.y, v.z, filename, line);
}

template <>
inline void print_value<float4>(const char* name, const float4& v, const char* filename, uint32_t line) {
  printf("Validation failed: %s (%f %f %f %f) at %s [%u]\n", name, v.x, v.y, v.z, v.w, filename, line);
}

}  // namespace etx

#else

# define ETX_SHARED_INLINE     inline
# define ETX_IN(type, name)    in type name
# define ETX_OUT(type, name)   out type name
# define ETX_INOUT(type, name) inout type name
# define ETX_ALIGNED           /* */
# define ETX_INIT(expr)

#endif
