#pragma once

// Only VCM serializes the trailing surface coefficient; other integrators keep their original record strides.
uint wavefront_path_vertex_stride() {
  return scene_path_mode_is_vcm() ? kGPUWavefrontPathVertexVCMStride : kGPUWavefrontPathVertexStride;
}

uint wavefront_light_path_vertex_stride() {
  return scene_path_mode_is_vcm() ? kGPUWavefrontLightPathVertexVCMStride : kGPUWavefrontLightPathVertexStride;
}

float wavefront_vcm_surface_factor() {
  return scene_path_mode_is_vcm() ? constants.vcm_vm_weight : 0.0f;
}

float wavefront_connection_mis(GPUWavefrontPathVertex vertex, float surface_factor) {
  return scene_path_mode_is_vcm() ? vertex.reverse_pdf + surface_factor * vertex.d_surface : vertex.reverse_pdf;
}

float wavefront_connection_mis(GPUWavefrontPathVertex vertex) {
  return wavefront_connection_mis(vertex, wavefront_vcm_surface_factor());
}

float wavefront_connection_mis(GPUWavefrontPathState state) {
  return scene_path_mode_is_vcm() ? state.reverse_pdf + wavefront_vcm_surface_factor() * state.d_surface : state.reverse_pdf;
}

float wavefront_merge_mis(GPUWavefrontPathVertex vertex, float inverse_surface_factor) {
  return vertex.d_vm * inverse_surface_factor + vertex.d_surface;
}
