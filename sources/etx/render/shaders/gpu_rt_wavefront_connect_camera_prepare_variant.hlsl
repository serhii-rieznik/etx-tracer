#include "bindless.hlsl"
#include <interop/image_filter_shared.hxx>
#include <access/bsdf_resource_gpu.hxx>

#define ETX_WAVEFRONT_BSDF_KIND_DIFFUSE          1
#define ETX_WAVEFRONT_BSDF_KIND_PLASTIC          2
#define ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR        3
#define ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC       4
#define ETX_WAVEFRONT_BSDF_KIND_COATED_CONDUCTOR 6

#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE)
# if ETX_ENABLE_VELVET_STAGE
#  include <interop/bsdf_velvet_shared.hxx>
# else
#  include <interop/bsdf_various_shared.hxx>
# endif
# if ETX_ENABLE_THINFILM_STAGE
#  include <interop/bsdf_dielectric_shared.hxx>
# endif
# define ETX_STAGE_BSDF_CLASS MaterialClass::Diffuse
# define ETX_STAGE_BSDF_EVAL  wavefront_connect_camera_stage_various_eval
# define ETX_STAGE_BSDF_PDF   wavefront_connect_camera_stage_various_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_PLASTIC)
# include <interop/bsdf_plastic_shared.hxx>
# define ETX_STAGE_BSDF_CLASS MaterialClass::Plastic
# define ETX_STAGE_BSDF_EVAL  bsdf_plastic_opaque_evaluate
# define ETX_STAGE_BSDF_PDF   bsdf_plastic_opaque_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR)
# if ETX_ENABLE_OPENPBR_STAGE
#  include <interop/bsdf_openpbr_shared.hxx>
# else
#  include <interop/bsdf_energy_compensated_shared.hxx>
# endif
# define ETX_STAGE_BSDF_CLASS MaterialClass::Conductor
# define ETX_STAGE_BSDF_EVAL  wavefront_connect_camera_stage_conductor_eval
# define ETX_STAGE_BSDF_PDF   wavefront_connect_camera_stage_conductor_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC)
# include <interop/bsdf_energy_compensated_shared.hxx>
# define ETX_STAGE_BSDF_CLASS MaterialClass::Dielectric
# define ETX_STAGE_BSDF_EVAL  bsdf_dielectric_energy_compensated_evaluate
# define ETX_STAGE_BSDF_PDF   bsdf_dielectric_energy_compensated_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_COATED_CONDUCTOR)
# include "gpu_rt_wavefront_coating_storage.hlsl"
# define ETX_STAGE_BSDF_CLASS MaterialClass::CoatedConductor
# define ETX_STAGE_BSDF_PDF   bsdf_coated_conductor_pdf
#endif

#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE)
BSDFEval wavefront_connect_camera_stage_various_eval(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  switch (material.cls) {
    case MaterialClass::Translucent:
      return bsdf_translucent_evaluate(context, data, outgoing_direction, material, sampler);
    case MaterialClass::Mirror:
      return bsdf_mirror_evaluate(context, data, outgoing_direction, material, sampler);
    case MaterialClass::Boundary:
      return bsdf_boundary_evaluate(context, data, outgoing_direction, material, sampler);
# if ETX_ENABLE_THINFILM_STAGE
    case MaterialClass::Thinfilm:
      return bsdf_thinfilm_evaluate(context, data, outgoing_direction, material, sampler);
# endif
# if ETX_ENABLE_VELVET_STAGE
    case MaterialClass::Velvet:
      return bsdf_velvet_evaluate(context, data, outgoing_direction, material, sampler);
# endif
    case MaterialClass::Void:
      return bsdf_void_evaluate(context, data, outgoing_direction, material, sampler);
    case MaterialClass::DiffractionGrating:
      return bsdf_diffraction_grating_evaluate(context, data, outgoing_direction, material, sampler);
    default:
      return bsdf_diffuse_evaluate(context, data, outgoing_direction, material, sampler);
  }
}

float wavefront_connect_camera_stage_various_pdf(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  switch (material.cls) {
    case MaterialClass::Translucent:
      return bsdf_translucent_pdf(context, data, outgoing_direction, material, sampler);
    case MaterialClass::Mirror:
      return bsdf_mirror_pdf(context, data, outgoing_direction, material, sampler);
    case MaterialClass::Boundary:
      return bsdf_boundary_pdf(context, data, outgoing_direction, material, sampler);
# if ETX_ENABLE_THINFILM_STAGE
    case MaterialClass::Thinfilm:
      return bsdf_thinfilm_pdf(context, data, outgoing_direction, material, sampler);
# endif
# if ETX_ENABLE_VELVET_STAGE
    case MaterialClass::Velvet:
      return bsdf_velvet_pdf(context, data, outgoing_direction, material, sampler);
# endif
    case MaterialClass::Void:
      return bsdf_void_pdf(context, data, outgoing_direction, material, sampler);
    case MaterialClass::DiffractionGrating:
      return bsdf_diffraction_grating_pdf(context, data, outgoing_direction, material, sampler);
    default:
      return bsdf_diffuse_pdf(context, data, outgoing_direction, material, sampler);
  }
}
#endif

#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR)
BSDFEval wavefront_connect_camera_stage_conductor_eval(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
# if ETX_ENABLE_OPENPBR_STAGE
  if (material.cls == MaterialClass::OpenPBR) {
    return bsdf_openpbr_evaluate(context, data, outgoing_direction, material, sampler);
  }
# endif
  return bsdf_conductor_energy_compensated_evaluate(context, data, outgoing_direction, material, sampler);
}

float wavefront_connect_camera_stage_conductor_pdf(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
# if ETX_ENABLE_OPENPBR_STAGE
  if (material.cls == MaterialClass::OpenPBR) {
    return bsdf_openpbr_pdf(context, data, outgoing_direction, material, sampler);
  }
# endif
  return bsdf_conductor_energy_compensated_pdf(context, data, outgoing_direction, material, sampler);
}
#endif

#if (ETX_BSDF_KIND != ETX_WAVEFRONT_BSDF_KIND_COATED_CONDUCTOR)
BSDFEval wavefront_connect_camera_stage_bsdf_eval(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  return ETX_STAGE_BSDF_EVAL(context, data, outgoing_direction, material, sampler);
}
#endif

float wavefront_connect_camera_stage_bsdf_pdf(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  return ETX_STAGE_BSDF_PDF(context, data, outgoing_direction, material, sampler);
}

bool wavefront_connect_camera_stage_matches_material(uint material_class) {
#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE)
  return (material_class == MaterialClass::Diffuse) || (material_class == MaterialClass::Translucent) || (material_class == MaterialClass::Mirror) ||
         (material_class == MaterialClass::Boundary) || (material_class == MaterialClass::Void) || (material_class == MaterialClass::DiffractionGrating)
# if ETX_ENABLE_THINFILM_STAGE
         || (material_class == MaterialClass::Thinfilm)
# endif
# if ETX_ENABLE_VELVET_STAGE
         || (material_class == MaterialClass::Velvet)
# endif
    ;
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR)
  return (material_class == MaterialClass::Conductor)
# if ETX_ENABLE_OPENPBR_STAGE
         || (material_class == MaterialClass::OpenPBR)
# endif
    ;
#else
  return material_class == ETX_STAGE_BSDF_CLASS;
#endif
}

#include "gpu_rt_wavefront_connect_camera_prepare_common.hlsl"

[numthreads(64, 1, 1)] void ETX_STAGE_ENTRY(uint3 dtid : SV_DispatchThreadID) {
  WavefrontConnectCameraPrepareInput input_value = (WavefrontConnectCameraPrepareInput)0;
  if (wavefront_load_connect_camera_prepare_input(dtid.x, input_value) == false) {
    return;
  }
#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_COATED_CONDUCTOR)
  const uint request_index = wavefront_coating_request_index(input_value.resources, false, input_value.path_index);
  const uint request_offset = wavefront_coating_request_offset(request_index);
  RWByteAddressBuffer request_buffer = bindless_rw_buffers[NonUniformResourceIndex(input_value.resources.coating_request_buffer)];
  RWByteAddressBuffer task_buffer = bindless_rw_buffers[NonUniformResourceIndex(input_value.resources.connect_camera_task_buffer)];
  const uint task_flags_offset = input_value.task_index * kGPUWavefrontConnectCameraTaskStride + kGPUWavefrontConnectCameraTaskFlagsOffset;
#endif

#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_COATED_CONDUCTOR) && ETX_COATING_RESOLVE
  if ((task_buffer.Load(task_flags_offset) & GPUWavefrontPointConnectionTaskFlags::CoatingPending) == 0u) {
    return;
  }
  task_buffer.Store(task_flags_offset, 0u);
  const uint film_seed = request_buffer.Load(request_offset + kGPUCoatingRequestConsumerSeedOffset);
  if (wavefront_sample_connect_camera_prepare_input(input_value, film_seed, false) == false) {
    return;
  }
  Sampler bsdf_sampler = (Sampler)0;
  BSDFEval bsdf_eval = wavefront_coating_load_evaluation(input_value.resources, request_index, bsdf_sampler);
#else
  const uint film_seed = wavefront_connect_camera_film_seed(input_value);
# if ETX_UPBP
  const bool update_path_sampler = scene_path_mode_is_upbp() == false;
# else
  const bool update_path_sampler = true;
# endif
  if (wavefront_sample_connect_camera_prepare_input(input_value, film_seed, update_path_sampler) == false) {
    return;
  }

  uint bsdf_seed = input_value.state.sampler_seed;
# if ETX_UPBP
  if (scene_path_mode_is_upbp()) {
    bsdf_seed = wavefront_connect_camera_upbp_evaluation_seed(input_value, kUPBPRandomDomainScatteringEvaluation);
  }
# endif
  Sampler bsdf_sampler = wavefront_connect_camera_make_bsdf_sampler(bsdf_seed);
  BSDFData bsdf_data =
    wavefront_connect_camera_make_surface_bsdf_data(input_value.hit.vertex, input_value.state.spect, input_value.hit.medium_index, input_value.current_vertex.w_i);
# if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_COATED_CONDUCTOR)
  wavefront_coating_begin_evaluation(wavefront_connect_camera_make_scene_bsdf_resource_gpu_context(), input_value.resources, request_index,
    input_value.current_vertex.material_index, bsdf_data, input_value.camera_sample.direction, input_value.material, bsdf_sampler);
  request_buffer.Store(request_offset + kGPUCoatingRequestConsumerSeedOffset, film_seed);
  task_buffer.Store(task_flags_offset, uint(GPUWavefrontPointConnectionTaskFlags::CoatingPending));
# else
  BSDFEval bsdf_eval = wavefront_connect_camera_stage_bsdf_eval(wavefront_connect_camera_make_scene_bsdf_resource_gpu_context(), bsdf_data, input_value.camera_sample.direction,
    input_value.material, bsdf_sampler);
# endif
#endif

#if (ETX_BSDF_KIND != ETX_WAVEFRONT_BSDF_KIND_COATED_CONDUCTOR) || ETX_COATING_RESOLVE
  float shading_fix = bsdf_fix_shading_normal(input_value.hit.geo_normal, input_value.hit.vertex.nrm, input_value.current_vertex.w_i, input_value.camera_sample.direction);
  bsdf_eval.bsdf = spectral_response_mul(bsdf_eval.bsdf, shading_fix);
  wavefront_store_connect_camera_prepare_task(input_value.task_index, input_value, bsdf_eval, bsdf_sampler);
#endif
}
