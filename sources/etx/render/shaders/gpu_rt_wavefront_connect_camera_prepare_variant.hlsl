#include "bindless.hlsl"
#include <interop/image_filter_shared.hxx>
#include <access/bsdf_resource_gpu.hxx>

#define ETX_WAVEFRONT_BSDF_KIND_DIFFUSE    1
#define ETX_WAVEFRONT_BSDF_KIND_PLASTIC    2
#define ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR  3
#define ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC 4

#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE)
# include <interop/bsdf_various_shared.hxx>
# define ETX_STAGE_BSDF_CLASS MaterialClass::Diffuse
# define ETX_STAGE_BSDF_EVAL  bsdf_diffuse_evaluate
# define ETX_STAGE_BSDF_PDF   bsdf_diffuse_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_PLASTIC)
# include <interop/bsdf_plastic_shared.hxx>
# define ETX_STAGE_BSDF_CLASS MaterialClass::Plastic
# define ETX_STAGE_BSDF_EVAL  bsdf_plastic_evaluate
# define ETX_STAGE_BSDF_PDF   bsdf_plastic_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR)
# include <interop/bsdf_conductor_shared.hxx>
# define ETX_STAGE_BSDF_CLASS MaterialClass::Conductor
# define ETX_STAGE_BSDF_EVAL  bsdf_conductor_evaluate
# define ETX_STAGE_BSDF_PDF   bsdf_conductor_pdf
#endif

#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC)
[numthreads(64, 1, 1)] void ETX_STAGE_ENTRY(uint3 dtid : SV_DispatchThreadID) {
  (void)dtid;
}
#else
BSDFEval wavefront_connect_camera_stage_bsdf_eval(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  return ETX_STAGE_BSDF_EVAL(context, data, outgoing_direction, material, sampler);
}

float wavefront_connect_camera_stage_bsdf_pdf(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  return ETX_STAGE_BSDF_PDF(context, data, outgoing_direction, material, sampler);
}

# include "gpu_rt_wavefront_connect_camera_prepare_common.hlsl"

[numthreads(64, 1, 1)] void ETX_STAGE_ENTRY(uint3 dtid : SV_DispatchThreadID) {
  wavefront_clear_connect_camera_task(dtid.x);

  WavefrontConnectCameraPrepareInput input_value = (WavefrontConnectCameraPrepareInput)0;
  if (wavefront_load_connect_camera_prepare_input(dtid.x, input_value) == false) {
    return;
  }
  if (input_value.material.cls != ETX_STAGE_BSDF_CLASS) {
    return;
  }

  Sampler bsdf_sampler = wavefront_connect_camera_make_bsdf_sampler(input_value.state.sampler_seed);
  BSDFData bsdf_data =
    wavefront_connect_camera_make_surface_bsdf_data(input_value.hit.vertex, input_value.state.spect, input_value.current_vertex.medium_index, input_value.current_vertex.w_i);
  BSDFEval bsdf_eval = wavefront_connect_camera_stage_bsdf_eval(wavefront_connect_camera_make_scene_bsdf_resource_gpu_context(), bsdf_data, input_value.camera_sample.direction,
    input_value.material, bsdf_sampler);
  float shading_fix = bsdf_fix_shading_normal(input_value.hit.geo_normal, input_value.hit.vertex.nrm, input_value.current_vertex.w_i, input_value.camera_sample.direction);
  if (isfinite(shading_fix) && (shading_fix > 0.0f)) {
    bsdf_eval.bsdf = spectral_response_mul(bsdf_eval.bsdf, shading_fix);
  }
  wavefront_store_connect_camera_prepare_task(dtid.x, input_value, bsdf_eval, bsdf_sampler);
}
#endif
