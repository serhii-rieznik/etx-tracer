#include "bindless.hlsl"
#include <interop/image_filter_shared.hxx>
#include <access/bsdf_resource_gpu.hxx>

#define ETX_WAVEFRONT_BSDF_KIND_DIFFUSE    1
#define ETX_WAVEFRONT_BSDF_KIND_PLASTIC    2
#define ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR  3
#define ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC 4
#define ETX_WAVEFRONT_BSDF_KIND_THINFILM   5

#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE)
# include <interop/bsdf_various_shared.hxx>
# define ETX_STAGE_BSDF_CLASS  MaterialClass::Diffuse
# define ETX_STAGE_BSDF_SAMPLE bsdf_diffuse_sample
# define ETX_STAGE_BSDF_PDF    bsdf_diffuse_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_PLASTIC)
# include <interop/bsdf_plastic_shared.hxx>
# define ETX_STAGE_BSDF_CLASS  MaterialClass::Plastic
# define ETX_STAGE_BSDF_SAMPLE bsdf_plastic_sample
# define ETX_STAGE_BSDF_PDF    bsdf_plastic_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR)
# include <interop/bsdf_various_shared.hxx>
# define ETX_STAGE_BSDF_CLASS  MaterialClass::Conductor
# define ETX_STAGE_BSDF_SAMPLE bsdf_diffuse_sample
# define ETX_STAGE_BSDF_PDF    bsdf_diffuse_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC)
# include <interop/bsdf_various_shared.hxx>
# define ETX_STAGE_BSDF_CLASS  MaterialClass::Dielectric
# define ETX_STAGE_BSDF_SAMPLE bsdf_diffuse_sample
# define ETX_STAGE_BSDF_PDF    bsdf_diffuse_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_THINFILM)
# include <interop/bsdf_dielectric_shared.hxx>
# define ETX_STAGE_BSDF_CLASS  MaterialClass::Thinfilm
# define ETX_STAGE_BSDF_SAMPLE bsdf_thinfilm_sample
# define ETX_STAGE_BSDF_PDF    bsdf_thinfilm_pdf
#endif

bool wavefront_surface_continue_stage_matches_material(uint material_class) {
  return material_class == ETX_STAGE_BSDF_CLASS;
}

BSDFSample wavefront_surface_continue_stage_bsdf_sample(BSDFResourceContext context, BSDFData data, Material material, inout Sampler sampler) {
  return ETX_STAGE_BSDF_SAMPLE(context, data, material, sampler);
}

float wavefront_surface_continue_stage_reverse_bsdf_pdf(BSDFResourceContext context, BSDFData input_data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  float3 reverse_w_o = -input_data.w_i;
  BSDFData reverse_data = input_data;
  reverse_data.w_i = -outgoing_direction;
  return ETX_STAGE_BSDF_PDF(context, reverse_data, reverse_w_o, material, sampler);
}

#include "gpu_rt_wavefront_surface_continue_prepare_common.hlsl"

[numthreads(64, 1, 1)] void wavefront_light_continue_prepare_diffuse_main(uint3 dtid : SV_DispatchThreadID) {
  wavefront_surface_continue_prepare_specialized(false, dtid.x);
}

  [numthreads(64, 1, 1)] void wavefront_light_continue_prepare_plastic_main(uint3 dtid : SV_DispatchThreadID) {
  wavefront_surface_continue_prepare_specialized(false, dtid.x);
}

[numthreads(64, 1, 1)] void wavefront_light_continue_prepare_conductor_main(uint3 dtid : SV_DispatchThreadID) {
  wavefront_surface_continue_prepare_specialized(false, dtid.x);
}

  [numthreads(64, 1, 1)] void wavefront_light_continue_prepare_dielectric_main(uint3 dtid : SV_DispatchThreadID) {
  wavefront_surface_continue_prepare_specialized(false, dtid.x);
}

[numthreads(64, 1, 1)] void wavefront_light_continue_prepare_thinfilm_main(uint3 dtid : SV_DispatchThreadID) {
  wavefront_surface_continue_prepare_specialized(false, dtid.x);
}
