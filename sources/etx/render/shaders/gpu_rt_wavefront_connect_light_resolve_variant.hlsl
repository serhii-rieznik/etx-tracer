#include "bindless.hlsl"
#include <interop/image_filter_shared.hxx>
#include <access/bsdf_resource_gpu.hxx>

#define ETX_WAVEFRONT_BSDF_KIND_DIFFUSE    1
#define ETX_WAVEFRONT_BSDF_KIND_PLASTIC    2
#define ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR  3
#define ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC 4

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
# define ETX_STAGE_BSDF_EVAL  wavefront_connect_light_stage_various_eval
# define ETX_STAGE_BSDF_PDF   wavefront_connect_light_stage_various_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_PLASTIC)
# include <interop/bsdf_plastic_shared.hxx>
# define ETX_STAGE_BSDF_CLASS MaterialClass::Plastic
# define ETX_STAGE_BSDF_EVAL  bsdf_plastic_evaluate
# define ETX_STAGE_BSDF_PDF   bsdf_plastic_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR)
# include <interop/bsdf_energy_compensated_shared.hxx>
# define ETX_STAGE_BSDF_CLASS MaterialClass::Conductor
# define ETX_STAGE_BSDF_EVAL  bsdf_conductor_energy_compensated_evaluate
# define ETX_STAGE_BSDF_PDF   bsdf_conductor_energy_compensated_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC)
# include <interop/bsdf_energy_compensated_shared.hxx>
# define ETX_STAGE_BSDF_CLASS MaterialClass::Dielectric
# define ETX_STAGE_BSDF_EVAL  bsdf_dielectric_energy_compensated_evaluate
# define ETX_STAGE_BSDF_PDF   bsdf_dielectric_energy_compensated_pdf
#endif

#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE)
BSDFEval wavefront_connect_light_stage_various_eval(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
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

float wavefront_connect_light_stage_various_pdf(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
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

BSDFEval wavefront_connect_light_stage_light_bsdf_eval(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  return ETX_STAGE_BSDF_EVAL(context, data, outgoing_direction, material, sampler);
}

float wavefront_connect_light_stage_light_bsdf_pdf(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  return ETX_STAGE_BSDF_PDF(context, data, outgoing_direction, material, sampler);
}

struct WavefrontConnectLightStagePrepared {
#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR) || (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC)
  BSDFEnergyCompensatedPreparedMaterial material;
#else
  uint unused;
#endif
};

WavefrontConnectLightStagePrepared wavefront_connect_light_stage_prepare_material(BSDFResourceContext context, BSDFData data, Material material, inout Sampler sampler) {
  WavefrontConnectLightStagePrepared result = (WavefrontConnectLightStagePrepared)0;
#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR) || (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC)
  result.material = bsdf_energy_compensated_prepare_material(context, data.spectrum_sample, material, data.tex, sampler);
#else
  (void)context;
  (void)data;
  (void)material;
  (void)sampler;
#endif
  return result;
}

BSDFEval wavefront_connect_light_stage_light_bsdf_eval_prepared(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material,
  WavefrontConnectLightStagePrepared prepared, inout Sampler sampler) {
#if ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR
  (void)sampler;
  return bsdf_conductor_energy_compensated_evaluate_prepared(context, data, outgoing_direction, material, prepared.material);
#elif ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC
  (void)sampler;
  return bsdf_dielectric_energy_compensated_evaluate_prepared(context, data, outgoing_direction, material, prepared.material);
#else
  (void)prepared;
  return wavefront_connect_light_stage_light_bsdf_eval(context, data, outgoing_direction, material, sampler);
#endif
}

float wavefront_connect_light_stage_light_bsdf_pdf_prepared(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material,
  WavefrontConnectLightStagePrepared prepared, inout Sampler sampler) {
#if ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR
  (void)sampler;
  return bsdf_conductor_energy_compensated_pdf_prepared(context, data, outgoing_direction, material, prepared.material);
#elif ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC
  (void)sampler;
  return bsdf_dielectric_energy_compensated_pdf_prepared(context, data, outgoing_direction, material, prepared.material);
#else
  (void)prepared;
  return wavefront_connect_light_stage_light_bsdf_pdf(context, data, outgoing_direction, material, sampler);
#endif
}

bool wavefront_connect_light_stage_matches_material(uint material_class) {
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
  return material_class == MaterialClass::Conductor;
#else
  return material_class == ETX_STAGE_BSDF_CLASS;
#endif
}

#define ETX_CONNECT_LIGHT_RESOLVE_STAGE 1
#include "gpu_rt_wavefront_connect_light_prepare_common.hlsl"

[numthreads(64, 1, 1)] void ETX_STAGE_ENTRY(uint3 dtid : SV_DispatchThreadID) {
#if ETX_ENABLE_WORK_QUEUES
  uint dispatch_index = 0u;
  uint batch_index = 0u;
  if (wavefront_connect_queue_load(wavefront_load_resources(), kGPUWavefrontConnectQueueFamilyCount + ETX_BSDF_KIND - 1u, dtid.x + dtid.y * (65535u * 64u), dispatch_index,
        batch_index)) {
    wavefront_resolve_connect_light_prepare_task(dispatch_index, batch_index);
  }
#else
  wavefront_resolve_connect_light_prepare_task(dtid.x, dtid.y);
#endif
}
