#include "bindless.hlsl"
#include <interop/image_filter_shared.hxx>
#include <access/bsdf_resource_gpu.hxx>

#define ETX_WAVEFRONT_BSDF_KIND_DIFFUSE    1
#define ETX_WAVEFRONT_BSDF_KIND_PLASTIC    2
#define ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR  3
#define ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC 4
#define ETX_WAVEFRONT_BSDF_KIND_THINFILM   5

#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE)
# if ETX_ENABLE_VELVET_STAGE
#  include <interop/bsdf_velvet_shared.hxx>
# else
#  include <interop/bsdf_various_shared.hxx>
# endif
# define ETX_STAGE_BSDF_CLASS  MaterialClass::Diffuse
# define ETX_STAGE_BSDF_SAMPLE wavefront_surface_continue_stage_various_sample
# define ETX_STAGE_BSDF_PDF    wavefront_surface_continue_stage_various_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_PLASTIC)
# include <interop/bsdf_plastic_shared.hxx>
# define ETX_STAGE_BSDF_CLASS  MaterialClass::Plastic
# define ETX_STAGE_BSDF_SAMPLE bsdf_plastic_sample
# define ETX_STAGE_BSDF_PDF    bsdf_plastic_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR)
# if ETX_ENABLE_OPENPBR_STAGE
#  include <interop/bsdf_openpbr_shared.hxx>
# else
#  include <interop/bsdf_energy_compensated_shared.hxx>
# endif
# define ETX_STAGE_BSDF_CLASS  MaterialClass::Conductor
# define ETX_STAGE_BSDF_SAMPLE wavefront_surface_continue_stage_conductor_sample
# define ETX_STAGE_BSDF_PDF    wavefront_surface_continue_stage_conductor_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC)
# include <interop/bsdf_energy_compensated_shared.hxx>
# define ETX_STAGE_BSDF_CLASS  MaterialClass::Dielectric
# define ETX_STAGE_BSDF_SAMPLE bsdf_dielectric_energy_compensated_sample
# define ETX_STAGE_BSDF_PDF    bsdf_dielectric_energy_compensated_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_THINFILM)
# include <interop/bsdf_dielectric_shared.hxx>
# define ETX_STAGE_BSDF_CLASS  MaterialClass::Thinfilm
# define ETX_STAGE_BSDF_SAMPLE bsdf_thinfilm_sample
# define ETX_STAGE_BSDF_PDF    bsdf_thinfilm_pdf
#endif

#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE)
BSDFSample wavefront_surface_continue_stage_various_sample(BSDFResourceContext context, BSDFData data, Material material, inout Sampler sampler) {
  switch (material.cls) {
    case MaterialClass::Translucent:
      return bsdf_translucent_sample(context, data, material, sampler);
    case MaterialClass::Mirror:
      return bsdf_mirror_sample(context, data, material, sampler);
    case MaterialClass::Boundary:
      return bsdf_boundary_sample(context, data, material, sampler);
# if ETX_ENABLE_VELVET_STAGE
    case MaterialClass::Velvet:
      return bsdf_velvet_sample(context, data, material, sampler);
# endif
    case MaterialClass::Void:
      return bsdf_void_sample(context, data, material, sampler);
    case MaterialClass::DiffractionGrating:
      return bsdf_diffraction_grating_sample(context, data, material, sampler);
    default:
      return bsdf_diffuse_sample(context, data, material, sampler);
  }
}

#if ETX_WAVEFRONT_PATH_TRACING_ONLY == 0
float wavefront_surface_continue_stage_various_pdf(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  switch (material.cls) {
    case MaterialClass::Translucent:
      return bsdf_translucent_pdf(context, data, outgoing_direction, material, sampler);
    case MaterialClass::Mirror:
      return bsdf_mirror_pdf(context, data, outgoing_direction, material, sampler);
    case MaterialClass::Boundary:
      return bsdf_boundary_pdf(context, data, outgoing_direction, material, sampler);
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
#endif

#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR)
BSDFSample wavefront_surface_continue_stage_conductor_sample(BSDFResourceContext context, BSDFData data, Material material, inout Sampler sampler) {
# if ETX_ENABLE_OPENPBR_STAGE
  if (material.cls == MaterialClass::OpenPBR) {
    return bsdf_openpbr_sample(context, data, material, sampler);
  }
# endif
  return bsdf_conductor_energy_compensated_sample(context, data, material, sampler);
}

#if ETX_WAVEFRONT_PATH_TRACING_ONLY == 0
float wavefront_surface_continue_stage_conductor_pdf(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
# if ETX_ENABLE_OPENPBR_STAGE
  if (material.cls == MaterialClass::OpenPBR) {
    return bsdf_openpbr_pdf(context, data, outgoing_direction, material, sampler);
  }
# endif
  return bsdf_conductor_energy_compensated_pdf(context, data, outgoing_direction, material, sampler);
}
#endif
#endif

bool wavefront_surface_continue_stage_matches_material(uint material_class) {
#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE)
  return (material_class == MaterialClass::Diffuse) || (material_class == MaterialClass::Translucent) || (material_class == MaterialClass::Mirror) ||
         (material_class == MaterialClass::Boundary) || (material_class == MaterialClass::Void) || (material_class == MaterialClass::DiffractionGrating)
# if ETX_ENABLE_VELVET_STAGE
         || (material_class == MaterialClass::Velvet)
# endif
         ;
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR)
# if ETX_ENABLE_OPENPBR_STAGE
  return (material_class == MaterialClass::Conductor) || (material_class == MaterialClass::OpenPBR);
# else
  return material_class == MaterialClass::Conductor;
# endif
#else
  return material_class == ETX_STAGE_BSDF_CLASS;
#endif
}

BSDFSample wavefront_surface_continue_stage_bsdf_sample(BSDFResourceContext context, BSDFData data, Material material, inout Sampler sampler) {
  return ETX_STAGE_BSDF_SAMPLE(context, data, material, sampler);
}

#if ETX_WAVEFRONT_PATH_TRACING_ONLY == 0
float wavefront_surface_continue_stage_reverse_bsdf_pdf(BSDFResourceContext context, BSDFData input_data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  float3 reverse_w_o = -input_data.w_i;
  BSDFData reverse_data = input_data;
  reverse_data.w_i = -outgoing_direction;
  if (input_data.path_source == PathSource::Camera) {
    reverse_data.path_source = PathSource::Light;
  } else if (input_data.path_source == PathSource::Light) {
    reverse_data.path_source = PathSource::Camera;
  }
  return ETX_STAGE_BSDF_PDF(context, reverse_data, reverse_w_o, material, sampler);
}
#endif

#include "gpu_rt_wavefront_surface_continue_prepare_common.hlsl"

[numthreads(64, 1, 1)] void ETX_STAGE_ENTRY(uint3 dtid : SV_DispatchThreadID) {
  wavefront_surface_continue_prepare_specialized(false, dtid.x);
}
