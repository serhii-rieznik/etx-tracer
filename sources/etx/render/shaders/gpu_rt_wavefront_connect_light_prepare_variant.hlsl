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
#if ETX_ENABLE_THINFILM_STAGE
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
#endif
#if ETX_ENABLE_VELVET_STAGE
    case MaterialClass::Velvet:
      return bsdf_velvet_evaluate(context, data, outgoing_direction, material, sampler);
#endif
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
#if ETX_ENABLE_THINFILM_STAGE
    case MaterialClass::Thinfilm:
      return bsdf_thinfilm_pdf(context, data, outgoing_direction, material, sampler);
#endif
#if ETX_ENABLE_VELVET_STAGE
    case MaterialClass::Velvet:
      return bsdf_velvet_pdf(context, data, outgoing_direction, material, sampler);
#endif
    case MaterialClass::Void:
      return bsdf_void_pdf(context, data, outgoing_direction, material, sampler);
    case MaterialClass::DiffractionGrating:
      return bsdf_diffraction_grating_pdf(context, data, outgoing_direction, material, sampler);
    default:
      return bsdf_diffuse_pdf(context, data, outgoing_direction, material, sampler);
  }
}
#endif

BSDFEval wavefront_connect_light_stage_camera_bsdf_eval(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  return ETX_STAGE_BSDF_EVAL(context, data, outgoing_direction, material, sampler);
}

float wavefront_connect_light_stage_camera_bsdf_pdf(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
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

BSDFEval wavefront_connect_light_stage_camera_bsdf_eval_prepared(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material,
  WavefrontConnectLightStagePrepared prepared, inout Sampler sampler) {
#if ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR
  (void)sampler;
  return bsdf_conductor_energy_compensated_evaluate_prepared(context, data, outgoing_direction, material, prepared.material);
#elif ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC
  (void)sampler;
  return bsdf_dielectric_energy_compensated_evaluate_prepared(context, data, outgoing_direction, material, prepared.material);
#else
  (void)prepared;
  return wavefront_connect_light_stage_camera_bsdf_eval(context, data, outgoing_direction, material, sampler);
#endif
}

float wavefront_connect_light_stage_camera_bsdf_pdf_prepared(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material,
  WavefrontConnectLightStagePrepared prepared, inout Sampler sampler) {
#if ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR
  (void)sampler;
  return bsdf_conductor_energy_compensated_pdf_prepared(context, data, outgoing_direction, material, prepared.material);
#elif ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC
  (void)sampler;
  return bsdf_dielectric_energy_compensated_pdf_prepared(context, data, outgoing_direction, material, prepared.material);
#else
  (void)prepared;
  return wavefront_connect_light_stage_camera_bsdf_pdf(context, data, outgoing_direction, material, sampler);
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

#define ETX_CONNECT_LIGHT_CAMERA_PREPARE_STAGE 1
#include "gpu_rt_wavefront_connect_light_prepare_common.hlsl"

[numthreads(64, 1, 1)] void ETX_STAGE_ENTRY(uint3 dtid : SV_DispatchThreadID) {
  const uint dispatch_index = dtid.x;
  const uint batch_index = dtid.y;

  const bool initialize_candidate = (constants.dispatch_item_offset & 2u) != 0u;
  uint initialized_light_vertex_index = kInvalidIndex;
  uint initialized_previous_light_vertex_index = kInvalidIndex;
  if (initialize_candidate) {
    wavefront_initialize_connect_light_prepare_candidate(dispatch_index, batch_index, initialized_light_vertex_index, initialized_previous_light_vertex_index);
  }

  WavefrontConnectLightPrepareInput input_value = (WavefrontConnectLightPrepareInput)0;
  if (wavefront_load_connect_light_prepare_input(dispatch_index, batch_index, initialize_candidate, initialized_light_vertex_index, initialized_previous_light_vertex_index,
        input_value) == false) {
    return;
  }
  if (wavefront_connect_light_stage_matches_vertex(input_value.camera_vertex, input_value.camera_material.cls) == false) {
    return;
  }

  float3 direction_to_light = input_value.light_vertex.position - input_value.camera_vertex.position;
  const float distance_squared = dot(direction_to_light, direction_to_light);
  if (distance_squared <= kRayEpsilon * kRayEpsilon) {
    return;
  }
  direction_to_light *= rsqrt(distance_squared);

  SpectralQuery spect = (SpectralQuery)0;
  spect.wavelength = input_value.camera_vertex.throughput.wavelength;
  spect.flags = input_value.camera_vertex.throughput.flags;

  BSDFEval bsdf_eval = (BSDFEval)0;
  if (wavefront_path_vertex_is_medium(input_value.camera_vertex)) {
    bsdf_eval = wavefront_connect_light_medium_eval(spect, input_value.camera_vertex, input_value.camera_vertex.w_i, direction_to_light);
  } else {
    Sampler bsdf_sampler = make_bsdf_sampler(scene_random_seed(input_value.task_index, constants.sample_index ^ (constants.path_iteration + 29u)));
    BSDFData bsdf_data = bsdf_data_make(wavefront_make_connect_path_vertex(input_value.camera_vertex), spect, kInvalidIndex, PathSource::Camera, input_value.camera_vertex.w_i);
    const BSDFResourceContext resource_context = make_scene_bsdf_resource_gpu_context();
    const WavefrontConnectLightStagePrepared prepared = wavefront_connect_light_stage_prepare_material(resource_context, bsdf_data, input_value.camera_material, bsdf_sampler);
    bsdf_eval = wavefront_connect_light_stage_camera_bsdf_eval_prepared(resource_context, bsdf_data, direction_to_light, input_value.camera_material, prepared, bsdf_sampler);
    wavefront_store_connect_light_camera_task(input_value, bsdf_eval, prepared);
    return;
  }
  wavefront_store_connect_light_camera_task(input_value, bsdf_eval, (WavefrontConnectLightStagePrepared)0);
}
