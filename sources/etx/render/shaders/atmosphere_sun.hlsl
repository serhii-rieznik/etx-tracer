#include "bindless.hlsl"

#include <interop/math_shared.hxx>
#include <interop/spectrum.hxx>
#include <interop/atmosphere_scattering_shared.hxx>

[[vk::push_constant]] AtmosphereSunPushConstants constants;

float atmosphere_sun_transmittance(float optical_depth) {
  return exp(-optical_depth);
}

float3 atmosphere_sun_extinction_xyz(float3 view_direction, float3 next_direction) {
  float3 origin = float3(0.0f, kScatteringPlanetRadius + constants.atmosphere_altitude, 0.0f);
  if (scattering_distance_to_sphere(origin, next_direction, float3(0.0f, 0.0f, 0.0f), kScatteringPlanetRadius) > 0.0f) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  float to_space = scattering_distance_to_sphere(origin, view_direction, float3(0.0f, 0.0f, 0.0f), kScatteringOuterSphereRadius);
  float3 density_scale = float3(constants.atmosphere_rayleigh_scale, constants.atmosphere_mie_scale, constants.atmosphere_ozone_scale);
  float3 view_optical_path = float3(0.0f, 0.0f, 0.0f);
  float t = 0.0f;

  while (t < to_space) {
    float dt = scattering_calculate_step_size_direct(t, to_space, origin, view_direction);
    float3 p = origin + view_direction * (t + 0.5f * dt);
    t += dt;
    float height_above_surface = length(p) - kScatteringPlanetRadius;
    float3 current_density = scattering_density_direct(height_above_surface);
    view_optical_path += dt * density_scale * current_density;
  }

  float first_wavelength = float(ShortestWavelength);
  float first_r = scattering_rayleigh(first_wavelength);
  float first_m = scattering_mie(first_wavelength);
  float first_o = scattering_ozone_absorption(first_wavelength);
  float first_transmittance = atmosphere_sun_transmittance(first_r * view_optical_path.x) * atmosphere_sun_transmittance(first_m * view_optical_path.y) *
                              atmosphere_sun_transmittance(first_o * view_optical_path.z);
  float3 previous_xyz = spectral_response_to_xyz(spectral_response_make(first_wavelength, first_transmittance));
  float3 xyz = float3(0.0f, 0.0f, 0.0f);

  for (uint wavelength = ShortestWavelength + 1u; wavelength <= LongestWavelength; ++wavelength) {
    float current_wavelength = float(wavelength);
    float r = scattering_rayleigh(current_wavelength);
    float m = scattering_mie(current_wavelength);
    float o = scattering_ozone_absorption(current_wavelength);

    float transmittance =
      atmosphere_sun_transmittance(r * view_optical_path.x) * atmosphere_sun_transmittance(m * view_optical_path.y) * atmosphere_sun_transmittance(o * view_optical_path.z);
    float3 current_xyz = spectral_response_to_xyz(spectral_response_make(current_wavelength, transmittance));
    xyz += previous_xyz + 0.5f * (current_xyz - previous_xyz);
    previous_xyz = current_xyz;
  }

  return xyz;
}

[numthreads(8, 8, 1)] void sun_main(uint3 dtid : SV_DispatchThreadID) {
  if ((dtid.x >= constants.width) || (dtid.y >= constants.height)) {
    return;
  }

  float2 dim = float2(float(constants.width), float(constants.height));
  float u = ((float(dtid.x) + 0.5f) / dim.x) * 2.0f - 1.0f;
  float v0 = ((float(dtid.y) + 0.5f) / dim.y) * 2.0f - 1.0f;
  float v1 = ((float(dtid.y) + 1.5f) / dim.y) * 2.0f - 1.0f;

  float3 light_direction = normalize(constants.light_direction);
  OrthonormalBasis basis = orthonormal_basis(light_direction);
  float tan_half_fov = tan(0.5f * constants.angular_size);

  float3 d0 = normalize(tan_half_fov * (u * basis.u + v0 * basis.v) + light_direction);
  float3 d1 = normalize(tan_half_fov * (u * basis.u + v1 * basis.v) + light_direction);

  float3 xyz = atmosphere_sun_extinction_xyz(d0, d1);
  float darkening = scattering_sun_disk_darkening(u, v0);
  float3 rgb = max(float3(0.0f, 0.0f, 0.0f), spectral_xyz_to_rgb(darkening * xyz));

  bindless_storage_textures[NonUniformResourceIndex(constants.output_texture_index)][int2(dtid.xy)] = float4(rgb, 1.0f);
}
