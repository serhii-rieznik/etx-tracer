namespace etx {
namespace subsurface {

struct Gather {
  Intersection intersection;
  SpectralResponse weight;
};

ETX_SHARED_INLINE void remap_channel(float color, const float scattering_distances, float& albedo, float& extinction, float& scattering) {
  constexpr float a = 1.826052378200f;
  constexpr float b = 4.985111943850f + 0.12735595943800f;
  constexpr float c = 1.096861024240f;
  constexpr float d = 0.496310210422f;
  constexpr float e = 4.231902997010f + 0.00310603949088f;
  constexpr float f = 2.406029994080f;
  constexpr float kMinScattering = 1.0f / 1024.0f;

  color = fmaxf(0.0f, color);

  float blend = powf(color, 0.25f);
  albedo = (1.0f - blend) * a * powf(atanf(b * color), c) + blend * d * powf(atanf(e * color), f);
  ETX_VALIDATE(albedo);
  albedo = clamp(albedo, 0.0f, 1.0f - kEpsilon);

  extinction = 1.0f / fmaxf(scattering_distances, kMinScattering);
  ETX_VALIDATE(extinction);

  scattering = extinction * albedo;
  ETX_VALIDATE(scattering);
}

ETX_SHARED_INLINE void remap(const float3& color, const float3& scattering_distances, float3& albedo, float3& extinction, float3& scattering) {
  remap_channel(color.x, scattering_distances.x, albedo.x, extinction.x, scattering.x);
  remap_channel(color.y, scattering_distances.y, albedo.y, extinction.y, scattering.y);
  remap_channel(color.z, scattering_distances.z, albedo.z, extinction.z, scattering.z);
}

}  // namespace subsurface
}  // namespace etx
