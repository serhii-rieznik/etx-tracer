#pragma once

#include "bsdf_resource_shared.hxx"

ETX_SHARED_INLINE float material_mix_amount(ETX_IN(BSDFResourceContext, context), ETX_IN(MaterialMix, mix), ETX_IN(float2, uv)) {
  return saturate(mix.amount.value.x * bsdf_resource_evaluate_sampled_image(context, mix.amount, uv, 1.0f));
}

#if (ETX_CPP)
template <typename MaterialContext>
ETX_SHARED_INLINE uint32_t material_mix_resolve(ETX_IN(MaterialContext, materials), ETX_IN(BSDFResourceContext, resources), uint32_t index, ETX_IN(float2, uv), float random) {
#else
ETX_SHARED_INLINE uint32_t material_mix_resolve(ETX_IN(MaterialAccessGPUContext, materials), ETX_IN(BSDFResourceContext, resources), uint32_t index, ETX_IN(float2, uv),
  float random) {
#endif
  Material material = ETX_ZERO(Material);
  while (material_access_try_load_full(materials, index, material)) {
    if (material.cls != MaterialClass::Mix) {
      return index;
    }
    const float amount = material_mix_amount(resources, material.mix, uv);
    if ((amount >= 1.0f) || ((amount > 0.0f) && (random < amount))) {
      index = material.mix.second;
      random = amount > 0.0f ? random / amount : 0.0f;
    } else {
      index = material.mix.first;
      random = amount < 1.0f ? (random - amount) / (1.0f - amount) : 0.0f;
    }
  }
  return kInvalidIndex;
}
