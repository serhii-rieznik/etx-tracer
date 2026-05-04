namespace etx {

namespace OpenPBRBSDF {

struct OpenPBRComponents {
  Material conductor;
  Material dielectric;
  Material plastic;
  float metalness = 0.0f;
  float transmission = 0.0f;
  float conductor_weight = 0.0f;
  float dielectric_weight = 0.0f;
  float plastic_weight = 1.0f;
};

struct OpenPBRSampleResult {
  BSDFSample sample;
  float component_weight = 0.0f;
};

ETX_SHARED_INLINE void setup_conductor_material(Material& material) {
  material.cls = MaterialClass::Conductor;
  material.int_ior.cls = SpectralDistribution::Conductor;
  material.int_ior.eta_index = default_conductor_eta_index();
  material.int_ior.k_index = default_conductor_k_index();
  material.reflectance = material.scattering;
  material.scattering.image_index = kInvalidIndex;
  material.energy_compensation_interface_index = material.conductor_energy_compensation_interface_index;
}

ETX_SHARED_INLINE void setup_dielectric_material(Material& material) {
  material.cls = MaterialClass::Dielectric;
  material.int_ior.cls = SpectralDistribution::Dielectric;
  if (material.int_ior.eta_index == kInvalidIndex) {
    material.int_ior.eta_index = default_dielectric_eta_index();
  }
  material.int_ior.k_index = kInvalidIndex;
  material.reflectance.image_index = kInvalidIndex;
  const uint32_t white_spectrum = scene_global_get().defaults.white_spectrum;
  if (white_spectrum != kInvalidIndex) {
    material.scattering.spectrum_index = white_spectrum;
    material.scattering.image_index = kInvalidIndex;
  } else {
    material.scattering = material.reflectance;
  }
}

ETX_SHARED_INLINE void setup_plastic_material(Material& material) {
  material.cls = MaterialClass::Plastic;
  material.int_ior.cls = SpectralDistribution::Dielectric;
  if (material.int_ior.eta_index == kInvalidIndex) {
    material.int_ior.eta_index = default_dielectric_eta_index();
  }
  material.int_ior.k_index = kInvalidIndex;
}

ETX_SHARED_INLINE OpenPBRComponents make_components(const BSDFData& data, const Material& material) {
  OpenPBRComponents result = {};
  result.conductor = material;
  result.dielectric = material;
  result.plastic = material;
  setup_conductor_material(result.conductor);
  setup_dielectric_material(result.dielectric);
  setup_plastic_material(result.plastic);

  result.metalness = clamp(evaluate_metalness(material, data.tex), 0.0f, 1.0f);
  result.transmission = clamp(evaluate_transmission(material, data.tex), 0.0f, 1.0f);
  result.conductor_weight = result.metalness;
  result.dielectric_weight = (1.0f - result.metalness) * result.transmission;
  result.plastic_weight = (1.0f - result.metalness) * (1.0f - result.transmission);
  return result;
}

ETX_SHARED_INLINE BSDFEval mix_eval(const BSDFData& data, const float3& w_o, const OpenPBRComponents& components, Sampler& smp) {
  BSDFEval result = {};
  result.func = SpectralResponse(data.spectrum_sample, 0.0f);
  result.bsdf = SpectralResponse(data.spectrum_sample, 0.0f);
  result.eta = 1.0f;
  result.medium_index = kInvalidIndex;

  if (components.conductor_weight > 0.0f) {
    const BSDFEval conductor = ConductorBSDF::evaluate(data, w_o, components.conductor, smp);
    result.func += conductor.func * components.conductor_weight;
    result.bsdf += conductor.bsdf * components.conductor_weight;
    result.pdf += conductor.pdf * components.conductor_weight;
    result.properties |= conductor.properties;
    result.medium_index = conductor.medium_index;
  }

  if (components.dielectric_weight > 0.0f) {
    const BSDFEval dielectric = DielectricBSDF::evaluate(data, w_o, components.dielectric, smp);
    result.func += dielectric.func * components.dielectric_weight;
    result.bsdf += dielectric.bsdf * components.dielectric_weight;
    result.pdf += dielectric.pdf * components.dielectric_weight;
    result.properties |= dielectric.properties;
    result.medium_index = dielectric.medium_index;
    result.eta = dielectric.eta;
  }

  if (components.plastic_weight > 0.0f) {
    const BSDFEval plastic = PlasticBSDF::evaluate(data, w_o, components.plastic, smp);
    result.func += plastic.func * components.plastic_weight;
    result.bsdf += plastic.bsdf * components.plastic_weight;
    result.pdf += plastic.pdf * components.plastic_weight;
    result.properties |= plastic.properties;
    if (result.medium_index == kInvalidIndex) {
      result.medium_index = plastic.medium_index;
    }
  }

  return result;
}

ETX_SHARED_INLINE float mix_pdf(const BSDFData& data, const float3& w_o, const OpenPBRComponents& components, Sampler& smp) {
  float result = 0.0f;
  if (components.conductor_weight > 0.0f) {
    result += ConductorBSDF::pdf(data, w_o, components.conductor, smp) * components.conductor_weight;
  }
  if (components.dielectric_weight > 0.0f) {
    result += DielectricBSDF::pdf(data, w_o, components.dielectric, smp) * components.dielectric_weight;
  }
  if (components.plastic_weight > 0.0f) {
    result += PlasticBSDF::pdf(data, w_o, components.plastic, smp) * components.plastic_weight;
  }
  return result;
}

ETX_SHARED_INLINE OpenPBRSampleResult sample_component(const BSDFData& data, const OpenPBRComponents& components, Sampler& smp) {
  OpenPBRSampleResult result = {};
  const float u = smp.next();
  if (u < components.conductor_weight) {
    result.sample = ConductorBSDF::sample(data, components.conductor, smp);
    result.component_weight = components.conductor_weight;
    return result;
  }

  if (u < (components.conductor_weight + components.dielectric_weight)) {
    result.sample = DielectricBSDF::sample(data, components.dielectric, smp);
    result.component_weight = components.dielectric_weight;
    return result;
  }

  result.sample = PlasticBSDF::sample(data, components.plastic, smp);
  result.component_weight = components.plastic_weight;
  return result;
}

ETX_SHARED_INLINE BSDFSample sample(const BSDFData& data, const Material& material, Sampler& smp) {
  const OpenPBRComponents components = make_components(data, material);
  const OpenPBRSampleResult component_result = sample_component(data, components, smp);
  BSDFSample result = component_result.sample;
  if (result.valid() == false) {
    return result;
  }

  if (result.is_delta()) {
    result.pdf *= component_result.component_weight;
    return result;
  }

  const BSDFEval eval = mix_eval(data, result.w_o, components, smp);
  result.weight = (eval.pdf > 0.0f) ? (eval.bsdf / eval.pdf) : SpectralResponse(data.spectrum_sample, 0.0f);
  result.pdf = eval.pdf;
  result.eta = eval.eta;
  result.properties = eval.properties;
  result.medium_index = eval.medium_index;
  return result;
}

ETX_SHARED_INLINE BSDFEval evaluate(const BSDFData& data, const float3& w_o, const Material& material, Sampler& smp) {
  const OpenPBRComponents components = make_components(data, material);
  return mix_eval(data, w_o, components, smp);
}

ETX_SHARED_INLINE float pdf(const BSDFData& data, const float3& w_o, const Material& material, Sampler& smp) {
  const OpenPBRComponents components = make_components(data, material);
  return mix_pdf(data, w_o, components, smp);
}

ETX_SHARED_INLINE bool is_delta(const Material& material, const float2& tex, Sampler& smp) {
  return false;
}

ETX_SHARED_INLINE SpectralResponse albedo(const BSDFData& data, const Material& material, Sampler& smp) {
  return apply_image(data.spectrum_sample, material.scattering, data.tex);
}

}  // namespace OpenPBRBSDF
}  // namespace etx
