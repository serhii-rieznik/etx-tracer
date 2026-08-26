#pragma once

#include <etx/rt/integrators/upbp_core.hxx>

#include <array>

namespace etx {

struct UPBPDensityMISContext {
  uint32_t light_vertex_count = 0u;
  UPBPVertexClass vertex_class = UPBPVertexClass::Surface;
  double forward_ray_factor = 0.0;
  double reverse_ray_factor = 0.0;
  double sin_theta = 0.0;
  bool delta = false;
  bool applicable = false;
};

struct UPBPDensityMISConfiguration {
  std::array<double, 6u> technique_factors = {};
  uint32_t enabled_techniques = 0u;
  bool photon_beams_long = false;
  bool camera_beams_long = true;

  bool enabled(const UPBPTechnique technique) const {
    return (enabled_techniques & static_cast<uint32_t>(technique)) != 0u;
  }

  double factor(const UPBPTechnique technique) const {
    switch (technique) {
      case UPBPTechnique::BPT:
        return technique_factors[0u];
      case UPBPTechnique::Surface:
        return technique_factors[1u];
      case UPBPTechnique::PP3D:
        return technique_factors[2u];
      case UPBPTechnique::PB2D:
        return technique_factors[3u];
      case UPBPTechnique::BP2D:
        return technique_factors[4u];
      case UPBPTechnique::BB1D:
        return technique_factors[5u];
    }
    return 0.0;
  }
};

struct UPBPDensityStrategyProbability {
  UPBPTechnique technique = UPBPTechnique::BPT;
  uint32_t light_vertex_count = 0u;
  double log_density = 0.0;
  bool applicable = false;
};

inline double upbp_density_strategy_factor(const UPBPDensityMISConfiguration& configuration, const UPBPDensityMISContext& context, const UPBPTechnique technique) {
  if ((technique == UPBPTechnique::BPT) || (configuration.enabled(technique) == false) || (context.applicable == false)) {
    return 0.0;
  }
  return upbp_density_competitor_factor({
    technique,
    context.vertex_class,
    configuration.factor(technique),
    context.forward_ray_factor,
    context.reverse_ray_factor,
    context.sin_theta,
    context.delta,
  });
}

template <typename BPTEnumerator>
inline bool upbp_enumerate_all_strategies(const UPBPPathProbabilityRecord& path, const std::vector<UPBPDensityMISContext>& density_contexts,
  const UPBPDensityMISConfiguration& configuration, BPTEnumerator&& enumerate_bpt, std::vector<UPBPDensityStrategyProbability>& result) {
  result.clear();
  std::vector<UPBPBPTStrategyProbability> bpt_strategies = {};
  if (enumerate_bpt(path, bpt_strategies) == false) {
    return false;
  }

  const bool bpt_enabled = configuration.enabled(UPBPTechnique::BPT);
  result.reserve(bpt_strategies.size() + density_contexts.size() * 5u);
  for (const UPBPBPTStrategyProbability& strategy : bpt_strategies) {
    result.emplace_back(UPBPDensityStrategyProbability{
      UPBPTechnique::BPT,
      strategy.light_vertex_count,
      strategy.log_density,
      strategy.applicable && bpt_enabled,
    });
  }

  constexpr std::array<UPBPTechnique, 5u> kDensityTechniques = {
    UPBPTechnique::Surface,
    UPBPTechnique::PP3D,
    UPBPTechnique::PB2D,
    UPBPTechnique::BP2D,
    UPBPTechnique::BB1D,
  };
  for (const UPBPDensityMISContext& context : density_contexts) {
    if ((context.light_vertex_count < 2u) || (context.light_vertex_count + 1u >= path.vertices.size())) {
      return false;
    }
    const UPBPBPTStrategyProbability& base = bpt_strategies[context.light_vertex_count];
    for (const UPBPTechnique technique : kDensityTechniques) {
      const double factor = upbp_density_strategy_factor(configuration, context, technique);
      result.emplace_back(UPBPDensityStrategyProbability{
        technique,
        context.light_vertex_count,
        factor > 0.0 ? base.log_density + std::log(factor) : -std::numeric_limits<double>::infinity(),
        base.applicable && (factor > 0.0),
      });
    }
  }
  return true;
}

inline bool upbp_enumerate_all_strategies_exhaustive(const UPBPPathProbabilityRecord& path, const std::vector<UPBPDensityMISContext>& density_contexts,
  const UPBPDensityMISConfiguration& configuration, std::vector<UPBPDensityStrategyProbability>& result) {
  return upbp_enumerate_all_strategies(path, density_contexts, configuration, upbp_enumerate_bpt_strategies_exhaustive, result);
}

inline bool upbp_enumerate_all_strategies_recursive(const UPBPPathProbabilityRecord& path, const std::vector<UPBPDensityMISContext>& density_contexts,
  const UPBPDensityMISConfiguration& configuration, std::vector<UPBPDensityStrategyProbability>& result) {
  return upbp_enumerate_all_strategies(path, density_contexts, configuration, upbp_enumerate_bpt_strategies_recursive, result);
}

inline double upbp_all_strategy_balance_weight(const std::vector<UPBPDensityStrategyProbability>& probabilities, const UPBPTechnique selected_technique,
  const uint32_t selected_light_vertex_count) {
  UPBPMISAccumulator accumulator = {};
  for (const UPBPDensityStrategyProbability& probability : probabilities) {
    const bool selected = (probability.technique == selected_technique) && (probability.light_vertex_count == selected_light_vertex_count);
    if (accumulator.append({probability.technique, probability.log_density, 1u, probability.applicable}, selected) == false) {
      return 0.0;
    }
  }
  return accumulator.weight();
}

}  // namespace etx
