#pragma once

#include "material.hxx"

struct HitPolicyMode {
  enum : uint32_t {
    KeepBoundaryHit = 0u,
    SkipBoundaryWithMediumTransition = 1u,
    MediumTransmittance = 2u,
  };
};

struct HitPolicyAction {
  enum : uint32_t {
    Ignore = 0u,
    CommitSurface = 1u,
    TransitionMedium = 2u,
    Occlude = 3u,
  };
};

struct ETX_ALIGNED HitPolicyDecision {
  uint32_t action ETX_INIT(HitPolicyAction::Ignore);
  uint32_t medium_index ETX_INIT(kInvalidIndex);
};

ETX_SHARED_INLINE uint32_t hit_policy_boundary_medium(bool entering_surface, uint32_t int_medium, uint32_t ext_medium) {
  return entering_surface ? int_medium : ext_medium;
}

ETX_SHARED_INLINE HitPolicyDecision hit_policy_evaluate(uint32_t mode, uint32_t material_class, bool alpha_rejected, bool entering_surface, uint32_t int_medium,
  uint32_t ext_medium) {
  HitPolicyDecision result;
  result.action = HitPolicyAction::Ignore;
  result.medium_index = kInvalidIndex;

  if ((material_class == MaterialClass::Void) || alpha_rejected) {
    return result;
  }

  if (material_class == MaterialClass::Boundary) {
    if (mode == HitPolicyMode::KeepBoundaryHit) {
      result.action = HitPolicyAction::CommitSurface;
    } else {
      result.action = HitPolicyAction::TransitionMedium;
      result.medium_index = hit_policy_boundary_medium(entering_surface, int_medium, ext_medium);
    }
    return result;
  }

  result.action = (mode == HitPolicyMode::MediumTransmittance) ? HitPolicyAction::Occlude : HitPolicyAction::CommitSurface;
  return result;
}
