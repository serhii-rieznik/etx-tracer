#pragma once

#include <etx/engine/options.hxx>
#include <etx/rt/integrators/upbp_core.hxx>

namespace etx {

struct UPBPOptions {
  static constexpr uint32_t kMaximumBoundaryCount = 4096u;
  static constexpr uint32_t kMaximumNullEventsPerInterval = 1048576u;
  static constexpr uint32_t kMaximumLightPathCount = 16777216u;
  static constexpr uint32_t kMinimumMemoryBudgetMiB = 64u;
  static constexpr uint32_t kMaximumMemoryBudgetMiB = 65536u;

  uint32_t technique_mask = static_cast<uint32_t>(UPBPTechnique::BPT) | static_cast<uint32_t>(UPBPTechnique::Surface) | static_cast<uint32_t>(UPBPTechnique::PP3D) |
                            static_cast<uint32_t>(UPBPTechnique::PB2D) | static_cast<uint32_t>(UPBPTechnique::BP2D) | static_cast<uint32_t>(UPBPTechnique::BB1D);
  UPBPKernel kernel = UPBPKernel::Epanechnikov;
  float initial_surface_radius = 0.0f;
  float initial_pp3d_radius = 0.0f;
  float initial_pb2d_radius = 0.0f;
  float initial_bp2d_radius = 0.0f;
  float initial_bb1d_radius = 0.0f;
  float radius_alpha = 0.75f;
  float beam_selection_probability = 1.0f;
  uint32_t maximum_boundary_count = 64u;
  uint32_t maximum_null_events_per_interval = 1024u;
  uint32_t maximum_light_path_count = 0u;
  uint32_t memory_budget_mb = 2048u;

  bool enabled(const UPBPTechnique technique) const {
    return (technique_mask & static_cast<uint32_t>(technique)) != 0u;
  }

  void load(const Options& options) {
    auto load_technique = [&options, this](const char* id, const UPBPTechnique technique) {
      const bool value = options.get_bool(id, enabled(technique));
      if (value) {
        technique_mask |= static_cast<uint32_t>(technique);
      } else {
        technique_mask &= ~static_cast<uint32_t>(technique);
      }
    };
    load_technique("upbp-bpt", UPBPTechnique::BPT);
    load_technique("upbp-surface", UPBPTechnique::Surface);
    load_technique("upbp-pp3d", UPBPTechnique::PP3D);
    load_technique("upbp-pb2d", UPBPTechnique::PB2D);
    load_technique("upbp-bp2d", UPBPTechnique::BP2D);
    load_technique("upbp-bb1d", UPBPTechnique::BB1D);
    kernel = options.get_integral("upbp-kernel", kernel);
    initial_surface_radius = options.get_float("upbp-surface-radius", initial_surface_radius);
    initial_pp3d_radius = options.get_float("upbp-pp3d-radius", initial_pp3d_radius);
    initial_pb2d_radius = options.get_float("upbp-pb2d-radius", initial_pb2d_radius);
    initial_bp2d_radius = options.get_float("upbp-bp2d-radius", initial_bp2d_radius);
    initial_bb1d_radius = options.get_float("upbp-bb1d-radius", initial_bb1d_radius);
    radius_alpha = options.get_float("upbp-radius-alpha", radius_alpha);
    beam_selection_probability = options.get_float("upbp-beam-selection-probability", beam_selection_probability);
    maximum_boundary_count = options.get_integral("upbp-maximum-boundaries", maximum_boundary_count);
    maximum_null_events_per_interval = options.get_integral("upbp-maximum-null-events", maximum_null_events_per_interval);
    maximum_light_path_count = options.get_integral("upbp-light-path-count", maximum_light_path_count);
    memory_budget_mb = options.get_integral("upbp-memory-budget-mb", memory_budget_mb);
  }

  void store(Options& options) const {
    options.options.clear();
    options.set_string("upbp-options", "UPBP Options", "UPBP Options");
    options.set_bool("upbp-bpt", enabled(UPBPTechnique::BPT), "BPT");
    options.set_bool("upbp-surface", enabled(UPBPTechnique::Surface), "Surface merging");
    options.set_bool("upbp-pp3d", enabled(UPBPTechnique::PP3D), "Point-point volume merging");
    options.set_bool("upbp-pb2d", enabled(UPBPTechnique::PB2D), "Point-beam volume estimation");
    options.set_bool("upbp-bp2d", enabled(UPBPTechnique::BP2D), "Beam-point volume estimation");
    options.set_bool("upbp-bb1d", enabled(UPBPTechnique::BB1D), "Beam-beam volume estimation");
    options.set_integral("upbp-kernel", kernel, "Kernel", Option::Meta::EnumValue, {UPBPKernel::TopHat, UPBPKernel::Epanechnikov}).name_getter = [](uint32_t index) {
      return index == static_cast<uint32_t>(UPBPKernel::TopHat) ? std::string{"Top Hat"} : std::string{"Epanechnikov"};
    };
    options.set_float("upbp-surface-radius", initial_surface_radius, "Initial surface radius", {0.0f, 1000.0f});
    options.set_float("upbp-pp3d-radius", initial_pp3d_radius, "Initial PP3D radius", {0.0f, 1000.0f});
    options.set_float("upbp-pb2d-radius", initial_pb2d_radius, "Initial PB2D radius", {0.0f, 1000.0f});
    options.set_float("upbp-bp2d-radius", initial_bp2d_radius, "Initial BP2D radius", {0.0f, 1000.0f});
    options.set_float("upbp-bb1d-radius", initial_bb1d_radius, "Initial BB1D radius", {0.0f, 1000.0f});
    options.set_float("upbp-radius-alpha", radius_alpha, "Radius alpha", {0.01f, 1.0f});
    options.set_float("upbp-beam-selection-probability", beam_selection_probability, "BB1D light-beam selection probability", {0.001f, 1.0f});
    options.set_integral("upbp-maximum-boundaries", maximum_boundary_count, "Maximum boundaries per segment", 0u, {1u, kMaximumBoundaryCount});
    options.set_integral("upbp-maximum-null-events", maximum_null_events_per_interval, "Maximum null events per medium interval", 0u, {1u, kMaximumNullEventsPerInterval});
    options.set_integral("upbp-light-path-count", maximum_light_path_count, "Maximum light paths per iteration (0 = memory budget)", 0u, {0u, kMaximumLightPathCount});
    options.set_integral("upbp-memory-budget-mb", memory_budget_mb, "Retained light storage budget (MiB)", 0u, {kMinimumMemoryBudgetMiB, kMaximumMemoryBudgetMiB});
  }
};

}  // namespace etx
