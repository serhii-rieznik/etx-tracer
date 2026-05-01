#pragma once

#include <cstdint>
#include <string>

namespace etx {

struct BSDFLutGenerationOptions {
  std::string output_directory = {};
  uint32_t sample_count = 512u;
};

bool generate_bsdf_energy_compensation_luts(const BSDFLutGenerationOptions& options);
bool pregenerate_named_bsdf_energy_compensation_lut_cache();

}  // namespace etx
