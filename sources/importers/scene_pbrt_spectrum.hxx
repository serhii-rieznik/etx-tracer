#pragma once

#include <etx/render/shared/spectrum.hxx>
#include <span>
#include <string_view>

namespace etx {

bool is_pbrt_named_spectrum(std::string_view name);
bool load_pbrt_named_spectrum(std::string_view name, SpectralDistribution& result);
float pbrt_photometric_response(const SpectralDistribution& spectrum);
SpectralDistribution pbrt_rgb_illuminant(const float3& rgb);
SpectralDistribution pbrt_sampled_spectrum(std::span<const float2> samples);
SpectralDistribution pbrt_multiply_spectra(const SpectralDistribution& a, const SpectralDistribution& b);

}  // namespace etx
