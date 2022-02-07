#pragma once

#include <etx/render/shared/base.hxx>
#include <etx/render/shared/spectrum.hxx>

namespace etx {

struct alignas(16) Material {
  enum class Class : uint32_t {
    Diffuse,
    Plastic,
    Conductor,
    MultiscatteringConductor,
    Dielectric,
    Thinfilm,
    Translucent,
    Mirror,
    Boundary,
    Generic,
    Coating,

    Count,
    Undefined = kInvalidIndex,
  };

  enum : uint32_t {
    DoubleSided = 1u << 0u,
  };

  struct alignas(16) Thinfilm {
    uint32_t image_index = kInvalidIndex;
    float min_thickness = 0.0f;
    float max_thickness = 0.0f;
    float pad;
  };

  Class cls = Class::Undefined;
  uint32_t diffuse_image_index = kInvalidIndex;
  uint32_t specular_image_index = kInvalidIndex;
  uint32_t normal_image_index = kInvalidIndex;

  uint32_t emissive_image_index = kInvalidIndex;
  uint32_t metal_roughness_image_index = kInvalidIndex;
  uint32_t int_medium = kInvalidIndex;
  uint32_t ext_medium = kInvalidIndex;

  SpectralDistribution diffuse = {};
  SpectralDistribution specular = {};
  SpectralDistribution transmittance = {};
  RefractiveIndex ext_ior = {};
  RefractiveIndex int_ior = {};

  Thinfilm thinfilm = {};

  float2 roughness = {0.0f, 0.0f};
  float metalness = 0.0f;
  float normal_scale = 0.5f;

  uint32_t options = {};
  uint32_t pad[3] = {};

  ETX_GPU_CODE bool double_sided() const {
    return (options & DoubleSided) == DoubleSided;
  }

  ETX_GPU_CODE bool is_delta() const {
    switch (cls) {
      case Class::Diffuse:
      case Class::Translucent:
      case Class::Boundary:
      case Class::Plastic:
        return false;

      case Class::Conductor:
      case Class::MultiscatteringConductor:
      case Class::Dielectric:
      case Class::Generic:
        return max(roughness.x, roughness.y) <= kDeltaAlphaTreshold;

      case Class::Thinfilm:
      case Class::Mirror:
        return true;

      default:
        return false;
    }
  }
};

}  // namespace etx
