#pragma once

#include <etx/render/host/scene_data.hxx>

namespace etx {

struct SpectrumTarget {
  uint32_t* spectrum_slot(Material& material) const;
  uint32_t* spectrum_slot(SceneData& scene) const;

  uint32_t material_index = kInvalidIndex;
  uint32_t spectrum_index = kInvalidIndex;
  enum class Channel : uint32_t {
    None,
    Scattering,
    Reflectance,
    Emission,
    InsideEta,
    InsideK,
    OutsideEta,
    OutsideK,
    Subsurface,
    ThinfilmEta,
    ThinfilmK,
    Absorption
  } channel = Channel::None;
  uint32_t medium_index = kInvalidIndex;
  uint32_t emitter_index = kInvalidIndex;
};

struct SpectrumEdit {
  std::vector<SpectrumTarget> targets;
  SpectrumSource source;
};

struct SpectrumDocument {
  std::vector<float2> points = {{kShortestWavelength, 0.5f}, {kLongestWavelength, 0.5f}};
  std::string classification = "reflectance";
  std::string title = "Custom spectrum";
  std::string path;

  bool validate(std::string& error) const;
  bool load(const std::string& path, std::string& error);
  bool save(const std::string& path, const std::string& protected_directory, std::string& error) const;
  SpectralDistribution distribution() const;
  float evaluate(float wavelength) const;
  bool simplify(float maximum_error);
  bool normalize();
};

struct SpectrumCurveEditor {
  SpectrumDocument document;
  int selected_point = 0;
  float wavelength = kShortestWavelength;
  float value = 0.5f;
  double plot_min = 0.0;
  double plot_max = 1.0;
  bool modified = false;
  std::string error;

  bool build();
  void select_point(int index);
  void fit();

 private:
  float _drag_min_wavelength = kShortestWavelength;
  float _drag_max_wavelength = kLongestWavelength;
  float _simplify_maximum_error = 0.01f;
};

}  // namespace etx
