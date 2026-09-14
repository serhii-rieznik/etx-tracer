#pragma once

#include <etx/render/shared/material.hxx>

namespace etx {

struct SpectrumTarget {
  uint32_t* spectrum_slot(Material& material) const;

  uint32_t material_index = kInvalidIndex;
  uint32_t spectrum_index = kInvalidIndex;
  enum class Channel : uint32_t { None, Scattering, Reflectance, Emission, InsideEta, InsideK, OutsideEta, OutsideK, Subsurface, ThinfilmEta, ThinfilmK } channel = Channel::None;
};

struct SpectrumDocument {
  std::vector<float2> points = {{kShortestWavelength, 0.5f}, {kLongestWavelength, 0.5f}};
  std::string classification = "reflectance";
  std::string title = "Custom spectrum";

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
  bool load_file(const std::string& path);
  float _drag_min_wavelength = kShortestWavelength;
  float _drag_max_wavelength = kLongestWavelength;
  std::string _pending_load_path;
  int _save_class = 0;
  float _simplify_maximum_error = 0.01f;
};

}  // namespace etx
