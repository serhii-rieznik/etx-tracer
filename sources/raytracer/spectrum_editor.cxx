#include "spectrum_editor.hxx"

#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/core/platform.hxx>
#include <imgui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>

namespace etx {
namespace {

ImU32 wavelength_color(float wavelength) {
  if ((wavelength < kRGBResponseShortestWavelength) || (wavelength > kRGBResponseLongestWavelength)) {
    return ImGui::GetColorU32(ImGuiCol_TextDisabled);
  }
  static const auto colors = []() {
    std::array<ImU32, RGBResponseWavelengthCount> result;
    for (uint32_t i = 0u; i < RGBResponseWavelengthCount; ++i) {
      float3 rgb = max(xyz_to_rgb(spectral_xyz(RGBResponseShortestWavelength - ShortestWavelength + i)), float3{});
      rgb = linear_to_gamma(rgb / max(rgb.x, max(rgb.y, rgb.z)));
      result[i] = ImGui::ColorConvertFloat4ToU32(ImVec4(rgb.x, rgb.y, rgb.z, 1.0f));
    }
    return result;
  }();
  return ImGui::GetColorU32(colors[static_cast<uint32_t>(wavelength - kRGBResponseShortestWavelength)]);
}

bool validate_point(const float2& point, std::string& error) {
  if ((std::isfinite(point.x) == false) || (point.x <= 0.0f)) {
    error = "Wavelengths must be positive and finite.";
    return false;
  }
  if ((std::isfinite(point.y) == false) || (point.y < 0.0f)) {
    error = "Values must be finite and non-negative.";
    return false;
  }
  return true;
}

}  // namespace

uint32_t* SpectrumTarget::spectrum_slot(Material& material) const {
  switch (channel) {
    case Channel::Scattering:
      return &material.scattering.spectrum_index;
    case Channel::Reflectance:
      return &material.reflectance.spectrum_index;
    case Channel::Emission:
      return &material.emission.spectrum_index;
    case Channel::Subsurface:
      return &material.subsurface.spectrum_index;
    case Channel::InsideEta:
      return &material.int_ior.eta_index;
    case Channel::InsideK:
      return &material.int_ior.k_index;
    case Channel::OutsideEta:
      return &material.ext_ior.eta_index;
    case Channel::OutsideK:
      return &material.ext_ior.k_index;
    case Channel::ThinfilmEta:
      return &material.thinfilm.ior.eta_index;
    case Channel::ThinfilmK:
      return &material.thinfilm.ior.k_index;
    default:
      return nullptr;
  }
}

bool SpectrumDocument::validate(std::string& error) const {
  if (points.empty()) {
    error = "A spectrum needs at least one point.";
    return false;
  }
  float previous = 0.0f;
  for (const float2& point : points) {
    if (validate_point(point, error) == false) {
      return false;
    }
    if (point.x <= previous) {
      error = "Wavelengths must be ordered and distinct.";
      return false;
    }
    previous = point.x;
  }
  error.clear();
  return true;
}

float SpectrumDocument::evaluate(float wavelength) const {
  const auto upper = std::upper_bound(points.begin(), points.end(), wavelength, [](float x, const float2& point) {
    return x < point.x;
  });
  if (upper == points.begin()) {
    return points.front().y;
  }
  if (upper == points.end()) {
    return points.back().y;
  }
  const float2& lower = *(upper - 1);
  const double t = (double(wavelength) - lower.x) / (double(upper->x) - lower.x);
  return static_cast<float>((1.0 - t) * lower.y + t * upper->y);
}

bool SpectrumDocument::simplify(float maximum_error) {
  if ((points.size() <= 2u) || (std::isfinite(maximum_error) == false) || (maximum_error < 0.0f)) {
    return false;
  }
  const size_t count = points.size();
  std::vector<size_t> point_counts(count, count);
  std::vector<size_t> previous(count);
  point_counts[0] = 1u;
  for (size_t begin = 0; (begin + 1u) < count; ++begin) {
    double min_slope = -std::numeric_limits<double>::infinity();
    double max_slope = std::numeric_limits<double>::infinity();
    for (size_t end = begin + 1u; end < count; ++end) {
      const double width = double(points[end].x) - points[begin].x;
      const double height = double(points[end].y) - points[begin].y;
      const double slope = height / width;
      // Each intervening sample bounds the slopes that stay within the value error.
      if ((slope >= min_slope) && (slope <= max_slope) && ((point_counts[begin] + 1u) <= point_counts[end])) {
        point_counts[end] = point_counts[begin] + 1u;
        previous[end] = begin;
      }
      min_slope = std::max(min_slope, (height - maximum_error) / width);
      max_slope = std::min(max_slope, (height + maximum_error) / width);
      if (min_slope > max_slope) {
        break;
      }
    }
  }
  if (point_counts.back() == count) {
    return false;
  }
  std::vector<float2> simplified(point_counts.back());
  size_t index = count - 1u;
  for (size_t i = simplified.size(); i > 0u; --i) {
    simplified[i - 1u] = points[index];
    index = previous[index];
  }
  points = std::move(simplified);
  return true;
}

SpectralDistribution SpectrumDocument::distribution() const {
  float2 samples[WavelengthCount];
  for (uint32_t i = 0; i < WavelengthCount; ++i) {
    const float wavelength = kShortestWavelength + float(i);
    samples[i] = {wavelength, evaluate(wavelength)};
  }
  return SpectralDistribution::from_samples(samples, WavelengthCount);
}

bool SpectrumDocument::load(const std::string& path, std::string& error) {
  std::ifstream file(std::filesystem::u8path(path));
  if (file.is_open() == false) {
    error = "Could not open the spectrum file.";
    return false;
  }
  SpectrumDocument loaded;
  loaded.classification.clear();
  loaded.points.clear();
  std::string line;
  uint32_t line_number = 0;
  bool wavelengths_in_nm = false;
  while (std::getline(file, line)) {
    ++line_number;
    const size_t begin = line.find_first_not_of(" \t\r");
    if (begin == std::string::npos) {
      continue;
    }
    if (line[begin] == '#') {
      const size_t colon = line.find(':', begin);
      if (colon != std::string::npos) {
        const std::string key = line.substr(begin, colon - begin);
        std::string text = line.substr(colon + 1);
        const size_t first = text.find_first_not_of(" \t\r");
        text = first == std::string::npos ? "" : text.substr(first, text.find_last_not_of(" \t\r") - first + 1);
        if (key == "#wavelength-unit") {
          if (text != "nm") {
            error = "Unsupported wavelength unit; use nm or the legacy headerless format.";
            return false;
          }
          wavelengths_in_nm = true;
        } else if (key == "#title") {
          loaded.title = text;
        } else if (key == "#class") {
          loaded.classification = text;
        }
      }
      continue;
    }
    std::istringstream row(line.substr(begin));
    row.imbue(std::locale::classic());
    float2 point = {};
    std::string extra;
    if (((row >> point.x >> point.y).fail()) || ((row >> extra) && (extra.front() != '#'))) {
      error = "Expected wavelength and value on line " + std::to_string(line_number) + ".";
      return false;
    }
    if ((std::isfinite(point.x) == false) || (point.x <= 0.0f)) {
      error = "Wavelengths must be positive and finite.";
      return false;
    }
    loaded.points.push_back(point);
  }
  if (file.bad()) {
    error = "Could not read the spectrum file.";
    return false;
  }
  std::sort(loaded.points.begin(), loaded.points.end(), [](const float2& a, const float2& b) {
    return a.x < b.x;
  });
  // Match the existing SPD loader's sub-100 wavelength-unit convention.
  if ((loaded.points.empty() == false) && (wavelengths_in_nm == false)) {
    double scale = 1.0;
    while ((double(loaded.points.front().x) * scale) < 100.0) {
      scale *= 10.0;
    }
    for (float2& point : loaded.points) {
      point.x = static_cast<float>(double(point.x) * scale);
    }
  }
  if (loaded.validate(error) == false) {
    return false;
  }
  *this = std::move(loaded);
  return true;
}

bool SpectrumDocument::save(const std::string& path, const std::string& protected_directory, std::string& error) const {
  if (validate(error) == false) {
    return false;
  }
  std::error_code ec;
  const auto destination = std::filesystem::weakly_canonical(std::filesystem::u8path(path), ec);
  if (ec) {
    error = "Could not resolve the save path.";
    return false;
  }
  const auto protected_path = std::filesystem::canonical(std::filesystem::u8path(protected_directory), ec);
  if (ec) {
    error = "Could not resolve the built-in spectrum directory.";
    return false;
  }
  for (auto parent = destination.parent_path(); parent.empty() == false; parent = parent.parent_path()) {
    if (std::filesystem::equivalent(parent, protected_path, ec)) {
      error = "Built-in spectra are read-only. Save to a different directory.";
      return false;
    }
    if (ec) {
      error = "Could not inspect the save directory.";
      return false;
    }
    if (parent == parent.root_path()) {
      break;
    }
  }
  auto staged = destination;
  staged += ".writing-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  std::ofstream file(staged, std::ios::out | std::ios::noreplace);
  if (file.is_open() == false) {
    error = "Could not create the spectrum save file.";
    return false;
  }
  file.imbue(std::locale::classic());
  file << "#title: " << title << '\n';
  if (classification.empty() == false) {
    file << "#class: " << classification << '\n';
  }
  file << "#wavelength-unit: nm\n#normalization: none\n";
  file << std::setprecision(std::numeric_limits<float>::max_digits10);
  for (const float2& point : points) {
    file << point.x << ' ' << point.y << '\n';
  }
  file.close();
  if (file.fail()) {
    std::filesystem::remove(staged, ec);
    error = "Could not save the spectrum file.";
    return false;
  }
  std::filesystem::rename(staged, destination, ec);
  if (ec) {
    std::filesystem::remove(staged, ec);
    error = "Could not replace the spectrum file. The previous file was preserved.";
    return false;
  }
  error.clear();
  return true;
}

void SpectrumCurveEditor::select_point(int index) {
  selected_point = index;
  wavelength = document.points[index].x;
  value = document.points[index].y;
}

void SpectrumCurveEditor::fit() {
  const auto bounds = std::minmax_element(document.points.begin(), document.points.end(), [](const float2& a, const float2& b) {
    return a.y < b.y;
  });
  plot_min = bounds.first->y;
  plot_max = bounds.second->y;
  if (plot_min == plot_max) {
    const double padding = plot_min > 0.0 ? plot_min * 0.05 : 0.5;
    plot_min = std::max(0.0, plot_min - padding);
    plot_max += padding;
  }
}

bool SpectrumCurveEditor::load_file(const std::string& path) {
  if (document.load(path, error) == false) {
    return false;
  }
  fit();
  select_point(0);
  modified = false;
  return true;
}

bool SpectrumCurveEditor::build() {
  bool changed = false;
  bool loaded_this_frame = false;
  const bool wide = ImGui::GetContentRegionAvail().x >= (ImGui::GetFontSize() * 55.0f);
  if (ImGui::Button("Load SPD...")) {
    const std::string path = open_file("spd", nullptr);
    if (path.empty() == false) {
      if (modified) {
        _pending_load_path = path;
        ImGui::OpenPopup("Discard spectrum edits?");
      } else {
        loaded_this_frame = load_file(path);
        changed |= loaded_this_frame;
      }
    }
  }
  ImGui::SameLine();
  if (ImGui::Button("Save SPD...")) {
    _save_class = 0;
    error.clear();
    ImGui::OpenPopup("Save spectrum");
  }
  if (ImGui::BeginPopupModal("Save spectrum", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("Original class: %s", document.classification.empty() ? "unspecified" : document.classification.c_str());
    constexpr const char* classes[] = {"Keep original", "reflectance", "illuminant", "dielectric", "conductor"};
    ImGui::Combo("File classification", &_save_class, classes, static_cast<int>(std::size(classes)));
    if (ImGui::Button("Save as...")) {
      const std::string path = save_file("spd", nullptr);
      if (path.empty() == false) {
        const std::string original_class = document.classification;
        if (_save_class != 0) {
          document.classification = classes[_save_class];
        }
        if (document.save(path, env().file_in_data("spectrum"), error)) {
          modified = false;
          ImGui::CloseCurrentPopup();
        } else {
          document.classification = original_class;
        }
      }
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
      ImGui::CloseCurrentPopup();
    }
    if (error.empty() == false) {
      ImGui::TextWrapped("%s", error.c_str());
    }
    ImGui::EndPopup();
  }
  ImGui::SameLine();
  ImGui::BeginDisabled(document.points.size() <= 2u);
  if (ImGui::Button("Simplify...")) {
    ImGui::OpenPopup("Simplify spectrum");
  }
  ImGui::EndDisabled();
  if (ImGui::BeginPopupModal("Simplify spectrum", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("%zu control points", document.points.size());
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * 12.0f);
    ImGui::InputFloat("Maximum error", &_simplify_maximum_error, 0.0f, 0.0f, "%.6g");
    ImGui::TextUnformatted("Absolute value error; endpoints are retained.");
    const bool valid_error = std::isfinite(_simplify_maximum_error) && (_simplify_maximum_error >= 0.0f);
    ImGui::BeginDisabled(valid_error == false);
    if (ImGui::Button("Simplify")) {
      if (document.simplify(_simplify_maximum_error)) {
        select_point(0);
        fit();
        changed = true;
        source_update_requested = true;
      }
      error.clear();
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
  if (ImGui::BeginPopupModal("Discard spectrum edits?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextUnformatted("Loading another spectrum will discard the unsaved draft.");
    if (ImGui::Button("Discard and load")) {
      loaded_this_frame = load_file(_pending_load_path);
      changed |= loaded_this_frame;
      _pending_load_path.clear();
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
      _pending_load_path.clear();
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
  if (wide == false) {
    ImGui::TextWrapped("%s%s", document.title.c_str(), modified ? " *" : "");
  }
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  const float footer_height = ImGui::GetTextLineHeightWithSpacing() + ImGui::GetFrameHeightWithSpacing() * 2.0f;
  const float plot_height = wide ? std::max(ImGui::GetFontSize() * 6.0f, ImGui::GetContentRegionAvail().y - footer_height) : ImGui::GetFontSize() * 12.0f;
  const ImVec2 size(std::max(120.0f, ImGui::GetContentRegionAvail().x), plot_height);
  const float margin = ImGui::GetFontSize() * 0.6f;
  const ImVec2 lo(origin.x + margin, origin.y + margin);
  const ImVec2 hi(origin.x + size.x - margin, origin.y + size.y - margin);
  ImGui::InvisibleButton("##spectrum_curve", size);
  const bool active = ImGui::IsItemActive();
  const bool hovered = ImGui::IsItemHovered();
  const bool activated = ImGui::IsItemActivated();
  if ((active == false) || activated) {
    fit();
    _drag_min_wavelength = document.points.front().x;
    _drag_max_wavelength = document.points.back().x;
    if (_drag_min_wavelength == _drag_max_wavelength) {
      _drag_min_wavelength = std::min(kShortestWavelength, _drag_min_wavelength);
      _drag_max_wavelength = std::max(kLongestWavelength, _drag_max_wavelength);
    }
  }
  const float min_wavelength = _drag_min_wavelength;
  const float max_wavelength = _drag_max_wavelength;
  const auto to_screen = [&](const float2& p) {
    return ImVec2(lo.x + (p.x - min_wavelength) / (max_wavelength - min_wavelength) * (hi.x - lo.x),
      hi.y - static_cast<float>(std::clamp((double(p.y) - plot_min) / (plot_max - plot_min), 0.0, 1.0)) * (hi.y - lo.y));
  };
  ImDrawList* draw = ImGui::GetWindowDrawList();
  draw->AddRectFilled(origin, ImVec2(origin.x + size.x, origin.y + size.y), ImGui::GetColorU32(ImGuiCol_FrameBg), ImGui::GetStyle().FrameRounding);
  for (int i = 0; i <= 4; ++i) {
    const float t = float(i) / 4.0f;
    draw->AddLine(ImVec2(lo.x, lo.y + t * (hi.y - lo.y)), ImVec2(hi.x, lo.y + t * (hi.y - lo.y)), ImGui::GetColorU32(ImGuiCol_Border));
    draw->AddLine(ImVec2(lo.x + t * (hi.x - lo.x), lo.y), ImVec2(lo.x + t * (hi.x - lo.x), hi.y), ImGui::GetColorU32(ImGuiCol_Border));
  }
  float line_distance = margin * margin;
  float insertion_wavelength = 0.0f;
  const int first_vertex = draw->VtxBuffer.Size;
  const ImU32 line_color = ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
  const auto draw_segment = [&](const ImVec2& a, const ImVec2& b) {
    if ((a.x == b.x) && (a.y == b.y)) {
      return;
    }
    const int steps = std::max(1, static_cast<int>(std::ceil((b.x - a.x) / 4.0f)));
    ImVec2 start = a;
    for (int i = 1; i <= steps; ++i) {
      const float t = float(i) / float(steps);
      const ImVec2 end(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y));
      draw->AddLine(start, end, line_color, 3.0f);
      start = end;
    }
    if (hovered == false) {
      return;
    }
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    const float dx = b.x - a.x;
    const float dy = b.y - a.y;
    const float length_squared = dx * dx + dy * dy;
    if (length_squared <= 0.0f) {
      return;
    }
    const float t = std::clamp(((mouse.x - a.x) * dx + (mouse.y - a.y) * dy) / length_squared, 0.0f, 1.0f);
    const ImVec2 closest(a.x + t * dx, a.y + t * dy);
    const float distance_squared = (mouse.x - closest.x) * (mouse.x - closest.x) + (mouse.y - closest.y) * (mouse.y - closest.y);
    if (distance_squared < line_distance) {
      line_distance = distance_squared;
      insertion_wavelength = std::clamp(min_wavelength + (closest.x - lo.x) / (hi.x - lo.x) * (max_wavelength - min_wavelength), min_wavelength, max_wavelength);
    }
  };
  ImVec2 previous = to_screen({min_wavelength, document.points.front().y});
  int nearest = -1;
  float distance = margin * margin * 4.0f;
  for (int i = 0; i < static_cast<int>(document.points.size()); ++i) {
    const ImVec2 p = to_screen(document.points[i]);
    draw_segment(previous, p);

    previous = p;
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    const float d = (p.x - mouse.x) * (p.x - mouse.x) + (p.y - mouse.y) * (p.y - mouse.y);
    if (d < distance) {
      nearest = i;
      distance = d;
    }
  }
  draw_segment(previous, to_screen({max_wavelength, document.points.back().y}));
  for (int i = first_vertex; i < draw->VtxBuffer.Size; ++i) {
    ImDrawVert& vertex = draw->VtxBuffer[i];
    const float wavelength = min_wavelength + (vertex.pos.x - lo.x) / (hi.x - lo.x) * (max_wavelength - min_wavelength);
    vertex.col = (vertex.col & IM_COL32_A_MASK) | (wavelength_color(wavelength) & ~IM_COL32_A_MASK);
  }
  for (int i = 0; i < static_cast<int>(document.points.size()); ++i) {
    const ImVec2 point = to_screen(document.points[i]);
    if (i == selected_point) {
      draw->AddCircle(point, 6.5f, ImGui::GetColorU32(ImGuiCol_Text), 0, 1.5f);
    }
    draw->AddCircleFilled(point, i == selected_point ? 4.5f : 3.5f, wavelength_color(document.points[i].x));
  }
  if (hovered && (nearest >= 0)) {
    ImGui::SetTooltip("%.6g nm: %.6g", document.points[nearest].x, document.points[nearest].y);
  } else if (hovered && (insertion_wavelength > 0.0f)) {
    ImGui::SetTooltip("Click to add a point at %.6g nm", insertion_wavelength);
  }
  if (activated) {
    if (nearest >= 0) {
      select_point(nearest);
    } else if (insertion_wavelength > 0.0f) {
      const auto position = std::lower_bound(document.points.begin(), document.points.end(), insertion_wavelength, [](const float2& point, float x) {
        return point.x < x;
      });
      const int index = static_cast<int>(position - document.points.begin());
      if ((position == document.points.end()) || (position->x != insertion_wavelength)) {
        const float power = document.evaluate(insertion_wavelength);
        document.points.insert(position, {insertion_wavelength, power});
        error.clear();
        changed = true;
      }
      select_point(index);
    } else {
      selected_point = -1;
    }
  }
  if (active && (selected_point >= 0) && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    const float lower = selected_point == 0 ? min_wavelength : std::nextafter(document.points[selected_point - 1].x, max_wavelength);
    const float upper = (selected_point + 1) == static_cast<int>(document.points.size()) ? max_wavelength : std::nextafter(document.points[selected_point + 1].x, 0.0f);
    float2 point = {std::clamp(min_wavelength + (mouse.x - lo.x) / (hi.x - lo.x) * (max_wavelength - min_wavelength), lower, upper),
      static_cast<float>(std::max(0.0, plot_min + (hi.y - mouse.y) / (hi.y - lo.y) * (plot_max - plot_min)))};
    if (validate_point(point, error)) {
      const float2 old = document.points[selected_point];
      document.points[selected_point] = point;
      select_point(selected_point);
      error.clear();
      changed |= (old.x != point.x) || (old.y != point.y);
    }
  }
  ImGui::TextDisabled("%.6g - %.6g nm | %.6g - %.6g | %zu points", min_wavelength, max_wavelength, plot_min, plot_max, document.points.size());
  if (wide) {
    ImGui::SameLine();
    ImGui::TextDisabled("%s%s", document.title.c_str(), modified ? " *" : "");
  } else {
    ImGui::TextWrapped("Click the line to add a point. Enter updates the selected point; Add creates a point.");
  }
  ImGui::SetNextItemWidth(wide ? ImGui::GetFontSize() * 9.0f : ImGui::GetContentRegionAvail().x * 0.55f);
  bool commit = ImGui::InputFloat("Wavelength (nm)", &wavelength, 0.0f, 0.0f, "%.6g", ImGuiInputTextFlags_EnterReturnsTrue);
  if (wide) {
    ImGui::SameLine();
  }
  ImGui::SetNextItemWidth(wide ? ImGui::GetFontSize() * 7.0f : ImGui::GetContentRegionAvail().x * 0.55f);
  commit |= ImGui::InputFloat("Value", &value, 0.0f, 0.0f, "%.6g", ImGuiInputTextFlags_EnterReturnsTrue);
  const auto commit_point = [&](bool insert) {
    const float2 point = {wavelength, value};
    if (validate_point(point, error) == false) {
      return;
    }
    const auto position = std::lower_bound(document.points.begin(), document.points.end(), wavelength, [](const float2& p, float x) {
      return p.x < x;
    });
    int index = static_cast<int>(position - document.points.begin());
    if ((position != document.points.end()) && (position->x == wavelength) && (insert || (index != selected_point))) {
      error = "A point already exists at this wavelength. Select it to edit.";
      return;
    }
    if (insert == false) {
      const float2 previous = document.points[selected_point];
      if ((previous.x == point.x) && (previous.y == point.y)) {
        error.clear();
        return;
      }
      document.points.erase(document.points.begin() + selected_point);
      if (index > selected_point) {
        --index;
      }
    }
    document.points.insert(document.points.begin() + index, point);
    select_point(index);
    error.clear();
    changed = true;
  };
  if (commit && (selected_point >= 0)) {
    commit_point(false);
  }
  if (wide) {
    ImGui::SameLine();
  }
  if (ImGui::Button("Add point")) {
    commit_point(true);
  }
  ImGui::SameLine();
  const bool can_delete = (selected_point > 0) && ((static_cast<size_t>(selected_point) + 1u) < document.points.size());
  ImGui::BeginDisabled(can_delete == false);
  if (ImGui::Button("Delete point") && can_delete) {
    document.points.erase(document.points.begin() + selected_point);
    select_point(std::min(selected_point, static_cast<int>(document.points.size()) - 1));
    error.clear();
    changed = true;
  }
  ImGui::EndDisabled();
  if (changed && (loaded_this_frame == false)) {
    modified = true;
  }
  if (error.empty() == false) {
    ImGui::TextWrapped("%s", error.c_str());
  }
  return changed;
}

}  // namespace etx
