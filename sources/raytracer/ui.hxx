#pragma once

#include <etx/util/options.hxx>
#include <etx/render/shared/base.hxx>
#include <etx/rt/integrators/integrator.hxx>
#include <etx/render/host/scene_loader.hxx>

#include "options.hxx"

#include <functional>
#include <string>

struct sapp_event;

namespace etx {

struct UI {
  UI() = default;
  ~UI() = default;

  void initialize(Film* film);
  void cleanup();

  void build(double dt, const std::vector<std::string>& recent_files, Scene& scene, Camera& camera, const SceneRepresentation::MaterialMapping& materials,
    const SceneRepresentation::MediumMapping& mediums);

  void set_integrator_list(Integrator* i[], uint64_t count) {
    _integrators = {i, count};
  }

  void set_current_integrator(Integrator*);

  bool handle_event(const sapp_event*);

  ViewOptions view_options() const;
  ViewOptions& mutable_view_options();

  struct {
    std::function<void(std::string)> reference_image_selected;
    std::function<void(std::string, SaveImageMode)> save_image_selected;
    std::function<void(std::string)> scene_file_selected;
    std::function<void(std::string)> save_scene_file_selected;
    std::function<void(Integrator*)> integrator_selected;
    std::function<void(bool)> stop_selected;
    std::function<void()> run_selected;
    std::function<void()> restart_selected;
    std::function<void()> reload_scene_selected;
    std::function<void()> reload_geometry_selected;
    std::function<void()> options_changed;
    std::function<void()> use_image_as_reference;
    std::function<void(uint32_t)> material_changed;
    std::function<void(uint32_t)> medium_changed;
    std::function<void(uint32_t)> emitter_changed;
    std::function<void(bool)> camera_changed;
    std::function<void()> scene_settings_changed;
    std::function<void()> denoise_selected;
    std::function<void(uint32_t)> view_scene;
  } callbacks;

 private:
  enum class SelectionKind : uint32_t {
    None,
    Material,
    Medium,
    Emitter,
    Camera,
    Scene,
  };

  bool build_options(Options&);
  void quit();
  void select_scene_file() const;
  void save_scene_file() const;
  void save_image(SaveImageMode mode) const;
  void load_image() const;
  bool build_material(Scene& scene, Material&);
  bool build_medium(Medium&);
  bool spectrum_picker(const char* name, SpectralDistribution& spd, bool linear);
  bool spectrum_picker(Scene& scene, const char* name, uint32_t spd_index, bool linear);
  bool ior_picker(Scene& scene, const char* name, RefractiveIndex& ior);

  void reset_selection();
  void reload_geometry();
  void reload_scene();
  void set_selection(SelectionKind kind, int32_t index, bool track_history = true);
  void navigate_history(int32_t step);
  bool can_navigate_back() const;
  bool can_navigate_forward() const;

 private:
  Integrator* _current_integrator = nullptr;
  Film* _film = nullptr;

  ArrayView<Integrator*> _integrators = {};
  ViewOptions _view_options = {};

  struct MappingRepresentation {
    std::vector<uint32_t> indices;
    std::vector<char> data;
    std::vector<const char*> names;

    uint64_t size() const {
      return indices.size();
    }

    bool empty() const {
      return indices.empty();
    }

    uint32_t at(const int32_t i) const {
      return indices.at(i);
    }

    void build(const std::unordered_map<std::string, uint32_t>&);
  };

  struct IORFile {
    std::string filename;
    std::string title;
    SpectralDistribution::Class cls = SpectralDistribution::Class::Invalid;
    SpectralDistribution eta = {};
    SpectralDistribution k = {};
  };

  enum UISetup : uint32_t {
    UIIntegrator = 1u << 0u,
    UIObjects = 1u << 1u,
    UIProperties = 1u << 2u,

    UIDefaults = UIIntegrator | UIObjects | UIProperties,
  };

  struct SelectionState {
    SelectionKind kind = SelectionKind::None;
    int32_t index = -1;
  };

  MappingRepresentation _material_mapping;
  MappingRepresentation _medium_mapping;
  std::vector<IORFile> _ior_files;
  SelectionState _selection;
  std::vector<SelectionState> _selection_history;
  int32_t _selection_history_cursor = -1;
  float _panel_width = 0.0f;
  uint32_t _ui_setup = UIDefaults;
  uint32_t _font_image = 0u;
  std::unordered_map<std::string, float3> _editor_values;
  uint64_t _material_mapping_hash = 0ull;
  uint64_t _medium_mapping_hash = 0ull;
};

}  // namespace etx
