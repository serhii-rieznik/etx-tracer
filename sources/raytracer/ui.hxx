#pragma once

#include <etx/util/options.hxx>
#include <etx/render/shared/base.hxx>
#include <etx/rt/integrators/integrator.hxx>
#include <etx/render/host/scene_representation.hxx>

#include "options.hxx"

#include <functional>
#include <string>
#include <unordered_map>

struct sapp_event;

namespace etx {

struct IORDatabase;

struct UI {
  UI() = default;
  ~UI() = default;

  void initialize(Film* film, const IORDatabase*);
  void cleanup();

  void build(double dt, const std::vector<std::string>& recent_files, Scene& scene, Camera& camera, const SceneRepresentation::MaterialMapping& materials,
    const SceneRepresentation::MediumMapping& mediums, const SceneRepresentation::MeshMapping& meshes);

  void set_integrator_list(Integrator* i[], uint64_t count) {
    _integrators = {i, count};
  }

  void set_current_integrator(Integrator*);

  bool handle_event(const sapp_event*);

  // Context structure for shared state during UI building
  struct BuildContext {
    std::vector<int32_t> emitter_primary_instance;
    std::function<const char*(uint32_t)> material_name_from_index;
    std::function<void(uint32_t, const char*, std::function<void()>&&)> with_window;
    float2 wpadding = {};
    float2 fpadding = {};
    float text_size = {};
    float button_size = {};
    float input_size = {};
    bool has_integrator = false;
    bool has_scene = false;
    bool scene_editable = false;
  };

  ViewOptions view_options() const;
  ViewOptions& mutable_view_options();

  struct {
    std::function<void(std::string)> reference_image_selected;
    std::function<void(std::string, SaveImageMode)> save_image_selected;
    std::function<void(std::string)> scene_file_selected;
    std::function<void(std::string)> save_scene_file_selected;
    std::function<void()> save_scene_file_as_selected;
    std::function<void(Integrator*)> integrator_selected;
    std::function<void(bool)> stop_selected;
    std::function<void()> run_selected;
    std::function<void()> restart_selected;
    std::function<void()> reload_scene_selected;
    std::function<void()> reload_geometry_selected;
    std::function<void()> options_changed;
    std::function<void()> use_image_as_reference;
    std::function<void(uint32_t)> material_changed;
    std::function<void()> medium_added;
    std::function<void(uint32_t)> medium_changed;
    std::function<void(uint32_t, uint32_t)> mesh_material_changed;  // mesh_index, new_material_index
    std::function<void(uint32_t)> emitter_changed;
    std::function<void(bool)> camera_changed;
    std::function<void()> scene_settings_changed;
    std::function<void()> denoise_selected;
    std::function<void(uint32_t)> view_scene;
    std::function<void()> clear_recent_files;
  } callbacks;

 private:
  // UI Helper Functions
  void full_width_item();
  bool labeled_control(const char* label, std::function<bool()>&& control_func);
  bool validated_float_control(const char* label, float& value, float min_val, float max_val, const char* format = "%.3f");
  bool validated_int_control(const char* label, int32_t& value, int32_t min_val, int32_t max_val);
  const char* format_string(const char* format, ...);

  enum class SelectionKind : uint32_t {
    None,
    Material,
    Medium,
    Mesh,
    Emitter,
    Camera,
    Scene,
    Integrator,
  };

  bool build_options(Options&);
  void quit();
  void select_scene_file() const;
  void save_scene_file() const;
  void save_scene_file_as() const;
  void save_image(SaveImageMode mode) const;
  void load_image() const;
  bool build_material(Scene& scene, Material&);
  bool build_medium(Scene& scene, Medium&, const char* name);
  bool spectrum_picker(const char* widget_id, SpectralDistribution& spd, bool linear, bool scale, bool show_color = true, bool show_scale = true);
  bool spectrum_picker(Scene& scene, const char* widget_id, uint32_t spd_index, bool linear, bool scale, bool show_color = true, bool show_scale = true);
  bool angle_editor(const char* label, float2& angles, float min_azimuth, float max_azimuth, float min_elevation, float max_elevation, float pole_threshold);
  bool ior_picker(Scene& scene, const char* name, RefractiveIndex& ior);
  bool emission_picker(Scene& scene, const char* label, const char* id_suffix, uint32_t& spectrum_index);
  bool medium_dropdown(const char* label, uint32_t& medium);

  void reset_selection();
  void reload_geometry();
  void reload_scene();
  void set_selection(SelectionKind kind, int32_t index, bool track_history = true);
  void validate_selections(const Scene& scene);
  void navigate_history(int32_t step);
  bool can_navigate_back() const;
  bool can_navigate_forward() const;

  void build_main_menu_bar(const std::vector<std::string>& recent_files);
  void build_toolbar(const BuildContext& ctx);
  void build_scene_objects_window(Scene& scene, const BuildContext& ctx, const SceneRepresentation::MaterialMapping& materials, const SceneRepresentation::MediumMapping& mediums,
    const SceneRepresentation::MeshMapping& meshes);
  void build_properties_window(Scene& scene, Camera& camera, const BuildContext& ctx);

  bool build_material_class_selector(Material& material);

  // Selection-specific property builders
  void build_material_selection_properties(Scene& scene, const BuildContext& ctx);
  void build_medium_selection_properties(Scene& scene, const BuildContext& ctx);
  void build_emitter_selection_properties(Scene& scene, const BuildContext& ctx);
  void build_mesh_selection_properties(Scene& scene, const BuildContext& ctx);
  void build_camera_selection_properties(Scene& scene, Camera& camera, const BuildContext& ctx);
  void build_scene_selection_properties(Scene& scene, const BuildContext& ctx);
  void build_integrator_selection_properties(Scene& scene, const BuildContext& ctx);

 private:
  Integrator* _current_integrator = nullptr;
  Film* _film = nullptr;

  ArrayView<Integrator*> _integrators = {};
  ViewOptions _view_options = {};

  struct MappingRepresentation {
    struct Entry {
      uint32_t index = kInvalidIndex;
      const char* name = nullptr;
    };
    std::vector<Entry> entries;
    std::vector<char> data;
    std::unordered_map<uint32_t, uint32_t> reverse;

    uint64_t size() const {
      return entries.size();
    }

    bool empty() const {
      return entries.empty();
    }

    uint32_t at(const int32_t i) const {
      return entries.at(i).index;
    }

    const char* name(int32_t i) const {
      return entries.at(i).name;
    }

    const Entry& entry(int32_t i) const {
      return entries.at(i);
    }

    const char* name_for(uint32_t index) const {
      auto it = reverse.find(index);
      return (it != reverse.end()) ? entries[it->second].name : nullptr;
    }

    void build(const std::unordered_map<std::string, uint32_t>&);
  };

  enum UISetup : uint32_t {
    UIObjects = 1u << 0u,
    UIProperties = 1u << 1u,

    UIDefaults = UIObjects | UIProperties,
  };

  struct SelectionState {
    SelectionKind kind = SelectionKind::None;
    int32_t index = -1;
  };

  struct SpectrumEditorState {
    float3 color = {};
    float scale = 1.0f;
    float temperature = 6500.0f;
    enum class Mode : uint32_t {
      Color,
      Temperature,
      Preset,
    } mode = Mode::Color;
  };

  MappingRepresentation _material_mapping;
  MappingRepresentation _medium_mapping;
  MappingRepresentation _mesh_mapping;
  SelectionState _selection;
  std::vector<SelectionState> _selection_history;
  int32_t _selection_history_cursor = -1;
  uint32_t _ui_setup = UIDefaults;
  uint32_t _font_image = 0u;
  std::unordered_map<std::string, SpectrumEditorState> _spectrum_editors;
  std::unordered_map<std::string, bool> _material_anisotropy;
  uint64_t _material_mapping_hash = 0ull;
  uint64_t _medium_mapping_hash = 0ull;
  uint64_t _mesh_mapping_hash = 0ull;
  const IORDatabase* _ior_database = nullptr;
  bool _auto_open_emission_section = false;
};

}  // namespace etx
