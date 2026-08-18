#pragma once

#include <etx/engine/options.hxx>
#include <etx/render/shared/base.hxx>
#include <etx/rt/integrators/integrator.hxx>
#include <etx/render/host/scene_representation.hxx>
#include <etx/rhi/rhi_imgui.hxx>

#include "renderer.hxx"
#include "options.hxx"

#include <algorithm>
#include <functional>
#include <string>
#include <unordered_map>

struct sapp_event;

namespace etx {

struct IORDatabase;

enum class MenuCommand : uint32_t {
  Quit,
  SelectCPURenderer,
  SelectRasterRenderer,
  SelectGPURenderer,
  OpenScene,
  ReloadScene,
  ReloadGeometry,
  OpenRecentScene,
  ClearRecentScenes,
  SaveScene,
  SaveSceneAs,
  SelectIntegrator,
  OpenReferenceImage,
  SaveImageRGB,
  SaveImageLDR,
  UseImageAsReference,
  RunRenderer,
  FinishRenderer,
  StopRenderer,
  RestartRenderer,
  ViewWholeScene,
  ViewPositiveX,
  ViewNegativeX,
  ViewPositiveY,
  ViewNegativeY,
  ViewPositiveZ,
  ViewNegativeZ,
  IncreaseExposure,
  DecreaseExposure,
  ToggleSceneObjects,
  ToggleProperties,
};

struct UI {
  struct FrameData {
    const IORDatabase& ior_database;
    const std::vector<std::string>& recent_files;
    const Film& film;
    float dt = 0.0f;
  };

  UI() = default;
  ~UI() = default;

  void build(SceneRepresentation& scene_rep, const FrameData& data);

  void set_integrator_list(Integrator* i[], uint64_t count) {
    _integrators = {i, count};
  }

  void set_current_integrator(Integrator*);

  void set_current_renderer_mode(RendererMode mode) {
    _current_renderer_mode = mode;
  }

  void set_current_renderer_status(const RendererPreparationStatus& status) {
    _current_renderer_status = status;
  }

  void set_current_renderer_stats(const RendererRuntimeStats& stats) {
    _current_renderer_stats = stats;
  }

  void set_current_renderer_controls(const RendererControlState& controls) {
    _current_renderer_controls = controls;
  }

  void set_gpu_kernel_timing_stats(const RendererKernelTimingStats& stats) {
    _gpu_kernel_timing_stats = stats;
  }

  void set_gpu_wavefront_steps_per_frame(uint32_t value) {
    _gpu_wavefront_steps_per_frame = std::clamp(value, 1u, 1024u);
  }

  uint32_t gpu_wavefront_steps_per_frame() const {
    return _gpu_wavefront_steps_per_frame;
  }

  void set_gpu_renderer_available(bool value) {
    _gpu_renderer_available = value;
    if ((_gpu_renderer_available == false) && (_current_renderer_mode == RendererMode::GPURaytracing)) {
      _current_renderer_mode = RendererMode::CPURaytracing;
    }
  }

  bool handle_event(const sapp_event*);
  void execute_menu_command(MenuCommand command, uint32_t argument = 0u, const std::string& value = {});

  void set_embedded_menu_enabled(bool value) {
    _embedded_menu_enabled = value;
  }

  void set_embedded_toolbar_enabled(bool value) {
    _embedded_toolbar_enabled = value;
  }

  bool gpu_renderer_available() const {
    return _gpu_renderer_available;
  }

  RendererMode current_renderer_mode() const {
    return _current_renderer_mode;
  }

  uint64_t integrator_count() const {
    return _integrators.count;
  }

  Integrator* integrator(uint64_t index) const {
    return index < _integrators.count ? _integrators[index] : nullptr;
  }

  Integrator* current_integrator() const {
    return _current_integrator;
  }

  bool scene_objects_visible() const {
    return (_ui_setup & UIObjects) != 0u;
  }

  bool properties_visible() const {
    return (_ui_setup & UIProperties) != 0u;
  }

  bool scene_view_commands_available() const {
    return static_cast<bool>(callbacks.view_scene);
  }

  const RendererControlState& renderer_controls() const {
    return _current_renderer_controls;
  }

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
  };

  void set_view_options(const ViewParameters& value) {
    _view_options = value;
  }

  struct {
    std::function<void()> quit_selected;
    std::function<void(std::string)> reference_image_selected;
    std::function<void(std::string, SaveImageMode)> save_image_selected;
    std::function<void(std::string)> scene_file_selected;
    std::function<void(std::string)> save_scene_file_selected;
    std::function<void(RendererMode)> renderer_selected;
    std::function<void(bool)> stop_selected;
    std::function<void()> run_selected;
    std::function<void()> restart_selected;
    std::function<void()> reload_scene_selected;
    std::function<void()> reload_geometry_selected;
    std::function<void()> reload_shaders_selected;
    std::function<void()> cancel_renderer_preparation_selected;
    std::function<void()> options_changed;
    std::function<void()> use_image_as_reference;
    std::function<void()> material_added;
    std::function<void(uint32_t, const std::string&)> material_renamed;
    std::function<void(uint32_t)> material_changed;
    std::function<void()> medium_added;
    std::function<void(uint32_t, const std::string&)> medium_renamed;
    std::function<void(uint32_t)> medium_changed;
    std::function<void(uint32_t, uint32_t)> mesh_material_changed;  // mesh_index, new_material_index
    std::function<void(uint32_t, const std::string&)> mesh_renamed;
    std::function<void(uint32_t)> emitter_changed;
    std::function<void(uint32_t)> emitter_added;  // 0=environment, 1=directional, 2=atmosphere
    std::function<bool(uint32_t)> emitter_deleted;
    std::function<void(uint2 /* viewport */, uint32_t /* pixel size*/)> camera_changed;
    std::function<void()> scene_settings_changed;
    std::function<void()> denoise_selected;
    std::function<void(uint32_t direction)> view_scene;
    std::function<void(Integrator::Type)> integrator_selected;
    std::function<void()> clear_recent_files;
    std::function<void(uint32_t)> camera_activated;
    std::function<void(uint32_t)> gpu_wavefront_steps_per_frame_changed;
    std::function<void(bool)> gpu_kernel_timing_enabled_changed;
    std::function<void(float)> exposure_changed;
    std::function<void(uint32_t)> view_layer_changed;
    std::function<void(uint32_t)> output_view_changed;
    std::function<void(uint32_t)> display_transform_changed;
  } callbacks;

 private:
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
    Rendering,  // Combined Scene + Integrator properties
  };

  bool build_options(Options&);
  void quit();
  void select_scene_file() const;
  void save_scene_file() const;
  void save_scene_file_as() const;
  void save_image(SaveImageMode mode) const;
  void load_image() const;
  bool build_material(SceneRepresentation& scene_rep, Material& material, const FrameData&);
  bool build_material(SceneRepresentation& scene_rep, Material& material, const FrameData&, const std::vector<uint32_t>& material_indices);
  bool build_medium(SceneRepresentation& scene_rep, Medium& medium);
  bool spectrum_picker(const char* widget_id, SpectralDistribution& spd, bool linear, bool scale, bool show_color = true, bool show_scale = true);
  bool spectrum_picker(SceneRepresentation& scene_rep, const char* widget_id, uint32_t spd_index, bool linear, bool scale, bool show_color = true, bool show_scale = true);
  bool image_picker(SceneRepresentation& scene_rep, const char* label, uint32_t& image_index, uint32_t image_options);
  bool sampled_image_picker(SceneRepresentation& scene_rep, const char* label, SampledImage& image, uint32_t image_options);
  bool angle_editor(const char* label, float2& angles, float min_azimuth, float max_azimuth, float min_elevation, float max_elevation, float pole_threshold);
  bool ior_picker(SceneRepresentation& scene_rep, const char* name, RefractiveIndex& ior, const FrameData&);
  bool ior_picker(SceneRepresentation& scene_rep, const char* name, RefractiveIndex& ior, const FrameData&, bool mixed, bool dielectric_only = false);
  bool emission_picker(SceneRepresentation& scene_rep, const char* label, const char* id_suffix, uint32_t& spectrum_index, const FrameData&);
  bool medium_dropdown(const char* label, uint32_t& medium);
  void update_name_buffer(SelectionKind kind, int32_t index, const char* current_name);

  void reset_selection();
  uint32_t selected_material_count() const;
  bool material_list_position_selected(int32_t index) const;
  void set_single_material_selection(int32_t index, bool track_history);
  void toggle_material_selection(int32_t index);
  void set_material_selection_range(int32_t index);
  std::vector<uint32_t> selected_material_indices(SceneRepresentation& scene_rep) const;
  void apply_material_changes(SceneRepresentation& scene_rep, const std::vector<uint32_t>& material_indices, const Material& before, const Material& after) const;
  void reload_geometry();
  void reload_scene();
  void set_selection(SelectionKind kind, int32_t index, bool track_history = true);
  void validate_selections(SceneRepresentation& scene_rep);
  void navigate_history(int32_t step);
  bool can_navigate_back() const;
  bool can_navigate_forward() const;

  void build_main_menu_bar(const std::vector<std::string>& recent_files);
#if defined(_WIN32)
  float build_title_bar_controls();
#endif
  void build_toolbar(const BuildContext& ctx);
  void build_status_bar(const BuildContext& ctx);
  void build_scene_objects_window(SceneRepresentation& scene_rep, const BuildContext& ctx);
  void build_properties_window(SceneRepresentation& scene_rep, Camera& camera, const BuildContext& ctx, const FrameData& data);

  bool build_material_class_selector(Material& material);
  bool build_material_class_selector(Material& material, bool mixed);

  void build_material_selection_properties(SceneRepresentation& scene_rep, const BuildContext& ctx, const FrameData& data);
  void build_medium_selection_properties(SceneRepresentation& scene_rep, const BuildContext& ctx, const FrameData& data);
  void build_emitter_selection_properties(SceneRepresentation& scene_rep, const BuildContext& ctx, const FrameData& data);
  void build_atmosphere_selection_properties(SceneRepresentation& scene_rep, const BuildContext& ctx);
  void build_mesh_selection_properties(SceneRepresentation& scene_rep, const BuildContext& ctx);
  void build_camera_selection_properties(SceneRepresentation& scene_rep, Camera& camera, uint32_t camera_index, const BuildContext& ctx, const FrameData& data);
  void build_scene_selection_properties(SceneRepresentation& scene_rep, const BuildContext& ctx, const FrameData& data);
  void build_integrator_selection_properties(SceneRepresentation& scene_rep, const BuildContext& ctx);
  void build_rendering_properties(SceneRepresentation& scene_rep, const BuildContext& ctx, const FrameData& data);

 private:
  Integrator* _current_integrator = nullptr;
  RendererMode _current_renderer_mode = RendererMode::CPURaytracing;
  RendererPreparationStatus _current_renderer_status = {};
  RendererRuntimeStats _current_renderer_stats = {};
  RendererControlState _current_renderer_controls = {};
  RendererKernelTimingStats _gpu_kernel_timing_stats = {};
  uint32_t _gpu_wavefront_steps_per_frame = 256u;
  bool _gpu_renderer_available = true;
  bool _embedded_toolbar_enabled = true;

  ArrayView<Integrator*> _integrators = {};
  ViewParameters _view_options = {
    .exposure = 1.0f,
    .view_option = (uint32_t)ViewOptions::Tonemapped,
    .view_image = (uint32_t)OutputView::OutputImage,
    .view_layer = (uint32_t)ViewLayer::Result,
  };

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

  struct PendingSelection {
    SelectionKind kind = SelectionKind::None;
    uint32_t index = kInvalidIndex;
    bool has = false;
  } _pending_selection;

  MappingRepresentation _material_mapping;
  MappingRepresentation _medium_mapping;
  MappingRepresentation _mesh_mapping;
  MappingRepresentation _camera_mapping;
  SelectionState _selection;
  SelectionState _name_edit_selection = {};
  char _name_edit_buffer[256] = {};
  std::vector<SelectionState> _selection_history;
  int32_t _selection_history_cursor = -1;
  uint32_t _ui_setup = UIDefaults;
  bool _embedded_menu_enabled = true;
  uint32_t _font_image = 0u;
  std::unordered_map<std::string, SpectrumEditorState> _spectrum_editors;
  std::unordered_map<std::string, bool> _material_anisotropy;
  std::vector<int32_t> _selected_material_positions;
  int32_t _material_selection_anchor = -1;
  bool _updating_material_multi_selection = false;
  const std::vector<uint32_t>* _editing_material_indices = nullptr;
  uint32_t _material_batch_changed_fields = 0u;
  uint64_t _material_mapping_hash = 0ull;
  uint64_t _medium_mapping_hash = 0ull;
  uint64_t _mesh_mapping_hash = 0ull;
  uint64_t _camera_mapping_hash = 0ull;
  bool _auto_open_emission_section = false;
  double _last_fps_update_time = 0.0;
  uint32_t _frame_count = 0;
  float _current_fps = 0.0f;
};

}  // namespace etx
