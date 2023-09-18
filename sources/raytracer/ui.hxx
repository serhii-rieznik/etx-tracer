#pragma once

#include <etx/core/options.hxx>

#include <etx/render/shared/base.hxx>
#include <etx/rt/integrators/integrator.hxx>

#include <etx/render/host/scene_loader.hxx>

#include <functional>

#include "options.hxx"

struct sapp_event;

namespace etx {

struct UI {
  void initialize();
  void cleanup();

  void build(double dt, const char* status);

  void set_integrator_list(Integrator* i[], uint64_t count) {
    _integrators = {i, count};
  }

  void set_current_integrator(Integrator*);
  void set_scene(Scene* scene, const SceneRepresentation::MaterialMapping&, const SceneRepresentation::MediumMapping&);

  const Options& integrator_options() const {
    return _integrator_options;
  }

  bool handle_event(const sapp_event*);

  ViewOptions view_options() const;

  struct {
    std::function<void(std::string)> reference_image_selected;
    std::function<void(std::string, SaveImageMode)> save_image_selected;
    std::function<void(std::string)> scene_file_selected;
    std::function<void(std::string)> save_scene_file_selected;
    std::function<void(Integrator*)> integrator_selected;
    std::function<void(bool)> stop_selected;
    std::function<void()> preview_selected;
    std::function<void()> run_selected;
    std::function<void()> reload_scene_selected;
    std::function<void()> reload_geometry_selected;
    std::function<void()> options_changed;
    std::function<void()> reload_integrator;
    std::function<void()> use_image_as_reference;
    std::function<void(uint32_t)> material_changed;
    std::function<void(uint32_t)> medium_changed;
    std::function<void(uint32_t)> emitter_changed;
    std::function<void()> camera_changed;
    std::function<void()> scene_settings_changed;
  } callbacks;

 private:
  bool build_options(Options&);
  void quit();
  void select_scene_file();
  void save_scene_file();
  void save_image(SaveImageMode mode);
  void load_image();
  bool build_material(Material&);
  bool build_medium(Medium&);
  bool spectrum_picker(const char* name, SpectralDistribution& spd, const Pointer<Spectrums> spectrums, bool linear);
  bool ior_picker(const char* name, RefractiveIndex& ior, const Pointer<Spectrums> spectrums);

 private:
  Integrator* _current_integrator = nullptr;
  Scene* _current_scene = nullptr;

  ArrayView<Integrator*> _integrators = {};
  ViewOptions _view_options = {};
  Options _integrator_options = {};

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
    SpectralDistribution::Class cls;
  };

  enum UISetup : uint32_t {
    UIIntegrator = 1u << 0u,
    UIMaterial = 1u << 1u,
    UIMedium = 1u << 2u,
    UIEmitters = 1u << 3u,
    UICamera = 1u << 4u,

    UIEverything = UIIntegrator,
  };

  MappingRepresentation _material_mapping;
  MappingRepresentation _medium_mapping;
  std::vector<IORFile> _ior_files;
  int32_t _selected_material = -1;
  int32_t _selected_medium = -1;
  int32_t _selected_emitter = -1;
  uint32_t _ui_setup = UIEverything;
  uint32_t _font_image = 0u;
};

}  // namespace etx
