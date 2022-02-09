#pragma once

#include <etx/core/options.hxx>

#include <etx/render/shared/base.hxx>
#include <etx/rt/integrators/integrator.hxx>

#include <functional>

#include "options.hxx"

struct sapp_event;

namespace etx {

struct UI {
  void initialize();
  void build(double dt, const char* status);

  void set_integrator_list(Integrator* i[], uint64_t count) {
    _integrators = {i, count};
  }

  void set_current_integrator(Integrator*);

  const Options& integrator_options() const {
    return _integrator_options;
  }

  bool handle_event(const sapp_event*);

  ViewOptions view_options() const;

  struct {
    std::function<void(std::string)> reference_image_selected;
    std::function<void(std::string, bool)> save_image_selected;
    std::function<void(std::string)> scene_file_selected;
    std::function<void(Integrator*)> integrator_selected;
    std::function<void(bool)> stop_selected;
    std::function<void()> preview_selected;
    std::function<void()> run_selected;
    std::function<void()> reload_scene_selected;
    std::function<void()> reload_geometry_selected;
    std::function<void()> options_changed;
  } callbacks;

 private:
  bool build_options(Options&);
  void select_scene_file();
  void save_image(bool xyz);
  void load_image();

 private:
  Integrator* _current_integrator = {};
  ArrayView<Integrator*> _integrators = {};
  Options _view_options = {};
  Options _integrator_options = {};
  const char* _integrator_name = {};
};

}  // namespace etx
