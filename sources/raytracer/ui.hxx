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
  void render();
  void cleanup();

  void set_integrator_list(Integrator* i[], uint64_t count) {
    _integrators = {i, count};
  }

  void set_current_integrator(Integrator*);
  void set_output_image_size(const uint2&);
  void update_camera_image(float4*);
  void update_light_image(float4*);

  const Options& integrator_options() const {
    return _integrator_options;
  }

  bool handle_event(const sapp_event*);

  ViewOptions view_options() const;

  struct {
    std::function<void(std::string)> reference_image_selected;
    std::function<void(std::string)> scene_file_selected;
    std::function<void(Integrator*)> integrator_selected;
  } callbacks;

 private:
  void build_options(Options&);

 private:
  ArrayView<Integrator*> _integrators = {};
  Options _view_options = {};
  Options _integrator_options = {};
  const char* _integrator_name = {};
};

}  // namespace etx
