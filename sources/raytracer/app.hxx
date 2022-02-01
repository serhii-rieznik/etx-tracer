#pragma once

#include <etx/core/core.hxx>
#include <etx/core/handle.hxx>

#include <etx/rt/integrators/path_tracing.hxx>

#include "ui.hxx"
#include "render.hxx"

namespace etx {

struct RTApplication {
  void init();
  void frame();
  void cleanup();
  void process_event(const sapp_event*);

 private:
  void on_referenece_image_selected(std::string);
  void on_scene_file_selected(std::string);
  void on_integrator_selected(Integrator*);

 private:
  RenderContext render;
  UI ui;
  TimeMeasure time_measure;

  struct {
    Integrator _test;
    CPUPathTracing _cpu_pt;
  } _integrators = {};

  Integrator* _integrator_array[2] = {
    &_integrators._test,
    &_integrators._cpu_pt,
  };

  Integrator* _current_integrator = nullptr;
  uint2 _scene_output_size = {1280u, 720u};
};

}  // namespace etx
