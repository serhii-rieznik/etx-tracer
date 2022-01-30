#pragma once

#include <etx/core/core.hxx>
#include <etx/core/handle.hxx>

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

 private:
  RenderContext render;
  UI ui;
  TimeMeasure time_measure;
};

}  // namespace etx
