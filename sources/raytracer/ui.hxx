#pragma once

#include <etx/core/options.hxx>
#include <functional>

#include "options.hxx"

struct sapp_event;

namespace etx {

struct UI {
  void initialize();
  void build(double dt);
  void render();
  void cleanup();

  void set_options(const Options&);

  bool handle_event(const sapp_event*);

  ViewOptions view_options() const;

  struct {
    std::function<void(std::string)> reference_image_selected;
  } callbacks;

 private:
  void build_options(Options&);

 private:
  Options _integrator_options = {};
  Options _view_options = {};
};

}  // namespace etx
