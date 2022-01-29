#pragma once

#include <etx/core/options.hxx>

struct sapp_event;

namespace etx {

struct UI {
  void initialize();
  void build(double dt);
  void render();
  void cleanup();

  void set_options(const Options&);

  bool handle_event(const sapp_event*);

 private:
  Options _integrator_options = {};
};

}  // namespace etx
