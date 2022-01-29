#pragma once

#include <etx/core/pimpl.hxx>

namespace etx {

struct RenderContext {
  RenderContext();
  ~RenderContext();

  void init();
  void cleanup();

  void start_frame();
  void end_frame();

 private:
  ETX_DECLARE_PIMPL(RenderContext, 256);
};

}  // namespace etx
