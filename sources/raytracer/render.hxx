#pragma once

namespace etx {

struct RenderContext {
  void init();
  void cleanup();

  void start_frame();
  void end_frame();
};

}  // namespace etx
