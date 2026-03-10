#pragma once

#include <etx/rhi/rhi.hxx>

namespace etx {

struct HeadlessRenderContext {
  void init();
  void cleanup();

  void begin_frame();
  void end_frame();

  RHIContext& context() {
    return _context;
  }

  const RHIContext& context() const {
    return _context;
  }

 private:
  RHIContext _context = {};
};

}  // namespace etx
