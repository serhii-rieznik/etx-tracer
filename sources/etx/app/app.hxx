#pragma once

#include <etx/core/pimpl.hxx>

namespace etx {

struct ApplicationImpl {
  virtual ~ApplicationImpl() = default;

  virtual void init() {
  }

  virtual void process() {
  }

  virtual void cleanup() {
  }
};

struct Application {
  Application(ApplicationImpl& impl);
  ~Application();

  int run(int argc, char* argv[]);

 private:
  ETX_DECLARE_PIMPL(Application, 256);
};

}  // namespace etx
