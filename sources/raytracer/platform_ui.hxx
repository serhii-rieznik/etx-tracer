#pragma once

#include <string>
#include <vector>

namespace etx {

struct UI;

enum class PlatformColorScheme {
  Light,
  Dark,
};

struct PlatformUI {
  void show_startup();
  void finish_startup(bool succeeded);
  void setup(UI& ui);
  void update(UI& ui, const std::vector<std::string>& recent_files);
  void shutdown();

  PlatformColorScheme color_scheme() const;
  bool defers_initialization() const;
};

PlatformUI& platform_ui();

}  // namespace etx
