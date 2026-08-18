#pragma once

#include <string>
#include <vector>

namespace etx {

struct UI;

enum class PlatformColorScheme {
  Light,
  Dark,
};

#if defined(_WIN32)
enum class PlatformTitleBarCommand {
  Minimize,
  ToggleMaximize,
  Close,
};
#endif

struct PlatformUI {
  void prepare_application();
  void show_startup();
  void finish_startup(bool succeeded);
  void setup(UI& ui);
  void update(UI& ui, const std::vector<std::string>& recent_files);
  void shutdown();

#if defined(_WIN32)
  void set_title_bar_layout(float menu_width, float controls_left, float height, float framebuffer_scale);
  void execute_title_bar_command(PlatformTitleBarCommand command);
  bool window_maximized() const;
#endif

  PlatformColorScheme color_scheme() const;
  bool defers_initialization() const;
};

PlatformUI& platform_ui();

}  // namespace etx
