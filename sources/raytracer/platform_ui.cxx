#include "platform_ui.hxx"

#include <etx/core/platform.hxx>

#if !defined(ETX_PLATFORM_APPLE)

namespace etx {

PlatformUI& platform_ui() {
  static PlatformUI instance = {};
  return instance;
}

void PlatformUI::prepare_application() {
}

void PlatformUI::show_startup() {
}

void PlatformUI::finish_startup(bool succeeded) {
  (void)succeeded;
}

void PlatformUI::setup(UI& ui) {
  (void)ui;
}

void PlatformUI::update(UI& ui, const std::vector<std::string>& recent_files) {
  (void)ui;
  (void)recent_files;
}

void PlatformUI::shutdown() {
}

PlatformColorScheme PlatformUI::color_scheme() const {
  return PlatformColorScheme::Dark;
}

bool PlatformUI::defers_initialization() const {
  return false;
}

}  // namespace etx

#endif
