#pragma once

#include <string>
#include <vector>

namespace etx {

struct UI;

void show_macos_startup_overlay();
void finish_macos_startup(bool succeeded);
void setup_macos_menu(UI& ui);
void update_macos_menu(UI& ui, const std::vector<std::string>& recent_files);
void shutdown_macos_menu();

}  // namespace etx
