#pragma once

#include <string>
#include <vector>

namespace etx {

struct UI;

void setup_macos_menu(UI& ui);
void update_macos_menu(UI& ui, const std::vector<std::string>& recent_files);
void shutdown_macos_menu();

}  // namespace etx
