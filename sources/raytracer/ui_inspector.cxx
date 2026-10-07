#include "ui.hxx"

#include <imgui.h>
#include <imgui_internal.h>

namespace etx {

void UI::property_label(const char* label) {
  const float start = ImGui::GetCursorPosX();
  const float available = ImGui::GetContentRegionAvail().x;
  const float font = ImGui::GetFontSize();
  ImGui::AlignTextToFramePadding();
  if (available < (font * 24.0f)) {
    ImGui::TextUnformatted(label);
    return;
  }

  const float label_width = font * 9.0f;
  const ImVec2 position = ImGui::GetCursorScreenPos();
  const ImVec2 size = ImGui::CalcTextSize(label);
  ImGui::Dummy(ImVec2(label_width, ImGui::GetFrameHeight()));
  const ImVec2 text_position(position.x, position.y + ImGui::GetStyle().FramePadding.y);
  const ImVec2 text_end(position.x + label_width, position.y + ImGui::GetFrameHeight());
  ImGui::RenderTextEllipsis(ImGui::GetWindowDrawList(), text_position, text_end, text_end.x, label, nullptr, &size);
  if ((size.x > label_width) && ImGui::IsItemHovered()) {
    ImGui::SetTooltip("%s", label);
  }
  ImGui::SameLine(start + label_width + ImGui::GetStyle().ItemSpacing.x);
}

bool UI::property_section(PropertySection section, const char* label, const char* summary, bool default_open) {
  static constexpr ImVec4 accents[] = {
    {0.32f, 0.62f, 0.85f, 1.0f},
    {0.42f, 0.72f, 0.48f, 1.0f},
    {0.68f, 0.48f, 0.81f, 1.0f},
    {0.84f, 0.64f, 0.28f, 1.0f},
    {0.28f, 0.68f, 0.70f, 1.0f},
    {0.58f, 0.61f, 0.65f, 1.0f},
    {0.42f, 0.59f, 0.81f, 1.0f},
    {0.55f, 0.48f, 0.76f, 1.0f},
    {0.58f, 0.61f, 0.65f, 1.0f},
    {0.72f, 0.53f, 0.38f, 1.0f},
  };
  const ImVec4 accent = accents[static_cast<uint32_t>(section)];
  const ImVec4 background = ImGui::GetStyleColorVec4(ImGuiCol_ChildBg);
  const auto tint = [&](float amount) {
    return ImVec4(background.x + (accent.x - background.x) * amount, background.y + (accent.y - background.y) * amount, background.z + (accent.z - background.z) * amount, 1.0f);
  };
  ImGui::Spacing();
  if (default_open) {
    ImGui::SetNextItemOpen(true, ImGuiCond_Once);
  }
  ImGui::PushStyleColor(ImGuiCol_Header, tint(0.14f));
  ImGui::PushStyleColor(ImGuiCol_HeaderHovered, tint(0.23f));
  ImGui::PushStyleColor(ImGuiCol_HeaderActive, tint(0.30f));
  const bool open = ImGui::CollapsingHeader(label, ImGuiTreeNodeFlags_Framed);
  ImGui::PopStyleColor(3);
  const ImVec2 item_min = ImGui::GetItemRectMin();
  const ImVec2 item_max = ImGui::GetItemRectMax();
  ImGui::GetWindowDrawList()->AddRectFilled(item_min, ImVec2(item_min.x + 3.0f, item_max.y), ImGui::ColorConvertFloat4ToU32(accent));
  if ((summary != nullptr) && (summary[0] != '\0')) {
    const ImVec2 size = ImGui::CalcTextSize(summary);
    const float padding = ImGui::GetStyle().FramePadding.x;
    const float start = item_max.x - size.x - padding;
    const float label_end = item_min.x + ImGui::GetTreeNodeToLabelSpacing() + ImGui::CalcTextSize(label, nullptr, true).x;
    if (start > (label_end + ImGui::GetStyle().ItemSpacing.x)) {
      ImGui::GetWindowDrawList()->AddText(ImVec2(start, item_min.y + ImGui::GetStyle().FramePadding.y), ImGui::GetColorU32(ImGuiCol_TextDisabled), summary);
    }
  }
  return open;
}

}  // namespace etx
