#include "platform_ui.hxx"

#include <etx/core/platform.hxx>

#if defined(ETX_PLATFORM_WINDOWS)

# if !defined(WIN32_LEAN_AND_MEAN)
#  define WIN32_LEAN_AND_MEAN 1
# endif

# include <Windows.h>
# include <commctrl.h>
# include <dwmapi.h>
# include <windowsx.h>

# include <sokol_app.h>

# include <algorithm>
# include <cmath>

namespace etx {

namespace {

constexpr UINT_PTR kPlatformUISubclassId = 1u;
constexpr DWORD kImmersiveDarkModeAttribute = 20u;
constexpr int32_t kDefaultTitleBarHeight = 40;

struct WindowsPlatformUIState {
  HWND window = nullptr;
  int32_t menu_width = 0;
  int32_t controls_left = 0;
  int32_t title_bar_height = kDefaultTitleBarHeight;
};

WindowsPlatformUIState g_platform_ui_state = {};

int32_t window_frame_size(HWND window, int32_t metric) {
  const UINT dpi = GetDpiForWindow(window);
  return GetSystemMetricsForDpi(metric, dpi) + GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi);
}

LRESULT hit_test_non_client_area(const WindowsPlatformUIState& state, LPARAM parameter) {
  POINT cursor = {
    .x = GET_X_LPARAM(parameter),
    .y = GET_Y_LPARAM(parameter),
  };

  RECT window_rectangle = {};
  if (GetWindowRect(state.window, &window_rectangle) == FALSE) {
    return HTNOWHERE;
  }

  const bool maximized = IsZoomed(state.window) != FALSE;
  if (maximized == false) {
    const int32_t frame_x = window_frame_size(state.window, SM_CXFRAME);
    const int32_t frame_y = window_frame_size(state.window, SM_CYFRAME);
    const bool left = cursor.x < (window_rectangle.left + frame_x);
    const bool right = cursor.x >= (window_rectangle.right - frame_x);
    const bool top = cursor.y < (window_rectangle.top + frame_y);
    const bool bottom = cursor.y >= (window_rectangle.bottom - frame_y);

    if (top && left) {
      return HTTOPLEFT;
    }
    if (top && right) {
      return HTTOPRIGHT;
    }
    if (bottom && left) {
      return HTBOTTOMLEFT;
    }
    if (bottom && right) {
      return HTBOTTOMRIGHT;
    }
    if (left) {
      return HTLEFT;
    }
    if (right) {
      return HTRIGHT;
    }
    if (top) {
      return HTTOP;
    }
    if (bottom) {
      return HTBOTTOM;
    }
  }

  if (ScreenToClient(state.window, &cursor) == FALSE) {
    return HTNOWHERE;
  }

  if ((cursor.y >= 0) && (cursor.y < state.title_bar_height)) {
    if (((cursor.x >= 0) && (cursor.x < state.menu_width)) || (cursor.x >= state.controls_left)) {
      return HTCLIENT;
    }
    return HTCAPTION;
  }

  return HTCLIENT;
}

LRESULT CALLBACK platform_ui_window_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam, UINT_PTR subclass_id, DWORD_PTR reference_data) {
  (void)subclass_id;
  auto& state = *reinterpret_cast<WindowsPlatformUIState*>(reference_data);

  if (sapp_is_fullscreen()) {
    return DefSubclassProc(window, message, wparam, lparam);
  }

  switch (message) {
    case WM_NCCALCSIZE: {
      if (wparam == TRUE) {
        auto* const parameters = reinterpret_cast<NCCALCSIZE_PARAMS*>(lparam);
        if (IsZoomed(window) != FALSE) {
          const int32_t frame_x = window_frame_size(window, SM_CXFRAME);
          const int32_t frame_y = window_frame_size(window, SM_CYFRAME);
          parameters->rgrc[0].left += frame_x;
          parameters->rgrc[0].top += frame_y;
          parameters->rgrc[0].right -= frame_x;
          parameters->rgrc[0].bottom -= frame_y;
        }
        return 0;
      }
      break;
    }

    case WM_NCHITTEST:
      return hit_test_non_client_area(state, lparam);

    case WM_NCDESTROY:
      RemoveWindowSubclass(window, platform_ui_window_proc, kPlatformUISubclassId);
      state.window = nullptr;
      break;

    default:
      break;
  }

  return DefSubclassProc(window, message, wparam, lparam);
}

}  // namespace

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

  auto& state = g_platform_ui_state;
  state.window = reinterpret_cast<HWND>(const_cast<void*>(sapp_win32_get_hwnd()));
  const UINT dpi = GetDpiForWindow(state.window);
  state.title_bar_height = MulDiv(kDefaultTitleBarHeight, static_cast<int32_t>(dpi), USER_DEFAULT_SCREEN_DPI);

  const BOOL use_dark_mode = TRUE;
  DwmSetWindowAttribute(state.window, kImmersiveDarkModeAttribute, &use_dark_mode, sizeof(use_dark_mode));
  const DWM_WINDOW_CORNER_PREFERENCE corner_preference = DWMWCP_ROUND;
  DwmSetWindowAttribute(state.window, DWMWA_WINDOW_CORNER_PREFERENCE, &corner_preference, sizeof(corner_preference));
  SetWindowSubclass(state.window, platform_ui_window_proc, kPlatformUISubclassId, reinterpret_cast<DWORD_PTR>(&state));

  SetWindowPos(state.window, nullptr, 0, 0, 0, 0, SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
}

void PlatformUI::update(UI& ui, const std::vector<std::string>& recent_files) {
  (void)ui;
  (void)recent_files;
}

void PlatformUI::shutdown() {
  auto& state = g_platform_ui_state;
  if (state.window != nullptr) {
    RemoveWindowSubclass(state.window, platform_ui_window_proc, kPlatformUISubclassId);
  }
  state = {};
}

void PlatformUI::set_title_bar_layout(float menu_width, float controls_left, float height, float framebuffer_scale) {
  auto& state = g_platform_ui_state;
  if (state.window == nullptr) {
    return;
  }

  state.menu_width = std::max(0, static_cast<int32_t>(std::ceil(menu_width * framebuffer_scale)));
  state.controls_left = std::max(state.menu_width, static_cast<int32_t>(std::floor(controls_left * framebuffer_scale)));
  const int32_t new_height = std::max(1, static_cast<int32_t>(std::ceil(height * framebuffer_scale)));
  if (state.title_bar_height != new_height) {
    state.title_bar_height = new_height;
  }
}

void PlatformUI::execute_title_bar_command(PlatformTitleBarCommand command) {
  const HWND window = g_platform_ui_state.window;
  if (window == nullptr) {
    return;
  }

  switch (command) {
    case PlatformTitleBarCommand::Minimize:
      PostMessageW(window, WM_SYSCOMMAND, SC_MINIMIZE, 0);
      break;

    case PlatformTitleBarCommand::ToggleMaximize:
      PostMessageW(window, WM_SYSCOMMAND, IsZoomed(window) != FALSE ? SC_RESTORE : SC_MAXIMIZE, 0);
      break;

    case PlatformTitleBarCommand::Close:
      PostMessageW(window, WM_SYSCOMMAND, SC_CLOSE, 0);
      break;
  }
}

bool PlatformUI::window_maximized() const {
  const HWND window = g_platform_ui_state.window;
  return (window != nullptr) && (IsZoomed(window) != FALSE);
}

PlatformColorScheme PlatformUI::color_scheme() const {
  return PlatformColorScheme::Dark;
}

bool PlatformUI::defers_initialization() const {
  return false;
}

}  // namespace etx

#endif
