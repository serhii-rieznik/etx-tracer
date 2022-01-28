#include <etx/log/log.hxx>

#include <etx/core/windows.hxx>

#include <stdarg.h>
#include <stdio.h>

namespace etx {

inline void set_console_color(log::Color clr) {
  auto con = GetStdHandle(STD_OUTPUT_HANDLE);
  switch (clr) {
    case log::Color::White: {
      SetConsoleTextAttribute(con, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
      break;
    }
    case log::Color::Yellow: {
      SetConsoleTextAttribute(con, FOREGROUND_RED | FOREGROUND_GREEN);
      break;
    }
    case log::Color::Red: {
      SetConsoleTextAttribute(con, FOREGROUND_RED);
      break;
    }
    case log::Color::Green: {
      SetConsoleTextAttribute(con, FOREGROUND_GREEN);
      break;
    }
    default:
      SetConsoleTextAttribute(con, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
  }
}

void log::output(Color clr, const char* fmt, ...) {
  set_console_color(clr);

  va_list list = {};
  va_start(list, fmt);
  vprintf(fmt, list);
  va_end(list);

  set_console_color(Color::White);
}

}  // namespace etx
