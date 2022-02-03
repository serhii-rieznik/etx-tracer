#include <etx/log/log.hxx>

#include <etx/core/windows.hxx>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

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
  constexpr int local_buffer_size = 1024;
  set_console_color(clr);

  va_list list = {};
  va_start(list, fmt);
  int required_size = _vscprintf(fmt, list) + 1;
  if (required_size + 1 < local_buffer_size) {
    char* buffer = (char*)_malloca(required_size + 1);
    vsnprintf(buffer, required_size, fmt, list);
    puts(buffer);
    _freea(buffer);
  } else {
    char* buffer = (char*)calloc(required_size + 1, 1);
    if (buffer != nullptr) {
      vsnprintf(buffer, required_size, fmt, list);
      puts(buffer);
      free(buffer);
    }
  }
  va_end(list);

  set_console_color(Color::White);
}

}  // namespace etx
