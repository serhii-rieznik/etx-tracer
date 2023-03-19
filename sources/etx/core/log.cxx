#include <etx/core/log.hxx>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

namespace etx {

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
