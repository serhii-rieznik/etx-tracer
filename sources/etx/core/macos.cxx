#include <etx/core/platform.hxx>

#if (ETX_PLATFORM_APPLE)

# include <dispatch/dispatch.h>
# include <sys/types.h>
# include <sys/user.h>
# include <unistd.h>

namespace etx {

static bool debugger_present = false;

void init_platform() {
  static dispatch_once_t predicate = {};
  
  dispatch_once(&predicate, ^{
    struct kinfo_proc info;
    size_t info_size = sizeof(info);
    int name[4] = {
      CTL_KERN,
      KERN_PROC,
      KERN_PROC_PID,
      getpid(),
    };

    if (sysctl(name, 4, &info, &info_size, NULL, 0) == -1) {
      debugger_present = false;
    }

    if (!debugger_present && (info.kp_proc.p_flag & P_TRACED) != 0)
      debugger_present = true;
  });
}

inline void log::set_console_color(log::Color clr) {
  if (debugger_present)
    return;

  switch (clr) {
    case log::Color::Red: {
      printf("\x1b[31m");
      break;
    }
    case log::Color::Green: {
      printf("\x1b[32m");
      break;
    }
    case log::Color::Yellow: {
      printf("\x1b[33m");
      break;
    }
    default:
      printf("\x1b[37m");
      break;
  }
}

}  // namespace etx

#endif
