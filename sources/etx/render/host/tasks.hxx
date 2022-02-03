#pragma once

#include <etx/core/pimpl.hxx>

#include <functional>

namespace etx {

struct Task {
  enum : uint32_t {
    InvalidHandle = ~0u,
  };

  struct Handle {
    uint32_t internal = InvalidHandle;
  };

  virtual ~Task() = default;

  virtual void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) = 0;
};

struct TaskScheduler {
  TaskScheduler();
  ~TaskScheduler();

  uint32_t max_thread_count();

  Task::Handle schedule(Task*, uint32_t range);

  void execute(Task*, uint32_t range);
  void execute(uint32_t range, std::function<void(uint32_t, uint32_t, uint32_t)> func);

  bool completed(Task::Handle);
  void wait(Task::Handle);

  void restart(Task::Handle);

 private:
  ETX_DECLARE_PIMPL(TaskScheduler, 256);
};

}  // namespace etx
