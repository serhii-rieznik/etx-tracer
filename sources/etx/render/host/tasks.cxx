#include <etx/core/handle.hxx>
#include <etx/core/profiler.hxx>
#include <etx/render/host/pool.hxx>
#include <etx/render/host/tasks.hxx>

#include <TaskScheduler.hxx>
#include <atomic>
#include <mutex>

#define ETX_ALWAYS_SINGLE_THREAD 0
#define ETX_DEBUG_SINGLE_THREAD  0

#if (ETX_DEBUG || ETX_ALWAYS_SINGLE_THREAD)
# define ETX_SINGLE_THREAD ETX_DEBUG_SINGLE_THREAD
#else
# define ETX_SINGLE_THREAD 0
#endif

namespace etx {

struct TaskWrapper : public enki::ITaskSet {
  Task* task = nullptr;
  std::atomic<bool> executed = false;
  bool background = false;
  enki::TaskSetPartition pinned_range = {0u, 1u};

  struct PinnedTask : public enki::IPinnedTask {
    TaskWrapper& owner;
    PinnedTask(TaskWrapper& wrapper)
      : owner(wrapper) {
    }
    void Execute() override {
      owner.ExecuteRange(owner.pinned_range, threadNum);
    }
  } pinned{*this};

  TaskWrapper(Task* t, uint32_t range, uint32_t min_size)
    : enki::ITaskSet(range, min_size)
    , task(t) {
  }

  void ExecuteRange(enki::TaskSetPartition range_, uint32_t threadnum_) override {
    executed.store(true, std::memory_order_release);
    task->execute_range(range_.start, range_.end, threadnum_);
  }

  enki::ICompletable* completable() {
    return background ? static_cast<enki::ICompletable*>(&pinned) : this;
  }
};

struct FunctionTask : public Task {
  using F = std::function<void(uint32_t, uint32_t, uint32_t)>;
  F func;

  FunctionTask(F f)
    : func(f) {
  }

  void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) override {
    func(begin, end, thread_id);
  }
};

struct TaskSchedulerImpl {
  enki::TaskScheduler scheduler;
  ObjectIndexPool<TaskWrapper> task_pool;
  ObjectIndexPool<FunctionTask> function_task_pool;
  std::map<uint32_t, uint32_t> task_to_function;
  std::mutex task_pool_lock;
  bool shutdown_requested = false;

  Task::Handle schedule_function(uint64_t range, FunctionTask::F func, bool background) {
    TaskWrapper* task = nullptr;
    uint32_t task_handle = Task::InvalidHandle;
    {
      std::scoped_lock lock(task_pool_lock);
      const uint32_t func_task_handle = function_task_pool.alloc(std::move(func));
      auto& func_task = function_task_pool.get(func_task_handle);
      try {
        task_handle = task_pool.alloc(&func_task, range, 1u);
        task = &task_pool.get(task_handle);
        task->background = background;
        task_to_function.emplace(task_handle, func_task_handle);
      } catch (const std::bad_alloc&) {
        if (task_handle != Task::InvalidHandle)
          task_pool.free(task_handle);
        function_task_pool.free(func_task_handle);
        throw;
      }
    }
    if (background) {
      task->pinned.threadNum = scheduler.GetNumTaskThreads() - 1u;
      scheduler.AddPinnedTask(&task->pinned);
    } else {
      scheduler.AddTaskSetToPipe(task);
    }
    return {task_handle};
  }

  TaskSchedulerImpl() {
    task_pool.init(1024u);
    function_task_pool.init(1024u);

    enki::TaskSchedulerConfig config = {};
    config.numExternalTaskThreads = 1u;
    config.numTaskThreadsToCreate = ETX_SINGLE_THREAD ? 1 : (enki::GetNumHardwareThreads() + 1u + config.numExternalTaskThreads);
    config.profilerCallbacks.threadStart = [](uint32_t thread_id) {
      ETX_PROFILER_REGISTER_THREAD(nullptr);
    };
    config.profilerCallbacks.threadStop = [](uint32_t thread_id) {
      ETX_PROFILER_EXIT_THREAD();
    };
    scheduler.Initialize(config);
  }

  ~TaskSchedulerImpl() {
    ETX_ASSERT(task_pool.alive_objects_count() == 0);
    task_pool.cleanup();
  }
};

TaskScheduler::TaskScheduler() {
  ETX_PIMPL_INIT(TaskScheduler);
}

TaskScheduler::~TaskScheduler() {
  ETX_PIMPL_CLEANUP(TaskScheduler);
}

uint32_t TaskScheduler::max_thread_count() {
  return _private->scheduler.GetConfig().numTaskThreadsToCreate + 2u;
}

void TaskScheduler::register_thread() {
  _private->scheduler.RegisterExternalTaskThread();
}

Task::Handle TaskScheduler::schedule(uint64_t range, Task* t) {
  TaskWrapper* task_wrapper = nullptr;
  uint32_t handle = Task::InvalidHandle;
  {
    std::scoped_lock lock(_private->task_pool_lock);
    handle = _private->task_pool.alloc(t, range, 1u);
    task_wrapper = &_private->task_pool.get(handle);
  }
  _private->scheduler.AddTaskSetToPipe(task_wrapper);
  return {handle};
}

Task::Handle TaskScheduler::schedule(uint64_t range, std::function<void(uint32_t, uint32_t, uint32_t)> func) {
  return _private->schedule_function(range, std::move(func), false);
}

Task::Handle TaskScheduler::schedule_background(std::function<void(uint32_t, uint32_t, uint32_t)> func) {
  return _private->schedule_function(1u, std::move(func), true);
}

void TaskScheduler::execute(uint64_t range, Task* t) {
  auto handle = schedule(range, t);
  wait_and_release(handle);
}

void TaskScheduler::execute_background(uint64_t range, Task* t) {
  std::vector<Task::Handle> handles;
  handles.reserve(range);
  try {
    const uint32_t first_worker = _private->scheduler.GetConfig().numExternalTaskThreads + 1u;
    const uint32_t worker_count = _private->scheduler.GetConfig().numTaskThreadsToCreate;
    for (uint32_t index = 0u; index < range; ++index) {
      TaskWrapper* task = nullptr;
      Task::Handle handle;
      {
        std::scoped_lock lock(_private->task_pool_lock);
        handle.data = _private->task_pool.alloc(t, 1u, 1u);
        task = &_private->task_pool.get(handle.data);
        task->background = true;
        task->pinned_range = {index, index + 1u};
        task->pinned.threadNum = first_worker + (index % worker_count);
      }
      handles.push_back(handle);
      _private->scheduler.AddPinnedTask(&task->pinned);
    }
  } catch (const std::bad_alloc&) {
    for (auto& handle : handles)
      wait_and_release(handle);
    throw;
  }
  for (auto& handle : handles)
    wait_and_release(handle);
}

void TaskScheduler::execute(uint64_t range, std::function<void(uint32_t, uint32_t, uint32_t)> func) {
  auto handle = schedule(range, func);
  wait_and_release(handle);
}

void TaskScheduler::execute_linear(uint64_t range, std::function<void(uint32_t, uint32_t, uint32_t)> func) {
  func(0u, static_cast<uint32_t>(range), 0u);
}

bool TaskScheduler::completed(Task::Handle handle) {
  if (handle.data == Task::InvalidHandle) {
    return true;
  }

  TaskWrapper* task_wrapper = nullptr;
  {
    std::scoped_lock lock(_private->task_pool_lock);
    task_wrapper = &_private->task_pool.get(handle.data);
  }
  return task_wrapper->executed.load(std::memory_order_acquire) && task_wrapper->completable()->GetIsComplete();
}

void TaskScheduler::wait_task(const Task::Handle& handle) {
  if (handle.data == Task::InvalidHandle) {
    return;
  }

  TaskWrapper* task_wrapper = nullptr;
  {
    std::scoped_lock lock(_private->task_pool_lock);
    task_wrapper = &_private->task_pool.get(handle.data);
  }
  _private->scheduler.WaitforTask(task_wrapper->completable());
}

void TaskScheduler::release(Task::Handle& handle) {
  if (handle.data == Task::InvalidHandle) {
    return;
  }
  {
    std::scoped_lock lock(_private->task_pool_lock);
    _private->task_pool.free(handle.data);

    auto func_task = _private->task_to_function.find(handle.data);
    if (func_task != _private->task_to_function.end()) {
      _private->function_task_pool.free(func_task->second);
      _private->task_to_function.erase(func_task);
    }
  }

  handle.data = Task::InvalidHandle;
}

void TaskScheduler::wait_and_release(Task::Handle& handle) {
  wait_task(handle);
  release(handle);
}

void TaskScheduler::restart(Task::Handle handle) {
  if (handle.data == Task::InvalidHandle) {
    return;
  }

  TaskWrapper* task_wrapper = nullptr;
  {
    std::scoped_lock lock(_private->task_pool_lock);
    task_wrapper = &_private->task_pool.get(handle.data);
  }
  _private->scheduler.WaitforTask(task_wrapper->completable());
  if (task_wrapper->background)
    _private->scheduler.AddPinnedTask(&task_wrapper->pinned);
  else
    _private->scheduler.AddTaskSetToPipe(task_wrapper);
}

void TaskScheduler::shutdown() {
  if (_private->shutdown_requested) {
    return;
  }
  _private->scheduler.WaitforAllAndShutdown();
  _private->shutdown_requested = true;
}

}  // namespace etx
