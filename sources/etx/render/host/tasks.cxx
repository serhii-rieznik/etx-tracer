#include <etx/core/handle.hxx>
#include <etx/core/profiler.hxx>
#include <etx/render/host/pool.hxx>
#include <etx/render/host/tasks.hxx>

#include <TaskScheduler.hxx>
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
  bool executed = false;

  TaskWrapper(Task* t, uint32_t range, uint32_t min_size)
    : enki::ITaskSet(range, min_size)
    , task(t) {
  }

  void ExecuteRange(enki::TaskSetPartition range_, uint32_t threadnum_) override {
    executed = true;
    task->execute_range(range_.start, range_.end, threadnum_);
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
  TaskWrapper* task = nullptr;
  uint32_t task_handle = Task::InvalidHandle;
  {
    std::scoped_lock lock(_private->task_pool_lock);
    const uint32_t func_task_handle = _private->function_task_pool.alloc(func);
    auto& func_task = _private->function_task_pool.get(func_task_handle);

    task_handle = _private->task_pool.alloc(&func_task, range, 1u);
    task = &_private->task_pool.get(task_handle);

    _private->task_to_function[task_handle] = func_task_handle;
  }
  _private->scheduler.AddTaskSetToPipe(task);

  return {task_handle};
}

void TaskScheduler::execute(uint64_t range, Task* t) {
  auto handle = schedule(range, t);
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
  return task_wrapper->executed && task_wrapper->GetIsComplete();
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
  _private->scheduler.WaitforTask(task_wrapper);
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
  _private->scheduler.WaitforTask(task_wrapper);
  _private->scheduler.AddTaskSetToPipe(task_wrapper);
}

}  // namespace etx
