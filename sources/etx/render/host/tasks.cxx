#include <etx/core/handle.hxx>
#include <etx/render/host/pool.hxx>
#include <etx/render/host/tasks.hxx>

#include <enkiTS/TaskScheduler.hxx>

namespace etx {

struct TaskWrapper : public enki::ITaskSet {
  Task* task = nullptr;

  TaskWrapper(Task* t, uint32_t range, uint32_t min_size)
    : enki::ITaskSet(range, min_size)
    , task(t) {
  }

  void ExecuteRange(enki::TaskSetPartition range_, uint32_t threadnum_) override {
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

  TaskSchedulerImpl() {
    task_pool.init(1024u);
    scheduler.Initialize(enki::GetNumHardwareThreads() + 1u);
  }

  ~TaskSchedulerImpl() {
    ETX_ASSERT(task_pool.count_alive() == 0);
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
  return _private->scheduler.GetConfig().numTaskThreadsToCreate;
}

Task::Handle TaskScheduler::schedule(Task* t, uint32_t range) {
  auto handle = _private->task_pool.alloc(t, range, 1u);
  auto& task_wrapper = _private->task_pool.get(handle);
  _private->scheduler.AddTaskSetToPipe(&task_wrapper);
  return {handle};
}

void TaskScheduler::execute(Task* t, uint32_t range) {
  wait(schedule(t, range));
}

void TaskScheduler::execute(uint32_t range, std::function<void(uint32_t, uint32_t, uint32_t)> func) {
  FunctionTask t(func);
  wait(schedule(&t, range));
}

bool TaskScheduler::completed(Task::Handle handle) {
  if (handle.internal == Task::InvalidHandle) {
    return true;
  }

  auto& task_wrapper = _private->task_pool.get(handle.internal);
  return task_wrapper.GetIsComplete();
}

void TaskScheduler::wait(Task::Handle handle) {
  if (handle.internal == Task::InvalidHandle) {
    return;
  }

  auto& task_wrapper = _private->task_pool.get(handle.internal);
  _private->scheduler.WaitforTaskSet(&task_wrapper);
  _private->task_pool.free(handle.internal);
}

void TaskScheduler::restart(Task::Handle handle, uint32_t new_rage) {
  if (handle.internal == Task::InvalidHandle) {
    return;
  }

  auto& task_wrapper = _private->task_pool.get(handle.internal);
  if (task_wrapper.GetIsComplete() == false) {
    _private->scheduler.WaitforTaskSet(&task_wrapper);
  }
  task_wrapper.m_SetSize = new_rage;
  _private->scheduler.AddTaskSetToPipe(&task_wrapper);
}

void TaskScheduler::restart(Task::Handle handle) {
  if (handle.internal == Task::InvalidHandle) {
    return;
  }
  auto& task_wrapper = _private->task_pool.get(handle.internal);
  restart(handle, task_wrapper.m_SetSize);
}

}  // namespace etx
