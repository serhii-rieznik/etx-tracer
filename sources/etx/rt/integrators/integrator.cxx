#include "integrator.hxx"

#include <etx/render/host/tasks.hxx>

#include <thread>
#include <mutex>
#include <deque>

namespace etx {

struct ITMessage {
  enum class Cls : uint32_t {
    Run,
    Stop,
  } cls;
  Integrator::Stop stop_option = Integrator::Stop::Immediate;
};

struct IntegratorThreadImpl {
  IntegratorThread* i = nullptr;
  std::thread thread;
  std::atomic<bool> running = {};
  std::deque<ITMessage> messages;
  std::mutex lock;

  TaskScheduler& scheduler;
  Integrator* integrator = nullptr;
  Integrator::State latest_state = Integrator::State::Stopped;
  Integrator::Status latest_status = {};

  bool async = false;

  IntegratorThreadImpl(TaskScheduler& sch, bool create_thread)
    : scheduler(sch)
    , async(create_thread)
    , running(true) {
    if (async) {
      thread = std::thread(&IntegratorThreadImpl::thread_function, this);
    }
  }

  ~IntegratorThreadImpl() {
    running = false;
    if (async && thread.joinable()) {
      thread.join();
    }
  }

  void post_message(const ITMessage& msg) {
    std::unique_lock l(lock);
    messages.push_back(msg);
  }

  void post_messages(const std::initializer_list<ITMessage>& msgs) {
    std::unique_lock l(lock);
    for (const auto& msg : msgs) {
      messages.push_back(msg);
    }
  }

  bool fetch_message(ITMessage& msg) {
    std::unique_lock l(lock);
    if (messages.empty())
      return false;

    msg = messages.front();
    messages.pop_front();
    return true;
  }

  bool has_messages() {
    std::unique_lock l(lock);
    return messages.empty() == false;
  }

  void process_messages() {
    ITMessage msg = {};
    while (fetch_message(msg)) {
      switch (msg.cls) {
        case ITMessage::Cls::Run: {
          if (integrator) {
            integrator->run();
          }
          break;
        }
        case ITMessage::Cls::Stop: {
          if (integrator) {
            integrator->stop(msg.stop_option);
          }
          break;
        }
        default:
          break;
      }
    }
  }

  void thread_function() {
    ETX_PROFILER_REGISTER_THREAD("integrator");
    scheduler.register_thread();

    while (running) {
      {
        ETX_PROFILER_NAMED_SCOPE("integrator::update");
        i->update();
      }

      if (latest_state == Integrator::State::Stopped) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
    ETX_PROFILER_EXIT_THREAD();
  }
};

IntegratorThread::IntegratorThread(TaskScheduler& scheduler, Mode mode) {
  ETX_PIMPL_INIT(IntegratorThread, scheduler, mode == Mode::Async);
}

IntegratorThread ::~IntegratorThread() {
  _private->running = false;
  ETX_PIMPL_CLEANUP(IntegratorThread);
}

void IntegratorThread::start(Integrator* i) {
  _private->integrator = i;
}

void IntegratorThread::terminate() {
}

Integrator* IntegratorThread::integrator() {
  return _private->integrator;
}

void IntegratorThread::set_integrator(Integrator* i) {
  stop(Integrator::Stop::Immediate);
  _private->integrator = i;
}

bool IntegratorThread::running() {
  return (_private->integrator != nullptr) && (_private->latest_state == Integrator::State::Running);
}

const Integrator::Status& IntegratorThread::status() const {
  return _private->latest_status;
}

void IntegratorThread::run() {
  _private->post_message({.cls = ITMessage::Cls::Run});
}

void IntegratorThread::stop(Integrator::Stop st) {
  _private->post_message({.cls = ITMessage::Cls::Stop, .stop_option = st});

  if (st == Integrator::Stop::Immediate) {
    while (_private->latest_state != Integrator::State::Stopped) {
      if (_private->async) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1u));
      } else {
        update();
      }
    }
  }
}

void IntegratorThread::restart() {
  _private->post_messages({
    {.cls = ITMessage::Cls::Stop, .stop_option = Integrator::Stop::Immediate},
    {.cls = ITMessage::Cls::Run},
  });
}

void IntegratorThread::update() {
  ETX_PROFILER_SCOPE();
  _private->process_messages();

  if (_private->integrator == nullptr)
    return;

  _private->integrator->update();
  _private->latest_state = _private->integrator->state();
  _private->latest_status = _private->integrator->status();
}

}  // namespace etx