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
    Options,
  } cls;
  Integrator::Stop stop_option = Integrator::Stop::Immediate;
  Options options = {};
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
  Options options = {};

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
            integrator->run(options);
          }
          break;
        }
        case ITMessage::Cls::Stop: {
          if (integrator) {
            integrator->stop(msg.stop_option);
          }
          break;
        }
        case ITMessage::Cls::Options: {
          options = msg.options;
          break;
        }
        default:
          break;
      }
    }
  }

  void thread_function() {
    scheduler.register_thread();

    while (running) {
      i->update();

      if (latest_state == Integrator::State::Stopped) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  }
};

IntegratorThread::IntegratorThread(TaskScheduler& scheduler, Mode mode) {
  ETX_PIMPL_INIT(IntegratorThread, scheduler, mode == Mode::Async);
}

IntegratorThread ::~IntegratorThread() {
  _private->running = false;
  ETX_PIMPL_CLEANUP(IntegratorThread);
}

void IntegratorThread::start() {
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

void IntegratorThread::run(const Options& opt) {
  _private->post_message({
    .cls = ITMessage::Cls::Options,
    .options = opt,
  });
  _private->post_message({
    .cls = ITMessage::Cls::Run,
  });
}

void IntegratorThread::stop(Integrator::Stop st) {
  _private->post_message({
    .cls = ITMessage::Cls::Stop,
    .stop_option = st,
  });

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
  _private->post_message({
    .cls = ITMessage::Cls::Stop,
    .stop_option = Integrator::Stop::Immediate,
  });
  _private->post_message({
    .cls = ITMessage::Cls::Run,
  });
}

void IntegratorThread::restart(const Options& opt) {
  _private->post_message({
    .cls = ITMessage::Cls::Stop,
    .stop_option = Integrator::Stop::Immediate,
  });
  _private->post_message({
    .cls = ITMessage::Cls::Options,
    .options = opt,
  });
  _private->post_message({
    .cls = ITMessage::Cls::Run,
  });
}

void IntegratorThread::update() {
  _private->process_messages();

  if (_private->integrator == nullptr)
    return;

  _private->integrator->update();
  _private->latest_state = _private->integrator->state();
  _private->latest_status = _private->integrator->status();
}

}  // namespace etx