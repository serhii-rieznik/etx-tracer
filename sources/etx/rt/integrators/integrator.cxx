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
  std::thread thread;
  std::atomic<bool> running = {};
  std::atomic<bool> synced = {};
  std::deque<ITMessage> messages;
  std::mutex lock;

  TaskScheduler& scheduler;
  Integrator* integrator = nullptr;
  Integrator::State latest_state = Integrator::State::Stopped;
  Integrator::Status latest_status = {};
  Options options = {};

  IntegratorThreadImpl(TaskScheduler& sch)
    : scheduler(sch) {
    running = true;
    thread = std::thread(&IntegratorThreadImpl::thread_function, this);
  }

  ~IntegratorThreadImpl() {
    running = false;
    thread.join();
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
      process_messages();

      if (integrator != nullptr) {
        integrator->update();
        latest_state = integrator->state();
        latest_status = integrator->status();

        if (latest_state == Integrator::State::Stopped) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
      }
    }
  }
};

IntegratorThread::IntegratorThread(TaskScheduler& scheduler) {
  ETX_PIMPL_INIT(IntegratorThread, scheduler);
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
      std::this_thread::sleep_for(std::chrono::milliseconds(1u));
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

}  // namespace etx