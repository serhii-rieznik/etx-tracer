#include <etx/core/core.hxx>
#include <etx/render/shared/scene.hxx>
#include <etx/rt/scene_global.hxx>

#include <atomic>
#include <mutex>

namespace etx {

namespace {

std::mutex g_scene_global_lock = {};
std::atomic<bool> g_scene_global_initialized = false;
std::atomic<const void*> g_scene_global_owner = nullptr;
std::atomic<const Scene*> g_scene_global_scene = nullptr;

}  // namespace

void scene_global_init() {
  std::scoped_lock lock(g_scene_global_lock);

  g_scene_global_owner.store(nullptr, std::memory_order_release);
  g_scene_global_scene.store(nullptr, std::memory_order_release);
  g_scene_global_initialized.store(true, std::memory_order_release);
}

void scene_global_deinit() {
  std::scoped_lock lock(g_scene_global_lock);

  g_scene_global_scene.store(nullptr, std::memory_order_release);
  g_scene_global_owner.store(nullptr, std::memory_order_release);
  g_scene_global_initialized.store(false, std::memory_order_release);
}

void scene_global_publish(const void* owner, const Scene* scene) {
  ETX_CRITICAL((owner != nullptr) && (scene != nullptr));

  const bool is_initialized = g_scene_global_initialized.load(std::memory_order_acquire);
  ETX_CRITICAL(is_initialized);

  std::scoped_lock lock(g_scene_global_lock);

  const bool still_initialized = g_scene_global_initialized.load(std::memory_order_acquire);
  ETX_CRITICAL(still_initialized);

  const void* current_owner = g_scene_global_owner.load(std::memory_order_acquire);
  ETX_CRITICAL((current_owner == nullptr) || (current_owner == owner));

  g_scene_global_owner.store(owner, std::memory_order_release);
  g_scene_global_scene.store(scene, std::memory_order_release);
}

void scene_global_clear(const void* owner) {
  ETX_CRITICAL(owner != nullptr);

  const bool is_initialized = g_scene_global_initialized.load(std::memory_order_acquire);
  if (is_initialized == false) {
    return;
  }

  std::scoped_lock lock(g_scene_global_lock);

  const bool still_initialized = g_scene_global_initialized.load(std::memory_order_acquire);
  if (still_initialized == false) {
    return;
  }

  const void* current_owner = g_scene_global_owner.load(std::memory_order_acquire);
  if (current_owner == nullptr) {
    return;
  }

  ETX_CRITICAL(current_owner == owner);

  g_scene_global_scene.store(nullptr, std::memory_order_release);
  g_scene_global_owner.store(nullptr, std::memory_order_release);
}

const Scene* scene_global_try_get() {
  return g_scene_global_scene.load(std::memory_order_acquire);
}

const Scene& scene_global_get() {
  const bool is_initialized = g_scene_global_initialized.load(std::memory_order_acquire);
  ETX_CRITICAL(is_initialized);

  const Scene* scene = scene_global_try_get();
  ETX_CRITICAL(scene != nullptr);
  return *scene;
}

bool scene_global_available() {
  return scene_global_try_get() != nullptr;
}

}  // namespace etx
