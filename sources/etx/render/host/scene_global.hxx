#pragma once

namespace etx {

struct Scene;

void scene_global_init();
void scene_global_deinit();

void scene_global_publish(const void* owner, const Scene* scene);
void scene_global_clear(const void* owner);

const Scene* scene_global_try_get();
const Scene& scene_global_get();
bool scene_global_available();

}  // namespace etx
