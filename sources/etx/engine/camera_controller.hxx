#pragma once

#include <sokol_app.h>
#include <etx/render/shared/camera.hxx>
#include <etx/render/host/scene_representation.hxx>
#include <cmath>

namespace etx {

struct CameraController {
  static constexpr float kMaxCameraDistance = 8192.0f;
  static constexpr float kMinCameraDistance = 1.0f / 255.0f;

  bool enable_inertia = true;

  CameraController(Camera& cam)
    : _camera(cam) {
  }

  bool update(double dt) {
    float dt_sec = clamp(float(dt), 0.0f, 0.1f);
    float dt_raw = float(dt);

    bool camera_changed = false;

    if (_scheduled.active) {
      _scheduled.active = false;
      _camera.position = _scheduled.pos;
      float3 view = _scheduled.center - _scheduled.pos;
      _camera.direction = normalize_safe(view, normalize_safe(_camera.direction, kWorldForward));
      _orbit_pivot = _scheduled.center;
      _orbit_distance = clamp(length(_orbit_pivot - _camera.position), kMinCameraDistance, kMaxCameraDistance);
      _pivot_initialized = true;
      _move_velocity = {};
      _orbit_velocity = {};
      _look_velocity = {};
      _pan_velocity = {};
      _dolly_velocity = 0.0f;
      _zoom_velocity = 0.0f;
      _move_scale_distance = max(_orbit_distance, 4.0f);
      _interpolation_initialized = false;
      camera_changed = true;
    }

    const bool shift_pressed = is_shift_pressed();
    const bool ctrl_pressed = is_ctrl_pressed();
    const bool left_pressed = (_mouse_buttons & MouseLeft) != 0u;
    const bool middle_pressed = (_mouse_buttons & MouseMiddle) != 0u;
    const bool right_pressed = (_mouse_buttons & MouseRight) != 0u;

    float3 target_orbit_velocity = {};
    float3 target_look_velocity = {};
    float3 target_pan_velocity = {};
    float target_dolly_velocity = 0.0f;
    float target_zoom_velocity = 0.0f;

    if (dt_raw > 1.0e-5f) {
      if (middle_pressed) {
        if (shift_pressed) {
          target_pan_velocity = {_mouse_delta.x / dt_raw, _mouse_delta.y / dt_raw, 0.0f};
        } else if (ctrl_pressed) {
          target_dolly_velocity = (_mouse_delta.y * _drag_dolly_speed) / dt_raw;
        } else {
          target_orbit_velocity = {_mouse_delta.x / dt_raw, _mouse_delta.y / dt_raw, 0.0f};
        }
      } else if (left_pressed || right_pressed) {
        target_look_velocity = {_mouse_delta.x / dt_raw, _mouse_delta.y / dt_raw, 0.0f};
      }

      if (_mouse_delta.z != 0.0f) {
        target_zoom_velocity = (_mouse_delta.z * _scroll_zoom_speed) / dt_raw;
      }
    }

    float mouse_alpha = enable_inertia ? ((dt_sec > 0.0f) ? (1.0f - expf(-_movement_response * dt_sec)) : 0.0f) : 1.0f;

    _orbit_velocity.x += (target_orbit_velocity.x - _orbit_velocity.x) * mouse_alpha;
    _orbit_velocity.y += (target_orbit_velocity.y - _orbit_velocity.y) * mouse_alpha;
    _look_velocity.x += (target_look_velocity.x - _look_velocity.x) * mouse_alpha;
    _look_velocity.y += (target_look_velocity.y - _look_velocity.y) * mouse_alpha;
    _pan_velocity.x += (target_pan_velocity.x - _pan_velocity.x) * mouse_alpha;
    _pan_velocity.y += (target_pan_velocity.y - _pan_velocity.y) * mouse_alpha;
    _dolly_velocity += (target_dolly_velocity - _dolly_velocity) * mouse_alpha;
    _zoom_velocity += (target_zoom_velocity - _zoom_velocity) * mouse_alpha;

    if (fabsf(_orbit_velocity.x) + fabsf(_orbit_velocity.y) < _min_velocity)
      _orbit_velocity = {};
    if (fabsf(_look_velocity.x) + fabsf(_look_velocity.y) < _min_velocity)
      _look_velocity = {};
    if (fabsf(_pan_velocity.x) + fabsf(_pan_velocity.y) < _min_velocity)
      _pan_velocity = {};
    if (fabsf(_dolly_velocity) < _min_velocity)
      _dolly_velocity = 0.0f;
    if (fabsf(_zoom_velocity) < _min_velocity)
      _zoom_velocity = 0.0f;

    if (_pan_velocity.x != 0.0f || _pan_velocity.y != 0.0f) {
      ensure_pivot_initialized();
      apply_pan(_pan_velocity.x * dt_sec, _pan_velocity.y * dt_sec);
      camera_changed = true;
    }
    if (_dolly_velocity != 0.0f) {
      ensure_pivot_initialized();
      apply_dolly(_dolly_velocity * dt_sec);
      camera_changed = true;
    }
    if (_orbit_velocity.x != 0.0f || _orbit_velocity.y != 0.0f) {
      ensure_pivot_initialized();
      apply_orbit(_orbit_velocity.x * dt_sec, _orbit_velocity.y * dt_sec);
      camera_changed = true;
    }
    if (_look_velocity.x != 0.0f || _look_velocity.y != 0.0f) {
      apply_look(_look_velocity.x * dt_sec, _look_velocity.y * dt_sec);
      sync_pivot_to_view_direction();
      camera_changed = true;
    }
    if (_zoom_velocity != 0.0f) {
      ensure_pivot_initialized();
      const float zoom_factor = expf(-_zoom_velocity * dt_sec);
      _orbit_distance = clamp(_orbit_distance * zoom_factor, kMinCameraDistance, kMaxCameraDistance);
      float3 forward = normalize_safe(_orbit_pivot - _camera.position, normalize_safe(_camera.direction, kWorldForward));
      _camera.position = _orbit_pivot - forward * _orbit_distance;
      _camera.direction = forward;
      camera_changed = true;
    }

    if (apply_keyboard_movement(dt_sec, shift_pressed, ctrl_pressed)) {
      camera_changed = true;
    }

    _mouse_delta = {};

    if (camera_changed) {
      clamp_position(_camera.position);

      if (_interpolation_initialized == false || enable_inertia == false) {
        _interpolated_position = _camera.position;
        _interpolated_direction = _camera.direction;
        _interpolation_initialized = true;
      }
    }

    if (_interpolation_initialized) {
      const float interpolation_response = _movement_response * 3.0f;
      float interp_alpha = enable_inertia ? ((dt_sec > 0.0f) ? (1.0f - expf(-interpolation_response * dt_sec)) : 0.0f) : 1.0f;
      _interpolated_position += (_camera.position - _interpolated_position) * interp_alpha;

      _interpolated_direction = normalize_safe(_interpolated_direction + (_camera.direction - _interpolated_direction) * interp_alpha, kWorldForward);

      build_camera(_camera, _interpolated_position, _interpolated_direction, kWorldUp, _camera.film_size, get_camera_fov(_camera));

      float3 pos_diff = _camera.position - _interpolated_position;
      float3 dir_diff = _camera.direction - _interpolated_direction;
      if (dot(pos_diff, pos_diff) > 1.0e-8f || dot(dir_diff, dir_diff) > 1.0e-8f) {
        return true;
      }
    } else {
      build_camera(_camera, _camera.position, _camera.direction, kWorldUp, _camera.film_size, get_camera_fov(_camera));
    }

    return camera_changed;
  }

  void handle_scroll(float scroll) {
#if (ETX_PLATFORM_APPLE)
    constexpr float kScrollScaleFactor = -1.0f / 32.0f;
#else
    constexpr float kScrollScaleFactor = 1.0f / 32.0f;
#endif
    _mouse_delta.z += kScrollScaleFactor * scroll;
  }

  void handle_event(const sapp_event* e) {
    switch (e->type) {
      case SAPP_EVENTTYPE_MOUSE_SCROLL: {
        handle_scroll(e->scroll_y);
        break;
      }

      case SAPP_EVENTTYPE_KEY_DOWN: {
        if ((e->key_code >= 0) && (e->key_code < 512)) {
          _keys[e->key_code] = true;
        }
        break;
      }

      case SAPP_EVENTTYPE_KEY_UP: {
        if ((e->key_code >= 0) && (e->key_code < 512)) {
          _keys[e->key_code] = false;
        }
        break;
      }

      case SAPP_EVENTTYPE_MOUSE_DOWN: {
        if (e->mouse_button == SAPP_MOUSEBUTTON_LEFT)
          _mouse_buttons = _mouse_buttons | MouseLeft;
        if (e->mouse_button == SAPP_MOUSEBUTTON_MIDDLE)
          _mouse_buttons = _mouse_buttons | MouseMiddle;
        if (e->mouse_button == SAPP_MOUSEBUTTON_RIGHT)
          _mouse_buttons = _mouse_buttons | MouseRight;
        break;
      }

      case SAPP_EVENTTYPE_MOUSE_UP: {
        if (e->mouse_button == SAPP_MOUSEBUTTON_LEFT)
          _mouse_buttons = _mouse_buttons & (~MouseLeft);
        if (e->mouse_button == SAPP_MOUSEBUTTON_MIDDLE)
          _mouse_buttons = _mouse_buttons & (~MouseMiddle);
        if (e->mouse_button == SAPP_MOUSEBUTTON_RIGHT)
          _mouse_buttons = _mouse_buttons & (~MouseRight);
        break;
      }

      case SAPP_EVENTTYPE_MOUSE_MOVE: {
        _mouse_delta.x += e->mouse_dx;
        _mouse_delta.y += e->mouse_dy;
        break;
      }

      case SAPP_EVENTTYPE_UNFOCUSED: {
        for (bool& k : _keys) {
          k = false;
        }
        _mouse_delta = {};
        _mouse_buttons = 0;
        reset_velocities();
        break;
      }

      default:
        break;
    }
  }

  void schedule(const float3& pos, const float3& view_center) {
    float3 safe_center = view_center;
    float3 view = safe_center - pos;
    const float len = length(view);
    if (len > kEpsilon) {
      float3 dir = view / len;
      if (fabsf(dir.y) > 0.999f) {
        safe_center.z += 0.001f * len;
      }
    }
    _scheduled = {pos, safe_center, true};
    _mouse_delta = {};
    reset_velocities();
  }

  void reset_velocities() {
    _move_velocity = {};
    _orbit_velocity = {};
    _look_velocity = {};
    _pan_velocity = {};
    _dolly_velocity = 0.0f;
    _zoom_velocity = 0.0f;
  }

 private:
  bool is_shift_pressed() const {
    return _keys[SAPP_KEYCODE_LEFT_SHIFT] || _keys[SAPP_KEYCODE_RIGHT_SHIFT];
  }

  bool is_ctrl_pressed() const {
    return _keys[SAPP_KEYCODE_LEFT_CONTROL] || _keys[SAPP_KEYCODE_RIGHT_CONTROL];
  }

  static void clamp_position(float3& p) {
    p.x = clamp(p.x, -kMaxCameraDistance, kMaxCameraDistance);
    p.y = clamp(p.y, -kMaxCameraDistance, kMaxCameraDistance);
    p.z = clamp(p.z, -kMaxCameraDistance, kMaxCameraDistance);
  }

  static float3 normalize_safe(const float3& v, const float3& fallback) {
    const float len = length(v);
    if (len < kEpsilon) {
      return fallback;
    }
    return v / len;
  }

  float estimate_navigation_scale() const {
    // Keep movement meaningful even when orbit distance is tiny:
    // use coarse world-space cues as additional scale hints.
    float scale = 4.0f;
    scale = max(scale, length(_camera.position) * 0.125f);

    const float dir_y = _camera.direction.y;
    if (fabsf(dir_y) > 1.0e-3f) {
      const float t_to_ground_plane = (-_camera.position.y) / dir_y;
      if (t_to_ground_plane > 0.0f) {
        scale = max(scale, t_to_ground_plane * 0.5f);
      }
    }

    return scale;
  }

  void ensure_pivot_initialized() {
    if (_pivot_initialized == false) {
      const float3 direction = normalize_safe(_camera.direction, kWorldForward);
      float default_distance = _camera.focal_distance;
      if (default_distance <= kMinCameraDistance) {
        default_distance = 4.0f;
      }
      _orbit_distance = clamp(default_distance, kMinCameraDistance, kMaxCameraDistance);
      _orbit_pivot = _camera.position + direction * _orbit_distance;
      _pivot_initialized = true;
      return;
    }

    _orbit_distance = clamp(length(_orbit_pivot - _camera.position), kMinCameraDistance, kMaxCameraDistance);
  }

  void sync_pivot_to_view_direction() {
    ensure_pivot_initialized();
    const float3 direction = normalize_safe(_camera.direction, kWorldForward);
    _orbit_pivot = _camera.position + direction * _orbit_distance;
  }

  void apply_look(float dx, float dy) {
    auto s = to_spherical(normalize_safe(_camera.direction, kWorldForward));
    s.phi += dx * _look_speed;
    const float max_theta = 89.99f * kPi / 180.0f;
    s.theta = clamp(s.theta - dy * _look_speed, -max_theta, max_theta);
    _camera.direction = normalize_safe(from_spherical(s), kWorldForward);
  }

  void apply_orbit(float dx, float dy) {
    float3 offset = _camera.position - _orbit_pivot;
    if (length(offset) < kEpsilon) {
      offset = -normalize_safe(_camera.direction, kWorldForward) * _orbit_distance;
    }

    auto s = to_spherical(offset);
    s.r = clamp(s.r, kMinCameraDistance, kMaxCameraDistance);
    s.phi += dx * _orbit_speed;
    const float max_theta = 89.99f * kPi / 180.0f;
    s.theta = clamp(s.theta + dy * _orbit_speed, -max_theta, max_theta);
    _orbit_distance = s.r;
    _camera.position = _orbit_pivot + from_spherical(s);
    _camera.direction = normalize_safe(_orbit_pivot - _camera.position, normalize_safe(_camera.direction, kWorldForward));
  }

  void apply_pan(float dx, float dy) {
    const float3 forward = normalize_safe(_orbit_pivot - _camera.position, normalize_safe(_camera.direction, kWorldForward));
    float3 side = cross(kWorldUp, forward);
    if (length(side) < kEpsilon) {
      side = cross(kWorldRight, forward);
    }
    side = normalize_safe(side, kWorldRight);
    const float3 up = normalize_safe(cross(forward, side), kWorldUp);
    const float pan_scale = _pan_speed * max(_orbit_distance, 1.0f);
    const float3 delta = (dy * up + dx * side) * pan_scale;
    _camera.position += delta;
    _orbit_pivot += delta;
    _camera.direction = forward;
  }

  void apply_dolly(float offset) {
    _orbit_distance = clamp(_orbit_distance + offset * max(_orbit_distance, 1.0f), kMinCameraDistance, kMaxCameraDistance);
    const float3 forward = normalize_safe(_orbit_pivot - _camera.position, normalize_safe(_camera.direction, kWorldForward));
    _camera.position = _orbit_pivot - forward * _orbit_distance;
    _camera.direction = forward;
  }

  bool apply_keyboard_movement(float dt_sec, bool shift_pressed, bool ctrl_pressed) {
    const float move_fwd = float(_keys[SAPP_KEYCODE_W] ? 1.0f : 0.0f) - float(_keys[SAPP_KEYCODE_S] ? 1.0f : 0.0f);
    const float move_side = float(_keys[SAPP_KEYCODE_D] ? 1.0f : 0.0f) - float(_keys[SAPP_KEYCODE_A] ? 1.0f : 0.0f);
    const float move_up = float(_keys[SAPP_KEYCODE_E] ? 1.0f : 0.0f) - float(_keys[SAPP_KEYCODE_Q] ? 1.0f : 0.0f);

    float3 target_velocity = {};
    if ((move_fwd != 0.0f) || (move_side != 0.0f) || (move_up != 0.0f)) {
      const float3 forward = normalize_safe(_camera.direction, kWorldForward);
      float3 right = cross(forward, kWorldUp);
      if (length(right) < kEpsilon) {
        right = _camera.side;
      }
      right = normalize_safe(right, kWorldRight);

      float3 move_direction = move_fwd * forward + move_side * right + move_up * kWorldUp;
      const float dir_len = length(move_direction);
      if (dir_len > 1.0f) {
        move_direction /= dir_len;
      }

      const float distance_ref = _pivot_initialized ? _orbit_distance : max(_camera.focal_distance, 4.0f);
      const float target_move_scale = max(distance_ref, estimate_navigation_scale());

      const float scale_response_up = 20.0f;
      const float scale_response_down = 2.0f;
      const float scale_response = (target_move_scale > _move_scale_distance) ? scale_response_up : scale_response_down;
      const float scale_alpha = (dt_sec > 0.0f) ? (1.0f - expf(-scale_response * dt_sec)) : 1.0f;
      _move_scale_distance += (target_move_scale - _move_scale_distance) * scale_alpha;

      float speed = _base_move_speed * max(1.0f, _move_scale_distance * 0.25f);
      if (shift_pressed && (ctrl_pressed == false)) {
        speed *= _fast_move_multiplier;
      } else if (ctrl_pressed && (shift_pressed == false)) {
        speed *= _slow_move_multiplier;
      }

      target_velocity = move_direction * speed;
    }

    float alpha = enable_inertia ? ((dt_sec > 0.0f) ? (1.0f - expf(-_movement_response * dt_sec)) : 0.0f) : 1.0f;
    _move_velocity += (target_velocity - _move_velocity) * alpha;

    if (length(_move_velocity) < _min_velocity) {
      _move_velocity = {};
      return false;
    }

    const float3 delta = _move_velocity * dt_sec;
    if (length(delta) < _min_translation) {
      return false;
    }

    _camera.position += delta;
    if (_pivot_initialized) {
      _orbit_pivot += delta;
    }
    return true;
  }

 private:
  enum : uint32_t {
    MouseLeft = 1u << 0u,
    MouseMiddle = 1u << 1u,
    MouseRight = 1u << 2u,
  };

  Camera& _camera;
  bool _keys[512] = {};
  float3 _mouse_delta = {};
  uint32_t _mouse_buttons = 0;
  float3 _move_velocity = {};
  float3 _orbit_velocity = {};
  float3 _look_velocity = {};
  float3 _pan_velocity = {};
  float _dolly_velocity = 0.0f;
  float _zoom_velocity = 0.0f;
  float3 _orbit_pivot = {};
  float _orbit_distance = 4.0f;
  float _move_scale_distance = 4.0f;
  bool _pivot_initialized = false;

  float3 _interpolated_position = {};
  float3 _interpolated_direction = {};
  bool _interpolation_initialized = false;

  float _base_move_speed = 2.5f;
  float _fast_move_multiplier = 3.0f;
  float _slow_move_multiplier = 0.2f;
  float _movement_response = 18.0f;
  float _min_velocity = 1.0e-4f;
  float _min_translation = 1.0e-6f;

  float _look_speed = 0.0025f;
  float _orbit_speed = 0.0030f;
  float _pan_speed = 0.0030f;
  float _drag_dolly_speed = 0.015f;
  float _scroll_zoom_speed = 0.8f;

  struct {
    float3 pos;
    float3 center;
    bool active = false;
  } _scheduled = {};
};

}  // namespace etx
