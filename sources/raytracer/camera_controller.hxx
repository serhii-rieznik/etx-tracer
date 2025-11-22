#include <sokol_app.h>

#include <etx/render/shared/camera.hxx>

#include <unordered_set>

namespace etx {

struct CameraController {
  static constexpr float kMaxCameraDistance = 8192.0f;

  CameraController(Camera& cam)
    : _camera(cam) {
  }

  bool update(double dt) {
    float move_fwd = float(_keys.count(SAPP_KEYCODE_W)) - float(_keys.count(SAPP_KEYCODE_S));
    float move_side = float(_keys.count(SAPP_KEYCODE_D)) - float(_keys.count(SAPP_KEYCODE_A));

    bool movement = (move_fwd != 0.0f) || (move_side != 0.0f);
    bool rotation = (mouse_buttons != 0) && ((_mouse_delta.x != 0.0f) || (_mouse_delta.y != 0.0f));
    bool zoom = (_mouse_delta.z != 0.0f);

    if (rotation) {
      if (mouse_buttons & MouseLeft) {
        float3 target = _camera.target();
        auto s = to_spherical(target - _camera.position);
        s.phi += _rotation_speed * (_mouse_delta.x * kPi / 180.0f);
        s.theta = clamp(s.theta - _rotation_speed * (_mouse_delta.y * kDoublePi / 180.0f), -kHalfPi + kPi / 180.0f, kHalfPi - kPi / 180.0f);
        target = _camera.position + from_spherical(s);
        _camera.direction = normalize(target - _camera.position);
      } else if (mouse_buttons & MouseMiddle) {
        if (_keys.count(SAPP_KEYCODE_LEFT_SHIFT)) {
          float3 direction = _camera.direction;
          float3 side = normalize(cross(kWorldUp, direction));
          float3 up = normalize(cross(direction, side));
          _camera.position += (_mouse_delta.y * up + _mouse_delta.x * side) * _move_speed * (1.0f + length(direction));
          // direction unchanged
        } else if (_keys.count(SAPP_KEYCODE_LEFT_CONTROL)) {
          float3 target = _camera.target();
          auto s = to_spherical(_camera.position - target);
          s.r = clamp(s.r + _mouse_delta.y / kPi, 1.0f / 255.0f, kMaxCameraDistance);
          _camera.position = target + from_spherical(s);
          _camera.direction = normalize(target - _camera.position);
        } else {
          float3 target = _camera.target();
          auto s = to_spherical(_camera.position - target);
          s.phi += _rotation_speed * (_mouse_delta.x * kPi / 180.0f);
          s.theta = clamp(s.theta + _rotation_speed * (_mouse_delta.y * kPi / 180.0f), -kHalfPi + kPi / 180.0f, kHalfPi - kPi / 180.0f);
          _camera.position = target + from_spherical(s);
          _camera.direction = normalize(target - _camera.position);
        }
      }

      _mouse_delta = {};
    }

    if (zoom) {
      float3 target = _camera.target();
      auto s = to_spherical(_camera.position - target);
      s.r = clamp(s.r + _mouse_delta.z * (1.0f + s.r), 1.0f / 255.0f, kMaxCameraDistance);
      _camera.position = target + from_spherical(s);
      _camera.direction = normalize(target - _camera.position);
      _mouse_delta.z = 0.0f;
    }

    if (movement) {
      float3 direction = _camera.direction;
      float3 side = cross(direction, kWorldUp);
      _camera.position += (move_fwd * direction + move_side * side) * _move_speed;
      // direction unchanged, target moves with position
    }

    if (scheduled.active) {
      scheduled.active = false;
      _camera.position = scheduled.pos;
      _camera.direction = normalize(scheduled.center - scheduled.pos);
      movement = true;
    }

    if (movement || rotation || zoom) {
      build_camera(_camera, _camera.position, _camera.direction, kWorldUp, _camera.film_size, get_camera_fov(_camera));
      return true;
    }

    return false;
  }

  void handle_scroll(float scroll) {
#if (ETX_PLATFORM_APPLE)
    float kScrollScaleFactor = -1.0f / 256.0f;
#else
    float kScrollScaleFactor = 1.0f / 256.0f;
#endif
    _mouse_delta.z = kScrollScaleFactor * scroll;
  }

  void handle_event(const sapp_event* e) {
    switch (e->type) {
      case SAPP_EVENTTYPE_MOUSE_SCROLL: {
        handle_scroll(e->scroll_y);
        break;
      }

      case SAPP_EVENTTYPE_KEY_DOWN: {
        _keys.insert(e->key_code);
        break;
      }

      case SAPP_EVENTTYPE_KEY_UP: {
        _keys.erase(e->key_code);
        break;
      }

      case SAPP_EVENTTYPE_MOUSE_DOWN: {
        _mouse_delta = {};
        mouse_buttons = mouse_buttons                                                         //
                        | MouseLeft * uint32_t(e->mouse_button == SAPP_MOUSEBUTTON_LEFT)      //
                        | MouseMiddle * uint32_t(e->mouse_button == SAPP_MOUSEBUTTON_MIDDLE)  //
                        | MouseRight * uint32_t(e->mouse_button == SAPP_MOUSEBUTTON_RIGHT);
        break;
      }

      case SAPP_EVENTTYPE_MOUSE_UP: {
        if (e->mouse_button == SAPP_MOUSEBUTTON_LEFT)
          mouse_buttons = mouse_buttons & (~MouseLeft);
        if (e->mouse_button == SAPP_MOUSEBUTTON_MIDDLE)
          mouse_buttons = mouse_buttons & (~MouseMiddle);
        if (e->mouse_button == SAPP_MOUSEBUTTON_RIGHT)
          mouse_buttons = mouse_buttons & (~MouseRight);
        break;
      }

      case SAPP_EVENTTYPE_MOUSE_MOVE: {
        _mouse_delta = {e->mouse_dx, e->mouse_dy};
        break;
      }

      default:
        break;
    }
  }

  void schedule(const float3& pos, const float3& view_center) {
    scheduled = {pos, view_center, true};
  }

 protected:
  enum : uint32_t {
    MouseLeft = 1u << 0u,
    MouseMiddle = 1u << 1u,
    MouseRight = 1u << 2u,
  };

  Camera& _camera;
  std::unordered_set<uint32_t> _keys;
  float3 _mouse_delta = {};
  uint32_t mouse_buttons = 0;
  float _move_speed = 1.0f / 100.0f;
  float _rotation_speed = 1.0f / 32.0f;

  struct {
    float3 pos;
    float3 center;
    bool active = false;
  } scheduled = {};
};

}  // namespace etx
