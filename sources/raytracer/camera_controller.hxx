#include <sokol_app.h>

#include <etx/render/shared/camera.hxx>

#include <unordered_set>

namespace etx {

struct CameraController {
  CameraController(Camera& cam)
    : _camera(cam) {
  }

  void handle_event(const sapp_event* e) {
    switch (e->type) {
      case SAPP_EVENTTYPE_MOUSE_SCROLL: {
#if (ETX_PLATFORM_APPLE)
        float scale_factor = -1.0f / 256.0f;
#else
        float scale_factor = 1.0f / 256.0f;
#endif
        _move_speed = clamp(_move_speed + e->scroll_y * scale_factor, 1.0f / 1000.0f, 1000.0f);
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
        if (e->mouse_button == SAPP_MOUSEBUTTON_LEFT) {
          _mouse_control = true;
          _mouse_delta = {};
        }
        break;
      }
      case SAPP_EVENTTYPE_MOUSE_UP: {
        if (e->mouse_button == SAPP_MOUSEBUTTON_LEFT) {
          _mouse_control = false;
        }
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

  bool update(double dt) {
    float move_fwd = float(_keys.count(SAPP_KEYCODE_W)) - float(_keys.count(SAPP_KEYCODE_S));
    float move_side = float(_keys.count(SAPP_KEYCODE_D)) - float(_keys.count(SAPP_KEYCODE_A));

    bool changed = (move_fwd != 0.0f) || (move_side != 0.0f);

    float3 direction = _camera.direction;
    float3 side = _camera.side;
    float3 up = {0.0f, 1.0f, 0.0f};

    if (_mouse_control && ((_mouse_delta.x != 0.0f) || (_mouse_delta.y != 0.0f))) {
      auto s = to_spherical(_camera.direction);
      s.phi += _rotation_speed * (_mouse_delta.x * kPi / 180.0f);
      s.theta = clamp(s.theta - _rotation_speed * (_mouse_delta.y * kDoublePi / 180.0f), -kHalfPi + kPi / 180.0f, kHalfPi - kPi / 180.0f);
      direction = from_spherical(s);  // phi_theta_to_direction(pt.x, pt.y);
      side = cross(direction, up);
      changed = true;
      _mouse_delta = {};
    }

    if (changed) {
      _camera.position += (move_fwd * direction + move_side * side) * _move_speed;
      update_camera(_camera, _camera.position, _camera.position + direction, up, _camera.image_size, get_camera_fov(_camera));
    }

    return changed;
  }

 private:
  Camera& _camera;
  std::unordered_set<uint32_t> _keys;
  float2 _mouse_delta = {};
  float _move_speed = 1.0f / 100.0f;
  float _rotation_speed = 1.0f / 32.0f;
  bool _mouse_control = false;
};

}  // namespace etx
