# ImGuizmo

This directory vendors the standalone `ImGuizmo` transform widget at commit
`b796ac3b861afc6e91ca74e4611effd9c9527367` (`ImGuizmo.h` and
`ImGuizmo.cpp`).

Local fixes:

- Correct `SetRect`'s vertical maximum to use the rectangle height.
- Balance the draw-list clip stack when a perspective gizmo is behind the camera.
- Scope gizmo activation to its own window and handle hit tests instead of
  rejecting clicks because an unrelated Dear ImGui item remains active or was
  hovered during the previous frame.
