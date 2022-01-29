add_executable(raytracer 
  main.cxx
  app.hxx
  app.cxx
  ui.hxx
  ui.cxx
  render.hxx
  render.cxx
)

target_link_libraries(raytracer sokol_imgui etx-core etx-log etx-rt)
