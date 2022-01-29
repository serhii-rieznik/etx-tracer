add_library(sokol_gfx STATIC 
  sokol_gfx.h 
  sokol_gfx.c
)
set_target_properties(sokol_gfx PROPERTIES FOLDER "thirdparty")
target_include_directories(sokol_gfx PUBLIC ${CMAKE_CURRENT_LIST_DIR})
