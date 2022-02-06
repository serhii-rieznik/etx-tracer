add_library(mikktspace
  mikktspace.h
  mikktspace.c
)
target_include_directories(mikktspace PUBLIC ${CMAKE_CURRENT_LIST_DIR})
set_target_properties(mikktspace PROPERTIES FOLDER "thirdparty")
