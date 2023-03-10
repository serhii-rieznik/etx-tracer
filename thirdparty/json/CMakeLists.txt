add_library(json INTERFACE json.hpp)
target_include_directories(json INTERFACE ./)
set_target_properties(json PROPERTIES FOLDER "thirdparty")
