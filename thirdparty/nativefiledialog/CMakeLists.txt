add_library(nativefiledialog STATIC
  src/include/nfd.h
)

if(WIN32)
  target_sources(nativefiledialog PRIVATE src/nfd_win.cpp)
elseif(APPLE)
  target_sources(nativefiledialog PRIVATE src/nfd_cocoa.m)

  find_library(NFD_APPKIT_FRAMEWORK AppKit REQUIRED)
  find_library(NFD_UNIFORM_TYPE_IDENTIFIERS_FRAMEWORK UniformTypeIdentifiers REQUIRED)
  target_compile_definitions(nativefiledialog PRIVATE NFD_MACOS_ALLOWEDCONTENTTYPES=1)
  target_link_libraries(nativefiledialog PRIVATE
    "${NFD_APPKIT_FRAMEWORK}"
    "${NFD_UNIFORM_TYPE_IDENTIFIERS_FRAMEWORK}"
  )
else()
  message(FATAL_ERROR "Native File Dialog Extended is configured only for Windows and macOS")
endif()

target_include_directories(nativefiledialog PUBLIC "${CMAKE_CURRENT_LIST_DIR}/src/include")
set_target_properties(nativefiledialog PROPERTIES FOLDER "thirdparty")
