if(NOT DEFINED APP_BUNDLE OR NOT IS_DIRECTORY "${APP_BUNDLE}")
  message(FATAL_ERROR "APP_BUNDLE must name an existing macOS app bundle")
endif()

if(NOT DEFINED SIGNING_IDENTITY OR SIGNING_IDENTITY STREQUAL "")
  message(FATAL_ERROR "SIGNING_IDENTITY must not be empty")
endif()

file(GLOB_RECURSE bundled_libraries "${APP_BUNDLE}/Contents/Frameworks/*.dylib")
foreach(library IN LISTS bundled_libraries)
  if(NOT IS_SYMLINK "${library}")
    execute_process(
      COMMAND /usr/bin/codesign --force --sign "${SIGNING_IDENTITY}" --timestamp=none "${library}"
      RESULT_VARIABLE sign_result
    )
    if(NOT sign_result EQUAL 0)
      message(FATAL_ERROR "Failed to code-sign ${library}")
    endif()
  endif()
endforeach()

execute_process(
  COMMAND /usr/bin/codesign --force --sign "${SIGNING_IDENTITY}" --timestamp=none "${APP_BUNDLE}"
  RESULT_VARIABLE sign_result
)
if(NOT sign_result EQUAL 0)
  message(FATAL_ERROR "Failed to code-sign ${APP_BUNDLE}")
endif()
