#
# FindDXC.cmake - Find or download DirectX Shader Compiler (DXC)
#
# This module defines the following variables:
#   DXC_EXECUTABLE      - Path to dxc executable (if found)
#   DXC_LIBRARY         - Path to DXC runtime library (if found)
#   DXC_INCLUDE_DIR     - Path to directory containing dxc/dxcapi.h
#   DXC_ROOT            - Root folder containing include/lib/bin for resolved DXC
#   DXC_FOUND           - True if DXC was found or installed
#   DXC_VERSION         - Version of DXC (if detectable)
#
# And the following targets:
#   DXC::DXC            - Imported target for DXC executable
#   DXC::Compiler       - Imported target for DXC runtime library
#

include(FindPackageHandleStandardArgs)

# Preferred in-repo vendor location.
set(DXC_VENDOR_ROOT "${CMAKE_SOURCE_DIR}/thirdparty/dxc")
set(DXC_VENDOR_PATHS "")

if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(_DXC_VENDOR_ARCH "x64")
    else()
        set(_DXC_VENDOR_ARCH "x86")
    endif()
    list(APPEND DXC_VENDOR_PATHS
        "${DXC_VENDOR_ROOT}/windows-${_DXC_VENDOR_ARCH}"
        "${DXC_VENDOR_ROOT}/windows"
        "${DXC_VENDOR_ROOT}/win-${_DXC_VENDOR_ARCH}"
        "${DXC_VENDOR_ROOT}/win"
    )
elseif(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
        set(_DXC_VENDOR_ARCH "arm64")
    else()
        set(_DXC_VENDOR_ARCH "x64")
    endif()
    list(APPEND DXC_VENDOR_PATHS
        "${DXC_VENDOR_ROOT}/macos-${_DXC_VENDOR_ARCH}"
        "${DXC_VENDOR_ROOT}/macos"
        "${DXC_VENDOR_ROOT}/darwin-${_DXC_VENDOR_ARCH}"
        "${DXC_VENDOR_ROOT}/darwin"
    )
else()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
        set(_DXC_VENDOR_ARCH "x64")
    else()
        set(_DXC_VENDOR_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    list(APPEND DXC_VENDOR_PATHS
        "${DXC_VENDOR_ROOT}/linux-${_DXC_VENDOR_ARCH}"
        "${DXC_VENDOR_ROOT}/linux"
    )
endif()

# Always include generic vendor roots as fallback.
list(APPEND DXC_VENDOR_PATHS
    "${DXC_VENDOR_ROOT}/${CMAKE_SYSTEM_NAME}"
    "${DXC_VENDOR_ROOT}"
)

# Set up search hints for DXC
set(DXC_SEARCH_PATHS
    ${DXC_VENDOR_PATHS}
    "${DXC_PATH}"
    "$ENV{DXC_PATH}"
    "$ENV{VULKAN_SDK}"
    "$ENV{VULKAN_SDK}/Bin"
    "$ENV{VULKAN_SDK}/bin"
    "$ENV{ProgramFiles}/Microsoft DirectX Shader Compiler"
    "$ENV{ProgramFiles\(x86\)}/Microsoft DirectX Shader Compiler"
    "$ENV{ProgramFiles}/dxc"
    "$ENV{ProgramFiles\(x86\)}/dxc"
    "/opt/homebrew/opt/directxshadercompiler"
    "/usr/local/opt/directxshadercompiler"
    "/opt/homebrew"
    "/usr/local"
    "/usr"
    "/opt/dxc"
)

list(REMOVE_DUPLICATES DXC_SEARCH_PATHS)

# If DXC_PATH points to a nested folder (for example bin/), probe parent roots too.
if(DEFINED DXC_PATH AND NOT "${DXC_PATH}" STREQUAL "")
    list(APPEND DXC_SEARCH_PATHS
        "${DXC_PATH}/.."
        "${DXC_PATH}/../.."
    )
endif()
if(DEFINED ENV{DXC_PATH} AND NOT "$ENV{DXC_PATH}" STREQUAL "")
    list(APPEND DXC_SEARCH_PATHS
        "$ENV{DXC_PATH}/.."
        "$ENV{DXC_PATH}/../.."
    )
endif()

# Set up DLL/library names based on platform
if(WIN32)
    set(DXC_LIBRARY_NAMES dxcompiler.dll)
    set(DXC_EXECUTABLE_NAMES dxc.exe)
elseif(APPLE)
    set(DXC_LIBRARY_NAMES libdxcompiler.dylib)
    set(DXC_EXECUTABLE_NAMES dxc)
else() # Linux
    set(DXC_LIBRARY_NAMES libdxcompiler.so)
    set(DXC_EXECUTABLE_NAMES dxc)
endif()

# Find DXC include directory
find_path(DXC_INCLUDE_DIR
    NAMES dxc/dxcapi.h
    PATHS ${DXC_SEARCH_PATHS}
    PATH_SUFFIXES include Include inc
    DOC "DXC include directory"
)

# Find DXC library
find_file(DXC_LIBRARY
    NAMES ${DXC_LIBRARY_NAMES}
    PATHS ${DXC_SEARCH_PATHS}
    PATH_SUFFIXES "" bin lib lib64
    DOC "DXC compiler library"
)

# Find DXC executable
find_program(DXC_EXECUTABLE
    NAMES ${DXC_EXECUTABLE_NAMES}
    PATHS ${DXC_SEARCH_PATHS}
    PATH_SUFFIXES "" bin
    DOC "DXC compiler executable"
)

# Check if we found DXC
if(DXC_LIBRARY AND DXC_INCLUDE_DIR)
    set(DXC_FOUND TRUE)
    set(DXC_ROOT "")
    get_filename_component(_DXC_INCLUDE_DIR_NAME "${DXC_INCLUDE_DIR}" NAME)
    if(_DXC_INCLUDE_DIR_NAME STREQUAL "include" OR _DXC_INCLUDE_DIR_NAME STREQUAL "Include" OR _DXC_INCLUDE_DIR_NAME STREQUAL "inc")
        get_filename_component(DXC_ROOT "${DXC_INCLUDE_DIR}" DIRECTORY)
    endif()

    if(NOT DXC_ROOT)
        get_filename_component(_DXC_LIBRARY_DIR "${DXC_LIBRARY}" DIRECTORY)
        get_filename_component(_DXC_LIBRARY_DIR_NAME "${_DXC_LIBRARY_DIR}" NAME)
        if(_DXC_LIBRARY_DIR_NAME STREQUAL "lib" OR _DXC_LIBRARY_DIR_NAME STREQUAL "lib64" OR _DXC_LIBRARY_DIR_NAME STREQUAL "bin")
            get_filename_component(DXC_ROOT "${_DXC_LIBRARY_DIR}" DIRECTORY)
        else()
            set(DXC_ROOT "${_DXC_LIBRARY_DIR}")
        endif()
    endif()

    if(DXC_EXECUTABLE)
        message(STATUS "Found DXC: ${DXC_EXECUTABLE} (${DXC_LIBRARY})")
    else()
        message(STATUS "Found DXC runtime library: ${DXC_LIBRARY}")
    endif()
    if(DXC_ROOT)
        message(STATUS "DXC root: ${DXC_ROOT}")
    endif()
else()
    set(DXC_FOUND FALSE)
    message(STATUS "DXC not found locally, will attempt fallback installation")
endif()

# Function to download and install DXC
function(install_dxc)
    message(STATUS "Attempting DXC fallback installation...")

    set(DXC_VERSION "v1.7.2308")
    set(DXC_BASE_URL "https://github.com/microsoft/DirectXShaderCompiler/releases/download/${DXC_VERSION}")

    if(WIN32)
        set(DXC_ARCHIVE "dxc_${DXC_VERSION}_windows_amd64.zip")
        set(DXC_EXTRACT_DIR "${CMAKE_BINARY_DIR}/dxc")
        set(DXC_INSTALL_DIR "${CMAKE_BINARY_DIR}/dxc")

        # Download DXC
        file(DOWNLOAD
            "${DXC_BASE_URL}/${DXC_ARCHIVE}"
            "${CMAKE_BINARY_DIR}/${DXC_ARCHIVE}"
            SHOW_PROGRESS
            STATUS download_status
        )

        list(GET download_status 0 download_result)
        if(NOT download_result EQUAL 0)
            message(FATAL_ERROR "Failed to download DXC: ${download_status}")
        endif()

        # Extract DXC
        file(ARCHIVE_EXTRACT
            INPUT "${CMAKE_BINARY_DIR}/${DXC_ARCHIVE}"
            DESTINATION "${DXC_EXTRACT_DIR}"
        )

        # Find and copy runtime libraries (layout differs across release archives)
        file(GLOB_RECURSE DXC_LIBRARY_FILES
            "${DXC_EXTRACT_DIR}/dxcompiler.dll"
            "${DXC_EXTRACT_DIR}/bin/dxcompiler.dll"
            "${DXC_EXTRACT_DIR}/bin/x64/dxcompiler.dll"
        )
        if(DXC_LIBRARY_FILES)
            list(GET DXC_LIBRARY_FILES 0 DXC_LIBRARY_FILE)
            configure_file("${DXC_LIBRARY_FILE}" "${DXC_INSTALL_DIR}/dxcompiler.dll" COPYONLY)
            set(DXC_LIBRARY "${DXC_INSTALL_DIR}/dxcompiler.dll" PARENT_SCOPE)
        else()
            message(FATAL_ERROR "dxcompiler.dll not found in extracted DXC archive")
        endif()

        file(GLOB_RECURSE DXIL_LIBRARY_FILES
            "${DXC_EXTRACT_DIR}/dxil.dll"
            "${DXC_EXTRACT_DIR}/bin/dxil.dll"
            "${DXC_EXTRACT_DIR}/bin/x64/dxil.dll"
        )
        if(DXIL_LIBRARY_FILES)
            list(GET DXIL_LIBRARY_FILES 0 DXIL_LIBRARY_FILE)
            configure_file("${DXIL_LIBRARY_FILE}" "${DXC_INSTALL_DIR}/dxil.dll" COPYONLY)
        endif()

        find_program(DXC_DOWNLOADED_EXECUTABLE
            NAMES dxc.exe
            PATHS "${DXC_EXTRACT_DIR}"
            PATH_SUFFIXES bin bin/x64
            NO_DEFAULT_PATH
        )
        if(DXC_DOWNLOADED_EXECUTABLE)
            set(DXC_EXECUTABLE "${DXC_DOWNLOADED_EXECUTABLE}" PARENT_SCOPE)
        else()
            set(DXC_EXECUTABLE "" PARENT_SCOPE)
        endif()

        find_path(DXC_DOWNLOADED_INCLUDE_DIR
            NAMES dxc/dxcapi.h
            PATHS "${DXC_EXTRACT_DIR}"
            PATH_SUFFIXES include inc
            NO_DEFAULT_PATH
        )
        if(DXC_DOWNLOADED_INCLUDE_DIR)
            set(DXC_INCLUDE_DIR "${DXC_DOWNLOADED_INCLUDE_DIR}" PARENT_SCOPE)
        else()
            message(FATAL_ERROR "dxc/dxcapi.h not found in extracted DXC archive")
        endif()

    elseif(APPLE)
        message(WARNING "DXC fallback for macOS is not implemented. Install DXC manually (build from source or use a prebuilt package) and set DXC_PATH.")
        return()
    else() # Linux
        message(WARNING "DXC fallback for Linux is not implemented. Please install DXC manually and set DXC_PATH if needed.")
        return()
    endif()

    message(STATUS "DXC installed to: ${DXC_INSTALL_DIR}")
endfunction()

# If DXC not found, try to install it
if(NOT DXC_FOUND)
    install_dxc()

    if(EXISTS "${DXC_LIBRARY}" AND EXISTS "${DXC_INCLUDE_DIR}/dxc/dxcapi.h")
        set(DXC_FOUND TRUE)
        message(STATUS "DXC installation successful")
    else()
        message(FATAL_ERROR "DXC resolution failed. Install DXC and ensure both runtime library and dxc/dxcapi.h are visible (set DXC_PATH if needed).")
    endif()
endif()

# Compute DXC root for downstream copy/package logic.
if(DXC_FOUND AND NOT DXC_ROOT)
    get_filename_component(_DXC_INCLUDE_DIR_NAME "${DXC_INCLUDE_DIR}" NAME)
    if(_DXC_INCLUDE_DIR_NAME STREQUAL "include" OR _DXC_INCLUDE_DIR_NAME STREQUAL "Include" OR _DXC_INCLUDE_DIR_NAME STREQUAL "inc")
        get_filename_component(DXC_ROOT "${DXC_INCLUDE_DIR}" DIRECTORY)
    endif()

    if(NOT DXC_ROOT)
        get_filename_component(_DXC_LIBRARY_DIR "${DXC_LIBRARY}" DIRECTORY)
        get_filename_component(_DXC_LIBRARY_DIR_NAME "${_DXC_LIBRARY_DIR}" NAME)
        if(_DXC_LIBRARY_DIR_NAME STREQUAL "lib" OR _DXC_LIBRARY_DIR_NAME STREQUAL "lib64" OR _DXC_LIBRARY_DIR_NAME STREQUAL "bin")
            get_filename_component(DXC_ROOT "${_DXC_LIBRARY_DIR}" DIRECTORY)
        else()
            set(DXC_ROOT "${_DXC_LIBRARY_DIR}")
        endif()
    endif()
endif()

# Create imported targets
if(DXC_FOUND)
    if(NOT TARGET DXC::Compiler)
        add_library(DXC::Compiler SHARED IMPORTED)
        set_target_properties(DXC::Compiler PROPERTIES
            IMPORTED_LOCATION "${DXC_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${DXC_INCLUDE_DIR}"
        )
    endif()

    if(NOT TARGET DXC::DXC AND DXC_EXECUTABLE)
        add_executable(DXC::DXC IMPORTED)
        set_target_properties(DXC::DXC PROPERTIES
            IMPORTED_LOCATION "${DXC_EXECUTABLE}"
        )
    endif()
endif()

# Handle standard arguments
find_package_handle_standard_args(DXC
    REQUIRED_VARS DXC_LIBRARY DXC_INCLUDE_DIR
    VERSION_VAR DXC_VERSION
)
