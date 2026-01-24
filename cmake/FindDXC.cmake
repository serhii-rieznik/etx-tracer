#
# FindDXC.cmake - Find or download DirectX Shader Compiler (DXC)
#
# This module defines the following variables:
#   DXC_EXECUTABLE     - Path to dxc.exe (if found)
#   DXC_LIBRARY        - Path to dxcompiler.dll/lib (if found)
#   DXC_FOUND          - True if DXC was found or downloaded
#   DXC_VERSION        - Version of DXC (if detectable)
#
# And the following targets:
#   DXC::DXC           - Imported target for DXC executable
#   DXC::Compiler      - Imported target for DXC library
#

include(FindPackageHandleStandardArgs)

# Set up search paths for DXC
set(DXC_SEARCH_PATHS
    "$ENV{ProgramFiles}/Microsoft DirectX Shader Compiler"
    "$ENV{ProgramFiles\(x86\)}/Microsoft DirectX Shader Compiler"
    "$ENV{ProgramFiles}/dxc"
    "$ENV{ProgramFiles\(x86\)}/dxc"
    "$ENV{DXC_PATH}"
    "$ENV{VULKAN_SDK}/Bin"
    "/usr/local/bin"
    "/usr/bin"
    "/opt/dxc/bin"
)

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

# Find DXC library
find_file(DXC_LIBRARY
    NAMES ${DXC_LIBRARY_NAMES}
    PATHS ${DXC_SEARCH_PATHS}
    PATH_SUFFIXES bin lib
    DOC "DXC compiler library"
)

# Find DXC executable
find_program(DXC_EXECUTABLE
    NAMES ${DXC_EXECUTABLE_NAMES}
    PATHS ${DXC_SEARCH_PATHS}
    PATH_SUFFIXES bin
    DOC "DXC compiler executable"
)

# Check if we found DXC
if(DXC_LIBRARY AND DXC_EXECUTABLE)
    set(DXC_FOUND TRUE)
    message(STATUS "Found DXC: ${DXC_EXECUTABLE}")
else()
    set(DXC_FOUND FALSE)
    message(STATUS "DXC not found locally, will attempt to download")
endif()

# Function to download and install DXC
function(install_dxc)
    message(STATUS "Installing DXC...")

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

        # Find and copy only the library file (DLL)
        file(GLOB DXC_LIBRARY_FILES "${DXC_EXTRACT_DIR}/bin/dxcompiler.dll")
        if(DXC_LIBRARY_FILES)
            configure_file("${DXC_LIBRARY_FILES}" "${DXC_INSTALL_DIR}/dxcompiler.dll" COPYONLY)
            set(DXC_LIBRARY "${DXC_INSTALL_DIR}/dxcompiler.dll" PARENT_SCOPE)
        else()
            message(FATAL_ERROR "dxcompiler.dll not found in extracted DXC archive")
        endif()

        # We don't need the executable for runtime compilation
        set(DXC_EXECUTABLE "" PARENT_SCOPE)

    elseif(APPLE)
        message(WARNING "DXC installation for macOS not implemented yet. Please install manually.")
        return()
    else() # Linux
        message(WARNING "DXC installation for Linux not implemented yet. Please install manually.")
        return()
    endif()

    message(STATUS "DXC installed to: ${DXC_INSTALL_DIR}")
endfunction()

# If DXC not found, try to install it
if(NOT DXC_FOUND)
    install_dxc()

    if(EXISTS "${DXC_LIBRARY}" AND EXISTS "${DXC_EXECUTABLE}")
        set(DXC_FOUND TRUE)
        message(STATUS "DXC installation successful")
    else()
        message(FATAL_ERROR "DXC installation failed. Please install DXC manually from: https://github.com/microsoft/DirectXShaderCompiler/releases")
    endif()
endif()

# Create imported targets
if(DXC_FOUND)
    if(NOT TARGET DXC::Compiler)
        add_library(DXC::Compiler SHARED IMPORTED)
        set_target_properties(DXC::Compiler PROPERTIES
            IMPORTED_LOCATION "${DXC_LIBRARY}"
            IMPORTED_IMPLIB "${DXC_LIBRARY}"
        )
    endif()

    if(NOT TARGET DXC::DXC)
        add_executable(DXC::DXC IMPORTED)
        set_target_properties(DXC::DXC PROPERTIES
            IMPORTED_LOCATION "${DXC_EXECUTABLE}"
        )
    endif()
endif()

# Handle standard arguments
find_package_handle_standard_args(DXC
    REQUIRED_VARS DXC_LIBRARY
    VERSION_VAR DXC_VERSION
)