cmake_minimum_required(VERSION 3.26)

project(ResearchRenderer VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(RR_ENABLE_VALIDATION "Enable Vulkan validation layers when available" ON)

# Vulkan and Slang are both shipped with the LunarG Vulkan SDK.
# We rely on $ENV{VULKAN_SDK} (set by the LunarG installer) rather than the
# vcpkg vulkan-loader/headers ports, because vcpkg's loader build is blocked
# by Device Guard / WDAC on this host.
if(NOT DEFINED ENV{VULKAN_SDK})
  message(FATAL_ERROR
    "VULKAN_SDK environment variable is not set. "
    "Install the LunarG Vulkan SDK and ensure VULKAN_SDK points to its root.")
endif()
set(RR_VULKAN_SDK_ROOT "$ENV{VULKAN_SDK}")

# Hint CMake's FindVulkan toward the SDK explicitly. With the vcpkg toolchain
# in play, default search paths are restricted and Vulkan_LIBRARY is otherwise
# missed even though the headers (and version) are detected.
set(Vulkan_INCLUDE_DIR "${RR_VULKAN_SDK_ROOT}/Include" CACHE PATH "" FORCE)
set(Vulkan_LIBRARY     "${RR_VULKAN_SDK_ROOT}/Lib/vulkan-1.lib" CACHE FILEPATH "" FORCE)

find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(efsw CONFIG REQUIRED)

find_path(SLANG_INCLUDE_DIR
  NAMES slang.h
  HINTS "${RR_VULKAN_SDK_ROOT}/Include/slang"
  NO_DEFAULT_PATH
  REQUIRED)

find_library(SLANG_LIBRARY
  NAMES slang
  HINTS "${RR_VULKAN_SDK_ROOT}/Lib"
  NO_DEFAULT_PATH
  REQUIRED)

find_program(SLANG_DLL
  NAMES slang.dll
  HINTS "${RR_VULKAN_SDK_ROOT}/Bin"
  NO_DEFAULT_PATH
  REQUIRED)

add_library(Slang::Slang SHARED IMPORTED)
set_target_properties(Slang::Slang PROPERTIES
  IMPORTED_LOCATION "${SLANG_DLL}"
  IMPORTED_IMPLIB   "${SLANG_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${SLANG_INCLUDE_DIR}")

message(STATUS "Vulkan SDK : ${RR_VULKAN_SDK_ROOT}")
message(STATUS "Slang lib  : ${SLANG_LIBRARY}")
message(STATUS "Slang dll  : ${SLANG_DLL}")
message(STATUS "Slang inc  : ${SLANG_INCLUDE_DIR}")

add_subdirectory(src)
