cmake_minimum_required(VERSION 3.26)

project(ResearchRenderer VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(RR_ENABLE_VALIDATION "Enable Vulkan validation layers when available" ON)
option(RR_BUILD_PYTHON "Build the research_renderer Python bindings" ON)

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

# ── Python bindings (nanobind) ────────────────────────────────────────────────
# nanobind is fetched via a hardened pip install into ./.python-deps (see
# tools/install-python-deps.ps1 and documents/vcpkg_supply_chain_hardening.md),
# not vcpkg, because the vcpkg nanobind port drags in a from-source CPython build.
# We locate its CMake config by asking the interpreter where the wheel put it.
if(RR_BUILD_PYTHON)
  find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

  # Resolve nanobind's CMake config dir from the project-local install. Setting
  # PYTHONPATH to .python-deps lets `python -m nanobind` work without a global
  # install. Falls back to the default search if the project-local copy is absent.
  set(_rr_python_deps "${CMAKE_SOURCE_DIR}/.python-deps")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${_rr_python_deps}"
            "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE NB_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _rr_nb_result)
  if(NOT _rr_nb_result EQUAL 0)
    message(FATAL_ERROR
      "Could not locate nanobind. Run tools/install-python-deps.ps1 first "
      "(it installs a hash-pinned nanobind into .python-deps).")
  endif()
  message(STATUS "nanobind cmake dir : ${NB_CMAKE_DIR}")
  list(APPEND CMAKE_PREFIX_PATH "${NB_CMAKE_DIR}")
  find_package(nanobind CONFIG REQUIRED)
endif()

add_subdirectory(src)
