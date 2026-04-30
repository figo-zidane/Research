add_library(core STATIC
  core/Log.cpp
)

target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(core PUBLIC fmt::fmt spdlog::spdlog)

find_package(volk CONFIG REQUIRED)

add_library(rhi STATIC
  rhi/Device.cpp
  rhi/Swapchain.cpp
  rhi/Frame.cpp
  rhi/CommandBuffer.cpp
)

target_include_directories(rhi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# volk owns the Vulkan function pointers; Vulkan::Headers gives us the SDK
# headers without linking vulkan-1.lib (volk loads it dynamically).
target_link_libraries(rhi PUBLIC core volk::volk_headers volk::volk glfw)
# VK_NO_PROTOTYPES ensures every translation unit goes through volk's
# function pointers instead of the static loader symbols.
target_compile_definitions(rhi PUBLIC GLFW_INCLUDE_NONE VK_NO_PROTOTYPES)

add_library(scene STATIC
  scene/Scene.cpp
)

target_include_directories(scene PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(scene PUBLIC core glm::glm)

add_library(render STATIC
  render/Renderer.cpp
)

target_include_directories(render PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(render PUBLIC core rhi scene)

add_library(app STATIC
  app/Application.cpp
)

target_include_directories(app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(app PUBLIC core render rhi glfw)

add_executable(research-renderer
  main.cpp
)

target_link_libraries(research-renderer PRIVATE app)

if(RR_ENABLE_VALIDATION)
  target_compile_definitions(rhi PRIVATE RR_ENABLE_VALIDATION=1)
else()
  target_compile_definitions(rhi PRIVATE RR_ENABLE_VALIDATION=0)
endif()

add_custom_command(
  TARGET research-renderer POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_SOURCE_DIR}/assets
          $<TARGET_FILE_DIR:research-renderer>/assets
)
