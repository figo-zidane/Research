add_library(core STATIC
  core/Log.cpp
)

target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(core PUBLIC fmt::fmt spdlog::spdlog)

find_package(volk CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

add_library(rhi STATIC
  rhi/vma_impl.cpp
  rhi/Device.cpp
  rhi/Swapchain.cpp
  rhi/Frame.cpp
  rhi/CommandBuffer.cpp
  rhi/Buffer.cpp
  rhi/Image.cpp
  rhi/Sampler.cpp
  rhi/BindlessRegistry.cpp
  rhi/Pipeline.cpp
)

target_include_directories(rhi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# volk owns the Vulkan function pointers; Vulkan::Headers gives us the SDK
# headers without linking vulkan-1.lib (volk loads it dynamically).
target_link_libraries(rhi PUBLIC
  core
  volk::volk_headers
  volk::volk
  glfw
  GPUOpen::VulkanMemoryAllocator
  Slang::Slang
)
# VK_NO_PROTOTYPES ensures every translation unit goes through volk's
# function pointers instead of the static loader symbols.
target_compile_definitions(rhi PUBLIC GLFW_INCLUDE_NONE VK_NO_PROTOTYPES)

add_library(shader STATIC
  shader/SlangSession.cpp
  shader/ShaderModule.cpp
  shader/ShaderReflection.cpp
)

target_include_directories(shader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(shader PUBLIC core Slang::Slang)

add_library(scene STATIC
  scene/Scene.cpp
)

target_include_directories(scene PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(scene PUBLIC core glm::glm)

add_library(render STATIC
  render/Renderer.cpp
  render/RenderGraph.cpp
)

target_include_directories(render PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(render PUBLIC core rhi scene shader)

# ── ImGui pass ───────────────────────────────────────────────────────────────
# The imgui_impl_vulkan backend is compiled with IMGUI_IMPL_VULKAN_USE_VOLK
# so that it includes volk.h instead of vulkan/vulkan.h, which is required
# because the rest of the project uses VK_NO_PROTOTYPES (set by rhi PUBLIC).
add_library(imgui_pass STATIC
  passes/imgui/ImGuiPass.cpp
  passes/imgui/imgui_impl_vulkan.cpp
  passes/imgui/imgui_impl_glfw.cpp
)

target_include_directories(imgui_pass PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(imgui_pass PUBLIC IMGUI_IMPL_VULKAN_USE_VOLK)
target_link_libraries(imgui_pass PUBLIC render rhi imgui::imgui glfw)

# ── Application / entry point ────────────────────────────────────────────────
add_library(app STATIC
  app/Application.cpp
  app/EditorUI.cpp
)

target_include_directories(app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(app PUBLIC core render rhi imgui_pass glfw imgui::imgui)

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

# Copy slang.dll next to the executable so runtime shader compilation works.
add_custom_command(
  TARGET research-renderer POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${SLANG_DLL}"
          $<TARGET_FILE_DIR:research-renderer>
)
