add_library(core STATIC
  core/Log.cpp
  core/FileWatcher.cpp
)

target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(core PUBLIC fmt::fmt spdlog::spdlog efsw::efsw)

find_package(volk CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

add_library(rhi STATIC
  rhi/vma_impl.cpp
  rhi/Device.cpp
  rhi/PlatformVulkan.cpp
  rhi/Swapchain.cpp
  rhi/Surface.cpp
  rhi/Frame.cpp
  rhi/CommandBuffer.cpp
  rhi/CommandRecorder.cpp
  rhi/Buffer.cpp
  rhi/Image.cpp
  rhi/Readback.cpp
  rhi/Sampler.cpp
  rhi/BindlessRegistry.cpp
  rhi/Pipeline.cpp
  rhi/AccelStructure.cpp
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
  shader/HotReload.cpp
)

target_include_directories(shader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(shader PUBLIC core Slang::Slang imgui::imgui)

add_library(scene STATIC
  scene/Scene.cpp
  scene/Camera.cpp
  scene/GltfLoader.cpp
)

target_include_directories(scene PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(scene PUBLIC core rhi glm::glm)

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

# ── GBuffer pass ──────────────────────────────────────────────────────────────
add_library(gbuffer_pass STATIC
  passes/gbuffer/GBufferPass.cpp
)
target_include_directories(gbuffer_pass PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gbuffer_pass PUBLIC render rhi imgui::imgui)

# ── PathTracer pass ───────────────────────────────────────────────────────────
add_library(pathtracer_pass STATIC
  passes/pathtracer_reference/PathTracerPass.cpp
)
target_include_directories(pathtracer_pass PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pathtracer_pass PUBLIC render rhi imgui::imgui)

# ── Accumulate pass ───────────────────────────────────────────────────────────
add_library(accumulate_pass STATIC
  passes/accumulate/AccumulatePass.cpp
)
target_include_directories(accumulate_pass PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(accumulate_pass PUBLIC render rhi imgui::imgui)

# ── Tonemap pass ──────────────────────────────────────────────────────────────
add_library(tonemap_pass STATIC
  passes/tonemap/TonemapPass.cpp
)
target_include_directories(tonemap_pass PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tonemap_pass PUBLIC render rhi imgui::imgui)

# ── ReSTIR DI pass ────────────────────────────────────────────────────────────
add_library(restir_di_pass STATIC
  passes/restir_di/ReSTIRDIPass.cpp
)
target_include_directories(restir_di_pass PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(restir_di_pass PUBLIC render rhi imgui::imgui)

# ── ReSTIR GI pass ────────────────────────────────────────────────────────────
add_library(restir_gi_pass STATIC
  passes/restir_gi/ReSTIRGIPass.cpp
)
target_include_directories(restir_gi_pass PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(restir_gi_pass PUBLIC render rhi imgui::imgui)

# ── ReSTIR PT pass ────────────────────────────────────────────────────────────
add_library(restir_pt_pass STATIC
  passes/restir_pt/ReSTIRPTPass.cpp
)
target_include_directories(restir_pt_pass PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(restir_pt_pass PUBLIC render rhi imgui::imgui)

# ── A-trous denoiser pass ─────────────────────────────────────────────────────
add_library(atrous_pass STATIC
  passes/denoise/AtrousPass.cpp
)
target_include_directories(atrous_pass PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(atrous_pass PUBLIC render rhi imgui::imgui)

# ── Application / entry point ────────────────────────────────────────────────
add_library(app STATIC
  app/Application.cpp
  app/EditorUI.cpp
)

target_include_directories(app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(app PUBLIC core render rhi scene shader
  imgui_pass gbuffer_pass pathtracer_pass accumulate_pass tonemap_pass
  restir_di_pass restir_gi_pass restir_pt_pass atrous_pass
  glfw imgui::imgui)

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

# ── Python bindings module ────────────────────────────────────────────────────
# Output is research_renderer.pyd, linking the same `app` static library the
# executable uses so UI and scripting share one code path.
if(RR_BUILD_PYTHON)
  nanobind_add_module(research_renderer_py
    NB_STATIC
    python/bindings.cpp
  )
  set_target_properties(research_renderer_py PROPERTIES OUTPUT_NAME research_renderer)
  target_link_libraries(research_renderer_py PRIVATE app)

  # The module needs assets/ and slang.dll alongside it, just like the exe,
  # so scripts can run from the module's directory.
  add_custom_command(TARGET research_renderer_py POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/assets
            $<TARGET_FILE_DIR:research_renderer_py>/assets)
  add_custom_command(TARGET research_renderer_py POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SLANG_DLL}"
            $<TARGET_FILE_DIR:research_renderer_py>)
endif()
