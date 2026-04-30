#pragma once

#include "app/EditorUI.h"
#include "render/Renderer.h"
#include "rhi/BindlessRegistry.h"
#include "rhi/CommandBuffer.h"
#include "rhi/Device.h"
#include "rhi/Frame.h"
#include "rhi/Swapchain.h"
#include "scene/Scene.h"

#include <GLFW/glfw3.h>

#include <chrono>
#include <string>

namespace rr::passes::imgui
{
class ImGuiPass;
}

namespace rr::app
{
class Application
{
public:
    Application();
    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
    Application(Application&&) = delete;
    Application& operator=(Application&&) = delete;
    ~Application();

    void run();

private:
    void initialize();
    void shutdown();
    void main_loop();
    void initialize_window();
    void initialize_vulkan();
    void initialize_renderer();
    void render_frame();
    void recreate_swapchain();

    static void glfw_resize_callback(GLFWwindow* window, int width, int height);

    GLFWwindow*               window_   = nullptr;
    rr::rhi::Device           device_;
    rr::rhi::Swapchain        swapchain_;
    rr::rhi::Frame            frame_;
    rr::rhi::CommandBuffer    command_buffer_;
    rr::rhi::BindlessRegistry bindless_registry_;
    rr::render::Renderer      renderer_;
    rr::scene::Scene          scene_;
    rr::app::EditorUI         editor_ui_;
    // Non-owning raw pointer to the ImGuiPass registered with the Renderer.
    // Lifetime is managed by the Renderer's RenderGraph.
    rr::passes::imgui::ImGuiPass* imgui_pass_ = nullptr;

    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    std::string  title_   = "Research Renderer";
    int          width_   = 1600;
    int          height_  = 900;
    bool         glfw_initialized_   = false;
    bool         framebuffer_resized_ = false;

    std::chrono::steady_clock::time_point last_frame_time_{};
    float delta_time_seconds_ = 0.0f;
};
}
