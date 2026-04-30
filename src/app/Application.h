#pragma once

#include "render/Renderer.h"
#include "rhi/CommandBuffer.h"
#include "rhi/Device.h"
#include "rhi/Frame.h"
#include "rhi/Swapchain.h"
#include "scene/Scene.h"

#include <GLFW/glfw3.h>

#include <string>

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
    void render_frame();
    void recreate_swapchain();

    static void glfw_resize_callback(GLFWwindow* window, int width, int height);

    GLFWwindow* window_ = nullptr;
    rr::rhi::Device device_;
    rr::rhi::Swapchain swapchain_;
    rr::rhi::Frame frame_;
    rr::rhi::CommandBuffer command_buffer_;
    rr::render::Renderer renderer_;
    rr::scene::Scene scene_;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    std::string title_ = "Research Renderer";
    int width_ = 1600;
    int height_ = 900;
    bool glfw_initialized_ = false;
    bool framebuffer_resized_ = false;
};
}
