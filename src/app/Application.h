#pragma once

#include "app/EditorUI.h"
#include "render/Renderer.h"
#include "rhi/BindlessRegistry.h"
#include "rhi/CommandBuffer.h"
#include "rhi/Device.h"
#include "rhi/Frame.h"
#include "rhi/Swapchain.h"
#include "scene/Camera.h"
#include "scene/Scene.h"
#include "shader/SlangSession.h"

#include <GLFW/glfw3.h>

#include <chrono>
#include <functional>
#include <string>

namespace rr::passes::imgui        { class ImGuiPass;        }
namespace rr::passes::gbuffer      { class GBufferPass;      }
namespace rr::passes::pathtracer   { class PathTracerPass;   }
namespace rr::passes::accumulate   { class AccumulatePass;   }
namespace rr::passes::tonemap      { class TonemapPass;      }

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

    // One-time GPU command submission helper.
    void one_time_submit(std::function<void(VkCommandBuffer)> fn);

    // GLFW callbacks
    static void glfw_resize_callback(GLFWwindow* window, int width, int height);
    static void glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void glfw_cursor_pos_callback(GLFWwindow* window, double x, double y);
    static void glfw_scroll_callback(GLFWwindow* window, double xoff, double yoff);

    GLFWwindow*               window_   = nullptr;
    rr::rhi::Device           device_;
    rr::rhi::Swapchain        swapchain_;
    rr::rhi::Frame            frame_;
    rr::rhi::CommandBuffer    command_buffer_;
    rr::rhi::BindlessRegistry bindless_registry_;
    rr::render::Renderer      renderer_;
    rr::scene::Scene          scene_;
    rr::scene::Camera         camera_;
    rr::shader::SlangSession  slang_session_;
    rr::app::EditorUI         editor_ui_;

    // Non-owning pointers — lifetime managed by Renderer's RenderGraph.
    rr::passes::imgui::ImGuiPass*       imgui_pass_      = nullptr;
    rr::passes::gbuffer::GBufferPass*   gbuffer_pass_    = nullptr;
    rr::passes::pathtracer::PathTracerPass* pathtracer_pass_ = nullptr;
    rr::passes::accumulate::AccumulatePass* accumulate_pass_ = nullptr;
    rr::passes::tonemap::TonemapPass*   tonemap_pass_    = nullptr;

    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    std::string  title_   = "Research Renderer";
    int          width_   = 1600;
    int          height_  = 900;
    bool         glfw_initialized_    = false;
    bool         framebuffer_resized_ = false;
    bool         camera_moved_        = true;  // true on first frame
    uint32_t     accumulated_spp_     = 0;

    std::chrono::steady_clock::time_point last_frame_time_{};
    float delta_time_seconds_ = 0.0f;

    // Screenshot
    bool save_screenshot_  = false;  // request set by EditorUI or auto at 4096 spp
    bool screenshot_saved_ = false;  // prevents repeated auto-save

    void capture_screenshot();
};
}
