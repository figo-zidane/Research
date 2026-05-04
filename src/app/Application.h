#pragma once

#include "app/EditorUI.h"
#include "render/Renderer.h"
#include "rhi/BindlessRegistry.h"
#include "rhi/Buffer.h"
#include "rhi/CommandRecorder.h"
#include "rhi/CommandBuffer.h"
#include "rhi/Device.h"
#include "rhi/Frame.h"
#include "rhi/Surface.h"
#include "rhi/Swapchain.h"
#include "scene/Camera.h"
#include "scene/Scene.h"
#include "shader/HotReload.h"
#include "shader/SlangSession.h"

#include <GLFW/glfw3.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace rr::passes::imgui        { class ImGuiPass;        }
namespace rr::passes::gbuffer      { class GBufferPass;      }
namespace rr::passes::pathtracer   { class PathTracerPass;   }
namespace rr::passes::accumulate   { class AccumulatePass;   }
namespace rr::passes::tonemap      { class TonemapPass;       }
namespace rr::passes::restir_di    { class ReSTIRDIPass;     }
namespace rr::passes::restir_gi    { class ReSTIRGIPass;     }
namespace rr::passes::denoise      { class AtrousPass;       }

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
    void one_time_submit(std::function<void(rr::rhi::CommandRecorder)> fn);

    // Screenshot: save accumulated image to PNG.
    void capture_screenshot();

    // MSE: compare the active realtime output vs accumulated PathTracer on CPU.
    // Reads back a 64×64 center crop from both images.
    void compute_mse();

    // Scene reload helpers.
    void reload_scene_cornell();
    void reload_scene_gltf(const std::string& path);

    // GLFW callbacks
    static void glfw_resize_callback(GLFWwindow* window, int width, int height);
    static void glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void glfw_cursor_pos_callback(GLFWwindow* window, double x, double y);
    static void glfw_scroll_callback(GLFWwindow* window, double xoff, double yoff);
    static void glfw_drop_callback(GLFWwindow* window, int count, const char** paths);

    // ── Core subsystems ───────────────────────────────────────────────────
    GLFWwindow*               window_   = nullptr;
    rr::rhi::Device           device_;
    rr::rhi::Surface          surface_;
    rr::rhi::Swapchain        swapchain_;
    rr::rhi::Frame            frame_;
    rr::rhi::CommandBuffer    command_buffer_;
    rr::rhi::BindlessRegistry bindless_registry_;
    rr::render::Renderer      renderer_;
    rr::scene::Scene          scene_;
    rr::scene::Camera         camera_;
    rr::shader::SlangSession  slang_session_;
    rr::shader::HotReload     hot_reload_;
    rr::app::EditorUI         editor_ui_;

    // ── Non-owning pass pointers (lifetime managed by Renderer) ──────────
    rr::passes::imgui::ImGuiPass*           imgui_pass_      = nullptr;
    rr::passes::gbuffer::GBufferPass*       gbuffer_pass_    = nullptr;
    rr::passes::pathtracer::PathTracerPass* pathtracer_pass_ = nullptr;
    rr::passes::accumulate::AccumulatePass* accumulate_pass_ = nullptr;
    rr::passes::tonemap::TonemapPass*       tonemap_pass_    = nullptr;
    rr::passes::restir_di::ReSTIRDIPass*    restir_di_pass_  = nullptr;
    rr::passes::restir_gi::ReSTIRGIPass*    restir_gi_pass_  = nullptr;
    rr::passes::denoise::AtrousPass*        atrous_pass_     = nullptr;

    std::string  title_   = "Research Renderer";
    int          width_   = 1600;
    int          height_  = 900;

    // ── Frame state ───────────────────────────────────────────────────────
    bool  framebuffer_resized_ = false;
    bool  glfw_initialized_    = false;
    float delta_time_seconds_  = 0.0f;
    std::chrono::steady_clock::time_point last_frame_time_;

    // ── Accumulation / camera state ───────────────────────────────────────
    uint32_t accumulated_spp_   = 0;
    bool     camera_moved_      = false;
    bool     save_screenshot_   = false;

    // ── Display mode ──────────────────────────────────────────────────────
    bool use_di_           = false;
    bool prev_use_di_      = false;
    bool use_gi_           = false;
    bool prev_use_gi_      = false;
    bool use_denoise_      = false;
    bool prev_use_denoise_ = false;
    bool mse_compare_      = false;

    // ── Scene state ────────────────────────────────────────────────────
    std::string current_scene_name_   = "Cornell Box";
    std::string gltf_path_input_;
    bool        load_cornell_request_ = false;
    bool        load_gltf_request_    = false;

    // ── MSE comparison ────────────────────────────────────────────────────
    static constexpr uint32_t    kMseCropSize   = 64;   // 64×64 pixel crop
    static constexpr uint32_t    kMseInterval   = 60;   // frames between updates
    static constexpr uint32_t    kMseHistoryLen = 128;
    uint32_t                     mse_frame_counter_ = 0;
    float                        mse_latest_        = -1.0f;
    std::array<float, kMseHistoryLen> mse_history_{};
    uint32_t                     mse_history_pos_   = 0;
    bool                         mse_auto_update_   = true;
    // Persistent staging buffers for MSE readback (allocated once, reused each update).
    rr::rhi::Buffer              mse_staging_pt_;
    rr::rhi::Buffer              mse_staging_ri_;
};
}
