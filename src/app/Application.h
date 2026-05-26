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
#include <string>
#include <vector>

namespace rr::passes::imgui        { class ImGuiPass;        }
namespace rr::passes::gbuffer      { class GBufferPass;      }
namespace rr::passes::pathtracer   { class PathTracerPass;   }
namespace rr::passes::accumulate   { class AccumulatePass;   }
namespace rr::passes::tonemap      { class TonemapPass;       }
namespace rr::passes::restir_di    { class ReSTIRDIPass;     }
namespace rr::passes::restir_gi    { class ReSTIRGIPass;     }
namespace rr::passes::restir_pt    { class ReSTIRPTPass;     }
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

    // ── Lifecycle ─────────────────────────────────────────────────────────
    // run() keeps the original blocking behaviour for research-renderer.exe.
    void run();

    // Explicit lifecycle for Python control: initialize() once, drive frames
    // with step() while !should_close(), then shutdown() (also called by dtor).
    void initialize();
    void shutdown();

    // Pump events + render exactly one frame. Safe to call after initialize().
    void step();
    [[nodiscard]] bool should_close() const;
    // Convenience: run the step loop on the C++ side until the window closes.
    void run_until_closed();

    // ── Scene ─────────────────────────────────────────────────────────────
    void load_cornell();
    void load_gltf(const std::string& path);
    [[nodiscard]] std::string scene_name() const { return current_scene_name_; }

    // ── Camera ────────────────────────────────────────────────────────────
    // eye/center/up are passed as plain float triples so glm does not leak
    // across the binding ABI.
    void set_camera_look_at(float ex, float ey, float ez,
                            float cx, float cy, float cz,
                            float ux, float uy, float uz);
    [[nodiscard]] std::array<float, 3> camera_position() const;
    [[nodiscard]] std::array<float, 3> camera_forward() const;

    // ── Pass enable/disable ───────────────────────────────────────────────
    // name: "restir_di" | "restir_gi" | "restir_pt" | "denoise".
    // PT vs GI exclusivity is handled by render_frame()'s existing logic.
    void set_pass_enabled(const std::string& name, bool enabled);

    // ── Tonemap ───────────────────────────────────────────────────────────
    void set_exposure(float exposure);
    [[nodiscard]] float exposure() const;

    // ── Input ─────────────────────────────────────────────────────────────
    // When disabled, mouse/WASD no longer override the script-set camera.
    void set_input_enabled(bool enabled) { input_enabled_ = enabled; }
    [[nodiscard]] bool input_enabled() const { return input_enabled_; }

    // ── Screenshot ────────────────────────────────────────────────────────
    // Save the currently displayed image (tonemapped LDR) to a PNG path.
    void capture_screenshot(const std::string& path);

    // ── Window ────────────────────────────────────────────────────────────
    [[nodiscard]] std::array<int, 2> window_size() const;

private:
    void main_loop();
    void initialize_window();
    void initialize_rhi();
    void initialize_renderer();
    void render_frame();
    void recreate_swapchain();
    void pre_transition_persistent_images(rr::rhi::CommandRecorder recorder);

    // Screenshot: save the currently displayed source to PNG.
    // Empty path auto-generates a timestamped filename (UI button path).
    void capture_screenshot_impl(const std::string& path);

    // Resolve the HDR image currently feeding the tonemap output, matching
    // render_frame()'s realtime/accumulate selection. Returns nullptr if none.
    [[nodiscard]] const rr::rhi::Image* current_display_image() const;

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
    // Keep device_ before surface_ so the surface is destroyed first.
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
    rr::passes::restir_pt::ReSTIRPTPass*    restir_pt_pass_  = nullptr;
    rr::passes::denoise::AtrousPass*        atrous_pass_     = nullptr;

    std::string  title_   = "Research Renderer";
    int          width_   = 1600;
    int          height_  = 900;

    // ── Frame state ───────────────────────────────────────────────────────
    bool  framebuffer_resized_ = false;
    bool  glfw_initialized_    = false;
    bool  input_enabled_       = true;
    // initialize() sets this; shutdown() is a no-op unless set, so the explicit
    // shutdown() from Python and the one in ~Application() don't double-free.
    bool  initialized_         = false;

    // Only one Application may exist at a time (GLFW + Vulkan are process-global
    // here). The constructor throws on a second instance; the dtor clears it.
    static inline bool s_instance_alive_ = false;
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
    bool use_pt_           = false;
    bool prev_use_pt_      = false;
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
