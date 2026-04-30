#pragma once

#include "render/RenderPass.h"

struct GLFWwindow;

namespace rr::rhi
{
class Device;
class Swapchain;
}

namespace rr::passes::imgui
{
// ImGuiPass integrates Dear ImGui (GLFW + Vulkan dynamic-rendering backends)
// as the last render pass in the graph.  It owns the ImGui context lifetime
// and issues all Vulkan draw commands inside execute().
//
// Usage:
//   1. Call initialize() once after the device and swapchain are up.
//   2. Call new_frame() at the start of each frame (before cmd recording).
//   3. Build UI content (EditorUI::build, or any ImGui:: call).
//   4. renderer_.render(ctx) — execute() finalises and submits the draw data.
//   5. Call shutdown() before destroying the device.
class ImGuiPass final : public rr::render::RenderPass
{
public:
    ImGuiPass() = default;
    ~ImGuiPass() override;

    ImGuiPass(const ImGuiPass&) = delete;
    ImGuiPass& operator=(const ImGuiPass&) = delete;

    // ── Lifecycle ────────────────────────────────────────────────────────

    // Initialise ImGui context and backends.  Must be called before the first
    // frame; device and swapchain must already be live.
    void initialize(rr::rhi::Device&          device,
                    GLFWwindow*               window,
                    const rr::rhi::Swapchain& swapchain);

    // Tear down ImGui backends and destroy the context.  Must be called while
    // the device is still alive (before Device::shutdown()).
    void shutdown();

    // ── Per-frame ────────────────────────────────────────────────────────

    // Drive ImGui's per-frame state machine.  Call once per frame, BEFORE
    // starting command buffer recording.
    void new_frame();

    // ── RenderPass interface ─────────────────────────────────────────────
    [[nodiscard]] const char* name() const override { return "ImGui"; }
    [[nodiscard]] Reflection  reflect() const override;
    void on_resize(VkExtent2D new_extent) override;
    // Update ImGui's minimum image count to match the current swapchain.
    // Call this from Application::recreate_swapchain() after swapchain recreation.
    void set_min_image_count(uint32_t count);
    void execute(rr::render::FrameContext& frame_context) override;

private:
    bool      initialized_  = false;
    VkFormat  color_format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{};
};
}
