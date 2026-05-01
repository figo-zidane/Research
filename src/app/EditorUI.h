#pragma once

#include <cstdint>

namespace rr::render
{
class Renderer;
}

namespace rr::app
{
// EditorUI builds the ImGui content for each frame.
// Call build() after ImGuiPass::new_frame() and before renderer_.render().
class EditorUI
{
public:
    // Build the full ImGui UI for this frame:
    //   - Stats overlay (FPS / frame time / SPP)
    //   - Each pass's renderUI() inside a collapsible panel
    //   - Screenshot button
    // accumulated_spp:      current accumulated sample count (for display + auto-trigger)
    // screenshot_request:   set to true by this function when the user clicks "Save Screenshot"
    void build(const rr::render::Renderer& renderer,
               float    delta_time_seconds,
               uint32_t accumulated_spp,
               bool&    screenshot_request);
};
}
