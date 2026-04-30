#pragma once

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
    //   - Stats overlay (FPS / frame time)
    //   - Each pass's renderUI()
    //   - ImGui Demo Window (Phase 3 validation helper)
    void build(const rr::render::Renderer& renderer, float delta_time_seconds);
};
}
