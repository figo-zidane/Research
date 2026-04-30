#include "app/EditorUI.h"

#include "render/RenderGraph.h"
#include "render/Renderer.h"

#include <imgui.h>

namespace rr::app
{
void EditorUI::build(const rr::render::Renderer& renderer, float delta_time_seconds)
{
    // ── Stats overlay ────────────────────────────────────────────────────
    constexpr ImGuiWindowFlags kOverlayFlags =
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize   |
        ImGuiWindowFlags_NoMove     |
        ImGuiWindowFlags_NoSavedSettings;

    ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(230.0f, 60.0f), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.6f);
    if (ImGui::Begin("##Stats", nullptr, kOverlayFlags))
    {
        const float fps = (delta_time_seconds > 0.0f) ? (1.0f / delta_time_seconds) : 0.0f;
        ImGui::Text("FPS : %.1f", fps);
        ImGui::Text("dt  : %.3f ms", delta_time_seconds * 1000.0f);
    }
    ImGui::End();

    // ── Per-pass panels ──────────────────────────────────────────────────
    for (const auto& pass : renderer.graph().passes())
    {
        pass->render_ui();
    }

}
}
