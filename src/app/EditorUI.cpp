#include "app/EditorUI.h"

#include "render/RenderGraph.h"
#include "render/Renderer.h"

#include <imgui.h>

namespace rr::app
{
void EditorUI::build(const rr::render::Renderer& renderer,
                     float    delta_time_seconds,
                     uint32_t accumulated_spp,
                     bool&    screenshot_request)
{
    // ── Stats overlay ────────────────────────────────────────────────────
    constexpr ImGuiWindowFlags kOverlayFlags =
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize   |
        ImGuiWindowFlags_NoMove     |
        ImGuiWindowFlags_NoSavedSettings;

    ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(260.0f, 90.0f), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.6f);
    if (ImGui::Begin("##Stats", nullptr, kOverlayFlags))
    {
        const float fps = (delta_time_seconds > 0.0f) ? (1.0f / delta_time_seconds) : 0.0f;
        ImGui::Text("FPS : %.1f", fps);
        ImGui::Text("dt  : %.3f ms", delta_time_seconds * 1000.0f);
        // SPP counter with progress bar toward 4096
        ImGui::Text("SPP : %u / 4096", accumulated_spp);
        const float progress = static_cast<float>(accumulated_spp) / 4096.0f;
        ImGui::ProgressBar(progress > 1.0f ? 1.0f : progress, ImVec2(-1.0f, 6.0f), "");
    }
    ImGui::End();

    // ── Render passes panel ──────────────────────────────────────────────
    ImGui::SetNextWindowPos(ImVec2(10.0f, 110.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(260.0f, 300.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(0.75f);
    if (ImGui::Begin("Passes"))
    {
        for (const auto& pass : renderer.graph().passes())
        {
            if (ImGui::CollapsingHeader(pass->name(), ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::PushID(pass->name());
                pass->render_ui();
                ImGui::PopID();
            }
        }

        ImGui::Separator();
        const bool can_save = (accumulated_spp > 0);
        if (!can_save) ImGui::BeginDisabled();
        if (ImGui::Button("Save Screenshot (PNG)"))
            screenshot_request = true;
        if (!can_save) ImGui::EndDisabled();
        if (accumulated_spp >= 4096)
            ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f), "4096 spp reached!");
    }
    ImGui::End();
}
}
