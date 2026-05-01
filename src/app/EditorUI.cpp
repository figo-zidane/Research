#include "app/EditorUI.h"

#include "render/RenderGraph.h"
#include "render/Renderer.h"
#include "shader/HotReload.h"

#include <imgui.h>

namespace rr::app
{
void EditorUI::build(const rr::render::Renderer& renderer,
                     float        delta_time_seconds,
                     uint32_t     accumulated_spp,
                     bool&        screenshot_request,
                     bool&        show_restir,
                     const rr::shader::HotReload* hot_reload,
                     const float*  mse_history,
                     uint32_t      mse_history_count,
                     uint32_t      mse_history_offset,
                     float         mse_latest,
                     bool*         mse_auto_update)
{
    // ── Stats overlay ────────────────────────────────────────────────────
    constexpr ImGuiWindowFlags kOverlayFlags =
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize   |
        ImGuiWindowFlags_NoMove     |
        ImGuiWindowFlags_NoSavedSettings;

    ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(280.0f, 100.0f), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.6f);
    if (ImGui::Begin("##Stats", nullptr, kOverlayFlags))
    {
        const float fps = (delta_time_seconds > 0.0f) ? (1.0f / delta_time_seconds) : 0.0f;
        ImGui::Text("FPS : %.1f", fps);
        ImGui::Text("dt  : %.3f ms", delta_time_seconds * 1000.0f);
        ImGui::Text("SPP : %u", accumulated_spp);
    }
    ImGui::End();

    // ── Main control panel ───────────────────────────────────────────────
    ImGui::SetNextWindowPos(ImVec2(10.0f, 120.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300.0f, 500.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(0.75f);
    if (ImGui::Begin("Renderer"))
    {
        // ── Display mode ──────────────────────────────────────────────────
        if (ImGui::CollapsingHeader("Display Mode", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::RadioButton("PathTracer (accumulated)", reinterpret_cast<int*>(&show_restir), 0);
            ImGui::RadioButton("ReSTIR DI (1 spp)",        reinterpret_cast<int*>(&show_restir), 1);
        }

        // ── MSE graph ────────────────────────────────────────────────────
        if (mse_history && mse_history_count > 0)
        {
            if (ImGui::CollapsingHeader("MSE (ReSTIR vs PathTracer)", ImGuiTreeNodeFlags_DefaultOpen))
            {
                if (mse_latest >= 0.0f)
                    ImGui::Text("Latest MSE: %.6f", mse_latest);
                else
                    ImGui::TextDisabled("MSE not yet computed.");

                if (mse_history_count > 1)
                {
                    char overlay[32];
                    snprintf(overlay, sizeof(overlay), "%.5f", mse_latest);
                    ImGui::PlotLines("##mse", mse_history,
                                     static_cast<int>(mse_history_count),
                                     static_cast<int>(mse_history_offset),
                                     overlay,
                                     0.0f, FLT_MAX,
                                     ImVec2(-1.0f, 60.0f));
                }

                if (mse_auto_update)
                {
                    bool au = *mse_auto_update;
                    if (ImGui::Checkbox("Auto-update MSE", &au))
                        *mse_auto_update = au;
                    ImGui::SameLine();
                    ImGui::TextDisabled("(every 60 frames)");
                }
            }
        }

        // ── Hot reload status ─────────────────────────────────────────────
        if (hot_reload)
        {
            if (ImGui::CollapsingHeader("Hot Reload"))
            {
                hot_reload->render_ui();
            }
        }

        // ── Render passes ─────────────────────────────────────────────────
        if (ImGui::CollapsingHeader("Passes"))
        {
            for (const auto& pass : renderer.graph().passes())
            {
                if (ImGui::CollapsingHeader(pass->name()))
                {
                    ImGui::PushID(pass->name());
                    pass->render_ui();
                    ImGui::PopID();
                }
            }
        }

        ImGui::Separator();
        const bool can_save = (accumulated_spp > 0);
        if (!can_save) ImGui::BeginDisabled();
        if (ImGui::Button("Save Screenshot (PNG)"))
            screenshot_request = true;
        if (!can_save) ImGui::EndDisabled();
    }
    ImGui::End();
}
}
