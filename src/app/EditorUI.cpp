#include "app/EditorUI.h"

#include "passes/denoise/AtrousPass.h"
#include "render/RenderGraph.h"
#include "render/Renderer.h"
#include "shader/HotReload.h"

#include <imgui.h>

#include <algorithm>
#include <cstring>

namespace rr::app
{
void EditorUI::build(const rr::render::Renderer& renderer,
                     float        delta_time_seconds,
                     uint32_t     accumulated_spp,
                     bool&        screenshot_request,
                     bool&        use_di,
                     bool&        use_gi,
                     bool&        use_pt,
                     bool&        use_denoise,
                     bool&        mse_compare,
                     std::string& gltf_path_input,
                     bool&        load_cornell_request,
                     bool&        load_gltf_request,
                     const std::string& current_scene_name,
                     rr::passes::denoise::AtrousPass* atrous_pass,
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
        // ── Scene loading ───────────────────────────────────────────────
        if (ImGui::CollapsingHeader("Scene"))
        {
            ImGui::Text("Current: %s", current_scene_name.c_str());
            ImGui::Spacing();
            if (ImGui::Button("Load Cornell Box"))
                load_cornell_request = true;
            ImGui::Spacing();
            ImGui::SetNextItemWidth(-1.0f);
            {
                char buf[512];
                const size_t copy_len = std::min(gltf_path_input.size(), sizeof(buf) - 1);
                std::memcpy(buf, gltf_path_input.data(), copy_len);
                buf[copy_len] = '\0';
                if (ImGui::InputText("##gltf_path", buf, sizeof(buf)))
                    gltf_path_input = buf;
            }
            ImGui::SameLine();
            if (ImGui::Button("Load glTF") && !gltf_path_input.empty())
                load_gltf_request = true;
            ImGui::TextDisabled("(Drag & Drop .gltf / .glb onto the window)");
            ImGui::TextDisabled("Note: each reload allocates new bindless slots.");
        }

        // ── Controls help ─────────────────────────────────────────────────
        if (ImGui::CollapsingHeader("Controls"))
        {
            ImGui::BulletText("W/A/S/D      : Move (while right-dragging)");
            ImGui::BulletText("E / Q        : Move up / down");
            ImGui::BulletText("Right drag   : Rotate view");
            ImGui::BulletText("Shift        : 4x speed");
        }

        // ── Display mode ──────────────────────────────────────────────────
        if (ImGui::CollapsingHeader("Display Mode", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Checkbox("Direct Lighting (ReSTIR DI)", &use_di);
            ImGui::Checkbox("Indirect Lighting (ReSTIR GI)", &use_gi);
            ImGui::Checkbox("Indirect Lighting (ReSTIR PT, multi-bounce)", &use_pt);
            if (use_pt && use_gi)
            {
                use_gi = false;  // PT and GI are mutually exclusive; PT wins.
            }
            const bool can_denoise = use_di || use_gi || use_pt;
            if (!can_denoise)
                use_denoise = false;
            if (!can_denoise)
                ImGui::BeginDisabled();
            ImGui::Checkbox("Denoise", &use_denoise);
            if (!can_denoise)
                ImGui::EndDisabled();
            if (!can_denoise)
                ImGui::TextDisabled("Enable ReSTIR DI or GI to denoise the realtime output.");
            ImGui::Spacing();
            ImGui::Checkbox("Compute both for MSE comparison", &mse_compare);
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("PathTracer stays active alongside the enabled realtime passes.\nUse when comparing MSE. Disables GPU optimisation.");
        }

        if (use_denoise && atrous_pass &&
            ImGui::CollapsingHeader("Denoiser", ImGuiTreeNodeFlags_DefaultOpen))
        {
            atrous_pass->render_ui();
        }

        // ── MSE graph ────────────────────────────────────────────────────
        if (mse_history && mse_history_count > 0)
        {
            if (ImGui::CollapsingHeader("MSE (Realtime vs PathTracer)", ImGuiTreeNodeFlags_DefaultOpen))
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
