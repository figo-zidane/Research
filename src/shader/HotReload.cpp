#include "shader/HotReload.h"

#include "core/FileWatcher.h"
#include "core/Log.h"
#include "rhi/Device.h"

#include <imgui.h>
#include <volk.h>

namespace rr::shader
{

HotReload::HotReload()  : watcher_(std::make_unique<rr::core::FileWatcher>()) {}
HotReload::~HotReload() { shutdown(); }

void HotReload::initialize(const std::filesystem::path& shader_dir)
{
    watcher_->start(shader_dir);
    initialized_ = true;
}

void HotReload::shutdown()
{
    if (!initialized_) return;
    watcher_->stop();
    initialized_ = false;
}

void HotReload::register_shader(const std::filesystem::path& path, RecompileCallback cb)
{
    entries_.push_back({path, std::move(cb)});
}

bool HotReload::pump(rr::rhi::Device& device)
{
    if (!initialized_) return false;

    auto changed = watcher_->poll();
    if (changed.empty()) return false;

    for (const auto& f : changed)
        core::log()->info("[HotReload] changed: {}", f.filename().string());

    // Ensure GPU is idle before we destroy and recreate any pipelines.
    vkDeviceWaitIdle(device.device());

    bool any_reloaded = false;
    last_error_.clear();

    for (auto& entry : entries_)
    {
        core::log()->info("[HotReload] recompiling '{}'", entry.path.filename().string());
        try
        {
            if (entry.callback())
            {
                last_reloaded_file_ = entry.path.filename().string();
                core::log()->info("[HotReload] OK: '{}'", entry.path.filename().string());
                any_reloaded = true;
            }
            else
            {
                last_error_ = "Recompile failed: " + entry.path.filename().string();
                core::log()->error("[HotReload] {}", last_error_);
            }
        }
        catch (const std::exception& e)
        {
            last_error_ = std::string(e.what());
            core::log()->error("[HotReload] exception: {}", last_error_);
        }
    }

    return any_reloaded;
}

void HotReload::render_ui() const
{
    if (has_error())
    {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
        ImGui::TextWrapped("HotReload error:");
        ImGui::TextWrapped("%s", last_error_.c_str());
        ImGui::PopStyleColor();
    }
    else if (!last_reloaded_file_.empty())
    {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.2f, 1.0f, 0.2f, 1.0f));
        ImGui::Text("Last reloaded: %s", last_reloaded_file_.c_str());
        ImGui::PopStyleColor();
    }
    else
    {
        ImGui::TextDisabled("No changes detected.");
    }
}

} // namespace rr::shader
