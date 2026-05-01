#pragma once

#include <cstdint>

namespace rr::render
{
class Renderer;
}

namespace rr::shader
{
class HotReload;
}

namespace rr::app
{
// EditorUI builds the ImGui content for each frame.
// Call build() after ImGuiPass::new_frame() and before renderer_.render().
class EditorUI
{
public:
    // Build the full ImGui UI for this frame.
    //   accumulated_spp      : current accumulated sample count
    //   screenshot_request   : set to true when the user clicks "Save Screenshot"
    //   show_restir          : toggle between PathTracer (false) and ReSTIR DI (true)
    //   hot_reload           : pointer to the HotReload object for status display
    //   mse_history          : circular buffer of MSE values (may be nullptr)
    //   mse_history_count    : number of valid entries in mse_history
    //   mse_latest           : most recent MSE value (-1 = not yet computed)
    //   mse_history_count    : number of valid entries in mse_history
    //   mse_history_offset   : circular buffer read start index
    //   mse_latest           : most recent MSE value (-1 = not yet computed)
    //   mse_auto_update      : toggle auto-MSE every N frames
    void build(const rr::render::Renderer& renderer,
               float                       delta_time_seconds,
               uint32_t                    accumulated_spp,
               bool&                       screenshot_request,
               bool&                       show_restir,
               bool&                       mse_compare,
               const rr::shader::HotReload* hot_reload        = nullptr,
               const float*                 mse_history       = nullptr,
               uint32_t                     mse_history_count  = 0,
               uint32_t                     mse_history_offset = 0,
               float                        mse_latest        = -1.0f,
               bool*                        mse_auto_update   = nullptr);
};
}
