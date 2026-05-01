#pragma once

#include "render/RenderPass.h"
#include "rhi/Image.h"
#include "rhi/Pipeline.h"
#include "shader/ShaderModule.h"
#include "shader/ShaderReflection.h"
#include "shader/SlangSession.h"

#include <cstdint>

namespace rr::passes::restir_di
{

// ReSTIRDIPass — ReSTIR DI (Direct Illumination) reference implementation.
//
// Three compute dispatches per frame:
//   cs_initial       : trace primary ray, build initial reservoir (M candidates)
//   cs_temporal      : merge with previous frame's reservoir at same pixel
//   cs_spatial_shade : gather neighbor reservoirs, trace shadow, write radiance
//
// Outputs a linear HDR radiance image (RGBA32F) for direct illumination.
// Connect output_texture_idx to TonemapPass::accumulated_texture_idx to display.
class ReSTIRDIPass : public rr::render::RenderPass
{
public:
    ReSTIRDIPass() = default;
    ~ReSTIRDIPass() override;

    void initialize(rr::rhi::Device&           device,
                    rr::shader::SlangSession&   session,
                    rr::rhi::BindlessRegistry&  registry,
                    VkExtent2D                  extent);

    void shutdown(rr::rhi::Device& device);

    // Hot-reload: recompile shader and recreate all three pipelines.
    // Returns true on success.
    bool reload_shader(rr::shader::SlangSession& session);

    // ── RenderPass interface ──────────────────────────────────────────────

    [[nodiscard]] const char* name() const override { return "ReSTIRDIPass"; }
    [[nodiscard]] Reflection  reflect() const override;
    void on_resize(VkExtent2D new_extent) override;
    void render_ui() override;
    void execute(rr::render::FrameContext& fc) override;

    // ── Output resource indices ───────────────────────────────────────────

    // gStorageImages[] index for cs_spatial_shade to write into.
    uint32_t output_storage_idx = UINT32_MAX;
    // gTextures[] index (SHADER_READ_ONLY_OPTIMAL) for TonemapPass to sample.
    uint32_t output_texture_idx = UINT32_MAX;

    // Returns the VkImage handle of the output image (used for FrameContext
    // accumulated_image barrier in TonemapPass when in ReSTIR DI mode).
    [[nodiscard]] VkImage output_image_handle() const;

    // Reset temporal history (call on mode switch to avoid stale reservoirs).
    void reset_history()
    {
        reservoir_flip_ = 0;
        first_execute_  = true;
        frame_count_    = 0;
    }

    // ── Per-pass settings (modifiable from ImGui) ─────────────────────────

    bool     temporal_enable        = true;
    bool     spatial_enable         = true;
    uint32_t num_initial_candidates = 16;  // M in RIS
    uint32_t spatial_num_neighbors  = 4;
    uint32_t spatial_radius         = 15;  // pixels

private:
    void create_images(rr::rhi::Device& device,
                       rr::rhi::BindlessRegistry& registry,
                       VkExtent2D extent);
    void destroy_images(rr::rhi::Device& device);

    void create_pipelines(rr::rhi::Device& device,
                          rr::rhi::BindlessRegistry& registry);

    // Issue a compute-to-compute image memory barrier for a set of images.
    // old_layout defaults to UNDEFINED (discard-and-transition). Pass GENERAL
    // for between-dispatch barriers that must preserve image contents.
    static void image_barrier_compute(VkCommandBuffer cmd,
                                      const VkImage* images, uint32_t count,
                                      VkAccessFlags2 src_access,
                                      VkAccessFlags2 dst_access,
                                      VkImageLayout  old_layout = VK_IMAGE_LAYOUT_UNDEFINED);

    rr::rhi::Device*           device_   = nullptr;
    rr::rhi::BindlessRegistry* registry_ = nullptr;

    // G-buffer
    rr::rhi::Image gbuf_pos_;   // float4(world_pos.xyz, ray_t)
    rr::rhi::Image gbuf_norm_;  // float4(normal.xyz, asfloat(material_index))
    uint32_t       gbuf_pos_idx_  = UINT32_MAX;
    uint32_t       gbuf_norm_idx_ = UINT32_MAX;

    // Reservoir ping-pong (A and B alternate as current/previous each frame)
    rr::rhi::Image reservoir_[2];
    uint32_t       reservoir_idx_[2] = {UINT32_MAX, UINT32_MAX};
    uint32_t       reservoir_flip_   = 0;  // 0 → write A / read B, 1 → write B / read A

    // Output radiance
    rr::rhi::Image output_img_;

    VkExtent2D extent_{};

    rr::shader::ShaderModule     shader_;
    rr::shader::ShaderReflection reflection_;

    // Entry-point indices in the compiled module
    static constexpr uint32_t kEntryInitial       = 0;
    static constexpr uint32_t kEntryTemporal      = 1;
    static constexpr uint32_t kEntrySpatialShade  = 2;

    rr::rhi::ComputePipeline initial_pipeline_;
    rr::rhi::ComputePipeline temporal_pipeline_;
    rr::rhi::ComputePipeline spatial_shade_pipeline_;

    bool initialized_ = false;

    // Set to false after the first execute(); used to determine correct
    // oldLayout for reservoir and output image barriers (UNDEFINED on first
    // frame, GENERAL on subsequent frames).
    bool     first_execute_ = true;
    // Monotonically increasing frame counter — used as frame_seed in the
    // push constants so the random number sequence is never periodic.
    uint32_t frame_count_   = 0;
};

} // namespace rr::passes::restir_di
