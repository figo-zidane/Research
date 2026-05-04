#pragma once

#include "render/RenderPass.h"
#include "rhi/CommandRecorder.h"
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
                    rr::rhi::Extent2D           extent);

    void shutdown(rr::rhi::Device& device);

    // Hot-reload: recompile shader and recreate all three pipelines.
    // Returns true on success.
    bool reload_shader(rr::shader::SlangSession& session);

    // ── RenderPass interface ──────────────────────────────────────────────

    [[nodiscard]] const char* name() const override { return "ReSTIRDIPass"; }
    [[nodiscard]] Reflection  reflect() const override;
    void on_resize(rr::rhi::Extent2D new_extent) override;
    void render_ui() override;
    void execute(rr::render::FrameContext& fc) override;

    // ── Output resource indices ───────────────────────────────────────────

    // gStorageImages[] index for cs_spatial_shade to write into.
    uint32_t output_storage_idx = UINT32_MAX;
    // gTextures[] index (SHADER_READ_ONLY_OPTIMAL) for TonemapPass to sample.
    uint32_t output_texture_idx = UINT32_MAX;

    // Returns the output image handle (used for FrameContext
    // accumulated_image barrier in TonemapPass when in ReSTIR DI mode).
    [[nodiscard]] rr::rhi::ImageHandle output_image_handle() const;
    [[nodiscard]] const rr::rhi::Image& output_image() const noexcept { return output_img_; }

    // Set G-buffer image handles and bindless indices produced by GBufferPass.
    // Must be called before the first execute().
    void set_gbuffer_indices(uint32_t pos_idx, rr::rhi::ImageHandle pos_image,
                             uint32_t norm_idx, rr::rhi::ImageHandle norm_image)
    {
        gbuf_pos_idx_    = pos_idx;
        gbuf_pos_image_  = pos_image;
        gbuf_norm_idx_   = norm_idx;
        gbuf_norm_image_ = norm_image;
    }

    // Transition all owned storage images from UNDEFINED → GENERAL.
    // Call once inside a one_time_submit after initialize() to satisfy the
    // Vulkan validation requirement that registered storage-image descriptors
    // are in the declared layout before the first frame executes.
    void pre_transition_to_general(rr::rhi::CommandRecorder recorder);

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
                       rr::rhi::Extent2D extent);
    void destroy_images(rr::rhi::Device& device);

    void create_pipelines(rr::rhi::Device& device,
                          rr::rhi::BindlessRegistry& registry);

    rr::rhi::Device*           device_   = nullptr;
    rr::rhi::BindlessRegistry* registry_ = nullptr;

    // G-buffer bindless indices and image handles (set via set_gbuffer_indices()).
    uint32_t gbuf_pos_idx_    = UINT32_MAX;
    uint32_t gbuf_norm_idx_   = UINT32_MAX;
    rr::rhi::ImageHandle gbuf_pos_image_  = 0;
    rr::rhi::ImageHandle gbuf_norm_image_ = 0;

    // Reservoir ping-pong (A and B alternate as current/previous each frame)
    rr::rhi::Image reservoir_[2];
    uint32_t       reservoir_idx_[2] = {UINT32_MAX, UINT32_MAX};
    uint32_t       reservoir_flip_   = 0;  // 0 → write A / read B, 1 → write B / read A

    // Output radiance
    rr::rhi::Image output_img_;

    rr::rhi::Extent2D extent_{};

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
