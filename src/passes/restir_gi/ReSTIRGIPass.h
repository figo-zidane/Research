#pragma once

#include "render/RenderPass.h"
#include "rhi/CommandRecorder.h"
#include "rhi/Image.h"
#include "rhi/Pipeline.h"
#include "shader/ShaderModule.h"
#include "shader/ShaderReflection.h"
#include "shader/SlangSession.h"

#include <cstdint>

namespace rr::passes::restir_gi
{

class ReSTIRGIPass : public rr::render::RenderPass
{
public:
    ReSTIRGIPass() = default;
    ~ReSTIRGIPass() override;

    void initialize(rr::rhi::Device&           device,
                    rr::shader::SlangSession&  session,
                    rr::rhi::BindlessRegistry& registry,
                    rr::rhi::Extent2D          extent);

    void shutdown(rr::rhi::Device& device);
    bool reload_shader(rr::shader::SlangSession& session);

    [[nodiscard]] const char* name() const override { return "ReSTIRGIPass"; }
    [[nodiscard]] Reflection  reflect() const override;
    void on_resize(rr::rhi::Extent2D new_extent) override;
    void render_ui() override;
    void execute(rr::render::FrameContext& fc) override;

    uint32_t output_storage_idx = UINT32_MAX;
    uint32_t output_texture_idx = UINT32_MAX;

    [[nodiscard]] rr::rhi::ImageHandle output_image_handle() const;

    void set_inputs(uint32_t gbuf_pos_idx,
                    uint32_t gbuf_norm_idx,
                    uint32_t direct_input_texture_idx,
                    rr::rhi::ImageHandle direct_input_image)
    {
        gbuf_pos_idx_           = gbuf_pos_idx;
        gbuf_norm_idx_          = gbuf_norm_idx;
        direct_input_texture_idx_ = direct_input_texture_idx;
        direct_input_image_     = direct_input_image;
    }

    void pre_transition_to_general(rr::rhi::CommandRecorder recorder);
    void reset_history()
    {
        reservoir_flip_     = 0;
        frame_count_        = 0;
        history_dirty_      = true;
    }
    [[nodiscard]] bool consume_history_dirty()
    {
        bool d = history_dirty_;
        history_dirty_ = false;
        return d;
    }

    bool  include_direct_lighting = true;
    bool  temporal_enable         = true;
    bool  spatial_enable          = true;
    float indirect_strength       = 1.0f;
    float ray_bias                = 1.0e-3f;
    uint32_t num_initial_candidates = 8;
    uint32_t spatial_num_neighbors  = 4;
    uint32_t spatial_radius         = 12;

private:
    void create_images(rr::rhi::Device& device,
                       rr::rhi::BindlessRegistry& registry,
                       rr::rhi::Extent2D extent);
    void destroy_images(rr::rhi::Device& device);
    void create_pipelines(rr::rhi::Device& device,
                          rr::rhi::BindlessRegistry& registry);

    rr::rhi::Device*           device_   = nullptr;
    rr::rhi::BindlessRegistry* registry_ = nullptr;

    uint32_t gbuf_pos_idx_             = UINT32_MAX;
    uint32_t gbuf_norm_idx_            = UINT32_MAX;
    uint32_t direct_input_texture_idx_ = UINT32_MAX;
    rr::rhi::ImageHandle direct_input_image_ = 0;

    rr::rhi::Image reservoir_pos_[2];
    rr::rhi::Image reservoir_rad_[2];
    rr::rhi::Image reservoir_meta_[2];
    rr::rhi::Image history_primary_pos_[2];
    rr::rhi::Image history_primary_norm_[2];
    uint32_t       reservoir_pos_idx_[2]  = {UINT32_MAX, UINT32_MAX};
    uint32_t       reservoir_rad_idx_[2]  = {UINT32_MAX, UINT32_MAX};
    uint32_t       reservoir_meta_idx_[2] = {UINT32_MAX, UINT32_MAX};
    uint32_t       history_primary_pos_idx_[2]  = {UINT32_MAX, UINT32_MAX};
    uint32_t       history_primary_norm_idx_[2] = {UINT32_MAX, UINT32_MAX};

    rr::rhi::Image output_img_;

    rr::rhi::Extent2D extent_{};

    rr::shader::ShaderModule     initial_shader_;
    rr::shader::ShaderModule     temporal_shader_;
    rr::shader::ShaderModule     spatial_shader_;
    rr::shader::ShaderReflection initial_reflection_;
    rr::shader::ShaderReflection temporal_reflection_;
    rr::shader::ShaderReflection spatial_reflection_;
    rr::rhi::ComputePipeline     initial_pipeline_;
    rr::rhi::ComputePipeline     temporal_pipeline_;
    rr::rhi::ComputePipeline     spatial_shade_pipeline_;

    bool     initialized_   = false;
    bool     history_dirty_ = true;
    uint32_t reservoir_flip_ = 0;
    uint32_t frame_count_   = 0;
};

} // namespace rr::passes::restir_gi