#pragma once

#include "render/RenderPass.h"
#include "rhi/CommandRecorder.h"
#include "rhi/Image.h"
#include "rhi/Pipeline.h"
#include "shader/ShaderModule.h"
#include "shader/ShaderReflection.h"
#include "shader/SlangSession.h"

#include <array>
#include <cstdint>

namespace rr::passes::denoise
{

class AtrousPass : public rr::render::RenderPass
{
public:
    AtrousPass() = default;
    ~AtrousPass() override;

    void initialize(rr::rhi::Device&           device,
                    rr::shader::SlangSession&  session,
                    rr::rhi::BindlessRegistry& registry,
                    rr::rhi::Extent2D          extent);

    void shutdown(rr::rhi::Device& device);
    bool reload_shader(rr::shader::SlangSession& session);

    [[nodiscard]] const char* name() const override { return "AtrousPass"; }
    [[nodiscard]] Reflection  reflect() const override;
    void on_resize(rr::rhi::Extent2D new_extent) override;
    void render_ui() override;
    void execute(rr::render::FrameContext& fc) override;

    void set_input(uint32_t input_texture_idx, rr::rhi::ImageHandle input_image)
    {
        input_texture_idx_ = input_texture_idx;
        input_image_       = input_image;
    }

    void set_gbuffer_indices(uint32_t gbuf_pos_idx, uint32_t gbuf_norm_idx)
    {
        gbuf_pos_idx_  = gbuf_pos_idx;
        gbuf_norm_idx_ = gbuf_norm_idx;
    }

    void pre_transition_to_general(rr::rhi::CommandRecorder recorder);

    [[nodiscard]] uint32_t output_texture_idx() const;
    [[nodiscard]] const rr::rhi::Image& output_image() const noexcept;
    [[nodiscard]] rr::rhi::ImageHandle output_image_handle() const;

    uint32_t iterations       = 5;
    float    sigma_position   = 0.25f;
    float    sigma_normal     = 0.20f;
    float    sigma_luminance  = 1.00f;

private:
    void create_images(rr::rhi::Device& device,
                       rr::rhi::BindlessRegistry& registry,
                       rr::rhi::Extent2D extent);
    void destroy_images(rr::rhi::Device& device);
    void create_pipeline(rr::rhi::Device& device,
                         rr::rhi::BindlessRegistry& registry);

    rr::rhi::Device*           device_   = nullptr;
    rr::rhi::BindlessRegistry* registry_ = nullptr;

    std::array<rr::rhi::Image, 2> ping_images_{};
    std::array<uint32_t, 2>       storage_indices_ = {UINT32_MAX, UINT32_MAX};
    std::array<uint32_t, 2>       texture_indices_ = {UINT32_MAX, UINT32_MAX};
    rr::rhi::Extent2D             extent_{};

    rr::shader::ShaderModule     shader_;
    rr::shader::ShaderReflection reflection_;
    rr::rhi::ComputePipeline     pipeline_;

    uint32_t input_texture_idx_ = UINT32_MAX;
    rr::rhi::ImageHandle input_image_ = 0;
    uint32_t gbuf_pos_idx_      = UINT32_MAX;
    uint32_t gbuf_norm_idx_     = UINT32_MAX;

    bool initialized_  = false;
    bool first_execute_ = true;
};

} // namespace rr::passes::denoise