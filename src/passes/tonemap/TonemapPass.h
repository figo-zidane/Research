#pragma once

#include "render/RenderPass.h"
#include "rhi/Pipeline.h"
#include "shader/ShaderModule.h"
#include "shader/ShaderReflection.h"
#include "shader/SlangSession.h"

namespace rr::passes::tonemap
{

// TonemapPass — fullscreen triangle + ACES tonemapping.
// Reads accumulated_image (a sampled texture) and writes to the swapchain.
class TonemapPass : public rr::render::RenderPass
{
public:
    TonemapPass() = default;
    ~TonemapPass() override;

    void initialize(rr::rhi::Device&           device,
                    rr::shader::SlangSession&   session,
                    rr::rhi::BindlessRegistry&  registry,
                    VkFormat                    swapchain_format);

    void shutdown(rr::rhi::Device& device);

    // Hot-reload: recompile shader and recreate pipeline. Returns true on success.
    bool reload_shader(rr::shader::SlangSession& session);

    [[nodiscard]] const char* name() const override { return "TonemapPass"; }
    [[nodiscard]] Reflection  reflect() const override;
    void on_resize(VkExtent2D new_extent) override;
    void render_ui() override;
    void execute(rr::render::FrameContext& fc) override;

    // Inputs (from AccumulatePass), set by Application before execute()
    uint32_t accumulated_texture_idx = UINT32_MAX;
    uint32_t linear_sampler_idx      = UINT32_MAX;
    float    exposure = 1.0f;

private:
    void create_pipeline(rr::rhi::Device& device, rr::rhi::BindlessRegistry& registry);

    rr::rhi::Device*           device_         = nullptr;
    rr::rhi::BindlessRegistry* registry_       = nullptr;

    VkExtent2D extent_{};
    VkFormat   swapchain_format_ = VK_FORMAT_UNDEFINED;

    rr::shader::ShaderModule     shader_;
    rr::shader::ShaderReflection reflection_;
    rr::rhi::GraphicsPipeline    pipeline_;

    bool initialized_ = false;
};

} // namespace rr::passes::tonemap
