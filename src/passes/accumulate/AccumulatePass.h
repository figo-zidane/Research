#pragma once

#include "render/RenderPass.h"
#include "rhi/Image.h"
#include "rhi/Pipeline.h"
#include "shader/ShaderModule.h"
#include "shader/ShaderReflection.h"
#include "shader/SlangSession.h"

#include <volk.h>

namespace rr::passes::accumulate
{

// AccumulatePass — temporal accumulation.
// Each frame: acc = acc * (spp / (spp+1)) + radiance / (spp+1)
// If camera moved, reset: acc = radiance, spp = 1.
class AccumulatePass : public rr::render::RenderPass
{
public:
    AccumulatePass() = default;
    ~AccumulatePass() override;

    void initialize(rr::rhi::Device&           device,
                    rr::shader::SlangSession&   session,
                    rr::rhi::BindlessRegistry&  registry,
                    VkExtent2D                  extent);

    void shutdown(rr::rhi::Device& device);

    // Hot-reload: recompile shader and recreate pipeline. Returns true on success.
    bool reload_shader(rr::shader::SlangSession& session);

    [[nodiscard]] const char* name() const override { return "AccumulatePass"; }
    [[nodiscard]] Reflection  reflect() const override;
    void on_resize(VkExtent2D new_extent) override;
    void render_ui() override;
    void execute(rr::render::FrameContext& fc) override;

    // Input from PathTracerPass
    uint32_t radiance_storage_idx  = UINT32_MAX;

    // Output — registered with BindlessRegistry as both storage and texture
    uint32_t accumulated_storage_idx  = UINT32_MAX;
    uint32_t accumulated_texture_idx  = UINT32_MAX;

    // Driven by Application
    bool     camera_moved   = false;
    uint32_t accumulated_spp = 0;

    // Returns the raw Vulkan image handle for use in FrameContext barriers.
    [[nodiscard]] VkImage accumulated_image_handle() const;

private:
    void create_images(rr::rhi::Device& device, rr::rhi::BindlessRegistry& registry,
                        VkExtent2D extent);
    void destroy_images(rr::rhi::Device& device);
    void create_pipeline(rr::rhi::Device& device, rr::rhi::BindlessRegistry& registry);

    rr::rhi::Device*           device_   = nullptr;
    rr::rhi::BindlessRegistry* registry_ = nullptr;

    rr::rhi::Image             accumulated_img_;  // RGBA32F
    VkExtent2D                 extent_{};

    rr::shader::ShaderModule     shader_;
    rr::shader::ShaderReflection reflection_;
    rr::rhi::ComputePipeline     pipeline_;

    bool initialized_ = false;
};

} // namespace rr::passes::accumulate
