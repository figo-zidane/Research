#pragma once

#include "render/RenderPass.h"
#include "rhi/Image.h"
#include "rhi/Pipeline.h"
#include "shader/ShaderModule.h"
#include "shader/ShaderReflection.h"
#include "shader/SlangSession.h"

namespace rr::rhi
{
class Device;
class BindlessRegistry;
} // namespace rr::rhi

namespace rr::passes::pathtracer
{

// PathTracerPass — naive reference path tracer using VK_KHR_ray_query.
// Runs as a compute dispatch, one thread per pixel.
// Outputs a linear HDR radiance image (RGBA32F).
class PathTracerPass : public rr::render::RenderPass
{
public:
    PathTracerPass() = default;
    ~PathTracerPass() override;

    void initialize(rr::rhi::Device&           device,
                    rr::shader::SlangSession&   session,
                    rr::rhi::BindlessRegistry&  registry,
                    rr::rhi::Extent2D           extent);

    void shutdown(rr::rhi::Device& device);
    // Hot-reload: recompile shader and recreate pipeline. Returns true on success.
    bool reload_shader(rr::shader::SlangSession& session);
    [[nodiscard]] const char* name() const override { return "PathTracerPass"; }
    [[nodiscard]] Reflection  reflect() const override;
    void on_resize(rr::rhi::Extent2D new_extent) override;
    void render_ui() override;
    void execute(rr::render::FrameContext& fc) override;

    // Index of the radiance image in gStorageImages[].
    uint32_t radiance_storage_idx  = UINT32_MAX;
    uint32_t radiance_texture_idx  = UINT32_MAX;

    uint32_t max_bounces = 5;

private:
    void create_images(rr::rhi::Device& device, rr::rhi::BindlessRegistry& registry,
                        rr::rhi::Extent2D extent);
    void destroy_images(rr::rhi::Device& device);
    void create_pipeline(rr::rhi::Device& device, rr::rhi::BindlessRegistry& registry);

    rr::rhi::Device*           device_   = nullptr;
    rr::rhi::BindlessRegistry* registry_ = nullptr;

    rr::rhi::Image             radiance_img_;  // RGBA32F
    rr::rhi::Extent2D          extent_{};

    rr::shader::ShaderModule     shader_;
    rr::shader::ShaderReflection reflection_;
    rr::rhi::ComputePipeline     pipeline_;

    uint32_t frame_count_ = 0;

    bool initialized_ = false;
};

} // namespace rr::passes::pathtracer
