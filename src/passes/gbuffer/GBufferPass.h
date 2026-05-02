#pragma once

#include "render/RenderPass.h"
#include "rhi/CommandRecorder.h"
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

namespace rr::passes::gbuffer
{

// GBufferPass rasterizes the scene into 3 off-screen render targets:
//   gBuffer0 (RGBA32_SFLOAT) : world-space position (32F for sub-mm precision)
//   gBuffer1 (RGBA32_SFLOAT) : world-space normal (.w = asfloat(material_id), 32F required)
//   gBuffer2 (R32_UINT)      : packed material ID + motion vector placeholder
// plus a D32_SFLOAT depth buffer.
//
// Output images are registered in the BindlessRegistry as storage images and
// their indices are written into FrameContext's gbuffer_* fields.
class GBufferPass : public rr::render::RenderPass
{
public:
    GBufferPass() = default;
    ~GBufferPass() override;

    // Initialize shaders and pipeline.  Must be called before first frame.
    void initialize(rr::rhi::Device&          device,
                    rr::shader::SlangSession& session,
                    rr::rhi::BindlessRegistry& registry,
                    rr::rhi::Extent2D        extent);

    void shutdown(rr::rhi::Device& device);

    // Transition all owned storage images from UNDEFINED → GENERAL.
    // Call once inside a one_time_submit after initialize().
    void pre_transition_to_general(rr::rhi::CommandRecorder recorder);

    // RenderPass interface
    [[nodiscard]] const char* name() const override { return "GBufferPass"; }
    [[nodiscard]] Reflection  reflect() const override;
    void on_resize(rr::rhi::Extent2D new_extent) override;
    void render_ui() override;
    void execute(rr::render::FrameContext& fc) override;

    // Image indices in the BindlessRegistry (written during initialize/resize).
    uint32_t position_storage_idx    = UINT32_MAX;
    uint32_t normal_storage_idx      = UINT32_MAX;
    uint32_t material_id_storage_idx = UINT32_MAX;

    // Image handles for barrier management by downstream passes.
    [[nodiscard]] rr::rhi::ImageHandle position_image_handle() const
    {
        return rr::rhi::to_handle(position_img_.handle());
    }
    [[nodiscard]] rr::rhi::ImageHandle normal_image_handle() const
    {
        return rr::rhi::to_handle(normal_img_.handle());
    }

private:
    void create_images(rr::rhi::Device& device, rr::rhi::BindlessRegistry& registry,
                        rr::rhi::Extent2D extent);
    void destroy_images(rr::rhi::Device& device);
    void create_pipeline(rr::rhi::Device& device, rr::rhi::BindlessRegistry& registry);

    rr::rhi::Device*          device_   = nullptr;
    rr::rhi::BindlessRegistry* registry_ = nullptr;

    // GBuffer images
    rr::rhi::Image position_img_;    // RGBA32_SFLOAT
    rr::rhi::Image normal_img_;      // RGBA32_SFLOAT (.w = asfloat(material_id))
    rr::rhi::Image material_id_img_; // R32_UINT
    rr::rhi::Image depth_img_;       // D32_SFLOAT

    rr::rhi::Extent2D extent_{};

    rr::shader::ShaderModule     shader_;
    rr::shader::ShaderReflection reflection_;
    rr::rhi::GraphicsPipeline    pipeline_;

    bool initialized_ = false;
};

} // namespace rr::passes::gbuffer
