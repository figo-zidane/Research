#include "passes/tonemap/TonemapPass.h"

#include "core/Log.h"
#include "render/FrameContext.h"
#include "rhi/BindlessRegistry.h"
#include "rhi/Device.h"

#include <imgui.h>

namespace rr::passes::tonemap
{

struct TonemapPushConstants
{
    uint32_t accumulated_idx;
    uint32_t linear_sampler_idx;
    float    exposure;
    uint32_t _pad;
};

TonemapPass::~TonemapPass() = default;

void TonemapPass::initialize(rr::rhi::Device& device,
                               rr::shader::SlangSession& session,
                               rr::rhi::BindlessRegistry& registry,
                               rr::rhi::Format swapchain_format)
{
    device_          = &device;
    registry_        = &registry;
    swapchain_format_ = swapchain_format;

    shader_.compile(session,
        "assets/shaders/passes/tonemap/tonemap.slang",
        {
            {"vs_main", rr::shader::ShaderStage::Vertex},
            {"fs_main", rr::shader::ShaderStage::Fragment}
        });
    reflection_ = rr::shader::ShaderReflection(shader_.program_layout());

    create_pipeline(device, registry);
    initialized_ = true;
    core::log()->info("[TonemapPass] Initialized");
}

void TonemapPass::shutdown(rr::rhi::Device& device)
{
    if (!initialized_) return;
    pipeline_.destroy(device);
    shader_.reset();
    initialized_ = false;
}

bool TonemapPass::reload_shader(rr::shader::SlangSession& session)
{
    if (!initialized_) return false;

    // Step 1: try compile new shader — old shader_ stays intact on failure.
    rr::shader::ShaderModule new_shader;
    try
    {
        new_shader.compile(session,
            "assets/shaders/passes/tonemap/tonemap.slang",
            {
                {"vs_main", rr::shader::ShaderStage::Vertex},
                {"fs_main", rr::shader::ShaderStage::Fragment}
            });
    }
    catch (const std::exception& e)
    {
        core::log()->error("[TonemapPass] Shader recompile failed: {}", e.what());
        return false;
    }
    rr::shader::ShaderReflection new_reflection(new_shader.program_layout());

    // Step 2: try create new pipeline using the new shader.
    rr::rhi::GraphicsPipeline new_pipeline;
    rr::rhi::GraphicsPipelineDesc desc{};
    desc.module        = &new_shader;
    desc.reflection    = &new_reflection;
    desc.vert_entry    = 0;
    desc.frag_entry    = 1;
    desc.registry      = registry_;
    desc.color_formats = {swapchain_format_};
    desc.depth_test    = false;
    desc.depth_write   = false;
    desc.cull_mode     = rr::rhi::CullMode::None;
    desc.topology      = rr::rhi::PrimitiveTopology::TriangleList;
    desc.debug_name    = "tonemap_pipeline";
    try
    {
        new_pipeline.create(*device_, desc);
    }
    catch (const std::exception& e)
    {
        core::log()->error("[TonemapPass] Pipeline recreate failed: {}", e.what());
        return false; // old pipeline_ still valid
    }

    // Step 3: both succeeded — destroy old, swap in new.
    pipeline_.destroy(*device_);
    pipeline_.swap(new_pipeline);
    shader_.swap(new_shader);
    reflection_ = new_reflection;
    return true;
}

void TonemapPass::create_pipeline(rr::rhi::Device& device,
                                    rr::rhi::BindlessRegistry& registry)
{
    rr::rhi::GraphicsPipelineDesc desc{};
    desc.module        = &shader_;
    desc.reflection    = &reflection_;
    desc.vert_entry    = 0;
    desc.frag_entry    = 1;
    desc.registry      = &registry;
    desc.color_formats = {swapchain_format_};
    // No depth
    desc.depth_test    = false;
    desc.depth_write   = false;
    desc.cull_mode     = rr::rhi::CullMode::None;
    desc.topology      = rr::rhi::PrimitiveTopology::TriangleList;
    desc.debug_name    = "tonemap_pipeline";
    pipeline_.create(device, desc);
}

void TonemapPass::on_resize(rr::rhi::Extent2D new_extent)
{
    extent_ = new_extent;
}

rr::render::RenderPass::Reflection TonemapPass::reflect() const
{
    return {};
}

void TonemapPass::render_ui()
{
    ImGui::SliderFloat("Exposure", &exposure, 0.01f, 10.0f, "%.2f",
                        ImGuiSliderFlags_Logarithmic);
}

void TonemapPass::execute(rr::render::FrameContext& fc)
{
    if (!pipeline_.is_valid()) return;
    if (accumulated_texture_idx == UINT32_MAX) return;

    const rr::rhi::CommandRecorder recorder = fc.command_recorder;

    // Transition accumulated image from GENERAL → SHADER_READ_ONLY
    {
        const rr::rhi::ImageBarrier barrier{
            .image_handle = fc.accumulated_image,
            .src_stage = rr::rhi::PipelineStage::ComputeShader,
            .src_access = rr::rhi::AccessFlags::ShaderWrite,
            .dst_stage = rr::rhi::PipelineStage::FragmentShader,
            .dst_access = rr::rhi::AccessFlags::ShaderRead,
            .old_layout = rr::rhi::ImageLayout::General,
            .new_layout = rr::rhi::ImageLayout::ShaderReadOnly,
            .subresource = {
                .aspect = rr::rhi::ImageAspect::Color,
                .base_mip = 0,
                .mip_count = rr::rhi::kRemainingMipLevels,
                .base_layer = 0,
                .layer_count = rr::rhi::kRemainingArrayLayers,
            },
        };
        recorder.pipeline_barrier({&barrier, 1});
    }

    const rr::rhi::ColorAttachment color_att{
        .image = nullptr,
        .image_view = fc.swapchain_image_view,
        .layout = rr::rhi::ImageLayout::ColorAttachment,
        .load_op = rr::rhi::LoadOp::DontCare,
        .store_op = rr::rhi::StoreOp::Store,
    };
    const rr::rhi::RenderingInfo rendering{
        .area = fc.swapchain_extent,
        .layer_count = 1,
        .color_attachments = {&color_att, 1},
        .depth_attachment = nullptr,
    };
    recorder.begin_rendering(rendering);

    recorder.bind_graphics_pipeline(pipeline_);
    recorder.set_viewport(0.0f, 0.0f,
                          static_cast<float>(fc.swapchain_extent.width),
                          static_cast<float>(fc.swapchain_extent.height));
    recorder.set_scissor(0, 0, fc.swapchain_extent.width, fc.swapchain_extent.height);

    TonemapPushConstants pc{};
    pc.accumulated_idx    = accumulated_texture_idx;
    pc.linear_sampler_idx = linear_sampler_idx;
    pc.exposure           = exposure;

    recorder.push_constants(&pc, sizeof(pc));

    recorder.draw(3, 1, 0, 0);

    recorder.end_rendering();

    // Transition accumulated image back to GENERAL for next frame's accumulation
    {
        const rr::rhi::ImageBarrier barrier{
            .image_handle = fc.accumulated_image,
            .src_stage = rr::rhi::PipelineStage::FragmentShader,
            .src_access = rr::rhi::AccessFlags::ShaderRead,
            .dst_stage = rr::rhi::PipelineStage::ComputeShader,
            .dst_access = rr::rhi::AccessFlags::ShaderWrite | rr::rhi::AccessFlags::ShaderRead,
            .old_layout = rr::rhi::ImageLayout::ShaderReadOnly,
            .new_layout = rr::rhi::ImageLayout::General,
            .subresource = {
                .aspect = rr::rhi::ImageAspect::Color,
                .base_mip = 0,
                .mip_count = rr::rhi::kRemainingMipLevels,
                .base_layer = 0,
                .layer_count = rr::rhi::kRemainingArrayLayers,
            },
        };
        recorder.pipeline_barrier({&barrier, 1});
    }
}

} // namespace rr::passes::tonemap
