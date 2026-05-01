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
                               VkFormat swapchain_format)
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
    desc.cull_mode     = VK_CULL_MODE_NONE;
    desc.topology      = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
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
    desc.cull_mode     = VK_CULL_MODE_NONE;
    desc.topology      = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    desc.debug_name    = "tonemap_pipeline";
    pipeline_.create(device, desc);
}

void TonemapPass::on_resize(VkExtent2D new_extent)
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

    VkCommandBuffer cmd = fc.command_buffer;

    // Transition accumulated image from GENERAL → SHADER_READ_ONLY
    {
        VkImageMemoryBarrier2 b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.srcAccessMask       = VK_ACCESS_2_SHADER_WRITE_BIT;
        b.dstStageMask        = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        b.dstAccessMask       = VK_ACCESS_2_SHADER_READ_BIT;
        b.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = fc.accumulated_image;  // set by Application
        b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                  VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        VkDependencyInfo dep{};
        dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // Dynamic rendering to swapchain
    VkRenderingAttachmentInfo color_att{};
    color_att.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color_att.imageView   = fc.swapchain_image_view;
    color_att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_att.loadOp      = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_att.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo rendering{};
    rendering.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering.renderArea           = {{0,0}, fc.swapchain_extent};
    rendering.layerCount           = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments    = &color_att;
    vkCmdBeginRendering(cmd, &rendering);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_.handle());

    VkViewport viewport{0.0f, 0.0f,
                         static_cast<float>(fc.swapchain_extent.width),
                         static_cast<float>(fc.swapchain_extent.height),
                         0.0f, 1.0f};
    VkRect2D scissor{{0,0}, fc.swapchain_extent};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    TonemapPushConstants pc{};
    pc.accumulated_idx    = accumulated_texture_idx;
    pc.linear_sampler_idx = linear_sampler_idx;
    pc.exposure           = exposure;

    VkPushDataInfoEXT push_info{};
    push_info.sType        = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
    push_info.offset       = 0;
    push_info.data.address = &pc;
    push_info.data.size    = sizeof(pc);
    vkCmdPushDataEXT(cmd, &push_info);

    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRendering(cmd);

    // Transition accumulated image back to GENERAL for next frame's accumulation
    {
        VkImageMemoryBarrier2 b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask        = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        b.srcAccessMask       = VK_ACCESS_2_SHADER_READ_BIT;
        b.dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask       = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT;
        b.oldLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = fc.accumulated_image;
        b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                  VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        VkDependencyInfo dep{};
        dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
    }
}

} // namespace rr::passes::tonemap
