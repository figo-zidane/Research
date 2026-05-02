#include "passes/accumulate/AccumulatePass.h"

#include "core/Log.h"
#include "render/FrameContext.h"
#include "rhi/BindlessRegistry.h"
#include "rhi/Device.h"

#include <imgui.h>

namespace rr::passes::accumulate
{

struct AccumulatePushConstants
{
    uint32_t radiance_idx;
    uint32_t accumulated_idx;
    uint32_t screen_width;
    uint32_t screen_height;
    uint32_t camera_moved;    // 1 = reset accumulation
    uint32_t accumulated_spp; // current spp before this frame
    uint32_t _pad[2];
};

AccumulatePass::~AccumulatePass() = default;

void AccumulatePass::initialize(rr::rhi::Device& device,
                                  rr::shader::SlangSession& session,
                                  rr::rhi::BindlessRegistry& registry,
                                  rr::rhi::Extent2D extent)
{
    device_   = &device;
    registry_ = &registry;
    extent_   = extent;

    shader_.compile(session,
        "assets/shaders/passes/accumulate/accumulate.slang",
        {{"cs_main", rr::shader::ShaderStage::Compute}});
    reflection_ = rr::shader::ShaderReflection(shader_.program_layout());

    create_images(device, registry, extent);
    create_pipeline(device, registry);
    initialized_ = true;
    core::log()->info("[AccumulatePass] Initialized {}x{}", extent.width, extent.height);
}

void AccumulatePass::shutdown(rr::rhi::Device& device)
{
    if (!initialized_) return;
    pipeline_.destroy(device);
    shader_.reset();
    destroy_images(device);
    initialized_ = false;
}

bool AccumulatePass::reload_shader(rr::shader::SlangSession& session)
{
    if (!initialized_) return false;

    // Step 1: try compile new shader — old shader_ stays intact on failure.
    rr::shader::ShaderModule new_shader;
    try
    {
        new_shader.compile(session,
            "assets/shaders/passes/accumulate/accumulate.slang",
            {{"cs_main", rr::shader::ShaderStage::Compute}});
    }
    catch (const std::exception& e)
    {
        core::log()->error("[AccumulatePass] Shader recompile failed: {}", e.what());
        return false;
    }
    rr::shader::ShaderReflection new_reflection(new_shader.program_layout());

    // Step 2: try create new pipeline using the new shader.
    rr::rhi::ComputePipeline new_pipeline;
    rr::rhi::ComputePipelineDesc d{};
    d.module      = &new_shader;
    d.reflection  = &new_reflection;
    d.entry_index = 0;
    d.registry    = registry_;
    d.debug_name  = "accumulate_pipeline";
    try
    {
        new_pipeline.create(*device_, d);
    }
    catch (const std::exception& e)
    {
        core::log()->error("[AccumulatePass] Pipeline recreate failed: {}", e.what());
        return false; // old pipeline_ still valid
    }

    // Step 3: both succeeded — destroy old, swap in new.
    pipeline_.destroy(*device_);
    pipeline_.swap(new_pipeline);
    shader_.swap(new_shader);
    reflection_ = new_reflection;
    return true;
}

void AccumulatePass::create_images(rr::rhi::Device& device,
                                     rr::rhi::BindlessRegistry& registry,
                                     rr::rhi::Extent2D ext)
{
    rr::rhi::ImageDesc d{};
    d.format     = VK_FORMAT_R32G32B32A32_SFLOAT;
    d.extent     = {ext.width, ext.height, 1};
    d.usage      = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                 | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;  // required for screenshot readback
    d.debug_name = "accumulated_image";
    accumulated_img_.create(device, d);

    accumulated_storage_idx = registry.register_storage_image(
        device, accumulated_img_.handle(), VK_FORMAT_R32G32B32A32_SFLOAT);
    accumulated_texture_idx = registry.register_texture(
        device, accumulated_img_.handle(),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_IMAGE_ASPECT_COLOR_BIT);
}

void AccumulatePass::destroy_images(rr::rhi::Device& device)
{
    accumulated_img_.destroy(device);
}

rr::rhi::ImageHandle AccumulatePass::accumulated_image_handle() const
{
    return rr::rhi::to_handle(accumulated_img_.handle());
}

void AccumulatePass::create_pipeline(rr::rhi::Device& device,
                                       rr::rhi::BindlessRegistry& registry)
{
    rr::rhi::ComputePipelineDesc desc{};
    desc.module      = &shader_;
    desc.reflection  = &reflection_;
    desc.entry_index  = 0;
    desc.registry    = &registry;
    desc.debug_name  = "accumulate_pipeline";
    pipeline_.create(device, desc);
}

void AccumulatePass::on_resize(rr::rhi::Extent2D new_extent)
{
    if (!initialized_) return;
    extent_ = new_extent;
    destroy_images(*device_);
    create_images(*device_, *registry_, new_extent);
    camera_moved    = true;
    accumulated_spp = 0;
}

rr::render::RenderPass::Reflection AccumulatePass::reflect() const
{
    Reflection r;
    r.outputs.push_back({"accumulated_image", ResourceDesc::Kind::Texture,
                          static_cast<rr::rhi::Format>(VK_FORMAT_R32G32B32A32_SFLOAT), extent_});
    return r;
}

void AccumulatePass::render_ui()
{
    ImGui::Text("Accumulate %ux%u  SPP=%u", extent_.width, extent_.height, accumulated_spp);
    ImGui::Text("  accumulated gStorageImages[%u]", accumulated_storage_idx);
}

void AccumulatePass::execute(rr::render::FrameContext& fc)
{
    if (!pipeline_.is_valid()) return;
    if (radiance_storage_idx == UINT32_MAX) return;

    VkCommandBuffer cmd = static_cast<VkCommandBuffer>(fc.command_recorder.handle());

    // Transition accumulated image to GENERAL
    {
        VkImageMemoryBarrier2 b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.srcAccessMask       = VK_ACCESS_2_SHADER_WRITE_BIT;
        b.dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask       = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        b.oldLayout           = camera_moved ? VK_IMAGE_LAYOUT_UNDEFINED
                                              : VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = accumulated_img_.handle();
        b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                  VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        VkDependencyInfo dep{};
        dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_.handle());

    AccumulatePushConstants pc{};
    pc.radiance_idx    = radiance_storage_idx;
    pc.accumulated_idx = accumulated_storage_idx;
    pc.screen_width    = extent_.width;
    pc.screen_height   = extent_.height;
    pc.camera_moved    = camera_moved ? 1u : 0u;
    pc.accumulated_spp = accumulated_spp;

    VkPushDataInfoEXT push_info{};
    push_info.sType        = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
    push_info.offset       = 0;
    push_info.data.address = &pc;
    push_info.data.size    = sizeof(pc);
    vkCmdPushDataEXT(cmd, &push_info);

    uint32_t gx = (extent_.width  + 7) / 8;
    uint32_t gy = (extent_.height + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);

    // Update spp for next frame
    if (camera_moved)
        accumulated_spp = 1;
    else
        accumulated_spp++;
    camera_moved = false;
}

} // namespace rr::passes::accumulate
