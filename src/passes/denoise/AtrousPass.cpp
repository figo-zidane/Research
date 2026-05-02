#include "passes/denoise/AtrousPass.h"

#include "core/Log.h"
#include "rhi/BindlessRegistry.h"
#include "rhi/Device.h"

#include <algorithm>
#include <imgui.h>
#include <stdexcept>
#include <vector>

namespace rr::passes::denoise
{

namespace
{

struct AtrousPushConstants
{
    uint32_t input_idx;
    uint32_t output_idx;
    uint32_t gbuf_pos_idx;
    uint32_t gbuf_norm_idx;
    uint32_t screen_width;
    uint32_t screen_height;
    uint32_t iter;
    uint32_t stride;
    float    sigma_position;
    float    sigma_normal;
    float    sigma_luminance;
    uint32_t _pad;
};

static_assert(sizeof(AtrousPushConstants) == 48);

} // namespace

AtrousPass::~AtrousPass() = default;

void AtrousPass::initialize(rr::rhi::Device& device,
                            rr::shader::SlangSession& session,
                            rr::rhi::BindlessRegistry& registry,
                            VkExtent2D extent)
{
    device_   = &device;
    registry_ = &registry;
    extent_   = extent;

    shader_.compile(session,
        "assets/shaders/passes/denoise/atrous.slang",
        {{"cs_main", rr::shader::ShaderStage::Compute}});
    reflection_ = rr::shader::ShaderReflection(shader_.program_layout());

    create_images(device, registry, extent);
    create_pipeline(device, registry);
    initialized_   = true;
    first_execute_ = true;
    core::log()->info("[AtrousPass] Initialized {}x{}", extent.width, extent.height);
}

void AtrousPass::shutdown(rr::rhi::Device& device)
{
    if (!initialized_) return;
    pipeline_.destroy(device);
    shader_.reset();
    destroy_images(device);
    initialized_ = false;
}

bool AtrousPass::reload_shader(rr::shader::SlangSession& session)
{
    if (!initialized_) return false;

    rr::shader::ShaderModule new_shader;
    try
    {
        new_shader.compile(session,
            "assets/shaders/passes/denoise/atrous.slang",
            {{"cs_main", rr::shader::ShaderStage::Compute}});
    }
    catch (const std::exception& e)
    {
        core::log()->error("[AtrousPass] Shader recompile failed: {}", e.what());
        return false;
    }
    rr::shader::ShaderReflection new_reflection(new_shader.program_layout());

    rr::rhi::ComputePipeline new_pipeline;
    rr::rhi::ComputePipelineDesc desc{};
    desc.module      = &new_shader;
    desc.reflection  = &new_reflection;
    desc.entry_index = 0;
    desc.registry    = registry_;
    desc.debug_name  = "atrous_pipeline";
    try
    {
        new_pipeline.create(*device_, desc);
    }
    catch (const std::exception& e)
    {
        core::log()->error("[AtrousPass] Pipeline recreate failed: {}", e.what());
        return false;
    }

    pipeline_.destroy(*device_);
    pipeline_.swap(new_pipeline);
    shader_.swap(new_shader);
    reflection_ = new_reflection;
    return true;
}

void AtrousPass::create_images(rr::rhi::Device& device,
                               rr::rhi::BindlessRegistry& registry,
                               VkExtent2D extent)
{
    for (size_t index = 0; index < ping_images_.size(); ++index)
    {
        rr::rhi::ImageDesc desc{};
        desc.format     = VK_FORMAT_R32G32B32A32_SFLOAT;
        desc.extent     = {extent.width, extent.height, 1};
        desc.usage      = VK_IMAGE_USAGE_STORAGE_BIT
                        | VK_IMAGE_USAGE_SAMPLED_BIT
                        | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        desc.debug_name = (index == 0) ? "atrous_ping_a" : "atrous_ping_b";
        ping_images_[index].create(device, desc);

        storage_indices_[index] = registry.register_storage_image(
            device, ping_images_[index].handle(), VK_FORMAT_R32G32B32A32_SFLOAT);
        texture_indices_[index] = registry.register_texture(
            device, ping_images_[index].handle(),
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_ASPECT_COLOR_BIT);
    }
}

void AtrousPass::destroy_images(rr::rhi::Device& device)
{
    for (auto& image : ping_images_)
        image.destroy(device);
}

void AtrousPass::create_pipeline(rr::rhi::Device& device,
                                 rr::rhi::BindlessRegistry& registry)
{
    rr::rhi::ComputePipelineDesc desc{};
    desc.module      = &shader_;
    desc.reflection  = &reflection_;
    desc.entry_index = 0;
    desc.registry    = &registry;
    desc.debug_name  = "atrous_pipeline";
    pipeline_.create(device, desc);
}

void AtrousPass::pre_transition_to_general(VkCommandBuffer cmd)
{
    const VkImage images[] = {
        ping_images_[0].handle(),
        ping_images_[1].handle(),
    };
    image_barrier_compute(
        cmd, images, static_cast<uint32_t>(std::size(images)),
        0,
        VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED);
}

uint32_t AtrousPass::output_texture_idx() const
{
    const uint32_t output_slot = (std::max(1u, iterations) - 1u) & 1u;
    return texture_indices_[output_slot];
}

VkImage AtrousPass::output_image_handle() const
{
    const uint32_t output_slot = (std::max(1u, iterations) - 1u) & 1u;
    return ping_images_[output_slot].handle();
}

rr::render::RenderPass::Reflection AtrousPass::reflect() const
{
    Reflection reflection;
    reflection.outputs.push_back({
        "atrous_output",
        ResourceDesc::Kind::Texture,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        extent_});
    return reflection;
}

void AtrousPass::on_resize(VkExtent2D new_extent)
{
    if (!initialized_) return;
    extent_ = new_extent;
    destroy_images(*device_);
    create_images(*device_, *registry_, new_extent);
    first_execute_ = true;
}

void AtrousPass::render_ui()
{
    ImGui::SliderInt("Iterations", reinterpret_cast<int*>(&iterations), 1, 7);
    ImGui::SliderFloat("Sigma Position", &sigma_position, 0.001f, 2.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Sigma Normal", &sigma_normal, 0.01f, 2.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Sigma Luminance", &sigma_luminance, 0.01f, 4.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ImGui::Text("  output gTextures[%u]", output_texture_idx());
}

void AtrousPass::image_barrier_compute(VkCommandBuffer cmd,
                                       const VkImage* images,
                                       uint32_t count,
                                       VkAccessFlags2 src_access,
                                       VkAccessFlags2 dst_access,
                                       VkImageLayout old_layout) const
{
    std::vector<VkImageMemoryBarrier2> barriers(count);
    for (uint32_t index = 0; index < count; ++index)
    {
        auto& barrier = barriers[index];
        barrier = {};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask       = src_access;
        barrier.dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask       = dst_access;
        barrier.oldLayout           = old_layout;
        barrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = images[index];
        barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                       VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    }

    VkDependencyInfo dep{};
    dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = count;
    dep.pImageMemoryBarriers    = barriers.data();
    vkCmdPipelineBarrier2(cmd, &dep);
}

void AtrousPass::execute(rr::render::FrameContext& fc)
{
    if (!initialized_ || !pipeline_.is_valid()) return;
    if (input_texture_idx_ == UINT32_MAX || input_image_ == VK_NULL_HANDLE) return;
    if (gbuf_pos_idx_ == UINT32_MAX || gbuf_norm_idx_ == UINT32_MAX) return;

    VkCommandBuffer cmd = fc.command_buffer;

    if (first_execute_)
    {
        const VkImage images[] = {
            ping_images_[0].handle(),
            ping_images_[1].handle(),
        };
        image_barrier_compute(
            cmd, images, static_cast<uint32_t>(std::size(images)),
            0,
            VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED);
        first_execute_ = false;
    }

    const uint32_t clamped_iterations = std::clamp(iterations, 1u, 7u);
    uint32_t read_texture_idx = input_texture_idx_;
    VkImage  read_image       = input_image_;

    for (uint32_t iteration = 0; iteration < clamped_iterations; ++iteration)
    {
        const uint32_t write_slot = iteration & 1u;
        const VkImage images[] = {
            read_image,
            ping_images_[write_slot].handle(),
        };
        image_barrier_compute(
            cmd, images, static_cast<uint32_t>(std::size(images)),
            VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
            VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
            VK_IMAGE_LAYOUT_GENERAL);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_.handle());

        AtrousPushConstants push_constants{};
        push_constants.input_idx        = read_texture_idx;
        push_constants.output_idx       = storage_indices_[write_slot];
        push_constants.gbuf_pos_idx     = gbuf_pos_idx_;
        push_constants.gbuf_norm_idx    = gbuf_norm_idx_;
        push_constants.screen_width     = extent_.width;
        push_constants.screen_height    = extent_.height;
        push_constants.iter             = iteration;
        push_constants.stride           = 1u << iteration;
        push_constants.sigma_position   = sigma_position;
        push_constants.sigma_normal     = sigma_normal;
        push_constants.sigma_luminance  = sigma_luminance;

        VkPushDataInfoEXT push_info{};
        push_info.sType        = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
        push_info.offset       = 0;
        push_info.data.address = &push_constants;
        push_info.data.size    = sizeof(push_constants);
        vkCmdPushDataEXT(cmd, &push_info);

        const uint32_t group_x = (extent_.width + 7u) / 8u;
        const uint32_t group_y = (extent_.height + 7u) / 8u;
        vkCmdDispatch(cmd, group_x, group_y, 1);

        read_texture_idx = texture_indices_[write_slot];
        read_image       = ping_images_[write_slot].handle();
    }
}

} // namespace rr::passes::denoise