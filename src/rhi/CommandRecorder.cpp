#include "rhi/CommandRecorder.h"

#include "rhi/Image.h"
#include "rhi/Pipeline.h"
#include "rhi/VulkanTypeCasts.h"

#include <array>
#include <stdexcept>

namespace rr::rhi
{
namespace
{
[[nodiscard]] VkCommandBuffer as_vk_cmd(const CommandRecorder& recorder)
{
    if (!recorder.is_valid())
    {
        throw std::runtime_error("CommandRecorder is not valid.");
    }
    return static_cast<VkCommandBuffer>(recorder.handle());
}
} // namespace

void CommandRecorder::bind_compute_pipeline(const ComputePipeline& pipeline) const
{
    vkCmdBindPipeline(as_vk_cmd(*this), VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
}

void CommandRecorder::bind_graphics_pipeline(const GraphicsPipeline& pipeline) const
{
    vkCmdBindPipeline(as_vk_cmd(*this), VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.handle());
}

void CommandRecorder::dispatch(uint32_t group_count_x, uint32_t group_count_y, uint32_t group_count_z) const
{
    vkCmdDispatch(as_vk_cmd(*this), group_count_x, group_count_y, group_count_z);
}

void CommandRecorder::draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance) const
{
    vkCmdDraw(as_vk_cmd(*this), vertex_count, instance_count, first_vertex, first_instance);
}

void CommandRecorder::push_constants(const void* data, uint32_t size, uint32_t offset) const
{
    VkPushDataInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
    info.offset = offset;
    info.data.address = data;
    info.data.size = size;
    vkCmdPushDataEXT(as_vk_cmd(*this), &info);
}

void CommandRecorder::set_viewport(float x, float y, float width, float height, float min_depth, float max_depth) const
{
    VkViewport viewport{x, y, width, height, min_depth, max_depth};
    vkCmdSetViewport(as_vk_cmd(*this), 0, 1, &viewport);
}

void CommandRecorder::set_scissor(int32_t x, int32_t y, uint32_t width, uint32_t height) const
{
    VkRect2D scissor{{x, y}, {width, height}};
    vkCmdSetScissor(as_vk_cmd(*this), 0, 1, &scissor);
}

void CommandRecorder::pipeline_barrier(std::span<const ImageBarrier> barriers) const
{
    std::vector<VkImageMemoryBarrier2> vk_barriers;
    vk_barriers.reserve(barriers.size());
    for (const ImageBarrier& barrier : barriers)
    {
        if (barrier.image == nullptr)
        {
            throw std::runtime_error("ImageBarrier requires a valid image.");
        }
        VkImageMemoryBarrier2 vk_barrier{};
        vk_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        vk_barrier.srcStageMask = to_vk_pipeline_stage(barrier.src_stage);
        vk_barrier.srcAccessMask = to_vk_access_flags(barrier.src_access);
        vk_barrier.dstStageMask = to_vk_pipeline_stage(barrier.dst_stage);
        vk_barrier.dstAccessMask = to_vk_access_flags(barrier.dst_access);
        vk_barrier.oldLayout = to_vk_image_layout(barrier.old_layout);
        vk_barrier.newLayout = to_vk_image_layout(barrier.new_layout);
        vk_barrier.image = barrier.image->handle();
        vk_barrier.subresourceRange = to_vk_image_subresource_range(barrier.subresource);
        vk_barriers.push_back(vk_barrier);
    }

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = static_cast<uint32_t>(vk_barriers.size());
    dep.pImageMemoryBarriers = vk_barriers.data();
    vkCmdPipelineBarrier2(as_vk_cmd(*this), &dep);
}

void CommandRecorder::begin_rendering(const RenderingInfo& rendering_info) const
{
    std::vector<VkRenderingAttachmentInfo> color_attachments;
    color_attachments.reserve(rendering_info.color_attachments.size());
    for (const ColorAttachment& attachment : rendering_info.color_attachments)
    {
        if (attachment.image == nullptr)
        {
            throw std::runtime_error("ColorAttachment requires a valid image.");
        }
        VkRenderingAttachmentInfo vk_attachment{};
        vk_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        vk_attachment.imageView = attachment.image->view();
        vk_attachment.imageLayout = to_vk_image_layout(attachment.layout);
        vk_attachment.loadOp = to_vk_load_op(attachment.load_op);
        vk_attachment.storeOp = to_vk_store_op(attachment.store_op);
        vk_attachment.clearValue.color = {{
            attachment.clear.float32[0],
            attachment.clear.float32[1],
            attachment.clear.float32[2],
            attachment.clear.float32[3],
        }};
        color_attachments.push_back(vk_attachment);
    }

    VkRenderingAttachmentInfo depth_attachment{};
    VkRenderingAttachmentInfo* depth_attachment_ptr = nullptr;
    if (rendering_info.depth_attachment != nullptr)
    {
        const DepthAttachment& attachment = *rendering_info.depth_attachment;
        if (attachment.image == nullptr)
        {
            throw std::runtime_error("DepthAttachment requires a valid image.");
        }
        depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depth_attachment.imageView = attachment.image->view();
        depth_attachment.imageLayout = to_vk_image_layout(attachment.layout);
        depth_attachment.loadOp = to_vk_load_op(attachment.load_op);
        depth_attachment.storeOp = to_vk_store_op(attachment.store_op);
        depth_attachment.clearValue.depthStencil = {attachment.clear_depth, attachment.clear_stencil};
        depth_attachment_ptr = &depth_attachment;
    }

    VkRenderingInfo vk_rendering_info{};
    vk_rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    vk_rendering_info.renderArea.extent = {rendering_info.area.width, rendering_info.area.height};
    vk_rendering_info.layerCount = rendering_info.layer_count;
    vk_rendering_info.colorAttachmentCount = static_cast<uint32_t>(color_attachments.size());
    vk_rendering_info.pColorAttachments = color_attachments.data();
    vk_rendering_info.pDepthAttachment = depth_attachment_ptr;
    vkCmdBeginRendering(as_vk_cmd(*this), &vk_rendering_info);
}

void CommandRecorder::end_rendering() const
{
    vkCmdEndRendering(as_vk_cmd(*this));
}

void CommandRecorder::clear_color_image(const Image& image,
                                        ImageLayout layout,
                                        const ClearColor& clear_color,
                                        std::span<const ImageSubresourceRange> ranges) const
{
    std::vector<VkImageSubresourceRange> vk_ranges;
    vk_ranges.reserve(ranges.size());
    for (const ImageSubresourceRange& range : ranges)
    {
        vk_ranges.push_back(to_vk_image_subresource_range(range));
    }

    VkClearColorValue value{};
    value.float32[0] = clear_color.float32[0];
    value.float32[1] = clear_color.float32[1];
    value.float32[2] = clear_color.float32[2];
    value.float32[3] = clear_color.float32[3];
    vkCmdClearColorImage(as_vk_cmd(*this), image.handle(), to_vk_image_layout(layout), &value,
                         static_cast<uint32_t>(vk_ranges.size()), vk_ranges.data());
}
} // namespace rr::rhi