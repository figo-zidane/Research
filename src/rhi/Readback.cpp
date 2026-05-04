#include "rhi/Readback.h"

#include "rhi/Buffer.h"
#include "rhi/CommandRecorder.h"
#include "rhi/Device.h"
#include "rhi/Image.h"
#include "rhi/internal/VulkanTypeCasts.h"

#include <volk.h>

#include <stdexcept>

namespace rr::rhi
{
namespace
{
void validate_region(const Image& src, const ImageRegion& region)
{
    const Extent3D src_extent = src.extent3d();
    if (!src.is_valid())
    {
        throw std::runtime_error("readback_image requires a valid source image.");
    }
    if (region.extent.width == 0 || region.extent.height == 0 || region.extent.depth == 0)
    {
        throw std::runtime_error("readback_image requires a non-empty region.");
    }
    if (region.offset.x < 0 || region.offset.y < 0 || region.offset.z < 0)
    {
        throw std::runtime_error("readback_image does not support negative offsets.");
    }

    const uint64_t end_x = static_cast<uint64_t>(region.offset.x) + region.extent.width;
    const uint64_t end_y = static_cast<uint64_t>(region.offset.y) + region.extent.height;
    const uint64_t end_z = static_cast<uint64_t>(region.offset.z) + region.extent.depth;
    if (end_x > src_extent.width || end_y > src_extent.height || end_z > src_extent.depth)
    {
        throw std::runtime_error("readback_image region exceeds the source image extent.");
    }
    if (region.mip >= src.mip_levels())
    {
        throw std::runtime_error("readback_image received an out-of-range mip level.");
    }
    if (region.layer >= src.array_layers())
    {
        throw std::runtime_error("readback_image received an out-of-range array layer.");
    }
}

void emit_image_barrier(VkCommandBuffer command_buffer,
                        VkImage image,
                        VkImageAspectFlags aspect_mask,
                        uint32_t mip,
                        uint32_t layer,
                        VkImageLayout old_layout,
                        VkImageLayout new_layout,
                        VkPipelineStageFlags2 src_stage,
                        VkAccessFlags2 src_access,
                        VkPipelineStageFlags2 dst_stage,
                        VkAccessFlags2 dst_access)
{
    VkImageMemoryBarrier2 barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask        = src_stage;
    barrier.srcAccessMask       = src_access;
    barrier.dstStageMask        = dst_stage;
    barrier.dstAccessMask       = dst_access;
    barrier.oldLayout           = old_layout;
    barrier.newLayout           = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = image;
    barrier.subresourceRange    = {aspect_mask, mip, 1, layer, 1};

    VkDependencyInfo dependency{};
    dependency.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers    = &barrier;
    vkCmdPipelineBarrier2(command_buffer, &dependency);
}
} // namespace

void readback_image(Device& device,
                    const Image& src,
                    ImageLayout src_current_layout,
                    Buffer& dst_staging,
                    const ImageRegion& region)
{
    validate_region(src, region);
    if (!dst_staging.is_valid())
    {
        throw std::runtime_error("readback_image requires a valid staging buffer.");
    }

    device.wait_idle();

    const VkImage image = from_handle<VkImage>(src.handle());
    const VkBuffer buffer = from_handle<VkBuffer>(dst_staging.handle());
    const VkImageLayout source_layout = to_vk_image_layout(src_current_layout);
    const VkImageAspectFlags aspect_mask = to_vk_image_aspect(region.aspect);

    device.one_time_submit([&](CommandRecorder recorder) {
        const VkCommandBuffer command_buffer = static_cast<VkCommandBuffer>(recorder.handle());

        if (src_current_layout != ImageLayout::TransferSrc)
        {
            emit_image_barrier(
                command_buffer,
                image,
                aspect_mask,
                region.mip,
                region.layer,
                source_layout,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_READ_BIT);
        }

        VkBufferImageCopy copy_region{};
        copy_region.bufferOffset      = 0;
        copy_region.bufferRowLength   = 0;
        copy_region.bufferImageHeight = 0;
        copy_region.imageSubresource  = {aspect_mask, region.mip, region.layer, 1};
        copy_region.imageOffset       = {region.offset.x, region.offset.y, region.offset.z};
        copy_region.imageExtent       = {region.extent.width, region.extent.height, region.extent.depth};

        vkCmdCopyImageToBuffer(command_buffer,
                               image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               buffer,
                               1,
                               &copy_region);

        if (src_current_layout != ImageLayout::TransferSrc)
        {
            emit_image_barrier(
                command_buffer,
                image,
                aspect_mask,
                region.mip,
                region.layer,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                source_layout,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
        }
    });
}
} // namespace rr::rhi