#include "rhi/CommandBuffer.h"

#include "rhi/Device.h"

#include <stdexcept>

namespace rr::rhi
{
CommandBuffer::~CommandBuffer()
{
    shutdown();
}

void CommandBuffer::initialize(Device& device)
{
    device_ = &device;

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = device.graphics_queue_family();
    // Per-frame pool: we reset the entire pool each frame instead of tracking
    // individual buffer reset. Transient hint lets the driver recycle memory.
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

    for (uint32_t i = 0; i < kFramesInFlight; ++i)
    {
        if (vkCreateCommandPool(device.device(), &pool_info, nullptr, &pools_[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create per-frame command pool.");
        }

        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = pools_[i];
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device.device(), &alloc_info, &buffers_[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate per-frame command buffer.");
        }
    }
}

void CommandBuffer::shutdown()
{
    if (device_ == nullptr || device_->device() == VK_NULL_HANDLE)
    {
        return;
    }
    for (uint32_t i = 0; i < kFramesInFlight; ++i)
    {
        if (pools_[i] != VK_NULL_HANDLE)
        {
            vkDestroyCommandPool(device_->device(), pools_[i], nullptr);
            pools_[i] = VK_NULL_HANDLE;
        }
        buffers_[i] = VK_NULL_HANDLE;
    }
    device_ = nullptr;
}

VkCommandBuffer CommandBuffer::begin_frame(uint32_t frame_index)
{
    vkResetCommandPool(device_->device(), pools_[frame_index], 0);

    VkCommandBuffer cmd = buffers_[frame_index];
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS)
    {
        throw std::runtime_error("vkBeginCommandBuffer failed.");
    }
    return cmd;
}

void CommandBuffer::end_frame(VkCommandBuffer cmd)
{
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
    {
        throw std::runtime_error("vkEndCommandBuffer failed.");
    }
}

void CommandBuffer::image_barrier(
    VkCommandBuffer cmd,
    VkImage image,
    VkImageLayout old_layout,
    VkImageLayout new_layout)
{
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    // Layout-driven scope selection. Phase 1B only needs UNDEFINED -> COLOR_ATTACHMENT
    // and COLOR_ATTACHMENT -> PRESENT_SRC; richer transitions are added later.
    if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
    {
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barrier.srcAccessMask = 0;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    }
    else if (old_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL && new_layout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
    {
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        barrier.dstAccessMask = 0;
    }
    else
    {
        // Conservative fallback for layouts we don't specialise yet.
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    }

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}
}
