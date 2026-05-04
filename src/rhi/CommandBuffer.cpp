#include "rhi/CommandBuffer.h"

#include "rhi/Device.h"
#include "rhi/internal/VulkanAccess.h"

#include <stdexcept>

namespace rr::rhi
{
namespace
{
[[nodiscard]] VkCommandPool as_vk_command_pool(CommandPoolHandle handle)
{
    return from_opaque_handle<VkCommandPool>(handle);
}

[[nodiscard]] VkCommandBuffer as_vk_command_buffer(CommandBufferHandle handle)
{
    return from_opaque_handle<VkCommandBuffer>(handle);
}
}

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
        VkCommandPool pool = VK_NULL_HANDLE;
        if (vkCreateCommandPool(vulkan::get_device(device), &pool_info, nullptr, &pool) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create per-frame command pool.");
        }
        pools_[i] = to_opaque_handle<CommandPoolHandle>(pool);

        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;
        VkCommandBuffer buffer = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(vulkan::get_device(device), &alloc_info, &buffer) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate per-frame command buffer.");
        }
        buffers_[i] = to_opaque_handle<CommandBufferHandle>(buffer);
    }
}

void CommandBuffer::shutdown()
{
    if (device_ == nullptr || device_->device() == nullptr)
    {
        return;
    }
    for (uint32_t i = 0; i < kFramesInFlight; ++i)
    {
        if (pools_[i] != nullptr)
        {
            vkDestroyCommandPool(vulkan::get_device(*device_), as_vk_command_pool(pools_[i]), nullptr);
            pools_[i] = nullptr;
        }
        buffers_[i] = nullptr;
    }
    device_ = nullptr;
}

CommandRecorder CommandBuffer::begin_frame(uint32_t frame_index)
{
    vkResetCommandPool(vulkan::get_device(*device_), as_vk_command_pool(pools_[frame_index]), 0);

    VkCommandBuffer cmd = as_vk_command_buffer(buffers_[frame_index]);
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS)
    {
        throw std::runtime_error("vkBeginCommandBuffer failed.");
    }
    return CommandRecorder{static_cast<void*>(cmd)};
}

void CommandBuffer::end_frame(CommandRecorder cmd)
{
    if (vkEndCommandBuffer(static_cast<VkCommandBuffer>(cmd.handle())) != VK_SUCCESS)
    {
        throw std::runtime_error("vkEndCommandBuffer failed.");
    }
}
}
