#include "rhi/Frame.h"

#include "rhi/Device.h"

#include <stdexcept>

namespace rr::rhi
{
namespace
{
[[nodiscard]] VkSemaphore as_vk_semaphore(SemaphoreHandle handle)
{
    return from_opaque_handle<VkSemaphore>(handle);
}

[[nodiscard]] VkFence as_vk_fence(FenceHandle handle)
{
    return from_opaque_handle<VkFence>(handle);
}
}

Frame::~Frame()
{
    shutdown();
}

void Frame::initialize(Device& device)
{
    device_ = &device;
    VkSemaphoreCreateInfo sem_info{};
    sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (uint32_t i = 0; i < kFramesInFlight; ++i)
    {
        VkSemaphore image_available = VK_NULL_HANDLE;
        VkFence in_flight = VK_NULL_HANDLE;
        if (vkCreateSemaphore(device_->device(), &sem_info, nullptr, &image_available) != VK_SUCCESS ||
            vkCreateFence(device_->device(), &fence_info, nullptr, &in_flight) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create per-frame sync primitives.");
        }
        image_available_[i] = to_opaque_handle<SemaphoreHandle>(image_available);
        in_flight_[i] = to_opaque_handle<FenceHandle>(in_flight);
    }
}

void Frame::wait_for_in_flight(uint32_t frame_index) const
{
    const VkFence fence = as_vk_fence(in_flight_[frame_index]);
    vkWaitForFences(device_->device(), 1, &fence, VK_TRUE, UINT64_MAX);
}

void Frame::reset_in_flight_fence(uint32_t frame_index) const
{
    const VkFence fence = as_vk_fence(in_flight_[frame_index]);
    vkResetFences(device_->device(), 1, &fence);
}

void Frame::shutdown()
{
    if (device_ == nullptr || device_->device() == VK_NULL_HANDLE)
    {
        return;
    }
    for (uint32_t i = 0; i < kFramesInFlight; ++i)
    {
        if (image_available_[i] != nullptr)
        {
            vkDestroySemaphore(device_->device(), as_vk_semaphore(image_available_[i]), nullptr);
            image_available_[i] = nullptr;
        }
        if (in_flight_[i] != nullptr)
        {
            vkDestroyFence(device_->device(), as_vk_fence(in_flight_[i]), nullptr);
            in_flight_[i] = nullptr;
        }
    }
    device_ = nullptr;
    current_frame_ = 0;
}

uint32_t Frame::advance() noexcept
{
    const uint32_t next = current_frame_;
    current_frame_ = (current_frame_ + 1) % kFramesInFlight;
    return next;
}
}
