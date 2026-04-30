#include "rhi/Frame.h"

#include "rhi/Device.h"

#include <stdexcept>

namespace rr::rhi
{
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

    for (auto& sync : syncs_)
    {
        if (vkCreateSemaphore(device_->device(), &sem_info, nullptr, &sync.image_available) != VK_SUCCESS ||
            vkCreateFence    (device_->device(), &fence_info, nullptr, &sync.in_flight)      != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create per-frame sync primitives.");
        }
    }
}

void Frame::shutdown()
{
    if (device_ == nullptr || device_->device() == VK_NULL_HANDLE)
    {
        return;
    }
    for (auto& sync : syncs_)
    {
        if (sync.image_available != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(device_->device(), sync.image_available, nullptr);
            sync.image_available = VK_NULL_HANDLE;
        }
        if (sync.in_flight != VK_NULL_HANDLE)
        {
            vkDestroyFence(device_->device(), sync.in_flight, nullptr);
            sync.in_flight = VK_NULL_HANDLE;
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
