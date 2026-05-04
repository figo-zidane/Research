#pragma once

#include "rhi/Device.h"

#include <volk.h>
#include <vma/vk_mem_alloc.h>

namespace rr::rhi::vulkan
{
[[nodiscard]] inline VkInstance get_instance(const Device& device) noexcept
{
    return from_opaque_handle<VkInstance>(device.instance());
}

[[nodiscard]] inline VkPhysicalDevice get_physical_device(const Device& device) noexcept
{
    return from_opaque_handle<VkPhysicalDevice>(device.physical_device());
}

[[nodiscard]] inline VkDevice get_device(const Device& device) noexcept
{
    return from_opaque_handle<VkDevice>(device.device());
}

[[nodiscard]] inline VkQueue get_graphics_queue(const Device& device) noexcept
{
    return from_opaque_handle<VkQueue>(device.graphics_queue());
}

[[nodiscard]] inline VmaAllocator get_allocator(const Device& device) noexcept
{
    return from_opaque_handle<VmaAllocator>(device.allocator());
}
}