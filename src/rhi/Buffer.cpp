#include "rhi/Buffer.h"

#include "core/Log.h"
#include "rhi/Device.h"

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <volk.h>
#include <vma/vk_mem_alloc.h>

#include <stdexcept>
#include <utility>

namespace rr::rhi
{

Buffer::Buffer(Buffer&& other) noexcept
    : buffer_(other.buffer_)
    , allocation_(other.allocation_)
    , size_(other.size_)
    , device_address_(other.device_address_)
    , mapped_(other.mapped_)
{
    other.buffer_          = VK_NULL_HANDLE;
    other.allocation_      = nullptr;
    other.size_            = 0;
    other.device_address_  = 0;
    other.mapped_          = nullptr;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept
{
    if (this != &other)
    {
        // Caller is responsible for calling destroy() before reassigning
        buffer_          = other.buffer_;
        allocation_      = other.allocation_;
        size_            = other.size_;
        device_address_  = other.device_address_;
        mapped_          = other.mapped_;
        other.buffer_          = VK_NULL_HANDLE;
        other.allocation_      = nullptr;
        other.size_            = 0;
        other.device_address_  = 0;
        other.mapped_          = nullptr;
    }
    return *this;
}

void Buffer::create(Device& device, const BufferDesc& desc)
{
    if (buffer_ != VK_NULL_HANDLE)
    {
        throw std::runtime_error("Buffer::create called on an already-created buffer.");
    }

    VkBufferCreateInfo buf_info{};
    buf_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.size        = desc.size;
    buf_info.usage       = desc.usage;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo alloc_info{};
    alloc_info.usage = static_cast<VmaMemoryUsage>(desc.memory_usage);
    alloc_info.flags = static_cast<VmaAllocationCreateFlags>(desc.alloc_flags);

    VmaAllocationInfo alloc_result{};
    if (vmaCreateBuffer(device.allocator(), &buf_info, &alloc_info,
                        &buffer_, &allocation_, &alloc_result) != VK_SUCCESS)
    {
        throw std::runtime_error("vmaCreateBuffer failed.");
    }

    size_   = desc.size;
    mapped_ = alloc_result.pMappedData; // non-null if MAPPED_BIT was set

    // Obtain device address if requested.
    if (desc.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
    {
        VkBufferDeviceAddressInfo addr_info{};
        addr_info.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addr_info.buffer = buffer_;
        device_address_ = vkGetBufferDeviceAddress(device.device(), &addr_info);
    }

    if (desc.debug_name)
    {
        VkDebugUtilsObjectNameInfoEXT name_info{};
        name_info.sType        = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        name_info.objectType   = VK_OBJECT_TYPE_BUFFER;
        name_info.objectHandle = reinterpret_cast<uint64_t>(buffer_);
        name_info.pObjectName  = desc.debug_name;
        vkSetDebugUtilsObjectNameEXT(device.device(), &name_info);
    }
}

void Buffer::destroy(Device& device)
{
    if (buffer_ == VK_NULL_HANDLE)
    {
        return;
    }
    vmaDestroyBuffer(device.allocator(), buffer_, allocation_);
    buffer_         = VK_NULL_HANDLE;
    allocation_     = nullptr;
    size_           = 0;
    device_address_ = 0;
    mapped_         = nullptr;
}

void* Buffer::map(Device& device)
{
    if (mapped_)
    {
        return mapped_;
    }
    if (vmaMapMemory(device.allocator(), allocation_, &mapped_) != VK_SUCCESS)
    {
        return nullptr;
    }
    return mapped_;
}

void Buffer::unmap(Device& device)
{
    if (!mapped_)
    {
        return;
    }
    vmaUnmapMemory(device.allocator(), allocation_);
    mapped_ = nullptr;
}

} // namespace rr::rhi
