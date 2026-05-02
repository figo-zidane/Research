#pragma once

#include "rhi/Types.h"

#include <cstdint>
#include <volk.h>

// Forward-declare VMA handle types so Buffer.h doesn't need to pull in the
// entire vk_mem_alloc.h header.  The actual definitions live in vma_impl.cpp.
VK_DEFINE_HANDLE(VmaAllocation)

namespace rr::rhi
{
class Device;

struct BufferDesc
{
    uint64_t      size         = 0;
    BufferUsage   usage        = BufferUsage::None;
    MemoryUsage   memory_usage = MemoryUsage::Auto;
    AllocFlags    alloc_flags  = AllocFlags::None;
    const char*   debug_name   = nullptr;
};

// Thin RAII wrapper around a VkBuffer + VmaAllocation.
// Ownership follows the create/destroy pattern used throughout rhi/.
class Buffer
{
public:
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    ~Buffer() = default; // must call destroy() explicitly before going out of scope

    void create(Device& device, const BufferDesc& desc);
    void destroy(Device& device);

    // Map the buffer memory for CPU access (only valid for host-visible buffers
    // that were not created with VMA_ALLOCATION_CREATE_MAPPED_BIT).
    // Returns nullptr if already mapped or if mapping fails.
    void* map(Device& device);
    void  unmap(Device& device);

    [[nodiscard]] BufferHandle handle() const noexcept { return to_handle(buffer_); }
    [[nodiscard]] uint64_t     size() const noexcept { return size_; }
    [[nodiscard]] uint64_t     device_address() const noexcept { return device_address_; }
    [[nodiscard]] void*        mapped() const noexcept { return mapped_; }
    [[nodiscard]] bool         is_valid() const noexcept { return buffer_ != VK_NULL_HANDLE; }

private:
    VkBuffer        buffer_         = VK_NULL_HANDLE;
    VmaAllocation   allocation_     = nullptr;
    uint64_t        size_           = 0;
    VkDeviceAddress device_address_ = 0;
    void*           mapped_         = nullptr;
};

} // namespace rr::rhi
