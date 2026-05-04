#pragma once

#include "rhi/Types.h"

#include <cstdint>

struct VmaAllocation_T;

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

// Thin RAII wrapper around a buffer resource plus its allocation.
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

    [[nodiscard]] BufferHandle handle() const noexcept { return buffer_; }
    [[nodiscard]] uint64_t     size() const noexcept { return size_; }
    [[nodiscard]] uint64_t     device_address() const noexcept { return device_address_; }
    [[nodiscard]] void*        mapped() const noexcept { return mapped_; }
    [[nodiscard]] bool         is_valid() const noexcept { return buffer_ != 0; }

private:
    BufferHandle    buffer_         = 0;
    VmaAllocation_T* allocation_    = nullptr;
    uint64_t        size_           = 0;
    uint64_t        device_address_ = 0;
    void*           mapped_         = nullptr;
};

} // namespace rr::rhi
