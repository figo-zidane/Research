#pragma once

#include "rhi/Buffer.h"

#include <cstdint>
#include <vector>
#include <volk.h>

namespace rr::rhi
{
class Device;

// ── Single acceleration structure (BLAS or TLAS) ─────────────────────────

class AccelStructure
{
public:
    AccelStructure() = default;
    AccelStructure(const AccelStructure&) = delete;
    AccelStructure& operator=(const AccelStructure&) = delete;
    AccelStructure(AccelStructure&& other) noexcept;
    AccelStructure& operator=(AccelStructure&& other) noexcept;
    ~AccelStructure() = default; // call destroy() explicitly

    void destroy(Device& device);

    [[nodiscard]] VkAccelerationStructureKHR handle()          const noexcept { return handle_; }
    [[nodiscard]] VkDeviceAddress            device_address()  const noexcept { return device_address_; }
    [[nodiscard]] bool                       is_valid()        const noexcept { return handle_ != VK_NULL_HANDLE; }

    // Non-owning pointer to the backing buffer (owned by AccelStructure)
    [[nodiscard]] const Buffer& buffer() const noexcept { return buffer_; }

    // ── Friends for builder functions ──────────────────────────────────────
    friend AccelStructure build_blas(Device& device,
                                     VkCommandBuffer cmd,
                                     VkDeviceAddress vertex_buffer_address,
                                     uint32_t        vertex_stride,
                                     uint32_t        vertex_count,
                                     VkDeviceAddress index_buffer_address,
                                     uint32_t        index_count,
                                     bool            allow_update,
                                     Buffer&         scratch_out);

    friend AccelStructure build_tlas(Device& device,
                                     VkCommandBuffer cmd,
                                     const std::vector<VkAccelerationStructureInstanceKHR>& instances,
                                     Buffer& scratch_out,
                                     Buffer& instance_buf_out);

private:
    VkAccelerationStructureKHR handle_         = VK_NULL_HANDLE;
    Buffer                     buffer_;         // backing memory
    VkDeviceAddress            device_address_ = 0;
};

// ── BLAS builder ──────────────────────────────────────────────────────────
// Builds a bottom-level AS for one mesh triangle geometry.
// scratch_out must be kept alive until the commands complete.
AccelStructure build_blas(Device&         device,
                           VkCommandBuffer cmd,
                           VkDeviceAddress vertex_buffer_address,
                           uint32_t        vertex_stride,
                           uint32_t        vertex_count,
                           VkDeviceAddress index_buffer_address,
                           uint32_t        index_count,
                           bool            allow_update,
                           Buffer&         scratch_out);

// ── TLAS builder ──────────────────────────────────────────────────────────
// Builds a top-level AS from the provided instance descriptors.
// scratch_out and instance_buf_out must be kept alive until the commands
// complete on the GPU.  Caller must call destroy() on both after wait-idle.
AccelStructure build_tlas(Device&         device,
                           VkCommandBuffer cmd,
                           const std::vector<VkAccelerationStructureInstanceKHR>& instances,
                           Buffer&         scratch_out,
                           Buffer&         instance_buf_out);

// ── Instance helper ───────────────────────────────────────────────────────
// Build a VkAccelerationStructureInstanceKHR from a 4x4 row-major transform,
// a BLAS device address, and a custom index (instance ID in shader).
VkAccelerationStructureInstanceKHR make_tlas_instance(
    const float       transform[16], // row-major mat4
    VkDeviceAddress   blas_address,
    uint32_t          instance_custom_index,
    uint32_t          hit_group_offset = 0,
    uint8_t           mask             = 0xFF);

} // namespace rr::rhi
