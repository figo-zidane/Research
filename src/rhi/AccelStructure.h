#pragma once

#include "rhi/Buffer.h"
#include "rhi/CommandRecorder.h"

#include <cstdint>
#include <span>
#include <vector>
#include <volk.h>

namespace rr::rhi
{
class Device;

struct BlasBuildInfo
{
    const Buffer* vertex_buffer = nullptr;
    uint64_t      vertex_offset = 0;
    uint32_t      vertex_stride = 0;
    uint32_t      vertex_count  = 0;
    const Buffer* index_buffer  = nullptr;
    uint64_t      index_offset  = 0;
    uint32_t      index_count   = 0;
    bool          allow_update  = false;
};

struct TlasInstance
{
    float    transform[16] = {};
    uint64_t blas_device_address = 0;
    uint32_t custom_index = 0;
    uint32_t hit_group_offset = 0;
    uint8_t  mask = 0xFF;
};

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

    [[nodiscard]] AccelStructureHandle handle() const noexcept { return to_handle(handle_); }
    [[nodiscard]] uint64_t             device_address() const noexcept { return device_address_; }
    [[nodiscard]] bool                 is_valid() const noexcept { return handle_ != VK_NULL_HANDLE; }

    // Non-owning pointer to the backing buffer (owned by AccelStructure)
    [[nodiscard]] const Buffer& buffer() const noexcept { return buffer_; }

    // ── Friends for builder functions ──────────────────────────────────────
    friend AccelStructure build_blas(Device& device,
                                     CommandRecorder recorder,
                                     const BlasBuildInfo& build_info,
                                     Buffer&         scratch_out);

    friend AccelStructure build_tlas(Device& device,
                                     CommandRecorder recorder,
                                     std::span<const TlasInstance> instances,
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
                           CommandRecorder recorder,
                           const BlasBuildInfo& build_info,
                           Buffer&         scratch_out);

// ── TLAS builder ──────────────────────────────────────────────────────────
// Builds a top-level AS from the provided instance descriptors.
// scratch_out and instance_buf_out must be kept alive until the commands
// complete on the GPU.  Caller must call destroy() on both after wait-idle.
AccelStructure build_tlas(Device&         device,
                           CommandRecorder recorder,
                           std::span<const TlasInstance> instances,
                           Buffer&         scratch_out,
                           Buffer&         instance_buf_out);

// ── Instance helper ───────────────────────────────────────────────────────
// Build a backend-agnostic TLAS instance from a 4x4 row-major transform.
TlasInstance make_tlas_instance(
    const float       transform[16],
    const AccelStructure& blas,
    uint32_t          instance_custom_index,
    uint32_t          hit_group_offset = 0,
    uint8_t           mask             = 0xFF);

} // namespace rr::rhi
