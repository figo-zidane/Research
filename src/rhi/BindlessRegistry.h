#pragma once

#include "rhi/Buffer.h"
#include "rhi/Image.h"

#include <cstdint>
#include <volk.h>

namespace rr::rhi
{
class Device;

// BindlessRegistry manages two VK_EXT_descriptor_heap buffers:
//
//   resource heap — holds image, storage-image, SSBO and TLAS descriptors
//   sampler heap  — holds sampler descriptors
//
// Callers obtain an opaque integer *index* for each registered resource and
// pass that index to shaders via push constants.  The heap binds are issued
// once at the start of each command buffer (bind_heaps).
//
// Heap layout (resource heap, byte offsets from buffer start):
//
//  [0 … reserved)                                       implementation reserved
//  [texture_base … + kMaxTextures * imgStride)          gTextures[]
//  [storage_base … + kMaxStorageImages * imgStride)     gStorageImages[]
//  [scene_base   … + kMaxSceneBuffers * bufStride)      gScene[]
//  [tlas_base    … + kMaxTlas * bufStride)              gTLAS[]
//
// Heap layout (sampler heap):
//  [0 … reserved)                                       implementation reserved
//  [sampler_base … + kMaxSamplers * smplStride)         gSamplers[]
//
// These base offsets and strides are exposed via accessors so Pipeline can
// supply VkDescriptorMappingSourceConstantOffsetEXT when creating pipelines.

class BindlessRegistry
{
public:
    // ── Capacity constants ────────────────────────────────────────────────
    static constexpr uint32_t kMaxTextures      = 65536;
    static constexpr uint32_t kMaxStorageImages = 4096;
    static constexpr uint32_t kMaxSceneBuffers  = 4096;
    static constexpr uint32_t kMaxTlas          = 16;
    static constexpr uint32_t kMaxSamplers      = 256;

    BindlessRegistry() = default;
    BindlessRegistry(const BindlessRegistry&) = delete;
    BindlessRegistry& operator=(const BindlessRegistry&) = delete;
    ~BindlessRegistry() = default;

    void initialize(Device& device);
    void shutdown(Device& device);

    // ── Resource registration ─────────────────────────────────────────────

    // Register a sampled texture (VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE).
    // Returns the gTextures[] index.
    uint32_t register_texture(Device&            device,
                        const Image&       image,
                        Format             format,
                        ImageLayout        layout,
                        ImageAspect        aspect,
                        ImageViewType      view_type = ImageViewType::View2D);

    // Register a storage image (VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).
    // Returns the gStorageImages[] index.
    uint32_t register_storage_image(Device&   device,
                            const Image& image,
                            Format       format,
                            ImageViewType view_type = ImageViewType::View2D);

    // Register a device-addressable buffer as an SSBO
    // (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).
    // Returns the gScene[] index.
    uint32_t register_buffer(Device&         device,
                             VkDeviceAddress address,
                             VkDeviceSize    size);

    // Register a TLAS (VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR).
    // Returns the gTLAS[] index.
    uint32_t register_accel_struct(Device&         device,
                                   VkDeviceAddress address,
                                   VkDeviceSize    size);

    // Register a sampler and write it to the sampler heap.
    // Returns the gSamplers[] index.
    uint32_t register_sampler(Device& device, const VkSamplerCreateInfo& info);

    // ── Per-frame command buffer setup ───────────────────────────────────

    // Bind both heaps.  Must be called once at the beginning of each
    // command buffer that accesses bindless resources.
    void bind_heaps(VkCommandBuffer cmd) const;

    // Issue a pipeline barrier to make newly-written heap data visible to the
    // GPU before commands that read from the heaps execute.
    void heap_write_barrier(VkCommandBuffer cmd) const;

    // ── Offset/stride accessors for Pipeline heap mapping ────────────────

    [[nodiscard]] VkDeviceSize texture_heap_offset()       const noexcept { return texture_base_; }
    [[nodiscard]] VkDeviceSize storage_image_heap_offset() const noexcept { return storage_base_; }
    [[nodiscard]] VkDeviceSize scene_buffer_heap_offset()  const noexcept { return scene_base_; }
    [[nodiscard]] VkDeviceSize tlas_heap_offset()          const noexcept { return tlas_base_; }
    [[nodiscard]] VkDeviceSize sampler_heap_offset()       const noexcept { return sampler_base_; }

    [[nodiscard]] VkDeviceSize image_descriptor_stride()   const noexcept { return img_stride_; }
    [[nodiscard]] VkDeviceSize buffer_descriptor_stride()  const noexcept { return buf_stride_; }
    [[nodiscard]] VkDeviceSize sampler_descriptor_stride() const noexcept { return smpl_stride_; }

private:
    struct HeapBuffer
    {
        Buffer       buf;
        void*        mapped = nullptr; // persistently mapped CPU pointer
        VkDeviceSize reserved_size = 0;
    };

    // Write a resource descriptor to the resource heap at a specific byte offset.
    void write_resource(Device&                          device,
                        const VkResourceDescriptorInfoEXT& info,
                        VkDeviceSize                     byte_offset) const;

    HeapBuffer resource_heap_{};
    HeapBuffer sampler_heap_{};

    // ── Descriptor sizes/alignments (queried from physical device) ────────
    VkDeviceSize img_stride_  = 0;
    VkDeviceSize buf_stride_  = 0;
    VkDeviceSize smpl_stride_ = 0;

    // ── Base byte offsets inside each heap buffer ─────────────────────────
    VkDeviceSize texture_base_ = 0;
    VkDeviceSize storage_base_ = 0;
    VkDeviceSize scene_base_   = 0;
    VkDeviceSize tlas_base_    = 0;
    VkDeviceSize sampler_base_ = 0;

    // ── Allocation counters (bump allocator; no free list for now) ────────
    uint32_t next_texture_      = 0;
    uint32_t next_storage_      = 0;
    uint32_t next_scene_buffer_ = 0;
    uint32_t next_tlas_         = 0;
    uint32_t next_sampler_      = 0;
};

} // namespace rr::rhi
