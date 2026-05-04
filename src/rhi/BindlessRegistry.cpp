#include "rhi/BindlessRegistry.h"

#include "core/Log.h"
#include "rhi/AccelStructure.h"
#include "rhi/Device.h"
#include "rhi/internal/VulkanAccess.h"
#include "rhi/VulkanTypeCasts.h"

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <volk.h>
#include <vma/vk_mem_alloc.h>

#include <cassert>
#include <stdexcept>

namespace
{
// Round up 'value' to the next multiple of 'alignment' (must be a power-of-2).
constexpr VkDeviceSize align_up(VkDeviceSize value, VkDeviceSize alignment) noexcept
{
    return (value + alignment - 1) & ~(alignment - 1);
}
} // anonymous namespace

namespace rr::rhi
{

void BindlessRegistry::initialize(Device& device)
{
    // ── 1. Query heap properties ──────────────────────────────────────────
    VkPhysicalDeviceDescriptorHeapPropertiesEXT heap_props{};
    heap_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT;

    VkPhysicalDeviceProperties2 dev_props2{};
    dev_props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    dev_props2.pNext = &heap_props;
    vkGetPhysicalDeviceProperties2(vulkan::get_physical_device(device), &dev_props2);

    img_stride_  = heap_props.imageDescriptorSize;
    buf_stride_  = heap_props.bufferDescriptorSize;
    smpl_stride_ = heap_props.samplerDescriptorSize;

    const VkDeviceSize img_align  = heap_props.imageDescriptorAlignment;
    const VkDeviceSize buf_align  = heap_props.bufferDescriptorAlignment;
    const VkDeviceSize smpl_align = heap_props.samplerDescriptorAlignment;

    rr::core::log()->info(
        "BindlessRegistry: imageDescSize={} bufDescSize={} samplerDescSize={}",
        img_stride_, buf_stride_, smpl_stride_);

    // ── 2. Compute resource heap layout ──────────────────────────────────
    const VkDeviceSize res_reserved = heap_props.minResourceHeapReservedRange;
    texture_base_ = align_up(res_reserved,         img_align);
    storage_base_ = texture_base_ + kMaxTextures      * img_stride_;
    storage_base_ = align_up(storage_base_,          img_align);
    const VkDeviceSize img_region_end = storage_base_ + kMaxStorageImages * img_stride_;

    scene_base_ = align_up(img_region_end, buf_align);
    tlas_base_  = scene_base_ + kMaxSceneBuffers * buf_stride_;
    tlas_base_  = align_up(tlas_base_, buf_align);
    const VkDeviceSize buf_region_end = tlas_base_ + kMaxTlas * buf_stride_;

    const VkDeviceSize res_heap_size = align_up(
        buf_region_end, heap_props.resourceHeapAlignment ? heap_props.resourceHeapAlignment : 256);

    // ── 3. Compute sampler heap layout ────────────────────────────────────
    const VkDeviceSize smpl_reserved = heap_props.minSamplerHeapReservedRange;
    sampler_base_ = align_up(smpl_reserved, smpl_align);
    const VkDeviceSize smpl_heap_size = align_up(
        sampler_base_ + kMaxSamplers * smpl_stride_,
        heap_props.samplerHeapAlignment ? heap_props.samplerHeapAlignment : 256);

    rr::core::log()->info(
        "BindlessRegistry: resourceHeap={}B samplerHeap={}B",
        res_heap_size, smpl_heap_size);

    // ── 4. Allocate resource heap buffer ──────────────────────────────────
    {
        BufferDesc desc{};
        desc.size        = res_heap_size;
        desc.usage       = BufferUsage::DescriptorHeap |
                   BufferUsage::ShaderDeviceAddress;
        desc.memory_usage = MemoryUsage::Auto;
        // Host-access write + persistent map so vkWriteResourceDescriptorsEXT
        // can write directly into the buffer's mapped memory.
        desc.alloc_flags = AllocFlags::HostAccessSequentialWrite |
                   AllocFlags::Mapped;
        desc.debug_name  = "BindlessResourceHeap";

        resource_heap_.buf.create(device, desc);
        resource_heap_.mapped = resource_heap_.buf.mapped();
        resource_heap_.reserved_size = res_reserved;

        if (!resource_heap_.mapped)
        {
            throw std::runtime_error(
                "BindlessRegistry: resource heap buffer could not be persistently mapped.");
        }
    }

    // ── 5. Allocate sampler heap buffer ───────────────────────────────────
    {
        BufferDesc desc{};
        desc.size        = smpl_heap_size;
        desc.usage       = BufferUsage::DescriptorHeap |
                   BufferUsage::ShaderDeviceAddress;
        desc.memory_usage = MemoryUsage::Auto;
        desc.alloc_flags = AllocFlags::HostAccessSequentialWrite |
                   AllocFlags::Mapped;
        desc.debug_name  = "BindlessSamplerHeap";

        sampler_heap_.buf.create(device, desc);
        sampler_heap_.mapped = sampler_heap_.buf.mapped();
        sampler_heap_.reserved_size = smpl_reserved;

        if (!sampler_heap_.mapped)
        {
            throw std::runtime_error(
                "BindlessRegistry: sampler heap buffer could not be persistently mapped.");
        }
    }

    rr::core::log()->info(
        "BindlessRegistry: initialized. "
        "textureBase={:#x} storageBase={:#x} sceneBase={:#x} tlasBase={:#x} samplerBase={:#x}",
        texture_base_, storage_base_, scene_base_, tlas_base_, sampler_base_);
}

void BindlessRegistry::shutdown(Device& device)
{
    sampler_heap_.buf.destroy(device);
    resource_heap_.buf.destroy(device);
    sampler_heap_.mapped = nullptr;
    resource_heap_.mapped = nullptr;
    next_texture_ = next_storage_ = next_scene_buffer_ = next_tlas_ = next_sampler_ = 0;
}

// ── Registration ─────────────────────────────────────────────────────────────

void BindlessRegistry::write_resource(Device&                            device,
                                       const VkResourceDescriptorInfoEXT& info,
                                       VkDeviceSize                       byte_offset) const
{
    VkHostAddressRangeEXT dest{};
    dest.address = static_cast<uint8_t*>(resource_heap_.mapped) + byte_offset;
    dest.size    = (info.type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
                    info.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                       ? img_stride_
                       : buf_stride_;

    if (vkWriteResourceDescriptorsEXT(vulkan::get_device(device), 1, &info, &dest) != VK_SUCCESS)
    {
        throw std::runtime_error("vkWriteResourceDescriptorsEXT failed.");
    }
}

uint32_t BindlessRegistry::register_texture(Device&            device,
                                             const Image&       image,
                                             Format             format,
                                             ImageLayout        layout,
                                             ImageAspect        aspect,
                                             ImageViewType      view_type)
{
    if (next_texture_ >= kMaxTextures)
    {
        throw std::runtime_error("BindlessRegistry: texture capacity exhausted.");
    }

    VkImageViewCreateInfo view_ci{};
    view_ci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_ci.image    = image.handle();
    view_ci.viewType = to_vk_image_view_type(view_type);
    view_ci.format   = to_vk_format(format);
    view_ci.subresourceRange = {to_vk_image_aspect(aspect), 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};

    VkImageDescriptorInfoEXT img_info{};
    img_info.sType  = VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT;
    img_info.pView  = &view_ci;
    img_info.layout = to_vk_image_layout(layout);

    VkResourceDescriptorInfoEXT res_info{};
    res_info.sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT;
    res_info.type  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    res_info.data.pImage = &img_info;

    const uint32_t index = next_texture_++;
    write_resource(device, res_info, texture_base_ + index * img_stride_);
    return index;
}

uint32_t BindlessRegistry::register_storage_image(Device&         device,
                                                    const Image&     image,
                                                    Format           format,
                                                    ImageViewType    view_type)
{
    if (next_storage_ >= kMaxStorageImages)
    {
        throw std::runtime_error("BindlessRegistry: storage image capacity exhausted.");
    }

    VkImageViewCreateInfo view_ci{};
    view_ci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_ci.image    = image.handle();
    view_ci.viewType = to_vk_image_view_type(view_type);
    view_ci.format   = to_vk_format(format);
    view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};

    VkImageDescriptorInfoEXT img_info{};
    img_info.sType  = VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT;
    img_info.pView  = &view_ci;
    img_info.layout = VK_IMAGE_LAYOUT_GENERAL;

    VkResourceDescriptorInfoEXT res_info{};
    res_info.sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT;
    res_info.type  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    res_info.data.pImage = &img_info;

    const uint32_t index = next_storage_++;
    write_resource(device, res_info, storage_base_ + index * img_stride_);
    return index;
}

uint32_t BindlessRegistry::register_buffer(Device& device, const Buffer& buffer)
{
    if (next_scene_buffer_ >= kMaxSceneBuffers)
    {
        throw std::runtime_error("BindlessRegistry: scene buffer capacity exhausted.");
    }

    VkDeviceAddressRangeEXT addr_range{buffer.device_address(), buffer.size()};

    VkResourceDescriptorInfoEXT res_info{};
    res_info.sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT;
    res_info.type  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    res_info.data.pAddressRange = &addr_range;

    const uint32_t index = next_scene_buffer_++;
    write_resource(device, res_info, scene_base_ + index * buf_stride_);
    return index;
}

uint32_t BindlessRegistry::register_accel_struct(Device& device, const AccelStructure& accel_structure)
{
    if (next_tlas_ >= kMaxTlas)
    {
        throw std::runtime_error("BindlessRegistry: TLAS capacity exhausted.");
    }

    VkDeviceAddressRangeEXT addr_range{accel_structure.device_address(), accel_structure.buffer().size()};

    VkResourceDescriptorInfoEXT res_info{};
    res_info.sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT;
    res_info.type  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    res_info.data.pAddressRange = &addr_range;

    const uint32_t index = next_tlas_++;
    write_resource(device, res_info, tlas_base_ + index * buf_stride_);
    return index;
}

uint32_t BindlessRegistry::register_sampler(Device& device, const SamplerDesc& desc)
{
    if (next_sampler_ >= kMaxSamplers)
    {
        throw std::runtime_error("BindlessRegistry: sampler capacity exhausted.");
    }

    VkHostAddressRangeEXT dest{};
    dest.address = static_cast<uint8_t*>(sampler_heap_.mapped) + sampler_base_ +
                   next_sampler_ * smpl_stride_;
    dest.size = smpl_stride_;

    const VkSamplerCreateInfo info = to_vk_sampler_create_info(desc);
    if (vkWriteSamplerDescriptorsEXT(vulkan::get_device(device), 1, &info, &dest) != VK_SUCCESS)
    {
        throw std::runtime_error("vkWriteSamplerDescriptorsEXT failed.");
    }

    return next_sampler_++;
}

// ── Per-frame ────────────────────────────────────────────────────────────────

void BindlessRegistry::bind_heaps(VkCommandBuffer cmd) const
{
    // Bind resource heap.
    VkDeviceAddressRangeEXT res_range{resource_heap_.buf.device_address(),
                                       resource_heap_.buf.size()};
    VkBindHeapInfoEXT res_bind{};
    res_bind.sType               = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT;
    res_bind.heapRange           = res_range;
    res_bind.reservedRangeOffset = 0;
    res_bind.reservedRangeSize   = resource_heap_.reserved_size;
    vkCmdBindResourceHeapEXT(cmd, &res_bind);

    // Bind sampler heap.
    VkDeviceAddressRangeEXT smpl_range{sampler_heap_.buf.device_address(),
                                        sampler_heap_.buf.size()};
    VkBindHeapInfoEXT smpl_bind{};
    smpl_bind.sType               = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT;
    smpl_bind.heapRange           = smpl_range;
    smpl_bind.reservedRangeOffset = 0;
    smpl_bind.reservedRangeSize   = sampler_heap_.reserved_size;
    vkCmdBindSamplerHeapEXT(cmd, &smpl_bind);
}

void BindlessRegistry::heap_write_barrier(VkCommandBuffer cmd) const
{
    VkMemoryBarrier2 barrier{};
    barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
    barrier.dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_RESOURCE_HEAP_READ_BIT_EXT |
                            VK_ACCESS_2_SAMPLER_HEAP_READ_BIT_EXT;

    VkDependencyInfo dep{};
    dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers    = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}

} // namespace rr::rhi
