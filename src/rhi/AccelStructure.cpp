#include "rhi/AccelStructure.h"

// Include VMA through the same path used by Buffer.cpp and vma_impl.cpp.
// VMA_IMPLEMENTATION must NOT be defined here; it's defined in vma_impl.cpp.
#include <volk.h>
#include <vk_mem_alloc.h>

#include "core/Log.h"
#include "rhi/Buffer.h"
#include "rhi/Device.h"

#include <stdexcept>
#include <cstring>

namespace rr::rhi
{

// ── AccelStructure move semantics ─────────────────────────────────────────

AccelStructure::AccelStructure(AccelStructure&& other) noexcept
    : handle_(other.handle_)
    , buffer_(std::move(other.buffer_))
    , device_address_(other.device_address_)
{
    other.handle_         = VK_NULL_HANDLE;
    other.device_address_ = 0;
}

AccelStructure& AccelStructure::operator=(AccelStructure&& other) noexcept
{
    if (this != &other)
    {
        handle_         = other.handle_;
        buffer_         = std::move(other.buffer_);
        device_address_ = other.device_address_;
        other.handle_         = VK_NULL_HANDLE;
        other.device_address_ = 0;
    }
    return *this;
}

void AccelStructure::destroy(Device& device)
{
    if (handle_ != VK_NULL_HANDLE)
    {
        vkDestroyAccelerationStructureKHR(device.device(), handle_, nullptr);
        handle_ = VK_NULL_HANDLE;
    }
    if (buffer_.is_valid())
        buffer_.destroy(device);
    device_address_ = 0;
}

// ── Internal helper ───────────────────────────────────────────────────────

namespace
{
// Allocate a scratch buffer for AS builds.
Buffer allocate_scratch(Device& device, VkDeviceSize size)
{
    BufferDesc desc{};
    desc.size        = size;
    desc.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    desc.memory_usage = 2; // VMA_MEMORY_USAGE_GPU_ONLY
    desc.alloc_flags  = 0;
    desc.debug_name   = "as_scratch";
    Buffer buf;
    buf.create(device, desc);
    return buf;
}
} // anonymous namespace

// ── BLAS builder ──────────────────────────────────────────────────────────

AccelStructure build_blas(Device&         device,
                           VkCommandBuffer cmd,
                           VkDeviceAddress vertex_buffer_address,
                           uint32_t        vertex_stride,
                           uint32_t        vertex_count,
                           VkDeviceAddress index_buffer_address,
                           uint32_t        index_count,
                           bool            allow_update,
                           Buffer&         scratch_out)
{
    // Geometry description
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat  = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress  = vertex_buffer_address;
    triangles.vertexStride  = vertex_stride;
    triangles.maxVertex     = vertex_count - 1;
    triangles.indexType     = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress   = index_buffer_address;

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.geometry.triangles = triangles;
    geometry.flags              = VK_GEOMETRY_OPAQUE_BIT_KHR;

    uint32_t primitive_count = index_count / 3;

    VkAccelerationStructureBuildGeometryInfoKHR build_info{};
    build_info.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build_info.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                               (allow_update ? VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR : 0u);
    build_info.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_info.geometryCount = 1;
    build_info.pGeometries   = &geometry;

    // Query sizes
    VkAccelerationStructureBuildSizesInfoKHR size_info{};
    size_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(
        device.device(),
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build_info, &primitive_count, &size_info);

    // Allocate AS backing buffer
    AccelStructure blas;
    {
        BufferDesc desc{};
        desc.size        = size_info.accelerationStructureSize;
        desc.usage       = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        desc.memory_usage = 2; // VMA_MEMORY_USAGE_GPU_ONLY
        desc.debug_name   = "blas_buffer";
        blas.buffer_.create(device, desc);
    }

    // Create AS object
    {
        VkAccelerationStructureCreateInfoKHR create_info{};
        create_info.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        create_info.buffer = blas.buffer_.handle();
        create_info.size   = size_info.accelerationStructureSize;
        create_info.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        if (vkCreateAccelerationStructureKHR(device.device(), &create_info, nullptr, &blas.handle_) != VK_SUCCESS)
            throw std::runtime_error("vkCreateAccelerationStructureKHR (BLAS) failed.");
    }

    // Get device address
    {
        VkAccelerationStructureDeviceAddressInfoKHR addr_info{};
        addr_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addr_info.accelerationStructure = blas.handle_;
        blas.device_address_ = vkGetAccelerationStructureDeviceAddressKHR(device.device(), &addr_info);
    }

    // Allocate scratch
    scratch_out = allocate_scratch(device, size_info.buildScratchSize);

    // Record build
    build_info.dstAccelerationStructure  = blas.handle_;
    build_info.scratchData.deviceAddress = scratch_out.device_address();

    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount  = primitive_count;
    range.primitiveOffset = 0;
    range.firstVertex     = 0;
    range.transformOffset = 0;

    const VkAccelerationStructureBuildRangeInfoKHR* ranges[1] = {&range};
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, ranges);

    // Barrier: AS build write → AS read
    VkMemoryBarrier2 barrier{};
    barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    VkDependencyInfo dep{};
    dep.sType               = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount  = 1;
    dep.pMemoryBarriers     = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);

    return blas;
}

// ── TLAS builder ──────────────────────────────────────────────────────────

AccelStructure build_tlas(Device&         device,
                           VkCommandBuffer cmd,
                           const std::vector<VkAccelerationStructureInstanceKHR>& instances,
                           Buffer&         scratch_out,
                           Buffer&         instance_buf_out)
{
    if (instances.empty())
        throw std::runtime_error("build_tlas: no instances provided.");

    // Upload instance data to a device-local buffer
    Buffer& instance_buf = instance_buf_out;
    {
        VkDeviceSize inst_bytes = sizeof(VkAccelerationStructureInstanceKHR) * instances.size();
        BufferDesc desc{};
        desc.size        = inst_bytes;
        desc.usage       = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        desc.memory_usage = 3; // VMA_MEMORY_USAGE_CPU_TO_GPU
        desc.alloc_flags  = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT;
        desc.debug_name   = "tlas_instances";
        instance_buf.create(device, desc);
        std::memcpy(instance_buf.mapped(), instances.data(), inst_bytes);
    }

    VkAccelerationStructureGeometryInstancesDataKHR inst_data{};
    inst_data.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    inst_data.data.deviceAddress = instance_buf.device_address();

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType       = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances = inst_data;

    uint32_t instance_count = static_cast<uint32_t>(instances.size());

    VkAccelerationStructureBuildGeometryInfoKHR build_info{};
    build_info.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    build_info.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build_info.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_info.geometryCount = 1;
    build_info.pGeometries   = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR size_info{};
    size_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(
        device.device(),
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build_info, &instance_count, &size_info);

    // Allocate TLAS backing buffer
    AccelStructure tlas;
    {
        BufferDesc desc{};
        desc.size        = size_info.accelerationStructureSize;
        desc.usage       = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        desc.memory_usage = 2;
        desc.debug_name   = "tlas_buffer";
        tlas.buffer_.create(device, desc);
    }

    {
        VkAccelerationStructureCreateInfoKHR create_info{};
        create_info.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        create_info.buffer = tlas.buffer_.handle();
        create_info.size   = size_info.accelerationStructureSize;
        create_info.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        if (vkCreateAccelerationStructureKHR(device.device(), &create_info, nullptr, &tlas.handle_) != VK_SUCCESS)
            throw std::runtime_error("vkCreateAccelerationStructureKHR (TLAS) failed.");
    }

    {
        VkAccelerationStructureDeviceAddressInfoKHR addr_info{};
        addr_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addr_info.accelerationStructure = tlas.handle_;
        tlas.device_address_ = vkGetAccelerationStructureDeviceAddressKHR(device.device(), &addr_info);
    }

    scratch_out = allocate_scratch(device, size_info.buildScratchSize);

    build_info.dstAccelerationStructure  = tlas.handle_;
    build_info.scratchData.deviceAddress = scratch_out.device_address();

    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = instance_count;
    const VkAccelerationStructureBuildRangeInfoKHR* ranges[1] = {&range};
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, ranges);

    // Barrier: AS build → ray tracing read
    VkMemoryBarrier2 barrier{};
    barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    VkDependencyInfo dep{};
    dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers    = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);

    // instance_buf_out (alias: instance_buf) must stay alive until the
    // queue finishes.  The caller is responsible for destroying it.

    return tlas;
}

// ── Instance helper ───────────────────────────────────────────────────────

VkAccelerationStructureInstanceKHR make_tlas_instance(
    const float       transform[16],
    VkDeviceAddress   blas_address,
    uint32_t          instance_custom_index,
    uint32_t          hit_group_offset,
    uint8_t           mask)
{
    VkAccelerationStructureInstanceKHR inst{};
    // 3x4 row-major transform (Vulkan uses this layout)
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 4; ++col)
            inst.transform.matrix[row][col] = transform[row * 4 + col];
    inst.instanceCustomIndex             = instance_custom_index & 0xFFFFFF;
    inst.mask                            = mask;
    inst.instanceShaderBindingTableRecordOffset = hit_group_offset;
    inst.flags                           = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    inst.accelerationStructureReference  = blas_address;
    return inst;
}

} // namespace rr::rhi
