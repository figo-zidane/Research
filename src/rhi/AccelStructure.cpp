#include "rhi/AccelStructure.h"

// Include VMA through the same path used by Buffer.cpp and vma_impl.cpp.
// VMA_IMPLEMENTATION must NOT be defined here; it's defined in vma_impl.cpp.
#include <volk.h>
#include <vk_mem_alloc.h>

#include "core/Log.h"
#include "rhi/Buffer.h"
#include "rhi/Device.h"
#include "rhi/VulkanTypeCasts.h"

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
VkCommandBuffer as_vk_cmd(CommandRecorder recorder)
{
    return static_cast<VkCommandBuffer>(recorder.handle());
}

// Allocate a scratch buffer for AS builds.
Buffer allocate_scratch(Device& device, VkDeviceSize size)
{
    BufferDesc desc{};
    desc.size        = size;
    desc.usage       = BufferUsage::Storage |
                       BufferUsage::ShaderDeviceAddress;
    desc.memory_usage = MemoryUsage::GpuOnly;
    desc.alloc_flags  = AllocFlags::None;
    desc.debug_name   = "as_scratch";
    Buffer buf;
    buf.create(device, desc);
    return buf;
}

VkAccelerationStructureInstanceKHR to_vk_tlas_instance(const TlasInstance& instance)
{
    VkAccelerationStructureInstanceKHR vk_instance{};
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 4; ++col)
            vk_instance.transform.matrix[row][col] = instance.transform[row * 4 + col];
    vk_instance.instanceCustomIndex = instance.custom_index & 0xFFFFFF;
    vk_instance.mask = instance.mask;
    vk_instance.instanceShaderBindingTableRecordOffset = instance.hit_group_offset;
    vk_instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    vk_instance.accelerationStructureReference = instance.blas_device_address;
    return vk_instance;
}
} // anonymous namespace

// ── BLAS builder ──────────────────────────────────────────────────────────

AccelStructure build_blas(Device& device,
                          CommandRecorder recorder,
                          const BlasBuildInfo& blas_info,
                          Buffer& scratch_out)
{
    if (!blas_info.vertex_buffer || !blas_info.index_buffer)
        throw std::runtime_error("build_blas: vertex/index buffer must be provided.");

    const VkCommandBuffer cmd = as_vk_cmd(recorder);
    const VkDeviceAddress vertex_buffer_address = blas_info.vertex_buffer->device_address() + blas_info.vertex_offset;
    const VkDeviceAddress index_buffer_address = blas_info.index_buffer->device_address() + blas_info.index_offset;

    // Geometry description
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat  = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress  = vertex_buffer_address;
    triangles.vertexStride  = blas_info.vertex_stride;
    triangles.maxVertex     = blas_info.vertex_count - 1;
    triangles.indexType     = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress   = index_buffer_address;

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.geometry.triangles = triangles;
    geometry.flags              = VK_GEOMETRY_OPAQUE_BIT_KHR;

    uint32_t primitive_count = blas_info.index_count / 3;

    VkAccelerationStructureBuildGeometryInfoKHR build_info{};
    build_info.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build_info.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                               (blas_info.allow_update ? VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR : 0u);
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
        desc.usage       = BufferUsage::AccelStructureStorage |
                           BufferUsage::ShaderDeviceAddress;
        desc.memory_usage = MemoryUsage::GpuOnly;
        desc.debug_name   = "blas_buffer";
        blas.buffer_.create(device, desc);
    }

    // Create AS object
    {
        VkAccelerationStructureCreateInfoKHR create_info{};
        create_info.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        create_info.buffer = from_handle<VkBuffer>(blas.buffer_.handle());
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

AccelStructure build_tlas(Device& device,
                          CommandRecorder recorder,
                          std::span<const TlasInstance> instances,
                          Buffer& scratch_out,
                          Buffer& instance_buf_out)
{
    if (instances.empty())
        throw std::runtime_error("build_tlas: no instances provided.");

    const VkCommandBuffer cmd = as_vk_cmd(recorder);
    std::vector<VkAccelerationStructureInstanceKHR> vk_instances;
    vk_instances.reserve(instances.size());
    for (const TlasInstance& instance : instances)
        vk_instances.push_back(to_vk_tlas_instance(instance));

    // Upload instance data to a device-local buffer
    Buffer& instance_buf = instance_buf_out;
    {
        const VkDeviceSize inst_bytes = sizeof(VkAccelerationStructureInstanceKHR) * vk_instances.size();
        BufferDesc desc{};
        desc.size        = inst_bytes;
        desc.usage       = BufferUsage::AccelStructureBuildInput |
                           BufferUsage::ShaderDeviceAddress;
        desc.memory_usage = MemoryUsage::CpuToGpu;
        desc.alloc_flags  = AllocFlags::HostAccessSequentialWrite |
                            AllocFlags::Mapped;
        desc.debug_name   = "tlas_instances";
        instance_buf.create(device, desc);
        std::memcpy(instance_buf.mapped(), vk_instances.data(), inst_bytes);
    }

    VkAccelerationStructureGeometryInstancesDataKHR inst_data{};
    inst_data.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    inst_data.data.deviceAddress = instance_buf.device_address();

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType       = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances = inst_data;

    uint32_t instance_count = static_cast<uint32_t>(vk_instances.size());

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
        desc.usage       = BufferUsage::AccelStructureStorage |
                           BufferUsage::ShaderDeviceAddress;
        desc.memory_usage = MemoryUsage::GpuOnly;
        desc.debug_name   = "tlas_buffer";
        tlas.buffer_.create(device, desc);
    }

    {
        VkAccelerationStructureCreateInfoKHR create_info{};
        create_info.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        create_info.buffer = from_handle<VkBuffer>(tlas.buffer_.handle());
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

TlasInstance make_tlas_instance(const float transform[16],
                                const AccelStructure& blas,
                                uint32_t instance_custom_index,
                                uint32_t hit_group_offset,
                                uint8_t mask)
{
    TlasInstance inst{};
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 4; ++col)
            inst.transform[row * 4 + col] = transform[row * 4 + col];
    inst.blas_device_address = blas.device_address();
    inst.custom_index = instance_custom_index;
    inst.hit_group_offset = hit_group_offset;
    inst.mask = mask;
    return inst;
}

} // namespace rr::rhi
