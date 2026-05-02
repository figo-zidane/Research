#pragma once

#include "rhi/Barrier.h"
#include "rhi/Rendering.h"
#include "rhi/SamplerDesc.h"
#include "rhi/Types.h"

#include <volk.h>
#include <vma/vk_mem_alloc.h>

#include <stdexcept>

namespace rr::rhi
{
[[nodiscard]] inline VkFormat to_vk_format(Format format)
{
    return static_cast<VkFormat>(format);
}

[[nodiscard]] inline Format from_vk_format(VkFormat format)
{
    return static_cast<Format>(format);
}

[[nodiscard]] inline VkExtent3D to_vk_extent3d(Extent3D extent)
{
    return VkExtent3D{extent.width, extent.height, extent.depth};
}

[[nodiscard]] inline VkImageType to_vk_image_type(ImageType type)
{
    switch (type)
    {
    case ImageType::Image1D: return VK_IMAGE_TYPE_1D;
    case ImageType::Image2D: return VK_IMAGE_TYPE_2D;
    case ImageType::Image3D: return VK_IMAGE_TYPE_3D;
    }
    throw std::runtime_error("Unsupported ImageType.");
}

[[nodiscard]] inline VkImageViewType to_vk_image_view_type(ImageViewType type)
{
    switch (type)
    {
    case ImageViewType::View1D: return VK_IMAGE_VIEW_TYPE_1D;
    case ImageViewType::View2D: return VK_IMAGE_VIEW_TYPE_2D;
    case ImageViewType::View3D: return VK_IMAGE_VIEW_TYPE_3D;
    case ImageViewType::ViewCube: return VK_IMAGE_VIEW_TYPE_CUBE;
    case ImageViewType::View1DArray: return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
    case ImageViewType::View2DArray: return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    case ImageViewType::ViewCubeArray: return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    }
    throw std::runtime_error("Unsupported ImageViewType.");
}

[[nodiscard]] inline VkImageLayout to_vk_image_layout(ImageLayout layout)
{
    switch (layout)
    {
    case ImageLayout::Undefined: return VK_IMAGE_LAYOUT_UNDEFINED;
    case ImageLayout::General: return VK_IMAGE_LAYOUT_GENERAL;
    case ImageLayout::ColorAttachment: return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    case ImageLayout::DepthStencilAttachment: return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    case ImageLayout::ShaderReadOnly: return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    case ImageLayout::TransferSrc: return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    case ImageLayout::TransferDst: return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    case ImageLayout::Present: return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    }
    throw std::runtime_error("Unsupported ImageLayout.");
}

[[nodiscard]] inline VkImageAspectFlags to_vk_image_aspect(ImageAspect aspect)
{
    VkImageAspectFlags flags = 0;
    if (has_any_flag(aspect, ImageAspect::Color))   flags |= VK_IMAGE_ASPECT_COLOR_BIT;
    if (has_any_flag(aspect, ImageAspect::Depth))   flags |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if (has_any_flag(aspect, ImageAspect::Stencil)) flags |= VK_IMAGE_ASPECT_STENCIL_BIT;
    return flags;
}

[[nodiscard]] inline ImageAspect from_vk_image_aspect(VkImageAspectFlags aspect)
{
    ImageAspect result = ImageAspect::None;
    if ((aspect & VK_IMAGE_ASPECT_COLOR_BIT) != 0)   result |= ImageAspect::Color;
    if ((aspect & VK_IMAGE_ASPECT_DEPTH_BIT) != 0)   result |= ImageAspect::Depth;
    if ((aspect & VK_IMAGE_ASPECT_STENCIL_BIT) != 0) result |= ImageAspect::Stencil;
    return result;
}

[[nodiscard]] inline VkImageUsageFlags to_vk_image_usage(ImageUsage usage)
{
    VkImageUsageFlags flags = 0;
    if (has_any_flag(usage, ImageUsage::TransferSrc))            flags |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (has_any_flag(usage, ImageUsage::TransferDst))            flags |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (has_any_flag(usage, ImageUsage::Sampled))                flags |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if (has_any_flag(usage, ImageUsage::Storage))                flags |= VK_IMAGE_USAGE_STORAGE_BIT;
    if (has_any_flag(usage, ImageUsage::ColorAttachment))        flags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (has_any_flag(usage, ImageUsage::DepthStencilAttachment)) flags |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (has_any_flag(usage, ImageUsage::HostTransfer))           flags |= VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT;
    return flags;
}

[[nodiscard]] inline VkBufferUsageFlags to_vk_buffer_usage(BufferUsage usage)
{
    VkBufferUsageFlags flags = 0;
    if (has_any_flag(usage, BufferUsage::TransferSrc))             flags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (has_any_flag(usage, BufferUsage::TransferDst))             flags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (has_any_flag(usage, BufferUsage::Vertex))                  flags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    if (has_any_flag(usage, BufferUsage::Index))                   flags |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if (has_any_flag(usage, BufferUsage::Storage))                 flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (has_any_flag(usage, BufferUsage::Uniform))                 flags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (has_any_flag(usage, BufferUsage::ShaderDeviceAddress))     flags |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    if (has_any_flag(usage, BufferUsage::AccelStructureBuildInput)) flags |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    if (has_any_flag(usage, BufferUsage::AccelStructureStorage))   flags |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
    if (has_any_flag(usage, BufferUsage::IndirectBuffer))          flags |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    if (has_any_flag(usage, BufferUsage::DescriptorHeap))          flags |= VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT;
    return flags;
}

[[nodiscard]] inline VmaMemoryUsage to_vma_memory_usage(MemoryUsage usage)
{
    switch (usage)
    {
    case MemoryUsage::GpuOnly: return VMA_MEMORY_USAGE_GPU_ONLY;
    case MemoryUsage::CpuToGpu: return VMA_MEMORY_USAGE_CPU_TO_GPU;
    case MemoryUsage::GpuToCpu: return VMA_MEMORY_USAGE_GPU_TO_CPU;
    case MemoryUsage::CpuOnly: return VMA_MEMORY_USAGE_CPU_ONLY;
    case MemoryUsage::Auto: return VMA_MEMORY_USAGE_AUTO;
    }
    throw std::runtime_error("Unsupported MemoryUsage.");
}

[[nodiscard]] inline VmaAllocationCreateFlags to_vma_allocation_flags(AllocFlags flags)
{
    VmaAllocationCreateFlags vk_flags = 0;
    if (has_any_flag(flags, AllocFlags::HostAccessSequentialWrite)) vk_flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    if (has_any_flag(flags, AllocFlags::HostAccessRandom))          vk_flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    if (has_any_flag(flags, AllocFlags::Mapped))                    vk_flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
    return vk_flags;
}

[[nodiscard]] inline VkCompareOp to_vk_compare_op(CompareOp op);

[[nodiscard]] inline VkFilter to_vk_sampler_filter(SamplerFilter filter)
{
    switch (filter)
    {
    case SamplerFilter::Nearest: return VK_FILTER_NEAREST;
    case SamplerFilter::Linear: return VK_FILTER_LINEAR;
    }
    throw std::runtime_error("Unsupported SamplerFilter.");
}

[[nodiscard]] inline VkSamplerMipmapMode to_vk_sampler_mipmap_mode(SamplerMipmapMode mode)
{
    switch (mode)
    {
    case SamplerMipmapMode::Nearest: return VK_SAMPLER_MIPMAP_MODE_NEAREST;
    case SamplerMipmapMode::Linear: return VK_SAMPLER_MIPMAP_MODE_LINEAR;
    }
    throw std::runtime_error("Unsupported SamplerMipmapMode.");
}

[[nodiscard]] inline VkSamplerAddressMode to_vk_sampler_address_mode(SamplerAddressMode mode)
{
    switch (mode)
    {
    case SamplerAddressMode::Repeat: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case SamplerAddressMode::ClampToEdge: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    }
    throw std::runtime_error("Unsupported SamplerAddressMode.");
}

[[nodiscard]] inline VkBorderColor to_vk_sampler_border_color(SamplerBorderColor color)
{
    switch (color)
    {
    case SamplerBorderColor::FloatTransparentBlack: return VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    }
    throw std::runtime_error("Unsupported SamplerBorderColor.");
}

[[nodiscard]] inline VkSamplerCreateInfo to_vk_sampler_create_info(const SamplerDesc& desc)
{
    VkSamplerCreateInfo info{};
    info.sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.magFilter               = to_vk_sampler_filter(desc.mag_filter);
    info.minFilter               = to_vk_sampler_filter(desc.min_filter);
    info.mipmapMode              = to_vk_sampler_mipmap_mode(desc.mipmap_mode);
    info.addressModeU            = to_vk_sampler_address_mode(desc.address_mode_u);
    info.addressModeV            = to_vk_sampler_address_mode(desc.address_mode_v);
    info.addressModeW            = to_vk_sampler_address_mode(desc.address_mode_w);
    info.mipLodBias              = desc.mip_lod_bias;
    info.anisotropyEnable        = desc.anisotropy_enable ? VK_TRUE : VK_FALSE;
    info.maxAnisotropy           = desc.max_anisotropy;
    info.compareEnable           = desc.compare_enable ? VK_TRUE : VK_FALSE;
    info.compareOp               = to_vk_compare_op(desc.compare_op);
    info.minLod                  = desc.min_lod;
    info.maxLod                  = desc.max_lod;
    info.borderColor             = to_vk_sampler_border_color(desc.border_color);
    info.unnormalizedCoordinates = desc.unnormalized_coordinates ? VK_TRUE : VK_FALSE;
    return info;
}

[[nodiscard]] inline VkSampleCountFlagBits to_vk_sample_count(SampleCount samples)
{
    switch (samples)
    {
    case SampleCount::Count1: return VK_SAMPLE_COUNT_1_BIT;
    case SampleCount::Count2: return VK_SAMPLE_COUNT_2_BIT;
    case SampleCount::Count4: return VK_SAMPLE_COUNT_4_BIT;
    case SampleCount::Count8: return VK_SAMPLE_COUNT_8_BIT;
    case SampleCount::Count16: return VK_SAMPLE_COUNT_16_BIT;
    case SampleCount::Count32: return VK_SAMPLE_COUNT_32_BIT;
    case SampleCount::Count64: return VK_SAMPLE_COUNT_64_BIT;
    }
    throw std::runtime_error("Unsupported SampleCount.");
}

[[nodiscard]] inline VkPipelineStageFlags2 to_vk_pipeline_stage(PipelineStage stage)
{
    VkPipelineStageFlags2 flags = 0;
    if (has_any_flag(stage, PipelineStage::TopOfPipe))             flags |= VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    if (has_any_flag(stage, PipelineStage::AllCommands))           flags |= VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    if (has_any_flag(stage, PipelineStage::ComputeShader))         flags |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    if (has_any_flag(stage, PipelineStage::FragmentShader))        flags |= VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    if (has_any_flag(stage, PipelineStage::ColorAttachmentOutput)) flags |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    if (has_any_flag(stage, PipelineStage::EarlyFragmentTests))    flags |= VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
    if (has_any_flag(stage, PipelineStage::LateFragmentTests))     flags |= VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    if (has_any_flag(stage, PipelineStage::RayTracingShader))      flags |= VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    if (has_any_flag(stage, PipelineStage::Transfer))              flags |= VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    if (has_any_flag(stage, PipelineStage::Host))                  flags |= VK_PIPELINE_STAGE_2_HOST_BIT;
    return flags;
}

[[nodiscard]] inline VkAccessFlags2 to_vk_access_flags(AccessFlags access)
{
    VkAccessFlags2 flags = 0;
    if (has_any_flag(access, AccessFlags::ShaderRead))                  flags |= VK_ACCESS_2_SHADER_READ_BIT;
    if (has_any_flag(access, AccessFlags::ShaderWrite))                 flags |= VK_ACCESS_2_SHADER_WRITE_BIT;
    if (has_any_flag(access, AccessFlags::ColorAttachmentWrite))        flags |= VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    if (has_any_flag(access, AccessFlags::DepthStencilAttachmentWrite)) flags |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    if (has_any_flag(access, AccessFlags::TransferRead))                flags |= VK_ACCESS_2_TRANSFER_READ_BIT;
    if (has_any_flag(access, AccessFlags::TransferWrite))               flags |= VK_ACCESS_2_TRANSFER_WRITE_BIT;
    if (has_any_flag(access, AccessFlags::HostWrite))                   flags |= VK_ACCESS_2_HOST_WRITE_BIT;
    if (has_any_flag(access, AccessFlags::ResourceHeapRead))            flags |= VK_ACCESS_2_RESOURCE_HEAP_READ_BIT_EXT;
    if (has_any_flag(access, AccessFlags::SamplerHeapRead))             flags |= VK_ACCESS_2_SAMPLER_HEAP_READ_BIT_EXT;
    return flags;
}

[[nodiscard]] inline VkAttachmentLoadOp to_vk_load_op(LoadOp op)
{
    switch (op)
    {
    case LoadOp::Load: return VK_ATTACHMENT_LOAD_OP_LOAD;
    case LoadOp::Clear: return VK_ATTACHMENT_LOAD_OP_CLEAR;
    case LoadOp::DontCare: return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    }
    throw std::runtime_error("Unsupported LoadOp.");
}

[[nodiscard]] inline VkAttachmentStoreOp to_vk_store_op(StoreOp op)
{
    switch (op)
    {
    case StoreOp::Store: return VK_ATTACHMENT_STORE_OP_STORE;
    case StoreOp::DontCare: return VK_ATTACHMENT_STORE_OP_DONT_CARE;
    }
    throw std::runtime_error("Unsupported StoreOp.");
}

[[nodiscard]] inline VkCullModeFlags to_vk_cull_mode(CullMode mode)
{
    switch (mode)
    {
    case CullMode::None: return VK_CULL_MODE_NONE;
    case CullMode::Front: return VK_CULL_MODE_FRONT_BIT;
    case CullMode::Back: return VK_CULL_MODE_BACK_BIT;
    case CullMode::FrontAndBack: return VK_CULL_MODE_FRONT_AND_BACK;
    }
    throw std::runtime_error("Unsupported CullMode.");
}

[[nodiscard]] inline VkFrontFace to_vk_front_face(FrontFace face)
{
    switch (face)
    {
    case FrontFace::CounterClockwise: return VK_FRONT_FACE_COUNTER_CLOCKWISE;
    case FrontFace::Clockwise: return VK_FRONT_FACE_CLOCKWISE;
    }
    throw std::runtime_error("Unsupported FrontFace.");
}

[[nodiscard]] inline VkPrimitiveTopology to_vk_primitive_topology(PrimitiveTopology topology)
{
    switch (topology)
    {
    case PrimitiveTopology::PointList: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    case PrimitiveTopology::LineList: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case PrimitiveTopology::LineStrip: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case PrimitiveTopology::TriangleList: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    case PrimitiveTopology::TriangleStrip: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    }
    throw std::runtime_error("Unsupported PrimitiveTopology.");
}

[[nodiscard]] inline VkCompareOp to_vk_compare_op(CompareOp op)
{
    switch (op)
    {
    case CompareOp::Never: return VK_COMPARE_OP_NEVER;
    case CompareOp::Less: return VK_COMPARE_OP_LESS;
    case CompareOp::Equal: return VK_COMPARE_OP_EQUAL;
    case CompareOp::LessOrEqual: return VK_COMPARE_OP_LESS_OR_EQUAL;
    case CompareOp::Greater: return VK_COMPARE_OP_GREATER;
    case CompareOp::NotEqual: return VK_COMPARE_OP_NOT_EQUAL;
    case CompareOp::GreaterOrEqual: return VK_COMPARE_OP_GREATER_OR_EQUAL;
    case CompareOp::Always: return VK_COMPARE_OP_ALWAYS;
    }
    throw std::runtime_error("Unsupported CompareOp.");
}

[[nodiscard]] inline VkImageSubresourceRange to_vk_image_subresource_range(const ImageSubresourceRange& range)
{
    return VkImageSubresourceRange{
        to_vk_image_aspect(range.aspect),
        range.base_mip,
        range.mip_count == kRemainingMipLevels ? VK_REMAINING_MIP_LEVELS : range.mip_count,
        range.base_layer,
        range.layer_count == kRemainingArrayLayers ? VK_REMAINING_ARRAY_LAYERS : range.layer_count,
    };
}
} // namespace rr::rhi