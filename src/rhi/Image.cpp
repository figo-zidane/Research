#include "rhi/Image.h"

#include "core/Log.h"
#include "rhi/Device.h"
#include "rhi/internal/VulkanAccess.h"
#include "rhi/internal/VulkanTypeCasts.h"

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <volk.h>
#include <vma/vk_mem_alloc.h>

#include <stdexcept>
#include <utility>

namespace rr::rhi
{

Image::Image(Image&& other) noexcept
    : image_(other.image_)
    , view_(other.view_)
    , allocation_(other.allocation_)
    , owns_image_(other.owns_image_)
    , owns_view_(other.owns_view_)
    , format_(other.format_)
    , extent_(other.extent_)
    , aspect_(other.aspect_)
    , mip_levels_(other.mip_levels_)
    , array_layers_(other.array_layers_)
{
    other.image_       = 0;
    other.view_        = 0;
    other.allocation_  = nullptr;
    other.owns_image_  = false;
    other.owns_view_   = false;
    other.format_      = Format::Undefined;
    other.extent_      = {};
    other.mip_levels_  = 1;
    other.array_layers_= 1;
}

Image& Image::operator=(Image&& other) noexcept
{
    if (this != &other)
    {
        image_        = other.image_;
        view_         = other.view_;
        allocation_   = other.allocation_;
        owns_image_   = other.owns_image_;
        owns_view_    = other.owns_view_;
        format_       = other.format_;
        extent_       = other.extent_;
        aspect_       = other.aspect_;
        mip_levels_   = other.mip_levels_;
        array_layers_ = other.array_layers_;
        other.image_       = 0;
        other.view_        = 0;
        other.allocation_  = nullptr;
        other.owns_image_  = false;
        other.owns_view_   = false;
        other.format_      = Format::Undefined;
        other.extent_      = {};
        other.mip_levels_  = 1;
        other.array_layers_= 1;
    }
    return *this;
}

void Image::create(Device& device, const ImageDesc& desc)
{
    if (image_ != 0)
    {
        throw std::runtime_error("Image::create called on an already-created image.");
    }

    VkImageCreateInfo img_info{};
    img_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_info.imageType     = to_vk_image_type(desc.type);
    img_info.format        = to_vk_format(desc.format);
    img_info.extent        = to_vk_extent3d(desc.extent);
    img_info.mipLevels     = desc.mip_levels;
    img_info.arrayLayers   = desc.array_layers;
    img_info.samples       = to_vk_sample_count(desc.samples);
    img_info.tiling        = VK_IMAGE_TILING_OPTIMAL;
    img_info.usage         = to_vk_image_usage(desc.usage);
    img_info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc_info{};
    alloc_info.usage = static_cast<VmaMemoryUsage>(desc.memory_usage);
    alloc_info.flags = static_cast<VmaAllocationCreateFlags>(desc.alloc_flags);

    VkImage raw_image = VK_NULL_HANDLE;
    VmaAllocation raw_allocation = nullptr;
    VkResult res = vmaCreateImage(vulkan::get_allocator(device), &img_info, &alloc_info,
                       &raw_image, &raw_allocation, nullptr);
    if (res != VK_SUCCESS)
    {
        throw std::runtime_error(std::string("vmaCreateImage failed: VkResult=") + std::to_string(static_cast<int>(res)));
    }

    format_       = desc.format;
    extent_       = desc.extent;
    aspect_       = desc.aspect;
    mip_levels_   = desc.mip_levels;
    array_layers_ = desc.array_layers;
    owns_image_   = true;
    owns_view_    = true;

    // Create the default image view covering all mips and layers.
    VkImageViewCreateInfo view_info{};
    view_info.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image    = raw_image;
    view_info.viewType = (desc.type == ImageType::Image3D)
                             ? VK_IMAGE_VIEW_TYPE_3D
                             : (desc.array_layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D);
    view_info.format   = to_vk_format(desc.format);
    view_info.subresourceRange.aspectMask     = to_vk_image_aspect(desc.aspect);
    view_info.subresourceRange.baseMipLevel   = 0;
    view_info.subresourceRange.levelCount     = desc.mip_levels;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount     = desc.array_layers;

    VkImageView raw_view = VK_NULL_HANDLE;
    if (vkCreateImageView(vulkan::get_device(device), &view_info, nullptr, &raw_view) != VK_SUCCESS)
    {
        vmaDestroyImage(vulkan::get_allocator(device), raw_image, raw_allocation);
        throw std::runtime_error("vkCreateImageView failed.");
    }

    image_      = to_handle(raw_image);
    view_       = to_handle(raw_view);
    allocation_ = raw_allocation;

    if (desc.debug_name)
    {
        auto set_name = [&](VkObjectType type, uint64_t handle, const char* name) {
            VkDebugUtilsObjectNameInfoEXT ni{};
            ni.sType        = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
            ni.objectType   = type;
            ni.objectHandle = handle;
            ni.pObjectName  = name;
            vkSetDebugUtilsObjectNameEXT(vulkan::get_device(device), &ni);
        };
        set_name(VK_OBJECT_TYPE_IMAGE,
                 image_, desc.debug_name);
        set_name(VK_OBJECT_TYPE_IMAGE_VIEW,
                 view_, desc.debug_name);
    }
}

void Image::attach_external(ImageHandle image,
                            ImageViewHandle view,
                            Format format,
                            Extent2D extent,
                            ImageAspect aspect,
                            uint32_t mip_levels,
                            uint32_t array_layers) noexcept
{
    image_        = image;
    view_         = view;
    allocation_   = nullptr;
    owns_image_   = false;
    owns_view_    = true;
    format_       = format;
    extent_       = {extent.width, extent.height, 1};
    aspect_       = aspect;
    mip_levels_   = mip_levels;
    array_layers_ = array_layers;
}

void Image::destroy(Device& device)
{
    if (image_ == 0)
    {
        return;
    }
    if (view_ != 0 && owns_view_)
    {
        vkDestroyImageView(vulkan::get_device(device), from_handle<VkImageView>(view_), nullptr);
    }
    if (image_ != 0 && owns_image_)
    {
        vmaDestroyImage(vulkan::get_allocator(device), from_handle<VkImage>(image_), allocation_);
    }
    view_       = 0;
    image_      = 0;
    allocation_ = nullptr;
    owns_image_ = false;
    owns_view_  = false;
}

void Image::upload_host(Device& device,
                        const void*   data,
                        uint64_t      data_size,
                        ImageLayout final_layout)
{
    // Use hostImageCopy (Vulkan 1.4 / VK_EXT_host_image_copy) to upload
    // pixel data directly without a staging buffer.
    //
    // Images created with VK_IMAGE_USAGE_HOST_TRANSFER_BIT may NOT have
    // VK_IMAGE_USAGE_TRANSFER_DST_BIT, so VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    // is illegal. VK_IMAGE_LAYOUT_GENERAL is always a valid host-copy destination.

    const VkImage raw_image = from_handle<VkImage>(image_);

    // Transition UNDEFINED → GENERAL.
    {
        VkHostImageLayoutTransitionInfo transition{};
        transition.sType            = VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO;
        transition.image            = raw_image;
        transition.oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
        transition.newLayout        = VK_IMAGE_LAYOUT_GENERAL;
        transition.subresourceRange = {to_vk_image_aspect(aspect_), 0, mip_levels_, 0, array_layers_};

        if (vkTransitionImageLayout(vulkan::get_device(device), 1, &transition) != VK_SUCCESS)
        {
            throw std::runtime_error("vkTransitionImageLayout (UNDEFINED→GENERAL) failed.");
        }
    }

    // Copy pixel data.
    {
        VkMemoryToImageCopy copy{};
        copy.sType             = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY;
        copy.pHostPointer      = data;
        copy.memoryRowLength   = 0; // tightly packed
        copy.memoryImageHeight = 0;
        copy.imageSubresource.mipLevel = 0;
        copy.imageSubresource.baseArrayLayer = 0;
        copy.imageSubresource.layerCount = array_layers_;
        copy.imageOffset       = {0, 0, 0};
        copy.imageSubresource.aspectMask = to_vk_image_aspect(aspect_);
        copy.imageExtent       = to_vk_extent3d(extent_);

        VkCopyMemoryToImageInfo copy_info{};
        copy_info.sType          = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO;
        copy_info.dstImage       = raw_image;
        copy_info.dstImageLayout = VK_IMAGE_LAYOUT_GENERAL;
        copy_info.regionCount    = 1;
        copy_info.pRegions       = &copy;

        if (vkCopyMemoryToImage(vulkan::get_device(device), &copy_info) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCopyMemoryToImage failed.");
        }
    }

    // Transition GENERAL → final_layout.
    {
        VkHostImageLayoutTransitionInfo transition{};
        transition.sType            = VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO;
        transition.image            = raw_image;
        transition.oldLayout        = VK_IMAGE_LAYOUT_GENERAL;
        transition.newLayout        = to_vk_image_layout(final_layout);
        transition.subresourceRange = {to_vk_image_aspect(aspect_), 0, mip_levels_, 0, array_layers_};

        if (vkTransitionImageLayout(vulkan::get_device(device), 1, &transition) != VK_SUCCESS)
        {
            throw std::runtime_error("vkTransitionImageLayout (GENERAL→final) failed.");
        }
    }
}

} // namespace rr::rhi
