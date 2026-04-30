#include "rhi/Image.h"

#include "core/Log.h"
#include "rhi/Device.h"

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
    , format_(other.format_)
    , extent_(other.extent_)
    , aspect_(other.aspect_)
    , mip_levels_(other.mip_levels_)
    , array_layers_(other.array_layers_)
{
    other.image_       = VK_NULL_HANDLE;
    other.view_        = VK_NULL_HANDLE;
    other.allocation_  = nullptr;
    other.format_      = VK_FORMAT_UNDEFINED;
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
        format_       = other.format_;
        extent_       = other.extent_;
        aspect_       = other.aspect_;
        mip_levels_   = other.mip_levels_;
        array_layers_ = other.array_layers_;
        other.image_       = VK_NULL_HANDLE;
        other.view_        = VK_NULL_HANDLE;
        other.allocation_  = nullptr;
        other.format_      = VK_FORMAT_UNDEFINED;
        other.extent_      = {};
        other.mip_levels_  = 1;
        other.array_layers_= 1;
    }
    return *this;
}

void Image::create(Device& device, const ImageDesc& desc)
{
    if (image_ != VK_NULL_HANDLE)
    {
        throw std::runtime_error("Image::create called on an already-created image.");
    }

    VkImageCreateInfo img_info{};
    img_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_info.imageType     = desc.type;
    img_info.format        = desc.format;
    img_info.extent        = desc.extent;
    img_info.mipLevels     = desc.mip_levels;
    img_info.arrayLayers   = desc.array_layers;
    img_info.samples       = desc.samples;
    img_info.tiling        = VK_IMAGE_TILING_OPTIMAL;
    img_info.usage         = desc.usage;
    img_info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc_info{};
    alloc_info.usage = static_cast<VmaMemoryUsage>(desc.memory_usage);
    alloc_info.flags = static_cast<VmaAllocationCreateFlags>(desc.alloc_flags);

    if (vmaCreateImage(device.allocator(), &img_info, &alloc_info,
                       &image_, &allocation_, nullptr) != VK_SUCCESS)
    {
        throw std::runtime_error("vmaCreateImage failed.");
    }

    format_       = desc.format;
    extent_       = desc.extent;
    aspect_       = desc.aspect;
    mip_levels_   = desc.mip_levels;
    array_layers_ = desc.array_layers;

    // Create the default image view covering all mips and layers.
    VkImageViewCreateInfo view_info{};
    view_info.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image    = image_;
    view_info.viewType = (desc.array_layers > 1)
                             ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                             : VK_IMAGE_VIEW_TYPE_2D;
    view_info.format   = desc.format;
    view_info.subresourceRange.aspectMask     = desc.aspect;
    view_info.subresourceRange.baseMipLevel   = 0;
    view_info.subresourceRange.levelCount     = desc.mip_levels;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount     = desc.array_layers;

    if (vkCreateImageView(device.device(), &view_info, nullptr, &view_) != VK_SUCCESS)
    {
        vmaDestroyImage(device.allocator(), image_, allocation_);
        image_      = VK_NULL_HANDLE;
        allocation_ = nullptr;
        throw std::runtime_error("vkCreateImageView failed.");
    }

    if (desc.debug_name)
    {
        auto set_name = [&](VkObjectType type, uint64_t handle, const char* name) {
            VkDebugUtilsObjectNameInfoEXT ni{};
            ni.sType        = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
            ni.objectType   = type;
            ni.objectHandle = handle;
            ni.pObjectName  = name;
            vkSetDebugUtilsObjectNameEXT(device.device(), &ni);
        };
        set_name(VK_OBJECT_TYPE_IMAGE,
                 reinterpret_cast<uint64_t>(image_), desc.debug_name);
        set_name(VK_OBJECT_TYPE_IMAGE_VIEW,
                 reinterpret_cast<uint64_t>(view_), desc.debug_name);
    }
}

void Image::destroy(Device& device)
{
    if (image_ == VK_NULL_HANDLE)
    {
        return;
    }
    if (view_ != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device.device(), view_, nullptr);
        view_ = VK_NULL_HANDLE;
    }
    vmaDestroyImage(device.allocator(), image_, allocation_);
    image_      = VK_NULL_HANDLE;
    allocation_ = nullptr;
}

void Image::upload_host(Device& device,
                        const void*   data,
                        VkDeviceSize  data_size,
                        VkImageLayout final_layout)
{
    // Use hostImageCopy (Vulkan 1.4 / VK_EXT_host_image_copy) to upload
    // pixel data directly without a staging buffer.

    // Transition UNDEFINED → TRANSFER_DST_OPTIMAL.
    {
        VkHostImageLayoutTransitionInfo transition{};
        transition.sType            = VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO;
        transition.image            = image_;
        transition.oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
        transition.newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        transition.subresourceRange = {aspect_, 0, mip_levels_, 0, array_layers_};

        if (vkTransitionImageLayout(device.device(), 1, &transition) != VK_SUCCESS)
        {
            throw std::runtime_error("vkTransitionImageLayout (UNDEFINED→TRANSFER_DST) failed.");
        }
    }

    // Copy pixel data.
    {
        VkMemoryToImageCopy copy{};
        copy.sType             = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY;
        copy.pHostPointer      = data;
        copy.memoryRowLength   = 0; // tightly packed
        copy.memoryImageHeight = 0;
        copy.imageSubresource  = {aspect_, 0, 0, array_layers_};
        copy.imageOffset       = {0, 0, 0};
        copy.imageExtent       = extent_;

        VkCopyMemoryToImageInfo copy_info{};
        copy_info.sType          = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO;
        copy_info.dstImage       = image_;
        copy_info.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        copy_info.regionCount    = 1;
        copy_info.pRegions       = &copy;

        if (vkCopyMemoryToImage(device.device(), &copy_info) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCopyMemoryToImage failed.");
        }
    }

    // Transition TRANSFER_DST_OPTIMAL → final_layout.
    {
        VkHostImageLayoutTransitionInfo transition{};
        transition.sType            = VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO;
        transition.image            = image_;
        transition.oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        transition.newLayout        = final_layout;
        transition.subresourceRange = {aspect_, 0, mip_levels_, 0, array_layers_};

        if (vkTransitionImageLayout(device.device(), 1, &transition) != VK_SUCCESS)
        {
            throw std::runtime_error("vkTransitionImageLayout (TRANSFER_DST→final) failed.");
        }
    }
}

} // namespace rr::rhi
