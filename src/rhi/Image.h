#pragma once

#include "rhi/Types.h"

#include <cstdint>
#include <volk.h>

VK_DEFINE_HANDLE(VmaAllocation)

namespace rr::rhi
{
class Device;

struct ImageDesc
{
    Format      format       = Format::Undefined;
    Extent3D    extent       = {1, 1, 1};
    ImageUsage  usage        = ImageUsage::None;
    ImageType   type         = ImageType::Image2D;
    ImageAspect aspect       = ImageAspect::Color;
    uint32_t    mip_levels   = 1;
    uint32_t    array_layers = 1;
    SampleCount samples      = SampleCount::Count1;
    int         memory_usage = 7; // VMA_MEMORY_USAGE_AUTO
    int         alloc_flags  = 0; // VmaAllocationCreateFlags
    const char* debug_name   = nullptr;
};

// Thin RAII wrapper around a VkImage + VkImageView + VmaAllocation.
// The image view covers the full mip/layer range with the configured aspect.
class Image
{
public:
    Image() = default;
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;
    ~Image() = default; // call destroy() explicitly

    void create(Device& device, const ImageDesc& desc);
    void destroy(Device& device);

    // Upload pixel data to the image using VK_KHR_copy_commands2 / hostImageCopy
    // (Vulkan 1.4 core). The image must have been created with
    // VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT in its usage flags.
    // data_size must match the complete mip 0 layer 0 size.
    void upload_host(Device& device,
                     const void* data,
                     VkDeviceSize data_size,
                     ImageLayout final_layout = ImageLayout::ShaderReadOnly);

    [[nodiscard]] VkImage            handle()   const noexcept { return image_; }
    [[nodiscard]] VkImageView        view()     const noexcept { return view_; }
    [[nodiscard]] Format             format()   const noexcept { return format_; }
    [[nodiscard]] Extent3D           extent3d() const noexcept { return extent_; }
    [[nodiscard]] rr::rhi::Extent2D  extent2d() const noexcept { return {extent_.width, extent_.height}; }
    [[nodiscard]] ImageAspect        aspect()   const noexcept { return aspect_; }
    [[nodiscard]] uint32_t           mip_levels()   const noexcept { return mip_levels_; }
    [[nodiscard]] uint32_t           array_layers() const noexcept { return array_layers_; }
    [[nodiscard]] bool               is_valid() const noexcept { return image_ != VK_NULL_HANDLE; }

private:
    VkImage            image_        = VK_NULL_HANDLE;
    VkImageView        view_         = VK_NULL_HANDLE;
    VmaAllocation      allocation_   = nullptr;
    Format             format_       = Format::Undefined;
    Extent3D           extent_       = {};
    ImageAspect        aspect_       = ImageAspect::Color;
    uint32_t           mip_levels_   = 1;
    uint32_t           array_layers_ = 1;
};

} // namespace rr::rhi
