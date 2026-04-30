#pragma once

#include <cstdint>
#include <volk.h>

VK_DEFINE_HANDLE(VmaAllocation)

namespace rr::rhi
{
class Device;

struct ImageDesc
{
    VkFormat               format       = VK_FORMAT_UNDEFINED;
    VkExtent3D             extent       = {1, 1, 1};
    VkImageUsageFlags      usage        = 0;
    VkImageType            type         = VK_IMAGE_TYPE_2D;
    VkImageAspectFlags     aspect       = VK_IMAGE_ASPECT_COLOR_BIT;
    uint32_t               mip_levels   = 1;
    uint32_t               array_layers = 1;
    VkSampleCountFlagBits  samples      = VK_SAMPLE_COUNT_1_BIT;
    int                    memory_usage = 4; // VMA_MEMORY_USAGE_AUTO
    int                    alloc_flags  = 0; // VmaAllocationCreateFlags
    const char*            debug_name   = nullptr;
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
                     VkImageLayout final_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    [[nodiscard]] VkImage            handle()   const noexcept { return image_; }
    [[nodiscard]] VkImageView        view()     const noexcept { return view_; }
    [[nodiscard]] VkFormat           format()   const noexcept { return format_; }
    [[nodiscard]] VkExtent3D         extent3d() const noexcept { return extent_; }
    [[nodiscard]] VkExtent2D         extent2d() const noexcept { return {extent_.width, extent_.height}; }
    [[nodiscard]] VkImageAspectFlags aspect()   const noexcept { return aspect_; }
    [[nodiscard]] uint32_t           mip_levels()   const noexcept { return mip_levels_; }
    [[nodiscard]] uint32_t           array_layers() const noexcept { return array_layers_; }
    [[nodiscard]] bool               is_valid() const noexcept { return image_ != VK_NULL_HANDLE; }

private:
    VkImage            image_        = VK_NULL_HANDLE;
    VkImageView        view_         = VK_NULL_HANDLE;
    VmaAllocation      allocation_   = nullptr;
    VkFormat           format_       = VK_FORMAT_UNDEFINED;
    VkExtent3D         extent_       = {};
    VkImageAspectFlags aspect_       = VK_IMAGE_ASPECT_COLOR_BIT;
    uint32_t           mip_levels_   = 1;
    uint32_t           array_layers_ = 1;
};

} // namespace rr::rhi
