#pragma once

#include "rhi/Types.h"

#include <cstdint>

struct VmaAllocation_T;

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

// Thin RAII wrapper around an image resource plus its default view.
// The view covers the full mip/layer range with the configured aspect.
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
    void attach_external(ImageHandle image,
                         ImageViewHandle view,
                         Format format,
                         Extent2D extent,
                         ImageAspect aspect = ImageAspect::Color,
                         uint32_t mip_levels = 1,
                         uint32_t array_layers = 1) noexcept;

    // Upload pixel data to the image using the backend's host image copy path.
    // The image must have been created with host-transfer usage enabled.
    // data_size must match the complete mip 0 layer 0 size.
    void upload_host(Device& device,
                     const void* data,
                     uint64_t data_size,
                     ImageLayout final_layout = ImageLayout::ShaderReadOnly);

    [[nodiscard]] ImageHandle        handle()   const noexcept { return image_; }
    [[nodiscard]] ImageViewHandle    view()     const noexcept { return view_; }
    [[nodiscard]] Format             format()   const noexcept { return format_; }
    [[nodiscard]] Extent3D           extent3d() const noexcept { return extent_; }
    [[nodiscard]] rr::rhi::Extent2D  extent2d() const noexcept { return {extent_.width, extent_.height}; }
    [[nodiscard]] ImageAspect        aspect()   const noexcept { return aspect_; }
    [[nodiscard]] uint32_t           mip_levels()   const noexcept { return mip_levels_; }
    [[nodiscard]] uint32_t           array_layers() const noexcept { return array_layers_; }
    [[nodiscard]] bool               is_valid() const noexcept { return image_ != 0; }

private:
    ImageHandle        image_        = 0;
    ImageViewHandle    view_         = 0;
    VmaAllocation_T*   allocation_   = nullptr;
    bool               owns_image_   = false;
    bool               owns_view_    = false;
    Format             format_       = Format::Undefined;
    Extent3D           extent_       = {};
    ImageAspect        aspect_       = ImageAspect::Color;
    uint32_t           mip_levels_   = 1;
    uint32_t           array_layers_ = 1;
};

} // namespace rr::rhi
