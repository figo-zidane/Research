#pragma once

#include "rhi/Types.h"

namespace rr::rhi
{
class Buffer;
class Device;
class Image;

struct ImageRegion
{
    Offset3D    offset{};
    Extent3D    extent{};
    ImageAspect aspect = ImageAspect::Color;
    uint32_t    mip = 0;
    uint32_t    layer = 0;
};

void readback_image(Device& device,
                    const Image& src,
                    ImageLayout src_current_layout,
                    Buffer& dst_staging,
                    const ImageRegion& region);
} // namespace rr::rhi