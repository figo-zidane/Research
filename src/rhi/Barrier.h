#pragma once

#include "rhi/Types.h"

#include <cstdint>

namespace rr::rhi
{
class Image;

constexpr uint32_t kRemainingMipLevels   = 0xffffffffu;
constexpr uint32_t kRemainingArrayLayers = 0xffffffffu;

struct ImageSubresourceRange
{
    ImageAspect aspect      = ImageAspect::Color;
    uint32_t    base_mip    = 0;
    uint32_t    mip_count   = kRemainingMipLevels;
    uint32_t    base_layer  = 0;
    uint32_t    layer_count = kRemainingArrayLayers;
};

struct ImageBarrier
{
    const Image*   image      = nullptr;
    PipelineStage  src_stage  = PipelineStage::None;
    AccessFlags    src_access = AccessFlags::None;
    PipelineStage  dst_stage  = PipelineStage::None;
    AccessFlags    dst_access = AccessFlags::None;
    ImageLayout    old_layout = ImageLayout::Undefined;
    ImageLayout    new_layout = ImageLayout::Undefined;
    ImageSubresourceRange subresource{};
};
} // namespace rr::rhi