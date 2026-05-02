#pragma once

#include "rhi/Types.h"

#include <array>
#include <span>

namespace rr::rhi
{
class Image;

struct ClearColor
{
    std::array<float, 4> float32 = {0.0f, 0.0f, 0.0f, 0.0f};
};

struct ColorAttachment
{
    const Image* image   = nullptr;
    ImageLayout  layout  = ImageLayout::ColorAttachment;
    LoadOp       load_op = LoadOp::Load;
    StoreOp      store_op = StoreOp::Store;
    ClearColor   clear{};
};

struct DepthAttachment
{
    const Image* image         = nullptr;
    ImageLayout  layout        = ImageLayout::DepthStencilAttachment;
    LoadOp       load_op       = LoadOp::Load;
    StoreOp      store_op      = StoreOp::Store;
    float        clear_depth   = 1.0f;
    uint32_t     clear_stencil = 0;
};

struct RenderingInfo
{
    Extent2D                         area{};
    uint32_t                         layer_count = 1;
    std::span<const ColorAttachment> color_attachments{};
    const DepthAttachment*           depth_attachment = nullptr;
};
} // namespace rr::rhi