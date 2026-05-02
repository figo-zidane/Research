#pragma once

#include "rhi/Types.h"

namespace rr::rhi
{
enum class SamplerFilter : uint8_t
{
    Nearest,
    Linear,
};

enum class SamplerMipmapMode : uint8_t
{
    Nearest,
    Linear,
};

enum class SamplerAddressMode : uint8_t
{
    Repeat,
    ClampToEdge,
};

enum class SamplerBorderColor : uint8_t
{
    FloatTransparentBlack,
};

struct SamplerDesc
{
    SamplerFilter      mag_filter               = SamplerFilter::Linear;
    SamplerFilter      min_filter               = SamplerFilter::Linear;
    SamplerMipmapMode  mipmap_mode              = SamplerMipmapMode::Linear;
    SamplerAddressMode address_mode_u           = SamplerAddressMode::Repeat;
    SamplerAddressMode address_mode_v           = SamplerAddressMode::Repeat;
    SamplerAddressMode address_mode_w           = SamplerAddressMode::Repeat;
    float              mip_lod_bias             = 0.0f;
    bool               anisotropy_enable        = false;
    float              max_anisotropy           = 1.0f;
    bool               compare_enable           = false;
    CompareOp          compare_op               = CompareOp::Always;
    float              min_lod                  = 0.0f;
    float              max_lod                  = 1000.0f;
    SamplerBorderColor border_color             = SamplerBorderColor::FloatTransparentBlack;
    bool               unnormalized_coordinates = false;
};
} // namespace rr::rhi