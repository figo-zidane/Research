#pragma once

#include <cstdint>
#include <type_traits>

namespace rr::rhi
{
struct Extent2D
{
    uint32_t width  = 0;
    uint32_t height = 0;
};

struct Extent3D
{
    uint32_t width  = 0;
    uint32_t height = 0;
    uint32_t depth  = 1;
};

enum class Format : int
{
    Undefined           = 0,
    R8_Unorm            = 9,
    R8G8B8A8_Unorm      = 37,
    R8G8B8A8_Srgb       = 43,
    B8G8R8A8_Unorm      = 44,
    B8G8R8A8_Srgb       = 50,
    R16G16B16A16_Sfloat = 97,
    R32_Uint            = 98,
    R32G32_Sfloat       = 103,
    R32G32B32_Sfloat    = 106,
    R32G32B32A32_Sfloat = 109,
    D32_Sfloat          = 126,
};

enum class ImageType : uint8_t
{
    Image1D,
    Image2D,
    Image3D,
};

enum class ImageViewType : uint8_t
{
    View1D,
    View2D,
    View3D,
    ViewCube,
    View1DArray,
    View2DArray,
    ViewCubeArray,
};

enum class ImageLayout : uint8_t
{
    Undefined,
    General,
    ColorAttachment,
    DepthStencilAttachment,
    ShaderReadOnly,
    TransferSrc,
    TransferDst,
    Present,
};

enum class ImageAspect : uint32_t
{
    None    = 0,
    Color   = 1u << 0,
    Depth   = 1u << 1,
    Stencil = 1u << 2,
};

enum class ImageUsage : uint32_t
{
    None                   = 0,
    TransferSrc            = 1u << 0,
    TransferDst            = 1u << 1,
    Sampled                = 1u << 2,
    Storage                = 1u << 3,
    ColorAttachment        = 1u << 4,
    DepthStencilAttachment = 1u << 5,
    HostTransfer           = 1u << 6,
};

enum class SampleCount : uint8_t
{
    Count1  = 1,
    Count2  = 2,
    Count4  = 4,
    Count8  = 8,
    Count16 = 16,
    Count32 = 32,
    Count64 = 64,
};

enum class PipelineStage : uint64_t
{
    None                  = 0,
    TopOfPipe             = 1ull << 0,
    AllCommands           = 1ull << 1,
    ComputeShader         = 1ull << 2,
    FragmentShader        = 1ull << 3,
    ColorAttachmentOutput = 1ull << 4,
    EarlyFragmentTests    = 1ull << 5,
    LateFragmentTests     = 1ull << 6,
    RayTracingShader      = 1ull << 7,
    Transfer              = 1ull << 8,
    Host                  = 1ull << 9,
};

enum class AccessFlags : uint64_t
{
    None                        = 0,
    ShaderRead                  = 1ull << 0,
    ShaderWrite                 = 1ull << 1,
    ColorAttachmentWrite        = 1ull << 2,
    DepthStencilAttachmentWrite = 1ull << 3,
    TransferRead                = 1ull << 4,
    TransferWrite               = 1ull << 5,
    HostWrite                   = 1ull << 6,
    ResourceHeapRead            = 1ull << 7,
    SamplerHeapRead             = 1ull << 8,
};

enum class LoadOp : uint8_t
{
    Load,
    Clear,
    DontCare,
};

enum class StoreOp : uint8_t
{
    Store,
    DontCare,
};

enum class CullMode : uint8_t
{
    None,
    Front,
    Back,
    FrontAndBack,
};

enum class FrontFace : uint8_t
{
    CounterClockwise,
    Clockwise,
};

enum class PrimitiveTopology : uint8_t
{
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
};

enum class CompareOp : uint8_t
{
    Never,
    Less,
    Equal,
    LessOrEqual,
    Greater,
    NotEqual,
    GreaterOrEqual,
    Always,
};

using ImageHandle     = uint64_t;
using ImageViewHandle = uint64_t;

template<typename Enum>
struct EnableBitMaskOperators : std::false_type {};

template<>
struct EnableBitMaskOperators<ImageAspect> : std::true_type {};

template<>
struct EnableBitMaskOperators<ImageUsage> : std::true_type {};

template<>
struct EnableBitMaskOperators<PipelineStage> : std::true_type {};

template<>
struct EnableBitMaskOperators<AccessFlags> : std::true_type {};

template<typename Enum>
constexpr bool kEnableBitMaskOperators = EnableBitMaskOperators<Enum>::value;

template<typename Enum>
[[nodiscard]] constexpr std::enable_if_t<kEnableBitMaskOperators<Enum>, Enum>
operator|(Enum lhs, Enum rhs) noexcept
{
    using Underlying = std::underlying_type_t<Enum>;
    return static_cast<Enum>(static_cast<Underlying>(lhs) | static_cast<Underlying>(rhs));
}

template<typename Enum>
constexpr std::enable_if_t<kEnableBitMaskOperators<Enum>, Enum&>
operator|=(Enum& lhs, Enum rhs) noexcept
{
    lhs = lhs | rhs;
    return lhs;
}

template<typename Enum>
[[nodiscard]] constexpr std::enable_if_t<kEnableBitMaskOperators<Enum>, Enum>
operator&(Enum lhs, Enum rhs) noexcept
{
    using Underlying = std::underlying_type_t<Enum>;
    return static_cast<Enum>(static_cast<Underlying>(lhs) & static_cast<Underlying>(rhs));
}

template<typename Enum>
[[nodiscard]] constexpr std::enable_if_t<kEnableBitMaskOperators<Enum>, bool>
has_any_flag(Enum value, Enum bits) noexcept
{
    using Underlying = std::underlying_type_t<Enum>;
    return (static_cast<Underlying>(value) & static_cast<Underlying>(bits)) != 0;
}

template<typename Handle>
[[nodiscard]] inline uint64_t to_handle(Handle handle) noexcept
{
    if constexpr (std::is_pointer_v<Handle>)
        return reinterpret_cast<uint64_t>(handle);
    else
        return static_cast<uint64_t>(handle);
}

template<typename Handle>
[[nodiscard]] inline Handle from_handle(uint64_t handle) noexcept
{
    if constexpr (std::is_pointer_v<Handle>)
        return reinterpret_cast<Handle>(handle);
    else
        return static_cast<Handle>(handle);
}
} // namespace rr::rhi