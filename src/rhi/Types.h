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

enum class Format : int
{
    Undefined           = 0,
    R8G8B8A8_Unorm      = 37,
    R8G8B8A8_Srgb       = 43,
    B8G8R8A8_Unorm      = 44,
    B8G8R8A8_Srgb       = 50,
    R16G16B16A16_Sfloat = 97,
    R32_Uint            = 98,
    R32G32B32A32_Sfloat = 109,
    D32_Sfloat          = 126,
};

using ImageHandle     = uint64_t;
using ImageViewHandle = uint64_t;

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