#pragma once

#include <cstdint>
#include <type_traits>

namespace rr::rhi
{
using CommandBufferHandle = struct CommandBufferImpl*;
using CommandPoolHandle   = struct CommandPoolImpl*;
using SemaphoreHandle     = struct SemaphoreImpl*;
using FenceHandle         = struct FenceImpl*;
using SurfaceHandle       = struct SurfaceImpl*;
using SwapchainHandle     = struct SwapchainImpl*;

static_assert(sizeof(void*) == 8, "rhi handles assume 64-bit non-dispatchable Vulkan handles");

template<typename Handle, typename RawHandle>
[[nodiscard]] inline Handle to_opaque_handle(RawHandle handle) noexcept
{
    static_assert(std::is_pointer_v<Handle>, "opaque handles must be pointer-like");
    static_assert(std::is_pointer_v<RawHandle> || std::is_integral_v<RawHandle>,
                  "raw handles must be pointer-like or integer-like");
    return reinterpret_cast<Handle>(handle);
}

template<typename RawHandle, typename Handle>
[[nodiscard]] inline RawHandle from_opaque_handle(Handle handle) noexcept
{
    static_assert(std::is_pointer_v<Handle>, "opaque handles must be pointer-like");
    static_assert(std::is_pointer_v<RawHandle> || std::is_integral_v<RawHandle>,
                  "raw handles must be pointer-like or integer-like");
    return reinterpret_cast<RawHandle>(handle);
}
}