#pragma once

#include "rhi/Handles.h"
#include "rhi/Platform.h"

#include <cstdint>

namespace rr::rhi
{
class Device;

class Surface
{
public:
    Surface() = default;
    Surface(const Surface&) = delete;
    Surface& operator=(const Surface&) = delete;
    Surface(Surface&&) = delete;
    Surface& operator=(Surface&&) = delete;
    ~Surface();

    void initialize(const Device& device, NativeWindowHandle window);
    void shutdown();

    [[nodiscard]] bool is_valid() const noexcept { return surface_ != nullptr; }

private:
    friend class Device;
    friend class Swapchain;

    uint64_t      instance_ = 0;
    SurfaceHandle surface_ = nullptr;
};
}