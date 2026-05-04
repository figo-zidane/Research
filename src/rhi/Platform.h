#pragma once

#include <cstdint>

namespace rr::rhi
{
struct NativeWindowHandle
{
    void* window = nullptr;
};

enum class PresentationSupport : uint8_t
{
    None,
    Window,
};
}