#pragma once

#include "rhi/Platform.h"

#include <cstdint>
#include <vector>

namespace rr::rhi::platform
{
[[nodiscard]] std::vector<const char*> required_instance_extensions(PresentationSupport presentation);
[[nodiscard]] uint64_t create_surface_handle(uint64_t instance_handle, NativeWindowHandle window);
void destroy_surface_handle(uint64_t instance_handle, uint64_t surface_handle) noexcept;
}