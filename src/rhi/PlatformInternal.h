#pragma once

#include "rhi/Handles.h"
#include "rhi/Platform.h"

#include <cstdint>
#include <vector>

namespace rr::rhi::platform
{
[[nodiscard]] std::vector<const char*> required_instance_extensions(PresentationSupport presentation);
[[nodiscard]] SurfaceHandle create_surface_handle(uint64_t instance_handle, NativeWindowHandle window);
void destroy_surface_handle(uint64_t instance_handle, SurfaceHandle surface_handle) noexcept;
}