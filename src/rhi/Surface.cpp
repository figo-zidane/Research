#include "rhi/Surface.h"

#include "rhi/Device.h"
#include "rhi/PlatformInternal.h"
#include "rhi/Types.h"

#include <stdexcept>

namespace rr::rhi
{
Surface::~Surface()
{
    shutdown();
}

void Surface::initialize(const Device& device, NativeWindowHandle window)
{
    if (surface_ != nullptr)
    {
        throw std::runtime_error("Surface is already initialized.");
    }
    if (device.instance() == VK_NULL_HANDLE)
    {
        throw std::runtime_error("Cannot create a surface before the Vulkan instance exists.");
    }

    instance_ = to_handle(device.instance());
    surface_ = platform::create_surface_handle(instance_, window);
}

void Surface::shutdown()
{
    platform::destroy_surface_handle(instance_, surface_);
    surface_ = nullptr;
    instance_ = 0;
}
}