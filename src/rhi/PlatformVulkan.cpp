#include "rhi/PlatformInternal.h"

#include "rhi/Types.h"

#include <volk.h>

#include <GLFW/glfw3.h>

#include <stdexcept>
#include <vector>

namespace rr::rhi::platform
{
std::vector<const char*> required_instance_extensions(PresentationSupport presentation)
{
    switch (presentation)
    {
    case PresentationSupport::None:
        return {};

    case PresentationSupport::Window:
    {
        uint32_t extension_count = 0;
        const char** extension_names = glfwGetRequiredInstanceExtensions(&extension_count);
        if (extension_names == nullptr || extension_count == 0)
        {
            throw std::runtime_error("GLFW did not provide Vulkan instance extensions.");
        }

        return {extension_names, extension_names + extension_count};
    }
    }

    throw std::runtime_error("Unsupported presentation mode.");
}

SurfaceHandle create_surface_handle(uint64_t instance_handle, NativeWindowHandle window)
{
    if (instance_handle == 0)
    {
        throw std::runtime_error("Cannot create a surface before the Vulkan instance exists.");
    }
    if (window.window == nullptr)
    {
        throw std::runtime_error("Cannot create a surface from a null window handle.");
    }

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    GLFWwindow* glfw_window = static_cast<GLFWwindow*>(window.window);
    if (glfwCreateWindowSurface(from_handle<VkInstance>(instance_handle), glfw_window, nullptr, &surface) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vulkan window surface.");
    }

    return to_opaque_handle<SurfaceHandle>(surface);
}

void destroy_surface_handle(uint64_t instance_handle, SurfaceHandle surface_handle) noexcept
{
    if (instance_handle == 0 || surface_handle == nullptr)
    {
        return;
    }

    vkDestroySurfaceKHR(
        from_handle<VkInstance>(instance_handle),
        from_opaque_handle<VkSurfaceKHR>(surface_handle),
        nullptr);
}
}