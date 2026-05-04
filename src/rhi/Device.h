#pragma once

#include "rhi/Platform.h"

#include <cstdint>
#include <string>
#include <vector>

#include <volk.h>

// Forward-declare VmaAllocator so users of Device.h don't pull in all of VMA.
VK_DEFINE_HANDLE(VmaAllocator)

namespace rr::rhi
{
class Surface;

class Device
{
public:
    struct CreateInfo
    {
        std::string application_name;
        PresentationSupport presentation = PresentationSupport::None;
        bool enable_validation = true;
    };

    struct QueueFamilies
    {
        uint32_t graphics_compute = UINT32_MAX;
    };

    Device() = default;
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) = delete;
    Device& operator=(Device&&) = delete;
    ~Device();

    // Two-phase init: the instance is needed to create a window surface, and
    // the surface is needed to pick a present-capable queue family before the
    // logical device is created.
    void create_instance(const CreateInfo& create_info);
    void create_device_with_surface(const Surface& surface);
    void create_device_with_surface(VkSurfaceKHR surface);
    void shutdown();

    [[nodiscard]] VkInstance instance() const noexcept { return instance_; }
    [[nodiscard]] VkPhysicalDevice physical_device() const noexcept { return physical_device_; }
    [[nodiscard]] VkDevice device() const noexcept { return device_; }
    [[nodiscard]] VkQueue graphics_queue() const noexcept { return graphics_queue_; }
    [[nodiscard]] uint32_t graphics_queue_family() const noexcept { return queue_families_.graphics_compute; }
    [[nodiscard]] VmaAllocator allocator() const noexcept { return allocator_; }

private:
    [[nodiscard]] bool validation_layers_available() const;
    void create_debug_messenger();
    void pick_physical_device(VkSurfaceKHR surface);
    void create_logical_device();
    void log_enabled_features() const;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphics_queue_ = VK_NULL_HANDLE;
    QueueFamilies queue_families_{};
    VkPhysicalDeviceProperties physical_device_properties_{};
    VmaAllocator allocator_ = nullptr;

    bool validation_enabled_ = false;
    std::vector<const char*> enabled_instance_extensions_;
    std::vector<const char*> enabled_device_extensions_;
};
}
