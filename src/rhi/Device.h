#pragma once

#include "rhi/Handles.h"
#include "rhi/Platform.h"

#include <functional>
#include <cstdint>
#include <string>
#include <vector>

namespace rr::rhi
{
class CommandRecorder;
class Frame;
class Surface;
class Swapchain;

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
    void one_time_submit(const std::function<void(CommandRecorder)>& record_commands) const;
    void submit_frame(CommandRecorder recorder, const Frame& frame, const Swapchain& swapchain, uint32_t image_index) const;
    void shutdown();
    void wait_idle() const;

    [[nodiscard]] InstanceHandle instance() const noexcept { return instance_; }
    [[nodiscard]] PhysicalDeviceHandle physical_device() const noexcept { return physical_device_; }
    [[nodiscard]] LogicalDeviceHandle device() const noexcept { return device_; }
    [[nodiscard]] QueueHandle graphics_queue() const noexcept { return graphics_queue_; }
    [[nodiscard]] uint32_t graphics_queue_family() const noexcept { return queue_families_.graphics_compute; }
    [[nodiscard]] AllocatorHandle allocator() const noexcept { return allocator_; }

private:
    [[nodiscard]] bool validation_layers_available() const;
    void create_device_with_surface_handle(SurfaceHandle surface);
    void create_debug_messenger();
    void pick_physical_device(SurfaceHandle surface);
    void create_logical_device();
    void log_enabled_features() const;

    InstanceHandle instance_ = nullptr;
    DebugMessengerHandle debug_messenger_ = nullptr;
    PhysicalDeviceHandle physical_device_ = nullptr;
    LogicalDeviceHandle device_ = nullptr;
    QueueHandle graphics_queue_ = nullptr;
    CommandPoolHandle one_time_pool_ = nullptr;
    QueueFamilies queue_families_{};
    std::string physical_device_name_;
    uint32_t physical_device_api_version_ = 0;
    AllocatorHandle allocator_ = nullptr;

    bool validation_enabled_ = false;
    std::vector<const char*> enabled_instance_extensions_;
    std::vector<const char*> enabled_device_extensions_;
};
}
