#pragma once

#include "rhi/Handles.h"
#include "rhi/Types.h"

#include <cstdint>
#include <vector>

namespace rr::rhi
{
class Device;
class Image;
class Surface;

class Swapchain
{
public:
    Swapchain() = default;
    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;
    Swapchain(Swapchain&&) = delete;
    Swapchain& operator=(Swapchain&&) = delete;
    ~Swapchain();

    void initialize(Device& device, const Surface& surface, uint32_t width, uint32_t height);
    void shutdown();
    // Recreate the swapchain in place (e.g. on window resize). Caller must
    // ensure the device is idle before calling.
    void recreate(uint32_t width, uint32_t height);

    [[nodiscard]] SwapchainHandle handle() const noexcept { return swapchain_; }
    [[nodiscard]] Format image_format() const noexcept { return image_format_; }
    [[nodiscard]] Extent2D extent() const noexcept { return extent_; }
    [[nodiscard]] uint32_t image_count() const noexcept;
    [[nodiscard]] const Image& image(uint32_t index) const;
    // Per-image present-completion semaphore. Binary semaphores cannot be
    // reused before their wait completes, and presentation has no fence to
    // synchronise with, so we tie one semaphore per swapchain image.
    [[nodiscard]] SemaphoreHandle render_finished(uint32_t image_index) const;

private:
    void create_swapchain(uint32_t width, uint32_t height);
    void destroy_swapchain();

    Device*         device_ = nullptr;
    SurfaceHandle   surface_ = nullptr;
    SwapchainHandle swapchain_ = nullptr;
    Format          image_format_ = Format::Undefined;
    uint32_t        color_space_ = 0;
    uint32_t        present_mode_ = 0;
    Extent2D        extent_{};
    std::vector<Image> images_;
    std::vector<SemaphoreHandle> render_finished_;
};
}
