#pragma once

#include <cstdint>
#include <vector>

#include <volk.h>

namespace rr::rhi
{
class Device;
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
    void initialize(Device& device, VkSurfaceKHR surface, uint32_t width, uint32_t height);
    void shutdown();
    // Recreate the swapchain in place (e.g. on window resize). Caller must
    // ensure the device is idle before calling.
    void recreate(uint32_t width, uint32_t height);

    [[nodiscard]] VkSwapchainKHR handle() const noexcept { return swapchain_; }
    [[nodiscard]] VkFormat image_format() const noexcept { return surface_format_.format; }
    [[nodiscard]] VkExtent2D extent() const noexcept { return extent_; }
    [[nodiscard]] uint32_t image_count() const noexcept { return static_cast<uint32_t>(images_.size()); }
    [[nodiscard]] VkImage image(uint32_t index) const { return images_[index]; }
    [[nodiscard]] VkImageView image_view(uint32_t index) const { return image_views_[index]; }
    // Per-image present-completion semaphore. Binary semaphores cannot be
    // reused before their wait completes, and presentation has no fence to
    // synchronise with, so we tie one semaphore per swapchain image.
    [[nodiscard]] VkSemaphore render_finished(uint32_t image_index) const { return render_finished_[image_index]; }

private:
    void create_swapchain(uint32_t width, uint32_t height);
    void destroy_swapchain();

    Device* device_ = nullptr;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkSurfaceFormatKHR surface_format_{};
    VkPresentModeKHR present_mode_ = VK_PRESENT_MODE_FIFO_KHR;
    VkExtent2D extent_{};
    std::vector<VkImage> images_;
    std::vector<VkImageView> image_views_;
    std::vector<VkSemaphore> render_finished_;
};
}
