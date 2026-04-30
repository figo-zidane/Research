#pragma once

#include <array>
#include <cstdint>

#include <volk.h>

namespace rr::rhi
{
class Device;

// 2 frames in flight is enough for desktop NVIDIA at vsync; we are also
// targeting low latency rather than maximum throughput.
inline constexpr uint32_t kFramesInFlight = 2;

class Frame
{
public:
    struct Sync
    {
        // Signaled by vkAcquireNextImageKHR; one per frame-in-flight slot. The
        // matching render_finished semaphore is owned by the swapchain (one
        // per image) because image indices and frame indices don't align.
        VkSemaphore image_available = VK_NULL_HANDLE;
        VkFence     in_flight       = VK_NULL_HANDLE;
    };

    Frame() = default;
    Frame(const Frame&) = delete;
    Frame& operator=(const Frame&) = delete;
    Frame(Frame&&) = delete;
    Frame& operator=(Frame&&) = delete;
    ~Frame();

    void initialize(Device& device);
    void shutdown();

    [[nodiscard]] const Sync& sync(uint32_t frame_index) const { return syncs_[frame_index]; }
    [[nodiscard]] uint32_t advance() noexcept;
    [[nodiscard]] uint32_t current() const noexcept { return current_frame_; }

private:
    Device* device_ = nullptr;
    std::array<Sync, kFramesInFlight> syncs_{};
    uint32_t current_frame_ = 0;
};
}
