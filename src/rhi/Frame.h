#pragma once

#include "rhi/Handles.h"

#include <array>
#include <cstdint>

namespace rr::rhi
{
class Device;

// 2 frames in flight is enough for desktop NVIDIA at vsync; we are also
// targeting low latency rather than maximum throughput.
inline constexpr uint32_t kFramesInFlight = 2;

class Frame
{
public:
    Frame() = default;
    Frame(const Frame&) = delete;
    Frame& operator=(const Frame&) = delete;
    Frame(Frame&&) = delete;
    Frame& operator=(Frame&&) = delete;
    ~Frame();

    void initialize(Device& device);
    void shutdown();
    void wait_for_in_flight(uint32_t frame_index) const;
    void reset_in_flight_fence(uint32_t frame_index) const;

    [[nodiscard]] SemaphoreHandle image_available_semaphore(uint32_t frame_index) const { return image_available_[frame_index]; }
    [[nodiscard]] FenceHandle in_flight_fence(uint32_t frame_index) const { return in_flight_[frame_index]; }
    [[nodiscard]] uint32_t advance() noexcept;
    [[nodiscard]] uint32_t current() const noexcept { return current_frame_; }

private:
    Device* device_ = nullptr;
    std::array<SemaphoreHandle, kFramesInFlight> image_available_{};
    std::array<FenceHandle, kFramesInFlight> in_flight_{};
    uint32_t current_frame_ = 0;
};
}
