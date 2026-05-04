#pragma once

#include "rhi/CommandRecorder.h"
#include <array>
#include <cstdint>

#include "rhi/Frame.h"

namespace rr::rhi
{
class Device;

class CommandBuffer
{
public:
    CommandBuffer() = default;
    CommandBuffer(const CommandBuffer&) = delete;
    CommandBuffer& operator=(const CommandBuffer&) = delete;
    CommandBuffer(CommandBuffer&&) = delete;
    CommandBuffer& operator=(CommandBuffer&&) = delete;
    ~CommandBuffer();

    void initialize(Device& device);
    void shutdown();

    // Reset the pool for the given frame slot and return a freshly-begun
    // primary command buffer ready for recording.
    [[nodiscard]] CommandRecorder begin_frame(uint32_t frame_index);
    void end_frame(CommandRecorder cmd);

    // Returns the command pool for frame slot 0 (suitable for one-time submissions
    // that don't need to align with a specific in-flight frame slot).
    [[nodiscard]] CommandPoolHandle pool() const { return pools_[0]; }

private:
    Device* device_ = nullptr;
    std::array<CommandPoolHandle, kFramesInFlight> pools_{};
    std::array<CommandBufferHandle, kFramesInFlight> buffers_{};
};
}
