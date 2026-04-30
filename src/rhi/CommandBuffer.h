#pragma once

#include <array>
#include <cstdint>

#include <volk.h>

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
    [[nodiscard]] VkCommandBuffer begin_frame(uint32_t frame_index);
    void end_frame(VkCommandBuffer cmd);

    // sync2 helper: transitions a single colour image's layout, picking
    // generic-but-correct stage/access masks for the source and dest layouts.
    static void image_barrier(
        VkCommandBuffer cmd,
        VkImage image,
        VkImageLayout old_layout,
        VkImageLayout new_layout);

private:
    Device* device_ = nullptr;
    std::array<VkCommandPool, kFramesInFlight> pools_{};
    std::array<VkCommandBuffer, kFramesInFlight> buffers_{};
};
}
