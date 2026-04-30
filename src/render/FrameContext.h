#pragma once

#include <cstdint>

#include <volk.h>

namespace rr::render
{
class Renderer;
}

namespace rr::rhi
{
class Device;
}

namespace rr::scene
{
class Scene;
}

namespace rr::render
{
struct FrameContext
{
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    rr::rhi::Device* device = nullptr;
    rr::scene::Scene* scene = nullptr;
    Renderer* renderer = nullptr;
    uint32_t frame_index = 0;
    float delta_time_seconds = 0.0f;
};
}
