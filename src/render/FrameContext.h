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
class BindlessRegistry;
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
    rr::rhi::BindlessRegistry* bindless_registry = nullptr;
    rr::scene::Scene* scene = nullptr;
    Renderer* renderer = nullptr;
    // Current swapchain image being rendered to.
    VkImageView swapchain_image_view = VK_NULL_HANDLE;
    VkExtent2D  swapchain_extent{};
    uint32_t    image_index = 0;
    uint32_t    frame_index = 0;
    float       delta_time_seconds = 0.0f;
};
}
