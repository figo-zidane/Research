#pragma once

#include <cstdint>

#include "rhi/CommandRecorder.h"
#include "rhi/Types.h"

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
    rr::rhi::CommandRecorder command_recorder{};
    rr::rhi::Device* device = nullptr;
    rr::rhi::BindlessRegistry* bindless_registry = nullptr;
    rr::scene::Scene* scene = nullptr;
    Renderer* renderer = nullptr;
    // Current swapchain image being rendered to.
    rr::rhi::ImageViewHandle swapchain_image_view = 0;
    rr::rhi::Extent2D        swapchain_extent{};
    uint32_t                 image_index = 0;
    uint32_t                 frame_index = 0;
    float                    delta_time_seconds = 0.0f;
    // Accumulated HDR image handle (for TonemapPass barrier).
    rr::rhi::ImageHandle     accumulated_image = 0;
};
}
