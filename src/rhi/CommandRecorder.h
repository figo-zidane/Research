#pragma once

#include "rhi/Barrier.h"
#include "rhi/Rendering.h"

#include <cstdint>
#include <span>

namespace rr::rhi
{
class ComputePipeline;
class GraphicsPipeline;
class Image;

class CommandRecorder
{
public:
    CommandRecorder() = default;
    explicit CommandRecorder(void* cmd) : cmd_(cmd) {}

    [[nodiscard]] void* handle() const noexcept { return cmd_; }
    [[nodiscard]] bool is_valid() const noexcept { return cmd_ != nullptr; }

    void bind_compute_pipeline(const ComputePipeline& pipeline) const;
    void bind_graphics_pipeline(const GraphicsPipeline& pipeline) const;
    void dispatch(uint32_t group_count_x, uint32_t group_count_y, uint32_t group_count_z) const;
    void draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance) const;
    void push_constants(const void* data, uint32_t size, uint32_t offset = 0) const;
    void set_viewport(float x, float y, float width, float height, float min_depth = 0.0f, float max_depth = 1.0f) const;
    void set_scissor(int32_t x, int32_t y, uint32_t width, uint32_t height) const;
    void pipeline_barrier(std::span<const ImageBarrier> barriers) const;
    void begin_rendering(const RenderingInfo& rendering_info) const;
    void end_rendering() const;
    void clear_color_image(const Image& image,
                           ImageLayout layout,
                           const ClearColor& clear_color,
                           std::span<const ImageSubresourceRange> ranges) const;

private:
    void* cmd_ = nullptr;
};
} // namespace rr::rhi