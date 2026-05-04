#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "rhi/Types.h"

namespace rr::rhi
{
class Device;
class BindlessRegistry;
} // namespace rr::rhi

namespace rr::shader
{
class ShaderModule;
class ShaderReflection;
} // namespace rr::shader

namespace rr::rhi
{

inline constexpr uint32_t kShaderUnused = ~0u;

// ── Compute Pipeline ──────────────────────────────────────────────────────────

struct ComputePipelineDesc
{
    const shader::ShaderModule*     module      = nullptr;
    const shader::ShaderReflection* reflection  = nullptr;
    // Entry point index within the module (default: 0).
    uint32_t                        entry_index = 0;
    // Optional heap-mapping descriptors.  If null and registry is provided,
    // the standard BindlessRegistry mapping is used.
    const BindlessRegistry*         registry    = nullptr;
    const char*                     debug_name  = nullptr;
};

class ComputePipeline
{
public:
    ComputePipeline() = default;
    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;
    ~ComputePipeline() = default;

    void create(Device& device, const ComputePipelineDesc& desc);
    void destroy(Device& device);

    // Swap pipeline handles (both must already be destroyed or not yet created).
    void swap(ComputePipeline& other) noexcept { std::swap(pipeline_, other.pipeline_); }

    [[nodiscard]] uint64_t handle() const noexcept { return pipeline_; }
    [[nodiscard]] bool     is_valid() const noexcept { return pipeline_ != 0; }

private:
    uint64_t pipeline_ = 0;
};

// ── Graphics Pipeline ─────────────────────────────────────────────────────────

struct GraphicsPipelineDesc
{
    const shader::ShaderModule*     module          = nullptr;
    const shader::ShaderReflection* reflection      = nullptr;
    uint32_t                        vert_entry      = 0;  // index in module
    uint32_t                        frag_entry      = 1;
    const BindlessRegistry*         registry        = nullptr;
    // Render attachments
    std::vector<Format>             color_formats;
    Format                          depth_format    = Format::Undefined;
    // Rasterization
    CullMode                        cull_mode       = CullMode::Back;
    FrontFace                       front_face      = FrontFace::CounterClockwise;
    PrimitiveTopology               topology        = PrimitiveTopology::TriangleList;
    bool                            depth_test      = true;
    bool                            depth_write     = true;
    CompareOp                       depth_compare   = CompareOp::Less;
    // No vertex input — all meshes are pulled via buffer device address.
    const char*                     debug_name      = nullptr;
};

class GraphicsPipeline
{
public:
    GraphicsPipeline() = default;
    GraphicsPipeline(const GraphicsPipeline&) = delete;
    GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;
    ~GraphicsPipeline() = default;

    void create(Device& device, const GraphicsPipelineDesc& desc);
    void destroy(Device& device);

    // Swap pipeline handles (both must already be destroyed or not yet created).
    void swap(GraphicsPipeline& other) noexcept { std::swap(pipeline_, other.pipeline_); }

    [[nodiscard]] uint64_t handle() const noexcept { return pipeline_; }
    [[nodiscard]] bool     is_valid() const noexcept { return pipeline_ != 0; }

private:
    uint64_t pipeline_ = 0;
};

// ── Ray-Tracing Pipeline ─────────────────────────────────────────────────────

struct RtShaderGroup
{
    // All indices refer to entry points within the linked module.
    // Leave an entry at kShaderUnused when the group does not use it.
    uint32_t raygen_index        = kShaderUnused;
    uint32_t miss_index          = kShaderUnused;
    uint32_t closest_hit_index   = kShaderUnused;
    uint32_t any_hit_index       = kShaderUnused;
    uint32_t intersection_index  = kShaderUnused;
};

struct RtPipelineDesc
{
    const shader::ShaderModule*     module      = nullptr;
    const shader::ShaderReflection* reflection  = nullptr;
    std::vector<RtShaderGroup>      groups;
    uint32_t                        max_recursion_depth = 2;
    uint32_t                        max_payload_size    = 32;  // bytes
    uint32_t                        max_attribute_size  = 8;   // bytes
    const BindlessRegistry*         registry    = nullptr;
    const char*                     debug_name  = nullptr;
};

class RtPipeline
{
public:
    RtPipeline() = default;
    RtPipeline(const RtPipeline&) = delete;
    RtPipeline& operator=(const RtPipeline&) = delete;
    ~RtPipeline() = default;

    void create(Device& device, const RtPipelineDesc& desc);
    void destroy(Device& device);

    [[nodiscard]] uint64_t handle() const noexcept { return pipeline_; }
    [[nodiscard]] bool     is_valid() const noexcept { return pipeline_ != 0; }

    // Returns the raw SBT handle for shader group at the given index.
    // The returned pointer is valid until the pipeline is destroyed.
    [[nodiscard]] const uint8_t* sbt_handle(uint32_t group_index) const;
    [[nodiscard]] uint32_t       sbt_handle_size() const noexcept { return sbt_handle_size_; }
    [[nodiscard]] uint32_t       group_count()     const noexcept { return group_count_; }

private:
    uint64_t             pipeline_        = 0;
    std::vector<uint8_t> sbt_handles_;
    uint32_t             sbt_handle_size_ = 0;
    uint32_t             group_count_     = 0;
};

} // namespace rr::rhi
