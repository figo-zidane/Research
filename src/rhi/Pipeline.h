#pragma once

#include <cstdint>
#include <utility>
#include <vector>
#include <volk.h>

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

    [[nodiscard]] VkPipeline       handle()  const noexcept { return pipeline_; }
    [[nodiscard]] bool             is_valid() const noexcept { return pipeline_ != VK_NULL_HANDLE; }

private:
    VkPipeline       pipeline_      = VK_NULL_HANDLE;
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
    std::vector<VkFormat>           color_formats;
    VkFormat                        depth_format    = VK_FORMAT_UNDEFINED;
    // Rasterization
    VkCullModeFlags                 cull_mode       = VK_CULL_MODE_BACK_BIT;
    VkFrontFace                     front_face      = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    VkPrimitiveTopology             topology        = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    bool                            depth_test      = true;
    bool                            depth_write     = true;
    VkCompareOp                     depth_compare   = VK_COMPARE_OP_LESS;
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

    [[nodiscard]] VkPipeline handle()   const noexcept { return pipeline_; }
    [[nodiscard]] bool       is_valid() const noexcept { return pipeline_ != VK_NULL_HANDLE; }

private:
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

// ── Ray-Tracing Pipeline ─────────────────────────────────────────────────────

struct RtShaderGroup
{
    // All indices refer to entry points within the linked module.
    // Set to VK_SHADER_UNUSED_KHR to leave unused.
    uint32_t raygen_index        = VK_SHADER_UNUSED_KHR;
    uint32_t miss_index          = VK_SHADER_UNUSED_KHR;
    uint32_t closest_hit_index   = VK_SHADER_UNUSED_KHR;
    uint32_t any_hit_index       = VK_SHADER_UNUSED_KHR;
    uint32_t intersection_index  = VK_SHADER_UNUSED_KHR;
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

    [[nodiscard]] VkPipeline handle()   const noexcept { return pipeline_; }
    [[nodiscard]] bool       is_valid() const noexcept { return pipeline_ != VK_NULL_HANDLE; }

    // Returns the raw SBT handle for shader group at the given index.
    // The returned pointer is valid until the pipeline is destroyed.
    [[nodiscard]] const uint8_t* sbt_handle(uint32_t group_index) const;
    [[nodiscard]] uint32_t       sbt_handle_size() const noexcept { return sbt_handle_size_; }
    [[nodiscard]] uint32_t       group_count()     const noexcept { return group_count_; }

private:
    VkPipeline           pipeline_        = VK_NULL_HANDLE;
    std::vector<uint8_t> sbt_handles_;
    uint32_t             sbt_handle_size_ = 0;
    uint32_t             group_count_     = 0;
};

// ── Helpers ───────────────────────────────────────────────────────────────────

// Build the VkShaderDescriptorSetAndBindingMappingInfoEXT pNext chain that
// maps the BindlessHeap arrays to the resource and sampler heaps.
// Returns a pNext chain that must be kept alive until vkCreateXxxPipeline
// returns.  All objects are stored in the provided vectors (caller owns them).
struct HeapMappingChain
{
    HeapMappingChain() = default;
    HeapMappingChain(HeapMappingChain&&) = default;
    HeapMappingChain& operator=(HeapMappingChain&&) = default;
    HeapMappingChain(const HeapMappingChain&) = delete;
    HeapMappingChain& operator=(const HeapMappingChain&) = delete;

    VkShaderDescriptorSetAndBindingMappingInfoEXT mapping_info{};
    std::vector<VkDescriptorSetAndBindingMappingEXT>  set_mappings;
    std::vector<VkDescriptorMappingSourceConstantOffsetEXT> offset_data;

    const void* pnext() const noexcept { return &mapping_info; }
};

// Build the heap-mapping chain for a pipeline that uses BindlessHeap.slang.
// Pass the chain's pnext() into the pipeline create-info's pNext.
HeapMappingChain build_heap_mapping(const BindlessRegistry& registry);

} // namespace rr::rhi
