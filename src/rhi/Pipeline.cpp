#include "rhi/Pipeline.h"

#include "core/Log.h"
#include "rhi/BindlessRegistry.h"
#include "rhi/Device.h"
#include "shader/ShaderModule.h"
#include "shader/ShaderReflection.h"

#include <stdexcept>
#include <string>

namespace rr::rhi
{

// ── Internal helpers ──────────────────────────────────────────────────────────

namespace
{
// Create a VkShaderModule from raw SPIR-V bytes.
VkShaderModule create_shader_module(VkDevice device, const std::vector<uint8_t>& spv)
{
    if (spv.empty())
    {
        throw std::runtime_error("Pipeline: SPIR-V blob is empty.");
    }
    VkShaderModuleCreateInfo info{};
    info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = spv.size();
    info.pCode    = reinterpret_cast<const uint32_t*>(spv.data());

    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &info, nullptr, &mod) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateShaderModule failed.");
    }
    return mod;
}

// Build a VkPipelineShaderStageCreateInfo.
VkPipelineShaderStageCreateInfo stage_info(VkShaderStageFlagBits stage,
                                           VkShaderModule        module,
                                           const char*           entry)
{
    VkPipelineShaderStageCreateInfo info{};
    info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage  = stage;
    info.module = module;
    info.pName  = entry;
    return info;
}

// Map ShaderStage → VkShaderStageFlagBits.
VkShaderStageFlagBits to_vk_stage(shader::ShaderStage s)
{
    using S = shader::ShaderStage;
    switch (s)
    {
    case S::Vertex:        return VK_SHADER_STAGE_VERTEX_BIT;
    case S::Fragment:      return VK_SHADER_STAGE_FRAGMENT_BIT;
    case S::Compute:       return VK_SHADER_STAGE_COMPUTE_BIT;
    case S::RayGeneration: return VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    case S::ClosestHit:    return VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    case S::Miss:          return VK_SHADER_STAGE_MISS_BIT_KHR;
    case S::AnyHit:        return VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    case S::Intersection:  return VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
    default:               return VK_SHADER_STAGE_ALL;
    }
}
} // namespace

// ── HeapMappingChain ──────────────────────────────────────────────────────────
// Bindings used in BindlessHeap.slang:
//   set 0, binding 0 → gTextures[]       (SAMPLED_IMAGE)
//   set 0, binding 1 → gStorageImages[]  (STORAGE_IMAGE)
//   set 0, binding 2 → gScene[]          (STORAGE_BUFFER)
//   set 0, binding 3 → gTLAS[]           (ACCELERATION_STRUCTURE)
//   set 1, binding 0 → gSamplers[]       (sampler heap)

HeapMappingChain build_heap_mapping(const BindlessRegistry& reg)
{
    HeapMappingChain chain;

    // We create one mapping entry per resource-heap array plus one for the
    // sampler heap.  Slang emits the arrays with sequential set/binding numbers
    // when using layout(descriptor_heap).
    //
    // resource heap → set 0
    // sampler heap  → set 1

    // Reserve to avoid reallocation (pointers must stay stable for pNext).
    chain.offset_data.resize(5);
    chain.set_mappings.resize(5);

    auto make_offset = [](VkDeviceSize heap_offset, VkDeviceSize stride) {
        VkDescriptorMappingSourceConstantOffsetEXT off{};
        off.heapOffset       = static_cast<uint32_t>(heap_offset);
        off.heapArrayStride  = static_cast<uint32_t>(stride);
        return off;
    };

    // gTextures (binding 0, resource heap)
    chain.offset_data[0] = make_offset(reg.texture_heap_offset(),       reg.image_descriptor_stride());
    // gStorageImages (binding 1, resource heap)
    chain.offset_data[1] = make_offset(reg.storage_image_heap_offset(), reg.image_descriptor_stride());
    // gSamplers (binding 0 on set 1, sampler heap) — handled separately below
    chain.offset_data[2] = make_offset(reg.sampler_heap_offset(),       reg.sampler_descriptor_stride());
    // gScene (binding 2, resource heap)
    chain.offset_data[3] = make_offset(reg.scene_buffer_heap_offset(),  reg.buffer_descriptor_stride());
    // gTLAS (binding 3, resource heap)
    chain.offset_data[4] = make_offset(reg.tlas_heap_offset(),          reg.buffer_descriptor_stride());

    auto make_mapping = [](uint32_t set, uint32_t binding, uint32_t count,
                           VkSpirvResourceTypeFlagsEXT resource_mask,
                           const VkDescriptorMappingSourceConstantOffsetEXT* offset_ptr) {
        VkDescriptorSetAndBindingMappingEXT m{};
        m.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT;
        m.descriptorSet = set;
        m.firstBinding  = binding;
        m.bindingCount  = count;
        m.resourceMask  = resource_mask;
        m.source        = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
        m.sourceData.constantOffset = *offset_ptr;
        return m;
    };

    chain.set_mappings[0] = make_mapping(0, 0, 1,
        VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT,
        &chain.offset_data[0]);
    chain.set_mappings[1] = make_mapping(0, 1, 1,
        VK_SPIRV_RESOURCE_TYPE_READ_WRITE_IMAGE_BIT_EXT,
        &chain.offset_data[1]);
    chain.set_mappings[2] = make_mapping(1, 0, 1,
        VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT,
        &chain.offset_data[2]);
    chain.set_mappings[3] = make_mapping(0, 2, 1,
        VK_SPIRV_RESOURCE_TYPE_READ_ONLY_STORAGE_BUFFER_BIT_EXT |
        VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT,
        &chain.offset_data[3]);
    chain.set_mappings[4] = make_mapping(0, 3, 1,
        VK_SPIRV_RESOURCE_TYPE_ACCELERATION_STRUCTURE_BIT_EXT,
        &chain.offset_data[4]);

    chain.mapping_info.sType        = VK_STRUCTURE_TYPE_SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT;
    chain.mapping_info.pNext        = nullptr;
    chain.mapping_info.mappingCount = static_cast<uint32_t>(chain.set_mappings.size());
    chain.mapping_info.pMappings    = chain.set_mappings.data();

    return chain;
}

// ── ComputePipeline ───────────────────────────────────────────────────────────

void ComputePipeline::create(Device& device, const ComputePipelineDesc& desc)
{
    if (pipeline_ != VK_NULL_HANDLE)
    {
        throw std::runtime_error("ComputePipeline::create called twice.");
    }
    if (!desc.module || !desc.module->is_valid())
    {
        throw std::runtime_error("ComputePipeline: ShaderModule is not compiled.");
    }
    if (!desc.reflection)
    {
        throw std::runtime_error("ComputePipeline: ShaderReflection is null.");
    }

    const auto& eps = desc.reflection->entry_points();
    if (desc.entry_index >= eps.size())
    {
        throw std::runtime_error("ComputePipeline: entry_index out of range.");
    }

    VkShaderModule vk_mod = create_shader_module(device.device(), desc.module->spv_code(desc.entry_index));

    // Build heap-mapping chain.  Per spec, VkShaderDescriptorSetAndBindingMappingInfoEXT
    // must be placed in VkPipelineShaderStageCreateInfo::pNext (not the layout).
    HeapMappingChain mapping_chain;
    if (desc.registry)
    {
        mapping_chain = build_heap_mapping(*desc.registry);
    }

    VkPipelineCreateFlags2CreateInfo flags2{};
    flags2.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO;
    flags2.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    // Slang emits SPIR-V with the entry point name "main" regardless of the
    // original function name.  Use "main" here to match.
    VkPipelineShaderStageCreateInfo stage_ci =
        stage_info(VK_SHADER_STAGE_COMPUTE_BIT, vk_mod, "main");
    if (desc.registry)
    {
        stage_ci.pNext = mapping_chain.pnext();
    }

    VkComputePipelineCreateInfo ci{};
    ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.pNext  = &flags2;
    ci.flags  = 0;
    ci.stage  = stage_ci;
    ci.layout = VK_NULL_HANDLE;

    const VkResult result = vkCreateComputePipelines(
        device.device(), VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline_);

    vkDestroyShaderModule(device.device(), vk_mod, nullptr);

    if (result != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateComputePipelines failed.");
    }

    if (desc.debug_name)
    {
        VkDebugUtilsObjectNameInfoEXT ni{};
        ni.sType        = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        ni.objectType   = VK_OBJECT_TYPE_PIPELINE;
        ni.objectHandle = reinterpret_cast<uint64_t>(pipeline_);
        ni.pObjectName  = desc.debug_name;
        vkSetDebugUtilsObjectNameEXT(device.device(), &ni);
    }

    rr::core::log()->info("ComputePipeline: created '{}'",
                          desc.debug_name ? desc.debug_name : "<unnamed>");
}

void ComputePipeline::destroy(Device& device)
{
    if (pipeline_ == VK_NULL_HANDLE) { return; }
    vkDestroyPipeline(device.device(), pipeline_, nullptr);
    pipeline_ = VK_NULL_HANDLE;
}

// ── GraphicsPipeline ──────────────────────────────────────────────────────────

void GraphicsPipeline::create(Device& device, const GraphicsPipelineDesc& desc)
{
    if (pipeline_ != VK_NULL_HANDLE)
    {
        throw std::runtime_error("GraphicsPipeline::create called twice.");
    }
    if (!desc.module || !desc.module->is_valid())
    {
        throw std::runtime_error("GraphicsPipeline: ShaderModule is not compiled.");
    }
    if (!desc.reflection) { throw std::runtime_error("GraphicsPipeline: reflection is null."); }

    const auto& eps = desc.reflection->entry_points();
    if (desc.vert_entry >= eps.size() || desc.frag_entry >= eps.size())
    {
        throw std::runtime_error("GraphicsPipeline: entry index out of range.");
    }

    // Build two VkShaderModules (one per entry point) since each has its own SPIR-V blob.
    // Slang emits SPIR-V with the entry point name "main" for every function.
    VkShaderModule vk_vs = create_shader_module(device.device(), desc.module->spv_code(desc.vert_entry));
    VkShaderModule vk_fs = create_shader_module(device.device(), desc.module->spv_code(desc.frag_entry));

    // Build heap-mapping chain.  Per spec, VkShaderDescriptorSetAndBindingMappingInfoEXT
    // must be placed in each VkPipelineShaderStageCreateInfo::pNext (not the layout).
    HeapMappingChain mapping_chain;
    if (desc.registry)
    {
        mapping_chain = build_heap_mapping(*desc.registry);
    }

    // Descriptor heap pipelines must use VK_NULL_HANDLE layout (spec requirement).
    VkPipelineShaderStageCreateInfo stages[] = {
        stage_info(VK_SHADER_STAGE_VERTEX_BIT,   vk_vs, "main"),
        stage_info(VK_SHADER_STAGE_FRAGMENT_BIT, vk_fs, "main"),
    };
    if (desc.registry)
    {
        stages[0].pNext = mapping_chain.pnext();
        stages[1].pNext = mapping_chain.pnext();
    }

    // No vertex input state — vertex data is fetched via buffer device addresses.
    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = desc.topology;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode    = desc.cull_mode;
    rs.frontFace   = desc.front_face;
    rs.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable  = desc.depth_test  ? VK_TRUE : VK_FALSE;
    ds.depthWriteEnable = desc.depth_write ? VK_TRUE : VK_FALSE;
    ds.depthCompareOp   = desc.depth_compare;

    // Default: opaque, no blending for all colour attachments.
    std::vector<VkPipelineColorBlendAttachmentState> blend_attachments(
        desc.color_formats.size());
    for (auto& att : blend_attachments)
    {
        att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                             VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    }

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = static_cast<uint32_t>(blend_attachments.size());
    cb.pAttachments    = blend_attachments.data();

    // Dynamic state: viewport and scissor are always dynamic.
    const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates    = dynamic_states;

    // Dynamic rendering (no VkRenderPass needed).
    VkPipelineRenderingCreateInfo rendering_info{};
    rendering_info.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering_info.colorAttachmentCount    = static_cast<uint32_t>(desc.color_formats.size());
    rendering_info.pColorAttachmentFormats = desc.color_formats.data();
    rendering_info.depthAttachmentFormat   = desc.depth_format;

    // flags2 leads the pNext chain so the driver sees it before rendering_info.
    VkPipelineCreateFlags2CreateInfo gfx_flags2{};
    gfx_flags2.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO;
    gfx_flags2.pNext = &rendering_info;
    gfx_flags2.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkGraphicsPipelineCreateInfo ci{};
    ci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    ci.pNext               = &gfx_flags2;
    ci.flags               = 0;
    ci.stageCount          = 2;
    ci.pStages             = stages;
    ci.pVertexInputState   = &vi;
    ci.pInputAssemblyState = &ia;
    ci.pViewportState      = &vp;
    ci.pRasterizationState = &rs;
    ci.pMultisampleState   = &ms;
    ci.pDepthStencilState  = &ds;
    ci.pColorBlendState    = &cb;
    ci.pDynamicState       = &dyn;
    ci.layout              = VK_NULL_HANDLE;
    ci.renderPass          = VK_NULL_HANDLE;

    const VkResult result = vkCreateGraphicsPipelines(
        device.device(), VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline_);

    vkDestroyShaderModule(device.device(), vk_vs, nullptr);
    vkDestroyShaderModule(device.device(), vk_fs, nullptr);

    if (result != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateGraphicsPipelines failed.");
    }

    if (desc.debug_name)
    {
        VkDebugUtilsObjectNameInfoEXT ni{};
        ni.sType        = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        ni.objectType   = VK_OBJECT_TYPE_PIPELINE;
        ni.objectHandle = reinterpret_cast<uint64_t>(pipeline_);
        ni.pObjectName  = desc.debug_name;
        vkSetDebugUtilsObjectNameEXT(device.device(), &ni);
    }

    rr::core::log()->info("GraphicsPipeline: created '{}'",
                          desc.debug_name ? desc.debug_name : "<unnamed>");
}

void GraphicsPipeline::destroy(Device& device)
{
    if (pipeline_ == VK_NULL_HANDLE) { return; }
    vkDestroyPipeline(device.device(), pipeline_, nullptr);
    pipeline_ = VK_NULL_HANDLE;
}

// ── RtPipeline ────────────────────────────────────────────────────────────────

void RtPipeline::create(Device& device, const RtPipelineDesc& desc)
{
    if (pipeline_ != VK_NULL_HANDLE)
    {
        throw std::runtime_error("RtPipeline::create called twice.");
    }
    if (!desc.module || !desc.module->is_valid())
    {
        throw std::runtime_error("RtPipeline: ShaderModule is not compiled.");
    }
    if (!desc.reflection || desc.groups.empty())
    {
        throw std::runtime_error("RtPipeline: reflection is null or no shader groups.");
    }

    const auto& eps = desc.reflection->entry_points();
    // Build stage array — each entry point has its own SPIR-V blob with "main" as the name.
    std::vector<VkShaderModule> vk_mods;
    vk_mods.reserve(eps.size());
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    for (size_t i = 0; i < eps.size(); ++i)
    {
        VkShaderModule m = create_shader_module(device.device(), desc.module->spv_code(static_cast<uint32_t>(i)));
        vk_mods.push_back(m);
        stages.push_back(stage_info(to_vk_stage(eps[i].stage), m, "main"));
    }

    // Heap mapping chain; attached to each stage's pNext per VK_EXT_descriptor_heap spec.
    HeapMappingChain mapping_chain;
    if (desc.registry)
    {
        mapping_chain = build_heap_mapping(*desc.registry);
        for (auto& s : stages)
        {
            s.pNext = mapping_chain.pnext();
        }
    }

    // Build shader groups.
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> vk_groups;
    vk_groups.reserve(desc.groups.size());
    for (const auto& g : desc.groups)
    {
        VkRayTracingShaderGroupCreateInfoKHR vkg{};
        vkg.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;

        if (g.raygen_index != VK_SHADER_UNUSED_KHR)
        {
            vkg.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
            vkg.generalShader      = g.raygen_index;
            vkg.closestHitShader   = VK_SHADER_UNUSED_KHR;
            vkg.anyHitShader       = VK_SHADER_UNUSED_KHR;
            vkg.intersectionShader = VK_SHADER_UNUSED_KHR;
        }
        else if (g.miss_index != VK_SHADER_UNUSED_KHR)
        {
            vkg.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
            vkg.generalShader      = g.miss_index;
            vkg.closestHitShader   = VK_SHADER_UNUSED_KHR;
            vkg.anyHitShader       = VK_SHADER_UNUSED_KHR;
            vkg.intersectionShader = VK_SHADER_UNUSED_KHR;
        }
        else
        {
            vkg.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
            vkg.generalShader      = VK_SHADER_UNUSED_KHR;
            vkg.closestHitShader   = g.closest_hit_index;
            vkg.anyHitShader       = g.any_hit_index;
            vkg.intersectionShader = g.intersection_index;
        }
        vk_groups.push_back(vkg);
    }

    // Descriptor heap pipelines must use VK_NULL_HANDLE layout (spec requirement).
    VkPipelineCreateFlags2CreateInfo rt_flags2{};
    rt_flags2.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO;
    rt_flags2.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    VkRayTracingPipelineCreateInfoKHR ci{};
    ci.sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    ci.pNext                        = &rt_flags2;
    ci.flags                        = 0;
    ci.stageCount                   = static_cast<uint32_t>(stages.size());
    ci.pStages                      = stages.data();
    ci.groupCount                   = static_cast<uint32_t>(vk_groups.size());
    ci.pGroups                      = vk_groups.data();
    ci.maxPipelineRayRecursionDepth = desc.max_recursion_depth;
    ci.layout                       = VK_NULL_HANDLE;

    const VkResult result = vkCreateRayTracingPipelinesKHR(
        device.device(), VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline_);

    for (VkShaderModule m : vk_mods)
        vkDestroyShaderModule(device.device(), m, nullptr);

    if (result != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateRayTracingPipelinesKHR failed.");
    }

    // Query SBT handles.
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_props{};
    rt_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &rt_props;
    vkGetPhysicalDeviceProperties2(device.physical_device(), &props2);

    sbt_handle_size_ = rt_props.shaderGroupHandleSize;
    group_count_     = static_cast<uint32_t>(vk_groups.size());
    sbt_handles_.resize(static_cast<size_t>(sbt_handle_size_) * group_count_);

    if (vkGetRayTracingShaderGroupHandlesKHR(
            device.device(), pipeline_, 0, group_count_,
            sbt_handles_.size(), sbt_handles_.data()) != VK_SUCCESS)
    {
        rr::core::log()->warn("RtPipeline: vkGetRayTracingShaderGroupHandlesKHR failed.");
    }

    if (desc.debug_name)
    {
        VkDebugUtilsObjectNameInfoEXT ni{};
        ni.sType        = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        ni.objectType   = VK_OBJECT_TYPE_PIPELINE;
        ni.objectHandle = reinterpret_cast<uint64_t>(pipeline_);
        ni.pObjectName  = desc.debug_name;
        vkSetDebugUtilsObjectNameEXT(device.device(), &ni);
    }

    rr::core::log()->info("RtPipeline: created '{}' ({} groups)",
                          desc.debug_name ? desc.debug_name : "<unnamed>",
                          group_count_);
}

void RtPipeline::destroy(Device& device)
{
    if (pipeline_ == VK_NULL_HANDLE) { return; }
    vkDestroyPipeline(device.device(), pipeline_, nullptr);
    pipeline_        = VK_NULL_HANDLE;
    sbt_handles_.clear();
    sbt_handle_size_ = 0;
    group_count_     = 0;
}

const uint8_t* RtPipeline::sbt_handle(uint32_t group_index) const
{
    if (group_index >= group_count_) { return nullptr; }
    return sbt_handles_.data() + static_cast<size_t>(group_index) * sbt_handle_size_;
}

} // namespace rr::rhi
