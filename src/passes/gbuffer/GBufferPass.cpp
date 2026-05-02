#include "passes/gbuffer/GBufferPass.h"

#include "core/Log.h"
#include "render/FrameContext.h"
#include "rhi/BindlessRegistry.h"
#include "rhi/CommandBuffer.h"
#include "rhi/Device.h"
#include "scene/GpuScene.h"
#include "scene/Scene.h"

#include <imgui.h>
#include <stdexcept>

// VMA for alloc flags
#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

namespace rr::passes::gbuffer
{

// Push constant sent once per draw call (per instance).
struct GBufferPushConstants
{
    uint32_t camera_buf_idx;
    uint32_t vertex_buf_idx;
    uint32_t index_buf_idx;
    uint32_t mesh_buf_idx;
    uint32_t material_buf_idx;
    uint32_t instance_buf_idx;
    uint32_t draw_instance_idx;
    uint32_t linear_sampler_idx;
};

GBufferPass::~GBufferPass()
{
    // Caller must call shutdown() explicitly.
}

void GBufferPass::initialize(rr::rhi::Device& device,
                               rr::shader::SlangSession& session,
                               rr::rhi::BindlessRegistry& registry,
                               rr::rhi::Extent2D extent)
{
    device_   = &device;
    registry_ = &registry;
    extent_   = extent;

    // Compile GBuffer shaders
    shader_.compile(session,
        "assets/shaders/passes/gbuffer/gbuffer.slang",
        {
            {"vs_main", rr::shader::ShaderStage::Vertex},
            {"fs_main", rr::shader::ShaderStage::Fragment}
        });
    reflection_ = rr::shader::ShaderReflection(shader_.program_layout());

    create_images(device, registry, extent);
    create_pipeline(device, registry);
    initialized_ = true;
    core::log()->info("[GBufferPass] Initialized {}x{}", extent.width, extent.height);
}

void GBufferPass::shutdown(rr::rhi::Device& device)
{
    if (!initialized_) return;
    pipeline_.destroy(device);
    shader_.reset();
    destroy_images(device);
    initialized_ = false;
}

void GBufferPass::create_images(rr::rhi::Device& device,
                                  rr::rhi::BindlessRegistry& registry,
                                  rr::rhi::Extent2D ext)
{
    const rr::rhi::Extent3D e3{ext.width, ext.height, 1};
    const rr::rhi::ImageUsage storage_usage =
        rr::rhi::ImageUsage::ColorAttachment |
        rr::rhi::ImageUsage::Storage         |
        rr::rhi::ImageUsage::Sampled;

    // World-space position (32F to avoid precision loss at mm scale)
    {
        rr::rhi::ImageDesc d{};
        d.format     = rr::rhi::Format::R32G32B32A32_Sfloat;
        d.extent     = e3;
        d.usage      = storage_usage;
        d.debug_name = "gbuffer_position";
        position_img_.create(device, d);
        position_storage_idx = registry.register_storage_image(
            device, position_img_, rr::rhi::Format::R32G32B32A32_Sfloat);
    }
    // World-space normal + material_id packed in .w as asfloat(material_id).
    // 32F required: material_id values produce denorm floats that 16F flushes to 0.
    {
        rr::rhi::ImageDesc d{};
        d.format     = rr::rhi::Format::R32G32B32A32_Sfloat;
        d.extent     = e3;
        d.usage      = storage_usage;
        d.debug_name = "gbuffer_normal";
        normal_img_.create(device, d);
        normal_storage_idx = registry.register_storage_image(
            device, normal_img_, rr::rhi::Format::R32G32B32A32_Sfloat);
    }
    // Material ID (R32_UINT as RGBA8 is not supported for storage everywhere)
    // Use R32_UINT for storage + sampled.
    {
        rr::rhi::ImageDesc d{};
        d.format     = rr::rhi::Format::R32_Uint;
        d.extent     = e3;
        d.usage      = rr::rhi::ImageUsage::ColorAttachment | rr::rhi::ImageUsage::Storage;
        d.debug_name = "gbuffer_material_id";
        material_id_img_.create(device, d);
        material_id_storage_idx = registry.register_storage_image(
            device, material_id_img_, rr::rhi::Format::R32_Uint);
    }
    // Depth
    {
        rr::rhi::ImageDesc d{};
        d.format     = rr::rhi::Format::D32_Sfloat;
        d.extent     = e3;
        d.usage      = rr::rhi::ImageUsage::DepthStencilAttachment | rr::rhi::ImageUsage::Sampled;
        d.aspect     = rr::rhi::ImageAspect::Depth;
        d.debug_name = "gbuffer_depth";
        depth_img_.create(device, d);
    }
}

void GBufferPass::destroy_images(rr::rhi::Device& device)
{
    depth_img_.destroy(device);
    material_id_img_.destroy(device);
    normal_img_.destroy(device);
    position_img_.destroy(device);
}

void GBufferPass::create_pipeline(rr::rhi::Device& device,
                                    rr::rhi::BindlessRegistry& registry)
{
    rr::rhi::GraphicsPipelineDesc desc{};
    desc.module       = &shader_;
    desc.reflection   = &reflection_;
    desc.vert_entry   = 0;
    desc.frag_entry   = 1;
    desc.registry     = &registry;
    desc.color_formats = {
        rr::rhi::Format::R32G32B32A32_Sfloat,  // position (32F for sub-mm precision)
        rr::rhi::Format::R32G32B32A32_Sfloat,  // normal + material_id packed in .w
        rr::rhi::Format::R32_Uint              // material id
    };
    desc.depth_format  = rr::rhi::Format::D32_Sfloat;
    desc.depth_test    = true;
    desc.depth_write   = true;
    desc.depth_compare = rr::rhi::CompareOp::Less;
    desc.cull_mode     = rr::rhi::CullMode::Back;
    desc.debug_name    = "gbuffer_pipeline";
    pipeline_.create(device, desc);
}

void GBufferPass::pre_transition_to_general(rr::rhi::CommandRecorder recorder)
{
    VkCommandBuffer cmd = static_cast<VkCommandBuffer>(recorder.handle());
    auto transition = [&](VkImage img) {
        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.srcStageMask        = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        b.srcAccessMask       = 0;
        b.dstStageMask        = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        b.dstAccessMask       = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = img;
        b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                  VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
    };
    transition(position_img_.handle());
    transition(normal_img_.handle());
    transition(material_id_img_.handle());
}

void GBufferPass::on_resize(rr::rhi::Extent2D new_extent)
{
    if (!initialized_) return;
    extent_ = new_extent;
    destroy_images(*device_);
    create_images(*device_, *registry_, new_extent);
    // Rebuild pipeline only if format changed (it doesn't here).
}

rr::render::RenderPass::Reflection GBufferPass::reflect() const
{
    Reflection r;
    r.outputs.push_back({"gbuffer_position",    ResourceDesc::Kind::Texture,
                          rr::rhi::Format::R32G32B32A32_Sfloat, extent_});
    r.outputs.push_back({"gbuffer_normal",      ResourceDesc::Kind::Texture,
                          rr::rhi::Format::R32G32B32A32_Sfloat, extent_});
    r.outputs.push_back({"gbuffer_material_id", ResourceDesc::Kind::Texture,
                          rr::rhi::Format::R32_Uint, extent_});
    return r;
}

void GBufferPass::render_ui()
{
    ImGui::Text("GBuffer %ux%u", extent_.width, extent_.height);
    ImGui::Text("  pos  gStorageImages[%u]", position_storage_idx);
    ImGui::Text("  norm gStorageImages[%u]", normal_storage_idx);
    ImGui::Text("  matid gStorageImages[%u]", material_id_storage_idx);
}

void GBufferPass::execute(rr::render::FrameContext& fc)
{
    const auto& scene = *fc.scene;
    if (!scene.is_uploaded() || scene.instance_count() == 0) return;
    if (!pipeline_.is_valid()) return;

    VkCommandBuffer cmd = static_cast<VkCommandBuffer>(fc.command_recorder.handle());
    const auto& handles = scene.gpu_handles();
    const VkExtent2D extent{extent_.width, extent_.height};

    // Transition GBuffer images to COLOR_ATTACHMENT_OPTIMAL
    auto barrier = [&](VkImage img, VkImageAspectFlags aspect,
                        VkImageLayout old_l, VkImageLayout new_l)
    {
        VkImageMemoryBarrier2 b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask        = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        b.srcAccessMask       = 0;
        b.dstStageMask        = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
        b.dstAccessMask       = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        b.oldLayout           = old_l;
        b.newLayout           = new_l;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = img;
        b.subresourceRange    = {aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
    };
    barrier(position_img_.handle(),    VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    barrier(normal_img_.handle(),      VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    barrier(material_id_img_.handle(), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    barrier(depth_img_.handle(),       VK_IMAGE_ASPECT_DEPTH_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    // Begin dynamic rendering
    std::array<VkRenderingAttachmentInfo, 3> color_attachments{};
    auto make_ca = [](VkImageView view) {
        VkRenderingAttachmentInfo a{};
        a.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        a.imageView   = view;
        a.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        a.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        a.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
        a.clearValue.color = {0.0f, 0.0f, 0.0f, 0.0f};
        return a;
    };
    color_attachments[0] = make_ca(position_img_.view());
    color_attachments[1] = make_ca(normal_img_.view());
    color_attachments[2] = make_ca(material_id_img_.view());
    color_attachments[2].clearValue.color.uint32[0] = 0xFFFFFFFF; // invalid material

    VkRenderingAttachmentInfo depth_att{};
    depth_att.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth_att.imageView   = depth_img_.view();
    depth_att.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depth_att.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_att.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    depth_att.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo rendering{};
    rendering.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering.renderArea           = {{0,0}, extent};
    rendering.layerCount           = 1;
    rendering.colorAttachmentCount = static_cast<uint32_t>(color_attachments.size());
    rendering.pColorAttachments    = color_attachments.data();
    rendering.pDepthAttachment     = &depth_att;
    vkCmdBeginRendering(cmd, &rendering);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_.handle());

    VkViewport viewport{0.0f, 0.0f,
                         static_cast<float>(extent_.width),
                         static_cast<float>(extent_.height),
                         0.0f, 1.0f};
    VkRect2D scissor{{0,0}, extent};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Bind global index buffer once (all meshes use the same global buffer)
    // Note: No VkIndexBuffer bind needed; VS manually reads index data via bindless buffer.

    // Per-instance draw loop
    GBufferPushConstants pc{};
    pc.camera_buf_idx      = handles.camera_buf_idx;
    pc.vertex_buf_idx      = handles.vertex_buf_idx;
    pc.index_buf_idx       = handles.index_buf_idx;
    pc.mesh_buf_idx        = handles.mesh_buf_idx;
    pc.material_buf_idx    = handles.material_buf_idx;
    pc.instance_buf_idx    = handles.instance_buf_idx;
    pc.linear_sampler_idx  = handles.linear_sampler_idx;

    const auto& meshes    = scene.meshes();
    const auto& instances = scene.instances();

    for (uint32_t i = 0; i < static_cast<uint32_t>(instances.size()); ++i)
    {
        pc.draw_instance_idx = i;
        VkPushDataInfoEXT push_info{};
        push_info.sType        = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
        push_info.offset       = 0;
        push_info.data.address = &pc;
        push_info.data.size    = sizeof(pc);
        vkCmdPushDataEXT(cmd, &push_info);

        const auto& inst = instances[i];
        const auto& mesh = meshes[inst.mesh_index];
        vkCmdDraw(cmd, mesh.index_count, 1, 0, 0);
    }

    vkCmdEndRendering(cmd);

    // Transition outputs to GENERAL layout (readable as storage images by compute passes)
    auto to_general = [&](VkImage img) {
        VkImageMemoryBarrier2 b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask        = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        b.srcAccessMask       = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        b.dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask       = VK_ACCESS_2_SHADER_READ_BIT;
        b.oldLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = img;
        b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                  VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
    };
    to_general(position_img_.handle());
    to_general(normal_img_.handle());
    to_general(material_id_img_.handle());
}

} // namespace rr::passes::gbuffer
