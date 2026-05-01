#include "passes/pathtracer_reference/PathTracerPass.h"

#include "core/Log.h"
#include "render/FrameContext.h"
#include "rhi/BindlessRegistry.h"
#include "rhi/Device.h"
#include "scene/GpuScene.h"
#include "scene/Scene.h"

#include <imgui.h>

namespace rr::passes::pathtracer
{

struct PathTracerPushConstants
{
    uint32_t camera_buf_idx;
    uint32_t tlas_idx;
    uint32_t radiance_img_idx;
    uint32_t vertex_buf_idx;
    uint32_t index_buf_idx;
    uint32_t mesh_buf_idx;
    uint32_t material_buf_idx;
    uint32_t instance_buf_idx;
    uint32_t light_buf_idx;
    uint32_t num_instances;
    uint32_t num_lights;
    uint32_t linear_sampler_idx;
    uint32_t frame_seed;
    uint32_t max_bounces;
    uint32_t _pad[2];
};

PathTracerPass::~PathTracerPass() = default;

void PathTracerPass::initialize(rr::rhi::Device& device,
                                  rr::shader::SlangSession& session,
                                  rr::rhi::BindlessRegistry& registry,
                                  VkExtent2D extent)
{
    device_   = &device;
    registry_ = &registry;
    extent_   = extent;

    shader_.compile(session,
        "assets/shaders/passes/pathtracer_reference/pathtracer.slang",
        {{"cs_main", rr::shader::ShaderStage::Compute}});
    reflection_ = rr::shader::ShaderReflection(shader_.program_layout());

    create_images(device, registry, extent);
    create_pipeline(device, registry);
    initialized_ = true;
    core::log()->info("[PathTracerPass] Initialized {}x{}", extent.width, extent.height);
}

void PathTracerPass::shutdown(rr::rhi::Device& device)
{
    if (!initialized_) return;
    pipeline_.destroy(device);
    shader_.reset();
    destroy_images(device);
    initialized_ = false;
}

bool PathTracerPass::reload_shader(rr::shader::SlangSession& session)
{
    if (!initialized_) return false;

    // Step 1: try compile new shader — old shader_ stays intact on failure.
    rr::shader::ShaderModule new_shader;
    try
    {
        new_shader.compile(session,
            "assets/shaders/passes/pathtracer_reference/pathtracer.slang",
            {{"cs_main", rr::shader::ShaderStage::Compute}});
    }
    catch (const std::exception& e)
    {
        core::log()->error("[PathTracerPass] Shader recompile failed: {}", e.what());
        return false;
    }
    rr::shader::ShaderReflection new_reflection(new_shader.program_layout());

    // Step 2: try create new pipeline using the new shader.
    rr::rhi::ComputePipeline new_pipeline;
    rr::rhi::ComputePipelineDesc d{};
    d.module      = &new_shader;
    d.reflection  = &new_reflection;
    d.entry_index = 0;
    d.registry    = registry_;
    d.debug_name  = "pathtracer_pipeline";
    try
    {
        new_pipeline.create(*device_, d);
    }
    catch (const std::exception& e)
    {
        core::log()->error("[PathTracerPass] Pipeline recreate failed: {}", e.what());
        return false; // old pipeline_ still valid
    }

    // Step 3: both succeeded — destroy old, swap in new.
    pipeline_.destroy(*device_);
    pipeline_.swap(new_pipeline);
    shader_.swap(new_shader);
    reflection_ = new_reflection;
    return true;
}

void PathTracerPass::create_images(rr::rhi::Device& device,
                                     rr::rhi::BindlessRegistry& registry,
                                     VkExtent2D ext)
{
    rr::rhi::ImageDesc d{};
    d.format     = VK_FORMAT_R32G32B32A32_SFLOAT;
    d.extent     = {ext.width, ext.height, 1};
    d.usage      = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    d.debug_name = "radiance_image";
    radiance_img_.create(device, d);

    radiance_storage_idx = registry.register_storage_image(
        device, radiance_img_.handle(), VK_FORMAT_R32G32B32A32_SFLOAT);
    radiance_texture_idx = registry.register_texture(
        device, radiance_img_.handle(),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_ASPECT_COLOR_BIT);
}

void PathTracerPass::destroy_images(rr::rhi::Device& device)
{
    radiance_img_.destroy(device);
}

void PathTracerPass::create_pipeline(rr::rhi::Device& device,
                                       rr::rhi::BindlessRegistry& registry)
{
    rr::rhi::ComputePipelineDesc desc{};
    desc.module      = &shader_;
    desc.reflection  = &reflection_;
    desc.entry_index  = 0;
    desc.registry    = &registry;
    desc.debug_name  = "pathtracer_pipeline";
    pipeline_.create(device, desc);
}

void PathTracerPass::on_resize(VkExtent2D new_extent)
{
    if (!initialized_) return;
    extent_ = new_extent;
    destroy_images(*device_);
    create_images(*device_, *registry_, new_extent);
}

rr::render::RenderPass::Reflection PathTracerPass::reflect() const
{
    Reflection r;
    r.outputs.push_back({"radiance_image", ResourceDesc::Kind::Texture,
                          VK_FORMAT_R32G32B32A32_SFLOAT, extent_});
    return r;
}

void PathTracerPass::render_ui()
{
    ImGui::Text("PathTracer %ux%u", extent_.width, extent_.height);
    ImGui::SliderInt("Max Bounces", reinterpret_cast<int*>(&max_bounces), 1, 16);
    ImGui::Text("  radiance gStorageImages[%u]", radiance_storage_idx);
}

void PathTracerPass::execute(rr::render::FrameContext& fc)
{
    const auto& scene = *fc.scene;
    if (!scene.is_uploaded()) return;
    if (!pipeline_.is_valid()) return;

    VkCommandBuffer cmd = fc.command_buffer;
    const auto& handles = scene.gpu_handles();

    // Transition radiance image to GENERAL
    {
        VkImageMemoryBarrier2 b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask        = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        b.srcAccessMask       = 0;
        b.dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask       = VK_ACCESS_2_SHADER_WRITE_BIT;
        b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = radiance_img_.handle();
        b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                  VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        VkDependencyInfo dep{};
        dep.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount  = 1;
        dep.pImageMemoryBarriers     = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_.handle());

    PathTracerPushConstants pc{};
    pc.camera_buf_idx    = handles.camera_buf_idx;
    pc.tlas_idx          = handles.tlas_idx;
    pc.radiance_img_idx  = radiance_storage_idx;
    pc.vertex_buf_idx    = handles.vertex_buf_idx;
    pc.index_buf_idx     = handles.index_buf_idx;
    pc.mesh_buf_idx      = handles.mesh_buf_idx;
    pc.material_buf_idx  = handles.material_buf_idx;
    pc.instance_buf_idx  = handles.instance_buf_idx;
    pc.light_buf_idx     = handles.light_buf_idx;
    pc.num_instances     = scene.instance_count();
    pc.num_lights        = scene.light_count();
    pc.linear_sampler_idx = handles.linear_sampler_idx;
    pc.frame_seed        = frame_count_++;
    pc.max_bounces       = max_bounces;

    VkPushDataInfoEXT push_info{};
    push_info.sType        = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
    push_info.offset       = 0;
    push_info.data.address = &pc;
    push_info.data.size    = sizeof(pc);
    vkCmdPushDataEXT(cmd, &push_info);

    // Dispatch one thread per pixel, 8x8 tiles
    uint32_t gx = (extent_.width  + 7) / 8;
    uint32_t gy = (extent_.height + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);
}

} // namespace rr::passes::pathtracer
