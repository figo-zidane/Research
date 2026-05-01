#include "passes/restir_di/ReSTIRDIPass.h"

#include "core/Log.h"
#include "render/FrameContext.h"
#include "rhi/BindlessRegistry.h"
#include "rhi/Device.h"
#include "scene/GpuScene.h"
#include "scene/Scene.h"

#include <imgui.h>
#include <stdexcept>

namespace rr::passes::restir_di
{

// Push-constant layout must match restir_di.slang exactly (96 bytes).
struct ReSTIRDIPushConstants
{
    uint32_t camera_buf_idx;
    uint32_t tlas_idx;
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
    uint32_t gbuf_pos_idx;
    uint32_t gbuf_norm_idx;
    uint32_t reservoir_curr_idx;
    uint32_t reservoir_prev_idx;
    uint32_t output_idx;
    uint32_t num_initial_candidates;
    uint32_t temporal_enable;
    uint32_t spatial_enable;
    uint32_t spatial_num_neighbors;
    uint32_t spatial_radius;
    uint32_t _pad0;
    uint32_t _pad1;
};
static_assert(sizeof(ReSTIRDIPushConstants) == 96);

// ── Lifecycle ─────────────────────────────────────────────────────────────────

ReSTIRDIPass::~ReSTIRDIPass() = default;

void ReSTIRDIPass::initialize(rr::rhi::Device& device,
                               rr::shader::SlangSession& session,
                               rr::rhi::BindlessRegistry& registry,
                               VkExtent2D extent)
{
    device_   = &device;
    registry_ = &registry;
    extent_   = extent;

    shader_.compile(session,
        "assets/shaders/passes/restir_di/restir_di.slang",
        {
            {"cs_initial",       rr::shader::ShaderStage::Compute},
            {"cs_temporal",      rr::shader::ShaderStage::Compute},
            {"cs_spatial_shade", rr::shader::ShaderStage::Compute},
        });
    reflection_ = rr::shader::ShaderReflection(shader_.program_layout());

    create_images(device, registry, extent);
    create_pipelines(device, registry);
    initialized_ = true;
    core::log()->info("[ReSTIRDIPass] Initialized {}x{}", extent.width, extent.height);
}

void ReSTIRDIPass::shutdown(rr::rhi::Device& device)
{
    if (!initialized_) return;
    initial_pipeline_.destroy(device);
    temporal_pipeline_.destroy(device);
    spatial_shade_pipeline_.destroy(device);
    shader_.reset();
    destroy_images(device);
    initialized_ = false;
}

bool ReSTIRDIPass::reload_shader(rr::shader::SlangSession& session)
{
    if (!initialized_) return false;

    // Step 1: try to compile new shader — leave old shader intact on failure.
    rr::shader::ShaderModule new_shader;
    try
    {
        new_shader.compile(session,
            "assets/shaders/passes/restir_di/restir_di.slang",
            {
                {"cs_initial",       rr::shader::ShaderStage::Compute},
                {"cs_temporal",      rr::shader::ShaderStage::Compute},
                {"cs_spatial_shade", rr::shader::ShaderStage::Compute},
            });
    }
    catch (const std::exception& e)
    {
        core::log()->error("[ReSTIRDIPass] Shader recompile failed: {}", e.what());
        return false; // old shader_ still valid
    }
    rr::shader::ShaderReflection new_reflection(new_shader.program_layout());

    // Step 2: try to create all three new pipelines using the new shader.
    auto make_desc = [&](uint32_t entry_idx, const char* name)
    {
        rr::rhi::ComputePipelineDesc d{};
        d.module      = &new_shader;
        d.reflection  = &new_reflection;
        d.entry_index = entry_idx;
        d.registry    = registry_;
        d.debug_name  = name;
        return d;
    };

    rr::rhi::ComputePipeline new_init, new_temp, new_spat;
    try
    {
        new_init.create(*device_, make_desc(kEntryInitial,      "restir_initial_pipeline"));
        new_temp.create(*device_, make_desc(kEntryTemporal,     "restir_temporal_pipeline"));
        new_spat.create(*device_, make_desc(kEntrySpatialShade, "restir_spatial_shade_pipeline"));
    }
    catch (const std::exception& e)
    {
        // Destroy any partially-created pipelines and keep old ones.
        new_init.destroy(*device_);
        new_temp.destroy(*device_);
        new_spat.destroy(*device_);
        core::log()->error("[ReSTIRDIPass] Pipeline recreate failed: {}", e.what());
        return false; // old pipelines still valid
    }

    // Step 3: both succeeded — swap in new resources and destroy old.
    initial_pipeline_.destroy(*device_);
    temporal_pipeline_.destroy(*device_);
    spatial_shade_pipeline_.destroy(*device_);

    initial_pipeline_.swap(new_init);
    temporal_pipeline_.swap(new_temp);
    spatial_shade_pipeline_.swap(new_spat);

    shader_.swap(new_shader);       // new_shader now holds old Slang objects
    reflection_ = new_reflection;   // ShaderReflection is copyable
    // new_shader destroyed at end of scope → releases old Slang COM objects
    return true;
}

// ── Image management ──────────────────────────────────────────────────────────

void ReSTIRDIPass::create_images(rr::rhi::Device& device,
                                   rr::rhi::BindlessRegistry& registry,
                                   VkExtent2D ext)
{
    auto make_storage = [&](rr::rhi::Image& img, const char* name) -> uint32_t
    {
        rr::rhi::ImageDesc d{};
        d.format     = VK_FORMAT_R32G32B32A32_SFLOAT;
        d.extent     = {ext.width, ext.height, 1};
        d.usage      = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        d.debug_name = name;
        img.create(device, d);
        return registry.register_storage_image(device, img.handle(), VK_FORMAT_R32G32B32A32_SFLOAT);
    };

    gbuf_pos_idx_  = make_storage(gbuf_pos_,       "restir_gbuf_pos");
    gbuf_norm_idx_ = make_storage(gbuf_norm_,      "restir_gbuf_norm");
    reservoir_idx_[0] = make_storage(reservoir_[0], "restir_reservoir_A");
    reservoir_idx_[1] = make_storage(reservoir_[1], "restir_reservoir_B");

    // Output: storage for shader writes + sampled texture for TonemapPass
    {
        rr::rhi::ImageDesc d{};
        d.format     = VK_FORMAT_R32G32B32A32_SFLOAT;
        d.extent     = {ext.width, ext.height, 1};
        d.usage      = VK_IMAGE_USAGE_STORAGE_BIT
                     | VK_IMAGE_USAGE_SAMPLED_BIT
                     | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        d.debug_name = "restir_output";
        output_img_.create(device, d);
    }
    output_storage_idx = registry.register_storage_image(
        device, output_img_.handle(), VK_FORMAT_R32G32B32A32_SFLOAT);
    // BUG-2 (Option A): register as GENERAL — the image stays in GENERAL
    // layout throughout the frame, so the descriptor layout must match.
    output_texture_idx = registry.register_texture(
        device, output_img_.handle(),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_ASPECT_COLOR_BIT);
}

void ReSTIRDIPass::destroy_images(rr::rhi::Device& device)
{
    gbuf_pos_.destroy(device);
    gbuf_norm_.destroy(device);
    reservoir_[0].destroy(device);
    reservoir_[1].destroy(device);
    output_img_.destroy(device);
}

VkImage ReSTIRDIPass::output_image_handle() const
{
    return output_img_.handle();
}

void ReSTIRDIPass::pre_transition_to_general(VkCommandBuffer cmd)
{
    VkImage imgs[] = {
        gbuf_pos_.handle(),
        gbuf_norm_.handle(),
        reservoir_[0].handle(),
        reservoir_[1].handle(),
        output_img_.handle(),
    };
    image_barrier_compute(cmd, imgs, 5, 0, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                           VK_IMAGE_LAYOUT_UNDEFINED);
}

// ── Pipeline management ───────────────────────────────────────────────────────

void ReSTIRDIPass::create_pipelines(rr::rhi::Device& device,
                                     rr::rhi::BindlessRegistry& registry)
{
    auto make = [&](rr::rhi::ComputePipeline& pl, uint32_t entry_idx, const char* name)
    {
        rr::rhi::ComputePipelineDesc d{};
        d.module      = &shader_;
        d.reflection  = &reflection_;
        d.entry_index = entry_idx;
        d.registry    = &registry;
        d.debug_name  = name;
        pl.create(device, d);
    };
    make(initial_pipeline_,      kEntryInitial,      "restir_initial_pipeline");
    make(temporal_pipeline_,     kEntryTemporal,     "restir_temporal_pipeline");
    make(spatial_shade_pipeline_, kEntrySpatialShade, "restir_spatial_shade_pipeline");
}

// ── RenderPass interface ──────────────────────────────────────────────────────

rr::render::RenderPass::Reflection ReSTIRDIPass::reflect() const
{
    Reflection r;
    r.outputs.push_back({"restir_di_output", ResourceDesc::Kind::Texture,
                          VK_FORMAT_R32G32B32A32_SFLOAT, extent_});
    return r;
}

void ReSTIRDIPass::on_resize(VkExtent2D new_extent)
{
    if (!initialized_) return;
    extent_ = new_extent;
    destroy_images(*device_);
    create_images(*device_, *registry_, new_extent);
    reservoir_flip_  = 0; // reset temporal history
    first_execute_   = true; // images are in UNDEFINED after re-creation
}

void ReSTIRDIPass::render_ui()
{
    ImGui::Text("ReSTIR DI  %ux%u", extent_.width, extent_.height);
    ImGui::Checkbox("Temporal reuse",  &temporal_enable);
    ImGui::Checkbox("Spatial reuse",   &spatial_enable);
    ImGui::SliderInt("Initial candidates (M)",
                     reinterpret_cast<int*>(&num_initial_candidates), 1, 64);
    ImGui::SliderInt("Spatial neighbours",
                     reinterpret_cast<int*>(&spatial_num_neighbors), 0, 16);
    ImGui::SliderInt("Spatial radius (px)",
                     reinterpret_cast<int*>(&spatial_radius), 1, 64);
    ImGui::Text("  output gStorageImages[%u] / gTextures[%u]",
                output_storage_idx, output_texture_idx);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

void ReSTIRDIPass::image_barrier_compute(VkCommandBuffer cmd,
                                          const VkImage*  images,
                                          uint32_t        count,
                                          VkAccessFlags2  src_access,
                                          VkAccessFlags2  dst_access,
                                          VkImageLayout   old_layout)
{
    std::vector<VkImageMemoryBarrier2> barriers(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        auto& b = barriers[i];
        b = {};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.srcAccessMask       = src_access;
        b.dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask       = dst_access;
        b.oldLayout           = old_layout;
        b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = images[i];
        b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                  VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    }
    VkDependencyInfo dep{};
    dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = count;
    dep.pImageMemoryBarriers    = barriers.data();
    vkCmdPipelineBarrier2(cmd, &dep);
}

// ── Execute ───────────────────────────────────────────────────────────────────

void ReSTIRDIPass::execute(rr::render::FrameContext& fc)
{
    if (!initialized_)              return;
    if (!initial_pipeline_.is_valid()
     || !temporal_pipeline_.is_valid()
     || !spatial_shade_pipeline_.is_valid()) return;

    const auto& scene = *fc.scene;
    if (!scene.is_uploaded()) return;

    VkCommandBuffer cmd     = fc.command_buffer;
    const auto&     handles = scene.gpu_handles();

    // Determine current / previous reservoir image indices for this frame
    uint32_t curr_res = reservoir_idx_[reservoir_flip_];
    uint32_t prev_res = reservoir_idx_[1 - reservoir_flip_];

    // Dispatch dimensions (8×8 tiles)
    const uint32_t gx = (extent_.width  + 7) / 8;
    const uint32_t gy = (extent_.height + 7) / 8;

    // Build push constants (shared by all three dispatches)
    ReSTIRDIPushConstants pc{};
    pc.camera_buf_idx         = handles.camera_buf_idx;
    pc.tlas_idx               = handles.tlas_idx;
    pc.vertex_buf_idx         = handles.vertex_buf_idx;
    pc.index_buf_idx          = handles.index_buf_idx;
    pc.mesh_buf_idx           = handles.mesh_buf_idx;
    pc.material_buf_idx       = handles.material_buf_idx;
    pc.instance_buf_idx       = handles.instance_buf_idx;
    pc.light_buf_idx          = handles.light_buf_idx;
    pc.num_instances          = scene.instance_count();
    pc.num_lights             = scene.light_count();
    pc.linear_sampler_idx     = handles.linear_sampler_idx;
    pc.frame_seed             = frame_count_++;
    pc.gbuf_pos_idx           = gbuf_pos_idx_;
    pc.gbuf_norm_idx          = gbuf_norm_idx_;
    pc.reservoir_curr_idx     = curr_res;
    pc.reservoir_prev_idx     = prev_res;
    pc.output_idx             = output_storage_idx;
    pc.num_initial_candidates = num_initial_candidates;
    pc.temporal_enable        = temporal_enable  ? 1u : 0u;
    pc.spatial_enable         = spatial_enable   ? 1u : 0u;
    pc.spatial_num_neighbors  = spatial_num_neighbors;
    pc.spatial_radius         = spatial_radius;

    VkPushDataInfoEXT push_info{};
    push_info.sType        = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
    push_info.offset       = 0;
    push_info.data.address = &pc;
    push_info.data.size    = sizeof(pc);

    // ── cs_initial ────────────────────────────────────────────────────────
    // G-Buffer: always discard (UNDEFINED → GENERAL).
    {
        VkImage imgs[] = {gbuf_pos_.handle(), gbuf_norm_.handle()};
        image_barrier_compute(cmd, imgs, 2, 0, VK_ACCESS_2_SHADER_WRITE_BIT,
                               VK_IMAGE_LAYOUT_UNDEFINED);
    }
    // reservoir_curr: first frame starts in UNDEFINED, subsequent in GENERAL
    // so temporal reuse can see the previous frame's content.
    {
        VkImage imgs[] = {reservoir_[reservoir_flip_].handle()};
        image_barrier_compute(cmd, imgs, 1, 0, VK_ACCESS_2_SHADER_WRITE_BIT,
                               first_execute_ ? VK_IMAGE_LAYOUT_UNDEFINED
                                              : VK_IMAGE_LAYOUT_GENERAL);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, initial_pipeline_.handle());
    vkCmdPushDataEXT(cmd, &push_info);
    vkCmdDispatch(cmd, gx, gy, 1);

    // Barrier: G-buffer and reservoir_curr written → readable by temporal pass.
    // Use GENERAL as oldLayout (images are already in GENERAL from above).
    {
        VkImage imgs[] = {gbuf_pos_.handle(), gbuf_norm_.handle(),
                          reservoir_[reservoir_flip_].handle()};
        image_barrier_compute(cmd, imgs, 3,
                               VK_ACCESS_2_SHADER_WRITE_BIT,
                               VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                               VK_IMAGE_LAYOUT_GENERAL);
    }

    // ── cs_temporal ───────────────────────────────────────────────────────
    // reservoir_prev needs to be readable. It was written by the previous frame
    // and is already in GENERAL layout from that frame's write barrier.
    // Re-issue a barrier to ensure visibility from this frame's perspective.
    {
        VkImageMemoryBarrier2 b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.srcAccessMask       = VK_ACCESS_2_SHADER_WRITE_BIT;
        b.dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask       = VK_ACCESS_2_SHADER_READ_BIT;
        // On first frame the previous reservoir is in UNDEFINED (no prior data).
        b.oldLayout           = first_execute_ ? VK_IMAGE_LAYOUT_UNDEFINED
                                               : VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = reservoir_[1 - reservoir_flip_].handle();
        b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                  VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        VkDependencyInfo dep{};
        dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, temporal_pipeline_.handle());
    vkCmdPushDataEXT(cmd, &push_info);
    vkCmdDispatch(cmd, gx, gy, 1);

    // Barrier: reservoir_curr updated → readable by spatial/shade pass.
    // Use GENERAL as oldLayout (preserve content written by temporal pass).
    {
        VkImage imgs[] = {reservoir_[reservoir_flip_].handle()};
        image_barrier_compute(cmd, imgs, 1,
                               VK_ACCESS_2_SHADER_WRITE_BIT,
                               VK_ACCESS_2_SHADER_READ_BIT,
                               VK_IMAGE_LAYOUT_GENERAL);
    }

    // ── cs_spatial_shade ──────────────────────────────────────────────────
    // output_img_: first frame starts in UNDEFINED, subsequent in GENERAL.
    {
        VkImage imgs[] = {output_img_.handle()};
        image_barrier_compute(cmd, imgs, 1, 0, VK_ACCESS_2_SHADER_WRITE_BIT,
                               first_execute_ ? VK_IMAGE_LAYOUT_UNDEFINED
                                              : VK_IMAGE_LAYOUT_GENERAL);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, spatial_shade_pipeline_.handle());
    vkCmdPushDataEXT(cmd, &push_info);
    vkCmdDispatch(cmd, gx, gy, 1);

    // Flip reservoir ping-pong for next frame.
    reservoir_flip_ ^= 1u;
    first_execute_   = false;
}

} // namespace rr::passes::restir_di
