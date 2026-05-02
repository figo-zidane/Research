#include "passes/restir_di/ReSTIRDIPass.h"

#include "core/Log.h"
#include "render/FrameContext.h"
#include "rhi/BindlessRegistry.h"
#include "rhi/Device.h"
#include "scene/GpuScene.h"
#include "scene/Scene.h"

#include <imgui.h>
#include <stdexcept>
#include <vector>

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

void image_barrier_compute(rr::rhi::CommandRecorder recorder,
                           const rr::rhi::ImageHandle* images,
                           uint32_t        count,
                           rr::rhi::AccessFlags src_access,
                           rr::rhi::AccessFlags dst_access,
                           rr::rhi::ImageLayout old_layout)
{
    std::vector<rr::rhi::ImageBarrier> barriers(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        auto& b = barriers[i];
        b.image_handle = images[i];
        b.src_stage = rr::rhi::PipelineStage::ComputeShader;
        b.src_access = src_access;
        b.dst_stage = rr::rhi::PipelineStage::ComputeShader;
        b.dst_access = dst_access;
        b.old_layout = old_layout;
        b.new_layout = rr::rhi::ImageLayout::General;
        b.subresource = {
            .aspect = rr::rhi::ImageAspect::Color,
            .base_mip = 0,
            .mip_count = rr::rhi::kRemainingMipLevels,
            .base_layer = 0,
            .layer_count = rr::rhi::kRemainingArrayLayers,
        };
    }
    recorder.pipeline_barrier(barriers);
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

ReSTIRDIPass::~ReSTIRDIPass() = default;

void ReSTIRDIPass::initialize(rr::rhi::Device& device,
                               rr::shader::SlangSession& session,
                               rr::rhi::BindlessRegistry& registry,
                               rr::rhi::Extent2D extent)
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
                                   rr::rhi::Extent2D ext)
{
    auto make_storage = [&](rr::rhi::Image& img, const char* name) -> uint32_t
    {
        rr::rhi::ImageDesc d{};
        d.format     = rr::rhi::Format::R32G32B32A32_Sfloat;
        d.extent     = {ext.width, ext.height, 1};
        d.usage      = rr::rhi::ImageUsage::Storage | rr::rhi::ImageUsage::TransferSrc;
        d.debug_name = name;
        img.create(device, d);
        return registry.register_storage_image(device, img, rr::rhi::Format::R32G32B32A32_Sfloat);
    };

    reservoir_idx_[0] = make_storage(reservoir_[0], "restir_reservoir_A");
    reservoir_idx_[1] = make_storage(reservoir_[1], "restir_reservoir_B");

    // Output: storage for shader writes + sampled texture for TonemapPass
    {
        rr::rhi::ImageDesc d{};
        d.format     = rr::rhi::Format::R32G32B32A32_Sfloat;
        d.extent     = {ext.width, ext.height, 1};
        d.usage      = rr::rhi::ImageUsage::Storage
                     | rr::rhi::ImageUsage::Sampled
                     | rr::rhi::ImageUsage::TransferSrc;
        d.debug_name = "restir_output";
        output_img_.create(device, d);
    }
    output_storage_idx = registry.register_storage_image(
        device, output_img_, rr::rhi::Format::R32G32B32A32_Sfloat);
    // BUG-2 (Option A): register as GENERAL — the image stays in GENERAL
    // layout throughout the frame, so the descriptor layout must match.
    output_texture_idx = registry.register_texture(
        device, output_img_,
        rr::rhi::Format::R32G32B32A32_Sfloat,
        rr::rhi::ImageLayout::General,
        rr::rhi::ImageAspect::Color);
}

void ReSTIRDIPass::destroy_images(rr::rhi::Device& device)
{
    reservoir_[0].destroy(device);
    reservoir_[1].destroy(device);
    output_img_.destroy(device);
}

rr::rhi::ImageHandle ReSTIRDIPass::output_image_handle() const
{
    return rr::rhi::to_handle(output_img_.handle());
}

void ReSTIRDIPass::pre_transition_to_general(rr::rhi::CommandRecorder recorder)
{
    rr::rhi::ImageHandle imgs[] = {
        rr::rhi::to_handle(reservoir_[0].handle()),
        rr::rhi::to_handle(reservoir_[1].handle()),
        rr::rhi::to_handle(output_img_.handle()),
    };
    image_barrier_compute(recorder, imgs, 3,
                          rr::rhi::AccessFlags::None,
                          rr::rhi::AccessFlags::ShaderRead | rr::rhi::AccessFlags::ShaderWrite,
                          rr::rhi::ImageLayout::Undefined);
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
                          rr::rhi::Format::R32G32B32A32_Sfloat, extent_});
    return r;
}

void ReSTIRDIPass::on_resize(rr::rhi::Extent2D new_extent)
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

// ── Execute ───────────────────────────────────────────────────────────────────

void ReSTIRDIPass::execute(rr::render::FrameContext& fc)
{
    if (!initialized_)              return;
    if (!initial_pipeline_.is_valid()
     || !temporal_pipeline_.is_valid()
     || !spatial_shade_pipeline_.is_valid()) return;

    const auto& scene = *fc.scene;
    if (!scene.is_uploaded()) return;

    const rr::rhi::CommandRecorder recorder = fc.command_recorder;
    const auto&     handles = scene.gpu_handles();
    const rr::rhi::ImageHandle gbuf_pos_image = gbuf_pos_image_;
    const rr::rhi::ImageHandle gbuf_norm_image = gbuf_norm_image_;

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

    // ── cs_initial ────────────────────────────────────────────────────────
    // G-buffer is written by cs_initial itself (ray-traced primary hits).
    // GBufferPass may have written the same slots beforehand; cs_initial
    // overwrites them.  Transition from GENERAL (post-GBufferPass) → GENERAL,
    // discarding old content (oldLayout = UNDEFINED for discard-transition).
    {
        rr::rhi::ImageHandle gbuf_imgs[] = {gbuf_pos_image, gbuf_norm_image};
        image_barrier_compute(recorder, gbuf_imgs, 2,
                              rr::rhi::AccessFlags::None,
                              rr::rhi::AccessFlags::ShaderWrite,
                              rr::rhi::ImageLayout::General);
    }
    // reservoir_curr: first frame starts in UNDEFINED, subsequent in GENERAL.
    {
        rr::rhi::ImageHandle imgs[] = {rr::rhi::to_handle(reservoir_[reservoir_flip_].handle())};
        image_barrier_compute(recorder, imgs, 1,
                              rr::rhi::AccessFlags::None,
                              rr::rhi::AccessFlags::ShaderWrite,
                              first_execute_ ? rr::rhi::ImageLayout::Undefined
                                             : rr::rhi::ImageLayout::General);
    }

    recorder.bind_compute_pipeline(initial_pipeline_);
    recorder.push_constants(&pc, sizeof(pc));
    recorder.dispatch(gx, gy, 1);

    // Barrier: G-buffer and reservoir_curr written → readable by temporal/spatial passes.
    {
        rr::rhi::ImageHandle imgs[] = {gbuf_pos_image, gbuf_norm_image,
                          rr::rhi::to_handle(reservoir_[reservoir_flip_].handle())};
        image_barrier_compute(recorder, imgs, 3,
                               rr::rhi::AccessFlags::ShaderWrite,
                               rr::rhi::AccessFlags::ShaderRead | rr::rhi::AccessFlags::ShaderWrite,
                               rr::rhi::ImageLayout::General);
    }

    // ── cs_temporal ───────────────────────────────────────────────────────
    // reservoir_prev needs to be readable. It was written by the previous frame
    // and is already in GENERAL layout from that frame's write barrier.
    // Re-issue a barrier to ensure visibility from this frame's perspective.
    {
        const rr::rhi::ImageBarrier barrier{
            .image = &reservoir_[1 - reservoir_flip_],
            .src_stage = rr::rhi::PipelineStage::ComputeShader,
            .src_access = rr::rhi::AccessFlags::ShaderWrite,
            .dst_stage = rr::rhi::PipelineStage::ComputeShader,
            .dst_access = rr::rhi::AccessFlags::ShaderRead,
            .old_layout = first_execute_ ? rr::rhi::ImageLayout::Undefined
                                         : rr::rhi::ImageLayout::General,
            .new_layout = rr::rhi::ImageLayout::General,
            .subresource = {
                .aspect = rr::rhi::ImageAspect::Color,
                .base_mip = 0,
                .mip_count = rr::rhi::kRemainingMipLevels,
                .base_layer = 0,
                .layer_count = rr::rhi::kRemainingArrayLayers,
            },
        };
        recorder.pipeline_barrier({&barrier, 1});
    }

    recorder.bind_compute_pipeline(temporal_pipeline_);
    recorder.push_constants(&pc, sizeof(pc));
    recorder.dispatch(gx, gy, 1);

    // Barrier: reservoir_curr updated → readable by spatial/shade pass.
    // Use GENERAL as oldLayout (preserve content written by temporal pass).
    {
        rr::rhi::ImageHandle imgs[] = {rr::rhi::to_handle(reservoir_[reservoir_flip_].handle())};
        image_barrier_compute(recorder, imgs, 1,
                               rr::rhi::AccessFlags::ShaderWrite,
                               rr::rhi::AccessFlags::ShaderRead,
                               rr::rhi::ImageLayout::General);
    }

    // ── cs_spatial_shade ──────────────────────────────────────────────────
    // output_img_: first frame starts in UNDEFINED, subsequent in GENERAL.
    {
        rr::rhi::ImageHandle imgs[] = {rr::rhi::to_handle(output_img_.handle())};
        image_barrier_compute(recorder, imgs, 1,
                               rr::rhi::AccessFlags::None,
                               rr::rhi::AccessFlags::ShaderWrite,
                               first_execute_ ? rr::rhi::ImageLayout::Undefined
                                              : rr::rhi::ImageLayout::General);
    }

    recorder.bind_compute_pipeline(spatial_shade_pipeline_);
    recorder.push_constants(&pc, sizeof(pc));
    recorder.dispatch(gx, gy, 1);

    // Flip reservoir ping-pong for next frame.
    reservoir_flip_ ^= 1u;
    first_execute_   = false;
}

} // namespace rr::passes::restir_di
