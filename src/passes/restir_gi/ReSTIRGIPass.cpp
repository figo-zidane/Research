#include "passes/restir_gi/ReSTIRGIPass.h"

#include "core/Log.h"
#include "render/FrameContext.h"
#include "rhi/BindlessRegistry.h"
#include "rhi/Device.h"
#include "scene/GpuScene.h"
#include "scene/Scene.h"

#include <imgui.h>
#include <stdexcept>
#include <vector>

namespace rr::passes::restir_gi
{

namespace
{

void compile_stage(rr::shader::ShaderModule& shader,
                   rr::shader::ShaderReflection& reflection,
                   rr::shader::SlangSession& session,
                   const char* entry_name)
{
    shader.compile(session,
        "assets/shaders/passes/restir_gi/restir_gi.slang",
        {
            {entry_name, rr::shader::ShaderStage::Compute},
        });
    reflection = rr::shader::ShaderReflection(shader.program_layout());
}

struct ReSTIRGIPushConstants
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
    uint32_t frame_seed;
    uint32_t gbuf_pos_idx;
    uint32_t gbuf_norm_idx;
    uint32_t direct_input_idx;
    uint32_t reservoir_curr_pos_idx;
    uint32_t reservoir_curr_rad_idx;
    uint32_t reservoir_curr_meta_idx;
    uint32_t reservoir_prev_pos_idx;
    uint32_t reservoir_prev_rad_idx;
    uint32_t reservoir_prev_meta_idx;
    uint32_t history_curr_pos_idx;
    uint32_t history_curr_norm_idx;
    uint32_t history_prev_pos_idx;
    uint32_t history_prev_norm_idx;
    uint32_t output_idx;
    uint32_t num_initial_candidates;
    uint32_t temporal_enable;
    uint32_t spatial_enable;
    uint32_t spatial_num_neighbors;
    uint32_t spatial_radius;
    uint32_t include_direct_lighting;
    float    indirect_strength;
    float    ray_bias;
    uint32_t _pad0;
    uint32_t _pad1;
};
static_assert(sizeof(ReSTIRGIPushConstants) == 140);

void image_barrier(rr::rhi::CommandRecorder recorder,
                   const rr::rhi::ImageHandle* images,
                   uint32_t        count,
                   rr::rhi::PipelineStage src_stage,
                   rr::rhi::AccessFlags  src_access,
                   rr::rhi::PipelineStage dst_stage,
                   rr::rhi::AccessFlags  dst_access,
                   rr::rhi::ImageLayout  old_layout,
                   rr::rhi::ImageLayout  new_layout = rr::rhi::ImageLayout::General)
{
    std::vector<rr::rhi::ImageBarrier> barriers(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        auto& b = barriers[i];
        b.image_handle = images[i];
        b.src_stage = src_stage;
        b.src_access = src_access;
        b.dst_stage = dst_stage;
        b.dst_access = dst_access;
        b.old_layout = old_layout;
        b.new_layout = new_layout;
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

} // namespace

ReSTIRGIPass::~ReSTIRGIPass() = default;

void ReSTIRGIPass::initialize(rr::rhi::Device& device,
                              rr::shader::SlangSession& session,
                              rr::rhi::BindlessRegistry& registry,
                              rr::rhi::Extent2D extent)
{
    device_   = &device;
    registry_ = &registry;
    extent_   = extent;

    compile_stage(initial_shader_, initial_reflection_, session, "cs_initial");
    compile_stage(temporal_shader_, temporal_reflection_, session, "cs_temporal");
    compile_stage(spatial_shader_, spatial_reflection_, session, "cs_spatial_shade");

    create_images(device, registry, extent);
    create_pipelines(device, registry);
    initialized_ = true;
    core::log()->info("[ReSTIRGIPass] Initialized {}x{}", extent.width, extent.height);
}

void ReSTIRGIPass::shutdown(rr::rhi::Device& device)
{
    if (!initialized_) return;
    initial_pipeline_.destroy(device);
    temporal_pipeline_.destroy(device);
    spatial_shade_pipeline_.destroy(device);
    initial_shader_.reset();
    temporal_shader_.reset();
    spatial_shader_.reset();
    destroy_images(device);
    initialized_ = false;
}

bool ReSTIRGIPass::reload_shader(rr::shader::SlangSession& session)
{
    if (!initialized_) return false;

    rr::shader::ShaderModule new_initial_shader;
    rr::shader::ShaderModule new_temporal_shader;
    rr::shader::ShaderModule new_spatial_shader;
    rr::shader::ShaderReflection new_initial_reflection;
    rr::shader::ShaderReflection new_temporal_reflection;
    rr::shader::ShaderReflection new_spatial_reflection;
    try
    {
        compile_stage(new_initial_shader, new_initial_reflection, session, "cs_initial");
        compile_stage(new_temporal_shader, new_temporal_reflection, session, "cs_temporal");
        compile_stage(new_spatial_shader, new_spatial_reflection, session, "cs_spatial_shade");
    }
    catch (const std::exception& e)
    {
        core::log()->error("[ReSTIRGIPass] Shader recompile failed: {}", e.what());
        return false;
    }

    auto make_desc = [](rr::shader::ShaderModule& shader,
                        rr::shader::ShaderReflection& reflection,
                        const char* name)
    {
        rr::rhi::ComputePipelineDesc desc{};
        desc.module      = &shader;
        desc.reflection  = &reflection;
        desc.entry_index = 0;
        desc.debug_name  = name;
        return desc;
    };

    rr::rhi::ComputePipeline new_initial;
    rr::rhi::ComputePipeline new_temporal;
    rr::rhi::ComputePipeline new_spatial;

    try
    {
        auto init_desc = make_desc(new_initial_shader, new_initial_reflection, "restir_gi_initial_pipeline");
        init_desc.registry = registry_;
        auto temporal_desc = make_desc(new_temporal_shader, new_temporal_reflection, "restir_gi_temporal_pipeline");
        temporal_desc.registry = registry_;
        auto spatial_desc = make_desc(new_spatial_shader, new_spatial_reflection, "restir_gi_spatial_shade_pipeline");
        spatial_desc.registry = registry_;
        new_initial.create(*device_, init_desc);
        new_temporal.create(*device_, temporal_desc);
        new_spatial.create(*device_, spatial_desc);
    }
    catch (const std::exception& e)
    {
        core::log()->error("[ReSTIRGIPass] Pipeline recreate failed: {}", e.what());
        new_initial.destroy(*device_);
        new_temporal.destroy(*device_);
        new_spatial.destroy(*device_);
        return false;
    }

    initial_pipeline_.destroy(*device_);
    temporal_pipeline_.destroy(*device_);
    spatial_shade_pipeline_.destroy(*device_);

    initial_pipeline_.swap(new_initial);
    temporal_pipeline_.swap(new_temporal);
    spatial_shade_pipeline_.swap(new_spatial);

    initial_shader_.swap(new_initial_shader);
    temporal_shader_.swap(new_temporal_shader);
    spatial_shader_.swap(new_spatial_shader);
    initial_reflection_ = new_initial_reflection;
    temporal_reflection_ = new_temporal_reflection;
    spatial_reflection_ = new_spatial_reflection;
    return true;
}

void ReSTIRGIPass::create_images(rr::rhi::Device& device,
                                 rr::rhi::BindlessRegistry& registry,
                                 rr::rhi::Extent2D ext)
{
    auto make_storage = [&](rr::rhi::Image& image, const char* name) -> uint32_t
    {
        rr::rhi::ImageDesc desc{};
        desc.format     = rr::rhi::Format::R32G32B32A32_Sfloat;
        desc.extent     = {ext.width, ext.height, 1};
        desc.usage      = rr::rhi::ImageUsage::Storage
                        | rr::rhi::ImageUsage::TransferSrc
                        | rr::rhi::ImageUsage::TransferDst;
        desc.debug_name = name;
        image.create(device, desc);
        return registry.register_storage_image(device, image, rr::rhi::Format::R32G32B32A32_Sfloat);
    };

    reservoir_pos_idx_[0]  = make_storage(reservoir_pos_[0], "restir_gi_reservoir_pos_A");
    reservoir_pos_idx_[1]  = make_storage(reservoir_pos_[1], "restir_gi_reservoir_pos_B");
    reservoir_rad_idx_[0]  = make_storage(reservoir_rad_[0], "restir_gi_reservoir_rad_A");
    reservoir_rad_idx_[1]  = make_storage(reservoir_rad_[1], "restir_gi_reservoir_rad_B");
    reservoir_meta_idx_[0] = make_storage(reservoir_meta_[0], "restir_gi_reservoir_meta_A");
    reservoir_meta_idx_[1] = make_storage(reservoir_meta_[1], "restir_gi_reservoir_meta_B");
    history_primary_pos_idx_[0]  = make_storage(history_primary_pos_[0], "restir_gi_history_primary_pos_A");
    history_primary_pos_idx_[1]  = make_storage(history_primary_pos_[1], "restir_gi_history_primary_pos_B");
    history_primary_norm_idx_[0] = make_storage(history_primary_norm_[0], "restir_gi_history_primary_norm_A");
    history_primary_norm_idx_[1] = make_storage(history_primary_norm_[1], "restir_gi_history_primary_norm_B");

    rr::rhi::ImageDesc d{};
    d.format     = rr::rhi::Format::R32G32B32A32_Sfloat;
    d.extent     = {ext.width, ext.height, 1};
    d.usage      = rr::rhi::ImageUsage::Storage
                 | rr::rhi::ImageUsage::Sampled
                 | rr::rhi::ImageUsage::TransferSrc;
    d.debug_name = "restir_gi_output";
    output_img_.create(device, d);

    output_storage_idx = registry.register_storage_image(
        device, output_img_, rr::rhi::Format::R32G32B32A32_Sfloat);
    output_texture_idx = registry.register_texture(
        device, output_img_,
        rr::rhi::Format::R32G32B32A32_Sfloat,
        rr::rhi::ImageLayout::General,
        rr::rhi::ImageAspect::Color);
}

void ReSTIRGIPass::destroy_images(rr::rhi::Device& device)
{
    history_primary_norm_[0].destroy(device);
    history_primary_norm_[1].destroy(device);
    history_primary_pos_[0].destroy(device);
    history_primary_pos_[1].destroy(device);
    reservoir_meta_[0].destroy(device);
    reservoir_meta_[1].destroy(device);
    reservoir_rad_[0].destroy(device);
    reservoir_rad_[1].destroy(device);
    reservoir_pos_[0].destroy(device);
    reservoir_pos_[1].destroy(device);
    output_img_.destroy(device);
}

void ReSTIRGIPass::create_pipelines(rr::rhi::Device& device,
                                    rr::rhi::BindlessRegistry& registry)
{
    auto make = [&](rr::rhi::ComputePipeline& pipeline, uint32_t entry_idx, const char* name)
    {
        rr::rhi::ComputePipelineDesc desc{};
        if (entry_idx == 0)
        {
            desc.module      = &initial_shader_;
            desc.reflection  = &initial_reflection_;
        }
        else if (entry_idx == 1)
        {
            desc.module      = &temporal_shader_;
            desc.reflection  = &temporal_reflection_;
        }
        else
        {
            desc.module      = &spatial_shader_;
            desc.reflection  = &spatial_reflection_;
        }
        desc.entry_index = 0;
        desc.registry    = &registry;
        desc.debug_name  = name;
        pipeline.create(device, desc);
    };

    make(initial_pipeline_, 0, "restir_gi_initial_pipeline");
    make(temporal_pipeline_, 1, "restir_gi_temporal_pipeline");
    make(spatial_shade_pipeline_, 2, "restir_gi_spatial_shade_pipeline");
}

rr::render::RenderPass::Reflection ReSTIRGIPass::reflect() const
{
    Reflection r;
    r.outputs.push_back({"restir_gi_output", ResourceDesc::Kind::Texture,
                         rr::rhi::Format::R32G32B32A32_Sfloat, extent_});
    return r;
}

void ReSTIRGIPass::on_resize(rr::rhi::Extent2D new_extent)
{
    if (!initialized_) return;
    extent_ = new_extent;
    destroy_images(*device_);
    create_images(*device_, *registry_, new_extent);
    reset_history();
}

void ReSTIRGIPass::render_ui()
{
    ImGui::Text("ReSTIR GI  %ux%u", extent_.width, extent_.height);
    ImGui::Checkbox("Include direct lighting", &include_direct_lighting);
    ImGui::Checkbox("Temporal reuse", &temporal_enable);
    ImGui::Checkbox("Spatial reuse", &spatial_enable);
    ImGui::SliderFloat("Indirect strength", &indirect_strength, 0.0f, 4.0f, "%.2f");
    ImGui::SliderFloat("Ray bias", &ray_bias, 1.0e-5f, 1.0e-2f, "%.5f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderInt("Initial candidates (M)", reinterpret_cast<int*>(&num_initial_candidates), 1, 32);
    ImGui::SliderInt("Spatial neighbours", reinterpret_cast<int*>(&spatial_num_neighbors), 0, 16);
    ImGui::SliderInt("Spatial radius (px)", reinterpret_cast<int*>(&spatial_radius), 1, 64);
    ImGui::Text("  output gStorageImages[%u] / gTextures[%u]",
                output_storage_idx, output_texture_idx);
}

rr::rhi::ImageHandle ReSTIRGIPass::output_image_handle() const
{
    return rr::rhi::to_handle(output_img_.handle());
}

void ReSTIRGIPass::pre_transition_to_general(rr::rhi::CommandRecorder recorder)
{
    // Images that will be cleared (reservoir + history; NOT output_img_).
    rr::rhi::ImageHandle clearable[] = {
        rr::rhi::to_handle(reservoir_pos_[0].handle()),
        rr::rhi::to_handle(reservoir_pos_[1].handle()),
        rr::rhi::to_handle(reservoir_rad_[0].handle()),
        rr::rhi::to_handle(reservoir_rad_[1].handle()),
        rr::rhi::to_handle(reservoir_meta_[0].handle()),
        rr::rhi::to_handle(reservoir_meta_[1].handle()),
        rr::rhi::to_handle(history_primary_pos_[0].handle()),
        rr::rhi::to_handle(history_primary_pos_[1].handle()),
        rr::rhi::to_handle(history_primary_norm_[0].handle()),
        rr::rhi::to_handle(history_primary_norm_[1].handle()),
    };
    constexpr uint32_t kClearCount = 10;

    // Step 1: UNDEFINED→GENERAL for clearable images (dstStage = CLEAR).
    {
        image_barrier(recorder, clearable, kClearCount,
                      rr::rhi::PipelineStage::TopOfPipe,
                      rr::rhi::AccessFlags::None,
                      rr::rhi::PipelineStage::Transfer,
                      rr::rhi::AccessFlags::TransferWrite,
                      rr::rhi::ImageLayout::Undefined,
                      rr::rhi::ImageLayout::General);
    }

    const rr::rhi::ImageSubresourceRange range{
        .aspect = rr::rhi::ImageAspect::Color,
        .base_mip = 0,
        .mip_count = rr::rhi::kRemainingMipLevels,
        .base_layer = 0,
        .layer_count = rr::rhi::kRemainingArrayLayers,
    };

    // Step 2: Clear history_primary_pos with w=-1 to mark all pixels invalid.
    rr::rhi::ClearColor invalid_pos{};
    invalid_pos.float32[3] = -1.0f;
    recorder.clear_color_image(history_primary_pos_[0], rr::rhi::ImageLayout::General, invalid_pos, {&range, 1});
    recorder.clear_color_image(history_primary_pos_[1], rr::rhi::ImageLayout::General, invalid_pos, {&range, 1});

    // Step 3: Zero-clear reservoir buffers so M==0 is guaranteed before first frame.
    rr::rhi::ClearColor zero{};
    recorder.clear_color_image(reservoir_pos_[0], rr::rhi::ImageLayout::General, zero, {&range, 1});
    recorder.clear_color_image(reservoir_pos_[1], rr::rhi::ImageLayout::General, zero, {&range, 1});
    recorder.clear_color_image(reservoir_rad_[0], rr::rhi::ImageLayout::General, zero, {&range, 1});
    recorder.clear_color_image(reservoir_rad_[1], rr::rhi::ImageLayout::General, zero, {&range, 1});
    recorder.clear_color_image(reservoir_meta_[0], rr::rhi::ImageLayout::General, zero, {&range, 1});
    recorder.clear_color_image(reservoir_meta_[1], rr::rhi::ImageLayout::General, zero, {&range, 1});
    // Also zero history_primary_norm (no sentinel needed, M==0 gate is sufficient).
    recorder.clear_color_image(history_primary_norm_[0], rr::rhi::ImageLayout::General, zero, {&range, 1});
    recorder.clear_color_image(history_primary_norm_[1], rr::rhi::ImageLayout::General, zero, {&range, 1});

    // Step 4: UNDEFINED→GENERAL for output_img_ (no clear needed, direct write by shader).
    {
        const rr::rhi::ImageBarrier output_barrier{
            .image = &output_img_,
            .src_stage = rr::rhi::PipelineStage::TopOfPipe,
            .src_access = rr::rhi::AccessFlags::None,
            .dst_stage = rr::rhi::PipelineStage::ComputeShader,
            .dst_access = rr::rhi::AccessFlags::ShaderWrite,
            .old_layout = rr::rhi::ImageLayout::Undefined,
            .new_layout = rr::rhi::ImageLayout::General,
            .subresource = range,
        };
        recorder.pipeline_barrier({&output_barrier, 1});
    }

    // Step 5: CLEAR → COMPUTE barrier for the 10 cleared images.
    {
        image_barrier(recorder, clearable, kClearCount,
                      rr::rhi::PipelineStage::Transfer,
                      rr::rhi::AccessFlags::TransferWrite,
                      rr::rhi::PipelineStage::ComputeShader,
                      rr::rhi::AccessFlags::ShaderRead | rr::rhi::AccessFlags::ShaderWrite,
                      rr::rhi::ImageLayout::General,
                      rr::rhi::ImageLayout::General);
    }
}

void ReSTIRGIPass::execute(rr::render::FrameContext& fc)
{
    if (!initialized_) return;
    if (!initial_pipeline_.is_valid()
     || !temporal_pipeline_.is_valid()
     || !spatial_shade_pipeline_.is_valid()) return;
    if (gbuf_pos_idx_ == UINT32_MAX || gbuf_norm_idx_ == UINT32_MAX) return;

    const auto& scene = *fc.scene;
    if (!scene.is_uploaded()) return;

    const rr::rhi::CommandRecorder recorder = fc.command_recorder;
    const auto&     handles = scene.gpu_handles();
    const rr::rhi::ImageHandle direct_input_image = direct_input_image_;

    const uint32_t curr = reservoir_flip_;
    const uint32_t prev = 1u - reservoir_flip_;

    const uint32_t gx = (extent_.width + 7) / 8;
    const uint32_t gy = (extent_.height + 7) / 8;

    ReSTIRGIPushConstants pc{};
    pc.camera_buf_idx          = handles.camera_buf_idx;
    pc.tlas_idx                = handles.tlas_idx;
    pc.vertex_buf_idx          = handles.vertex_buf_idx;
    pc.index_buf_idx           = handles.index_buf_idx;
    pc.mesh_buf_idx            = handles.mesh_buf_idx;
    pc.material_buf_idx        = handles.material_buf_idx;
    pc.instance_buf_idx        = handles.instance_buf_idx;
    pc.light_buf_idx           = handles.light_buf_idx;
    pc.num_instances           = scene.instance_count();
    pc.num_lights              = scene.light_count();
    pc.frame_seed              = frame_count_++;
    pc.gbuf_pos_idx            = gbuf_pos_idx_;
    pc.gbuf_norm_idx           = gbuf_norm_idx_;
    pc.direct_input_idx        = direct_input_texture_idx_;
    pc.reservoir_curr_pos_idx  = reservoir_pos_idx_[curr];
    pc.reservoir_curr_rad_idx  = reservoir_rad_idx_[curr];
    pc.reservoir_curr_meta_idx = reservoir_meta_idx_[curr];
    pc.reservoir_prev_pos_idx  = reservoir_pos_idx_[prev];
    pc.reservoir_prev_rad_idx  = reservoir_rad_idx_[prev];
    pc.reservoir_prev_meta_idx = reservoir_meta_idx_[prev];
    pc.history_curr_pos_idx    = history_primary_pos_idx_[curr];
    pc.history_curr_norm_idx   = history_primary_norm_idx_[curr];
    pc.history_prev_pos_idx    = history_primary_pos_idx_[prev];
    pc.history_prev_norm_idx   = history_primary_norm_idx_[prev];
    pc.output_idx              = output_storage_idx;
    pc.num_initial_candidates  = num_initial_candidates;
    pc.temporal_enable         = temporal_enable ? 1u : 0u;
    pc.spatial_enable          = spatial_enable ? 1u : 0u;
    pc.spatial_num_neighbors   = spatial_num_neighbors;
    pc.spatial_radius          = spatial_radius;
    pc.include_direct_lighting = include_direct_lighting ? 1u : 0u;
    pc.indirect_strength       = indirect_strength;
    pc.ray_bias                = ray_bias;

    // If history was just reset, clear the prev-side buffers so cs_temporal sees
    // an explicitly invalid history (M=0, history_pos.w=-1) instead of stale data
    // from before the reset.
    if (consume_history_dirty())
    {
        const rr::rhi::ImageSubresourceRange range{
            .aspect = rr::rhi::ImageAspect::Color,
            .base_mip = 0,
            .mip_count = rr::rhi::kRemainingMipLevels,
            .base_layer = 0,
            .layer_count = rr::rhi::kRemainingArrayLayers,
        };
        rr::rhi::ImageHandle clear_targets[] = {
            rr::rhi::to_handle(reservoir_pos_[prev].handle()),
            rr::rhi::to_handle(reservoir_rad_[prev].handle()),
            rr::rhi::to_handle(reservoir_meta_[prev].handle()),
            rr::rhi::to_handle(history_primary_pos_[prev].handle()),
            rr::rhi::to_handle(history_primary_norm_[prev].handle()),
        };

        // Pre-clear barrier: ensure prior shader writes complete before transfer clears.
        {
            image_barrier(recorder, clear_targets, 5,
                          rr::rhi::PipelineStage::ComputeShader,
                          rr::rhi::AccessFlags::ShaderRead | rr::rhi::AccessFlags::ShaderWrite,
                          rr::rhi::PipelineStage::Transfer,
                          rr::rhi::AccessFlags::TransferWrite,
                          rr::rhi::ImageLayout::General,
                          rr::rhi::ImageLayout::General);
        }

        rr::rhi::ClearColor zero{};
        rr::rhi::ClearColor invalid_pos{};
        invalid_pos.float32[3] = -1.0f;
        recorder.clear_color_image(reservoir_pos_[prev], rr::rhi::ImageLayout::General, zero, {&range, 1});
        recorder.clear_color_image(reservoir_rad_[prev], rr::rhi::ImageLayout::General, zero, {&range, 1});
        recorder.clear_color_image(reservoir_meta_[prev], rr::rhi::ImageLayout::General, zero, {&range, 1});
        recorder.clear_color_image(history_primary_pos_[prev], rr::rhi::ImageLayout::General, invalid_pos, {&range, 1});
        recorder.clear_color_image(history_primary_norm_[prev], rr::rhi::ImageLayout::General, zero, {&range, 1});

        // Post-clear barrier: clear writes must be visible before subsequent shader access.
        {
            image_barrier(recorder, clear_targets, 5,
                          rr::rhi::PipelineStage::Transfer,
                          rr::rhi::AccessFlags::TransferWrite,
                          rr::rhi::PipelineStage::ComputeShader,
                          rr::rhi::AccessFlags::ShaderRead | rr::rhi::AccessFlags::ShaderWrite,
                          rr::rhi::ImageLayout::General,
                          rr::rhi::ImageLayout::General);
        }
    }

    {
        rr::rhi::ImageHandle curr_imgs[] = {
            rr::rhi::to_handle(reservoir_pos_[curr].handle()),
            rr::rhi::to_handle(reservoir_rad_[curr].handle()),
            rr::rhi::to_handle(reservoir_meta_[curr].handle()),
            rr::rhi::to_handle(history_primary_pos_[curr].handle()),
            rr::rhi::to_handle(history_primary_norm_[curr].handle()),
        };
        image_barrier(recorder,
                              curr_imgs,
                              5,
                              rr::rhi::PipelineStage::ComputeShader,
                              rr::rhi::AccessFlags::ShaderRead | rr::rhi::AccessFlags::ShaderWrite,
                              rr::rhi::PipelineStage::ComputeShader,
                              rr::rhi::AccessFlags::ShaderWrite,
                              rr::rhi::ImageLayout::General,
                              rr::rhi::ImageLayout::General);
    }

    recorder.bind_compute_pipeline(initial_pipeline_);
    recorder.push_constants(&pc, sizeof(pc));
    recorder.dispatch(gx, gy, 1);

    {
        rr::rhi::ImageHandle curr_imgs[] = {
            rr::rhi::to_handle(reservoir_pos_[curr].handle()),
            rr::rhi::to_handle(reservoir_rad_[curr].handle()),
            rr::rhi::to_handle(reservoir_meta_[curr].handle()),
            rr::rhi::to_handle(history_primary_pos_[curr].handle()),
            rr::rhi::to_handle(history_primary_norm_[curr].handle()),
        };
        image_barrier(recorder,
                              curr_imgs,
                              5,
                              rr::rhi::PipelineStage::ComputeShader,
                              rr::rhi::AccessFlags::ShaderWrite,
                              rr::rhi::PipelineStage::ComputeShader,
                              rr::rhi::AccessFlags::ShaderRead | rr::rhi::AccessFlags::ShaderWrite,
                              rr::rhi::ImageLayout::General,
                              rr::rhi::ImageLayout::General);
    }

    {
        rr::rhi::ImageHandle prev_imgs[] = {
            rr::rhi::to_handle(reservoir_pos_[prev].handle()),
            rr::rhi::to_handle(reservoir_rad_[prev].handle()),
            rr::rhi::to_handle(reservoir_meta_[prev].handle()),
            rr::rhi::to_handle(history_primary_pos_[prev].handle()),
            rr::rhi::to_handle(history_primary_norm_[prev].handle()),
        };
        image_barrier(recorder,
                              prev_imgs,
                              5,
                              rr::rhi::PipelineStage::ComputeShader,
                              rr::rhi::AccessFlags::ShaderWrite,
                              rr::rhi::PipelineStage::ComputeShader,
                              rr::rhi::AccessFlags::ShaderRead,
                              rr::rhi::ImageLayout::General,
                              rr::rhi::ImageLayout::General);
    }

    recorder.bind_compute_pipeline(temporal_pipeline_);
    recorder.push_constants(&pc, sizeof(pc));
    recorder.dispatch(gx, gy, 1);

    {
        rr::rhi::ImageHandle curr_imgs[] = {
            rr::rhi::to_handle(reservoir_pos_[curr].handle()),
            rr::rhi::to_handle(reservoir_rad_[curr].handle()),
            rr::rhi::to_handle(reservoir_meta_[curr].handle()),
        };
        image_barrier(recorder,
                              curr_imgs,
                              3,
                              rr::rhi::PipelineStage::ComputeShader,
                              rr::rhi::AccessFlags::ShaderWrite,
                              rr::rhi::PipelineStage::ComputeShader,
                              rr::rhi::AccessFlags::ShaderRead,
                              rr::rhi::ImageLayout::General,
                              rr::rhi::ImageLayout::General);
    }

    if (include_direct_lighting && direct_input_image_ != 0)
    {
        rr::rhi::ImageHandle direct_imgs[] = {direct_input_image};
        image_barrier(recorder,
                              direct_imgs,
                              1,
                              rr::rhi::PipelineStage::ComputeShader,
                              rr::rhi::AccessFlags::ShaderWrite,
                              rr::rhi::PipelineStage::ComputeShader,
                              rr::rhi::AccessFlags::ShaderRead,
                              rr::rhi::ImageLayout::General,
                              rr::rhi::ImageLayout::General);
    }

    {
        rr::rhi::ImageHandle output_imgs[] = {rr::rhi::to_handle(output_img_.handle())};
        image_barrier(recorder,
                              output_imgs,
                              1,
                              rr::rhi::PipelineStage::ComputeShader,
                              rr::rhi::AccessFlags::ShaderRead | rr::rhi::AccessFlags::ShaderWrite,
                              rr::rhi::PipelineStage::ComputeShader,
                              rr::rhi::AccessFlags::ShaderWrite,
                              rr::rhi::ImageLayout::General,
                              rr::rhi::ImageLayout::General);
    }

    recorder.bind_compute_pipeline(spatial_shade_pipeline_);
    recorder.push_constants(&pc, sizeof(pc));
    recorder.dispatch(gx, gy, 1);

    reservoir_flip_ ^= 1u;
}

} // namespace rr::passes::restir_gi