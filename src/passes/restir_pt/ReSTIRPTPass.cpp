#include "passes/restir_pt/ReSTIRPTPass.h"

#include "core/Log.h"
#include "render/FrameContext.h"
#include "rhi/BindlessRegistry.h"
#include "rhi/Device.h"
#include "scene/GpuScene.h"
#include "scene/Scene.h"

#include <cstring>
#include <imgui.h>
#include <stdexcept>
#include <vector>

namespace rr::passes::restir_pt
{

namespace
{

void compile_stage(rr::shader::ShaderModule& shader,
                   rr::shader::ShaderReflection& reflection,
                   rr::shader::SlangSession& session,
                   const char* entry_name)
{
    shader.compile(session,
        "assets/shaders/passes/restir_pt/restir_pt.slang",
        {
            {entry_name, rr::shader::ShaderStage::Compute},
        });
    reflection = rr::shader::ShaderReflection(shader.program_layout());
}

struct ReSTIRPTPushConstants
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
    uint32_t reservoir_curr_nrm_idx;
    uint32_t reservoir_curr_rad_idx;
    uint32_t reservoir_curr_pre_idx;
    uint32_t reservoir_prev_pos_idx;
    uint32_t reservoir_prev_nrm_idx;
    uint32_t reservoir_prev_rad_idx;
    uint32_t reservoir_prev_pre_idx;
    uint32_t history_curr_pos_idx;
    uint32_t history_curr_norm_idx;
    uint32_t history_prev_pos_idx;
    uint32_t history_prev_norm_idx;
    uint32_t output_idx;
    uint32_t num_initial_candidates;
    uint32_t max_bounces;
    uint32_t temporal_enable;
    uint32_t spatial_enable;
    uint32_t spatial_num_neighbors;
    uint32_t spatial_radius;
    uint32_t include_direct_lighting;
    float    indirect_strength;
    float    ray_bias;
    uint32_t guiding_accum_idx;
    uint32_t guiding_fitted_idx;
    uint32_t guiding_enable;
    float    guiding_alpha;
    float    grid_base_cell_size;
    float    grid_distance_slope;
    uint32_t grid_cell_count;
    uint32_t grid_min_samples;
    float    grid_kappa_max;
    uint32_t _pad0;
};
static_assert(sizeof(ReSTIRPTPushConstants) == 184);

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

ReSTIRPTPass::~ReSTIRPTPass() = default;

void ReSTIRPTPass::initialize(rr::rhi::Device& device,
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
    compile_stage(fit_shader_, fit_reflection_, session, "cs_fit_guiding");

    create_images(device, registry, extent);
    create_guiding_buffers(device, registry);
    create_pipelines(device, registry);
    initialized_ = true;
    core::log()->info("[ReSTIRPTPass] Initialized {}x{}", extent.width, extent.height);
}

void ReSTIRPTPass::shutdown(rr::rhi::Device& device)
{
    if (!initialized_) return;
    initial_pipeline_.destroy(device);
    temporal_pipeline_.destroy(device);
    spatial_shade_pipeline_.destroy(device);
    fit_pipeline_.destroy(device);
    initial_shader_.reset();
    temporal_shader_.reset();
    spatial_shader_.reset();
    fit_shader_.reset();
    destroy_guiding_buffers(device);
    destroy_images(device);
    initialized_ = false;
}

bool ReSTIRPTPass::reload_shader(rr::shader::SlangSession& session)
{
    if (!initialized_) return false;

    rr::shader::ShaderModule new_initial_shader;
    rr::shader::ShaderModule new_temporal_shader;
    rr::shader::ShaderModule new_spatial_shader;
    rr::shader::ShaderModule new_fit_shader;
    rr::shader::ShaderReflection new_initial_reflection;
    rr::shader::ShaderReflection new_temporal_reflection;
    rr::shader::ShaderReflection new_spatial_reflection;
    rr::shader::ShaderReflection new_fit_reflection;
    try
    {
        compile_stage(new_initial_shader, new_initial_reflection, session, "cs_initial");
        compile_stage(new_temporal_shader, new_temporal_reflection, session, "cs_temporal");
        compile_stage(new_spatial_shader, new_spatial_reflection, session, "cs_spatial_shade");
        compile_stage(new_fit_shader, new_fit_reflection, session, "cs_fit_guiding");
    }
    catch (const std::exception& e)
    {
        core::log()->error("[ReSTIRPTPass] Shader recompile failed: {}", e.what());
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
    rr::rhi::ComputePipeline new_fit;

    try
    {
        auto init_desc = make_desc(new_initial_shader, new_initial_reflection, "restir_pt_initial_pipeline");
        init_desc.registry = registry_;
        auto temporal_desc = make_desc(new_temporal_shader, new_temporal_reflection, "restir_pt_temporal_pipeline");
        temporal_desc.registry = registry_;
        auto spatial_desc = make_desc(new_spatial_shader, new_spatial_reflection, "restir_pt_spatial_shade_pipeline");
        spatial_desc.registry = registry_;
        auto fit_desc = make_desc(new_fit_shader, new_fit_reflection, "restir_pt_fit_guiding_pipeline");
        fit_desc.registry = registry_;
        new_initial.create(*device_, init_desc);
        new_temporal.create(*device_, temporal_desc);
        new_spatial.create(*device_, spatial_desc);
        new_fit.create(*device_, fit_desc);
    }
    catch (const std::exception& e)
    {
        core::log()->error("[ReSTIRPTPass] Pipeline recreate failed: {}", e.what());
        new_initial.destroy(*device_);
        new_temporal.destroy(*device_);
        new_spatial.destroy(*device_);
        new_fit.destroy(*device_);
        return false;
    }

    initial_pipeline_.destroy(*device_);
    temporal_pipeline_.destroy(*device_);
    spatial_shade_pipeline_.destroy(*device_);
    fit_pipeline_.destroy(*device_);

    initial_pipeline_.swap(new_initial);
    temporal_pipeline_.swap(new_temporal);
    spatial_shade_pipeline_.swap(new_spatial);
    fit_pipeline_.swap(new_fit);

    initial_shader_.swap(new_initial_shader);
    temporal_shader_.swap(new_temporal_shader);
    spatial_shader_.swap(new_spatial_shader);
    fit_shader_.swap(new_fit_shader);
    initial_reflection_ = new_initial_reflection;
    temporal_reflection_ = new_temporal_reflection;
    spatial_reflection_ = new_spatial_reflection;
    fit_reflection_ = new_fit_reflection;
    return true;
}

void ReSTIRPTPass::create_images(rr::rhi::Device& device,
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

    reservoir_pos_idx_[0] = make_storage(reservoir_pos_[0], "restir_pt_reservoir_pos_A");
    reservoir_pos_idx_[1] = make_storage(reservoir_pos_[1], "restir_pt_reservoir_pos_B");
    reservoir_nrm_idx_[0] = make_storage(reservoir_nrm_[0], "restir_pt_reservoir_nrm_A");
    reservoir_nrm_idx_[1] = make_storage(reservoir_nrm_[1], "restir_pt_reservoir_nrm_B");
    reservoir_rad_idx_[0] = make_storage(reservoir_rad_[0], "restir_pt_reservoir_rad_A");
    reservoir_rad_idx_[1] = make_storage(reservoir_rad_[1], "restir_pt_reservoir_rad_B");
    reservoir_pre_idx_[0] = make_storage(reservoir_pre_[0], "restir_pt_reservoir_pre_A");
    reservoir_pre_idx_[1] = make_storage(reservoir_pre_[1], "restir_pt_reservoir_pre_B");
    history_primary_pos_idx_[0]  = make_storage(history_primary_pos_[0], "restir_pt_history_primary_pos_A");
    history_primary_pos_idx_[1]  = make_storage(history_primary_pos_[1], "restir_pt_history_primary_pos_B");
    history_primary_norm_idx_[0] = make_storage(history_primary_norm_[0], "restir_pt_history_primary_norm_A");
    history_primary_norm_idx_[1] = make_storage(history_primary_norm_[1], "restir_pt_history_primary_norm_B");

    rr::rhi::ImageDesc d{};
    d.format     = rr::rhi::Format::R32G32B32A32_Sfloat;
    d.extent     = {ext.width, ext.height, 1};
    d.usage      = rr::rhi::ImageUsage::Storage
                 | rr::rhi::ImageUsage::Sampled
                 | rr::rhi::ImageUsage::TransferSrc;
    d.debug_name = "restir_pt_output";
    output_img_.create(device, d);

    output_storage_idx = registry.register_storage_image(
        device, output_img_, rr::rhi::Format::R32G32B32A32_Sfloat);
    output_texture_idx = registry.register_texture(
        device, output_img_,
        rr::rhi::Format::R32G32B32A32_Sfloat,
        rr::rhi::ImageLayout::General,
        rr::rhi::ImageAspect::Color);
}

void ReSTIRPTPass::destroy_images(rr::rhi::Device& device)
{
    history_primary_norm_[0].destroy(device);
    history_primary_norm_[1].destroy(device);
    history_primary_pos_[0].destroy(device);
    history_primary_pos_[1].destroy(device);
    reservoir_pre_[0].destroy(device);
    reservoir_pre_[1].destroy(device);
    reservoir_rad_[0].destroy(device);
    reservoir_rad_[1].destroy(device);
    reservoir_nrm_[0].destroy(device);
    reservoir_nrm_[1].destroy(device);
    reservoir_pos_[0].destroy(device);
    reservoir_pos_[1].destroy(device);
    output_img_.destroy(device);
}

void ReSTIRPTPass::create_guiding_buffers(rr::rhi::Device& device,
                                          rr::rhi::BindlessRegistry& registry)
{
    // accum  : 4 x uint32 per cell (acc_dir xyz fixed-point + count)
    // fitted : 4 x float  per cell (mu xyz + kappa)
    constexpr uint64_t kAccumStride  = 16;
    constexpr uint64_t kFittedStride = 16;
    const uint64_t accum_bytes  = uint64_t(kGridCellCount) * kAccumStride;
    const uint64_t fitted_bytes = uint64_t(kGridCellCount) * kFittedStride;

    auto make_buf = [&](rr::rhi::Buffer& buf, uint64_t bytes, const char* name)
    {
        rr::rhi::BufferDesc desc{};
        desc.size         = bytes;
        desc.usage        = rr::rhi::BufferUsage::Storage
                          | rr::rhi::BufferUsage::ShaderDeviceAddress;
        desc.memory_usage = rr::rhi::MemoryUsage::CpuToGpu;
        desc.alloc_flags  = rr::rhi::AllocFlags::HostAccessSequentialWrite
                          | rr::rhi::AllocFlags::Mapped;
        desc.debug_name   = name;
        buf.create(device, desc);
        if (buf.mapped() != nullptr)
            std::memset(buf.mapped(), 0, bytes);   // zero accumulators / lobes
    };

    make_buf(guiding_accum_,  accum_bytes,  "restir_pt_guiding_accum");
    make_buf(guiding_fitted_, fitted_bytes, "restir_pt_guiding_fitted");

    guiding_accum_idx_  = registry.register_buffer(device, guiding_accum_);
    guiding_fitted_idx_ = registry.register_buffer(device, guiding_fitted_);
}

void ReSTIRPTPass::destroy_guiding_buffers(rr::rhi::Device& device)
{
    guiding_accum_.destroy(device);
    guiding_fitted_.destroy(device);
    guiding_accum_idx_  = UINT32_MAX;
    guiding_fitted_idx_ = UINT32_MAX;
}

void ReSTIRPTPass::create_pipelines(rr::rhi::Device& device,
                                    rr::rhi::BindlessRegistry& registry)
{
    auto make = [&](rr::rhi::ComputePipeline& pipeline, uint32_t entry_idx, const char* name)
    {
        rr::rhi::ComputePipelineDesc desc{};
        if (entry_idx == 0)
        {
            desc.module     = &initial_shader_;
            desc.reflection = &initial_reflection_;
        }
        else if (entry_idx == 1)
        {
            desc.module     = &temporal_shader_;
            desc.reflection = &temporal_reflection_;
        }
        else if (entry_idx == 2)
        {
            desc.module     = &spatial_shader_;
            desc.reflection = &spatial_reflection_;
        }
        else
        {
            desc.module     = &fit_shader_;
            desc.reflection = &fit_reflection_;
        }
        desc.entry_index = 0;
        desc.registry    = &registry;
        desc.debug_name  = name;
        pipeline.create(device, desc);
    };

    make(initial_pipeline_, 0, "restir_pt_initial_pipeline");
    make(temporal_pipeline_, 1, "restir_pt_temporal_pipeline");
    make(spatial_shade_pipeline_, 2, "restir_pt_spatial_shade_pipeline");
    make(fit_pipeline_, 3, "restir_pt_fit_guiding_pipeline");
}

rr::render::RenderPass::Reflection ReSTIRPTPass::reflect() const
{
    Reflection r;
    r.outputs.push_back({"restir_pt_output", ResourceDesc::Kind::Texture,
                         rr::rhi::Format::R32G32B32A32_Sfloat, extent_});
    return r;
}

void ReSTIRPTPass::on_resize(rr::rhi::Extent2D new_extent)
{
    if (!initialized_) return;
    extent_ = new_extent;
    destroy_images(*device_);
    create_images(*device_, *registry_, new_extent);
    reset_history();
}

void ReSTIRPTPass::render_ui()
{
    ImGui::Text("ReSTIR PT  %ux%u", extent_.width, extent_.height);
    ImGui::Checkbox("Include direct lighting", &include_direct_lighting);
    ImGui::Checkbox("Temporal reuse", &temporal_enable);
    ImGui::Checkbox("Spatial reuse", &spatial_enable);
    ImGui::SliderFloat("Indirect strength", &indirect_strength, 0.0f, 4.0f, "%.2f");
    ImGui::SliderFloat("Ray bias", &ray_bias, 1.0e-5f, 1.0e-2f, "%.5f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderInt("Initial candidates (M)", reinterpret_cast<int*>(&num_initial_candidates), 1, 32);
    ImGui::SliderInt("Max bounces", reinterpret_cast<int*>(&max_bounces), 1, 12);
    ImGui::SliderInt("Spatial neighbours", reinterpret_cast<int*>(&spatial_num_neighbors), 0, 16);
    ImGui::SliderInt("Spatial radius (px)", reinterpret_cast<int*>(&spatial_radius), 1, 64);

    ImGui::Separator();
    ImGui::Text("Path Guiding (ReSTIR PG)");
    bool prev_guiding = guiding_enable;
    ImGui::Checkbox("Enable guiding", &guiding_enable);
    if (guiding_enable != prev_guiding)
        grid_dirty_ = true;   // clear stale lobes when toggling
    ImGui::SliderFloat("Guiding alpha (MIS)", &guiding_alpha, 0.0f, 1.0f, "%.2f");
    if (ImGui::SliderFloat("Grid base cell size", &grid_base_cell_size,
                           0.01f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic))
        grid_dirty_ = true;
    if (ImGui::SliderFloat("Grid distance slope", &grid_distance_slope,
                           0.0f, 0.2f, "%.4f"))
        grid_dirty_ = true;
    ImGui::SliderInt("Grid min samples", reinterpret_cast<int*>(&grid_min_samples), 1, 64);
    ImGui::SliderFloat("vMF kappa max", &grid_kappa_max, 16.0f, 2048.0f, "%.0f",
                       ImGuiSliderFlags_Logarithmic);
    ImGui::Text("  grid cells: %u  accum gScene[%u] fitted gScene[%u]",
                kGridCellCount, guiding_accum_idx_, guiding_fitted_idx_);
    if (ImGui::Button("Clear guiding grid"))
        grid_dirty_ = true;

    ImGui::Text("  output gStorageImages[%u] / gTextures[%u]",
                output_storage_idx, output_texture_idx);
}

rr::rhi::ImageHandle ReSTIRPTPass::output_image_handle() const
{
    return rr::rhi::to_handle(output_img_.handle());
}

void ReSTIRPTPass::pre_transition_to_general(rr::rhi::CommandRecorder recorder)
{
    rr::rhi::ImageHandle clearable[] = {
        rr::rhi::to_handle(reservoir_pos_[0].handle()),
        rr::rhi::to_handle(reservoir_pos_[1].handle()),
        rr::rhi::to_handle(reservoir_nrm_[0].handle()),
        rr::rhi::to_handle(reservoir_nrm_[1].handle()),
        rr::rhi::to_handle(reservoir_rad_[0].handle()),
        rr::rhi::to_handle(reservoir_rad_[1].handle()),
        rr::rhi::to_handle(reservoir_pre_[0].handle()),
        rr::rhi::to_handle(reservoir_pre_[1].handle()),
        rr::rhi::to_handle(history_primary_pos_[0].handle()),
        rr::rhi::to_handle(history_primary_pos_[1].handle()),
        rr::rhi::to_handle(history_primary_norm_[0].handle()),
        rr::rhi::to_handle(history_primary_norm_[1].handle()),
    };
    constexpr uint32_t kClearCount = 12;

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

    // history_primary_pos uses w=-1 as the "invalid surface" sentinel.
    rr::rhi::ClearColor invalid_pos{};
    invalid_pos.float32[3] = -1.0f;
    recorder.clear_color_image(history_primary_pos_[0], rr::rhi::ImageLayout::General, invalid_pos, {&range, 1});
    recorder.clear_color_image(history_primary_pos_[1], rr::rhi::ImageLayout::General, invalid_pos, {&range, 1});

    // Zero-clear reservoir buffers so M==0 / valid==0 before the first frame.
    rr::rhi::ClearColor zero{};
    recorder.clear_color_image(reservoir_pos_[0], rr::rhi::ImageLayout::General, zero, {&range, 1});
    recorder.clear_color_image(reservoir_pos_[1], rr::rhi::ImageLayout::General, zero, {&range, 1});
    recorder.clear_color_image(reservoir_nrm_[0], rr::rhi::ImageLayout::General, zero, {&range, 1});
    recorder.clear_color_image(reservoir_nrm_[1], rr::rhi::ImageLayout::General, zero, {&range, 1});
    recorder.clear_color_image(reservoir_rad_[0], rr::rhi::ImageLayout::General, zero, {&range, 1});
    recorder.clear_color_image(reservoir_rad_[1], rr::rhi::ImageLayout::General, zero, {&range, 1});
    recorder.clear_color_image(reservoir_pre_[0], rr::rhi::ImageLayout::General, zero, {&range, 1});
    recorder.clear_color_image(reservoir_pre_[1], rr::rhi::ImageLayout::General, zero, {&range, 1});
    recorder.clear_color_image(history_primary_norm_[0], rr::rhi::ImageLayout::General, zero, {&range, 1});
    recorder.clear_color_image(history_primary_norm_[1], rr::rhi::ImageLayout::General, zero, {&range, 1});

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

void ReSTIRPTPass::execute(rr::render::FrameContext& fc)
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

    ReSTIRPTPushConstants pc{};
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
    pc.reservoir_curr_nrm_idx  = reservoir_nrm_idx_[curr];
    pc.reservoir_curr_rad_idx  = reservoir_rad_idx_[curr];
    pc.reservoir_curr_pre_idx  = reservoir_pre_idx_[curr];
    pc.reservoir_prev_pos_idx  = reservoir_pos_idx_[prev];
    pc.reservoir_prev_nrm_idx  = reservoir_nrm_idx_[prev];
    pc.reservoir_prev_rad_idx  = reservoir_rad_idx_[prev];
    pc.reservoir_prev_pre_idx  = reservoir_pre_idx_[prev];
    pc.history_curr_pos_idx    = history_primary_pos_idx_[curr];
    pc.history_curr_norm_idx   = history_primary_norm_idx_[curr];
    pc.history_prev_pos_idx    = history_primary_pos_idx_[prev];
    pc.history_prev_norm_idx   = history_primary_norm_idx_[prev];
    pc.output_idx              = output_storage_idx;
    pc.num_initial_candidates  = num_initial_candidates;
    pc.max_bounces             = max_bounces;
    pc.temporal_enable         = temporal_enable ? 1u : 0u;
    pc.spatial_enable          = spatial_enable ? 1u : 0u;
    pc.spatial_num_neighbors   = spatial_num_neighbors;
    pc.spatial_radius          = spatial_radius;
    pc.include_direct_lighting = include_direct_lighting ? 1u : 0u;
    pc.indirect_strength       = indirect_strength;
    pc.ray_bias                = ray_bias;
    pc.guiding_accum_idx       = guiding_accum_idx_;
    pc.guiding_fitted_idx      = guiding_fitted_idx_;
    pc.guiding_enable          = guiding_enable ? 1u : 0u;
    pc.guiding_alpha           = guiding_alpha;
    pc.grid_base_cell_size     = grid_base_cell_size;
    pc.grid_distance_slope     = grid_distance_slope;
    pc.grid_cell_count         = kGridCellCount;
    pc.grid_min_samples        = grid_min_samples;
    pc.grid_kappa_max          = grid_kappa_max;

    // Grid reset: zero the accumulator and invalidate every fitted lobe so
    // the next frame's cs_initial falls back to pure cosine sampling until
    // the grid re-converges. reset_history() is invoked from camera-move /
    // scene-reload / mode-switch sites where the renderer also resets
    // accumulation, so a host-side clear of these mapped buffers is safe.
    if (consume_grid_dirty())
    {
        constexpr uint64_t kAccumStride  = 16;
        constexpr uint64_t kFittedStride = 16;
        if (guiding_accum_.mapped() != nullptr)
            std::memset(guiding_accum_.mapped(), 0,
                        uint64_t(kGridCellCount) * kAccumStride);
        if (guiding_fitted_.mapped() != nullptr)
            std::memset(guiding_fitted_.mapped(), 0,
                        uint64_t(kGridCellCount) * kFittedStride);
    }

    // If history was just reset, clear the prev-side buffers so cs_temporal
    // sees an explicitly invalid history instead of stale data.
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
            rr::rhi::to_handle(reservoir_nrm_[prev].handle()),
            rr::rhi::to_handle(reservoir_rad_[prev].handle()),
            rr::rhi::to_handle(reservoir_pre_[prev].handle()),
            rr::rhi::to_handle(history_primary_pos_[prev].handle()),
            rr::rhi::to_handle(history_primary_norm_[prev].handle()),
        };

        {
            image_barrier(recorder, clear_targets, 6,
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
        recorder.clear_color_image(reservoir_nrm_[prev], rr::rhi::ImageLayout::General, zero, {&range, 1});
        recorder.clear_color_image(reservoir_rad_[prev], rr::rhi::ImageLayout::General, zero, {&range, 1});
        recorder.clear_color_image(reservoir_pre_[prev], rr::rhi::ImageLayout::General, zero, {&range, 1});
        recorder.clear_color_image(history_primary_pos_[prev], rr::rhi::ImageLayout::General, invalid_pos, {&range, 1});
        recorder.clear_color_image(history_primary_norm_[prev], rr::rhi::ImageLayout::General, zero, {&range, 1});

        {
            image_barrier(recorder, clear_targets, 6,
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
            rr::rhi::to_handle(reservoir_nrm_[curr].handle()),
            rr::rhi::to_handle(reservoir_rad_[curr].handle()),
            rr::rhi::to_handle(reservoir_pre_[curr].handle()),
            rr::rhi::to_handle(history_primary_pos_[curr].handle()),
            rr::rhi::to_handle(history_primary_norm_[curr].handle()),
        };
        image_barrier(recorder, curr_imgs, 6,
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
            rr::rhi::to_handle(reservoir_nrm_[curr].handle()),
            rr::rhi::to_handle(reservoir_rad_[curr].handle()),
            rr::rhi::to_handle(reservoir_pre_[curr].handle()),
            rr::rhi::to_handle(history_primary_pos_[curr].handle()),
            rr::rhi::to_handle(history_primary_norm_[curr].handle()),
        };
        image_barrier(recorder, curr_imgs, 6,
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
            rr::rhi::to_handle(reservoir_nrm_[prev].handle()),
            rr::rhi::to_handle(reservoir_rad_[prev].handle()),
            rr::rhi::to_handle(reservoir_pre_[prev].handle()),
            rr::rhi::to_handle(history_primary_pos_[prev].handle()),
            rr::rhi::to_handle(history_primary_norm_[prev].handle()),
        };
        image_barrier(recorder, prev_imgs, 6,
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
            rr::rhi::to_handle(reservoir_nrm_[curr].handle()),
            rr::rhi::to_handle(reservoir_rad_[curr].handle()),
            rr::rhi::to_handle(reservoir_pre_[curr].handle()),
        };
        image_barrier(recorder, curr_imgs, 4,
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
        image_barrier(recorder, direct_imgs, 1,
                      rr::rhi::PipelineStage::ComputeShader,
                      rr::rhi::AccessFlags::ShaderWrite,
                      rr::rhi::PipelineStage::ComputeShader,
                      rr::rhi::AccessFlags::ShaderRead,
                      rr::rhi::ImageLayout::General,
                      rr::rhi::ImageLayout::General);
    }

    {
        rr::rhi::ImageHandle output_imgs[] = {rr::rhi::to_handle(output_img_.handle())};
        image_barrier(recorder, output_imgs, 1,
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

    // ── Fit guiding lobes ──────────────────────────────────────
    // cs_spatial_shade splatted ReSTIR-selected bounce directions into the
    // accumulator buffer (via gSceneRW atomics). cs_fit_guiding then turns
    // each cell into a single vMF lobe for next frame's cs_initial.
    //
    // Ordering: the splat (ComputeShader writes) must complete before the fit
    // (ComputeShader reads/writes the same buffer). The RHI currently exposes
    // only image-based pipeline barriers, so we issue a ComputeShader→
    // ComputeShader barrier on the output image as the execution dependency
    // between the two compute dispatches — the same convention the existing
    // ReSTIR passes use for inter-dispatch ordering. (A dedicated buffer
    // memory barrier would be a future RHI addition; in practice the broad
    // execution dependency plus the device-coherent mapped allocation make
    // the accumulator writes visible to the fit pass on this single queue.)
    if (guiding_enable && fit_pipeline_.is_valid())
    {
        const rr::rhi::ImageSubresourceRange grange{
            .aspect = rr::rhi::ImageAspect::Color,
            .base_mip = 0,
            .mip_count = rr::rhi::kRemainingMipLevels,
            .base_layer = 0,
            .layer_count = rr::rhi::kRemainingArrayLayers,
        };
        const rr::rhi::ImageBarrier splat_to_fit{
            .image = &output_img_,
            .src_stage = rr::rhi::PipelineStage::ComputeShader,
            .src_access = rr::rhi::AccessFlags::ShaderWrite,
            .dst_stage = rr::rhi::PipelineStage::ComputeShader,
            .dst_access = rr::rhi::AccessFlags::ShaderRead
                        | rr::rhi::AccessFlags::ShaderWrite,
            .old_layout = rr::rhi::ImageLayout::General,
            .new_layout = rr::rhi::ImageLayout::General,
            .subresource = grange,
        };
        recorder.pipeline_barrier({&splat_to_fit, 1});

        const uint32_t fit_groups = (kGridCellCount + 63u) / 64u;
        recorder.bind_compute_pipeline(fit_pipeline_);
        recorder.push_constants(&pc, sizeof(pc));
        recorder.dispatch(fit_groups, 1, 1);
    }

    reservoir_flip_ ^= 1u;
}

} // namespace rr::passes::restir_pt
