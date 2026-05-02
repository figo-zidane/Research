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
                                  rr::rhi::Extent2D extent)
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
                                     rr::rhi::Extent2D ext)
{
    rr::rhi::ImageDesc d{};
    d.format     = rr::rhi::Format::R32G32B32A32_Sfloat;
    d.extent     = {ext.width, ext.height, 1};
    d.usage      = rr::rhi::ImageUsage::Storage | rr::rhi::ImageUsage::Sampled;
    d.debug_name = "radiance_image";
    radiance_img_.create(device, d);

    radiance_storage_idx = registry.register_storage_image(
        device, radiance_img_, rr::rhi::Format::R32G32B32A32_Sfloat);
    radiance_texture_idx = registry.register_texture(
        device, radiance_img_,
        rr::rhi::Format::R32G32B32A32_Sfloat,
        rr::rhi::ImageLayout::General,
        rr::rhi::ImageAspect::Color);
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

void PathTracerPass::on_resize(rr::rhi::Extent2D new_extent)
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
                          rr::rhi::Format::R32G32B32A32_Sfloat, extent_});
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

    const rr::rhi::CommandRecorder recorder = fc.command_recorder;
    const auto& handles = scene.gpu_handles();

    // Transition radiance image to GENERAL
    {
        const rr::rhi::ImageBarrier barrier{
            .image = &radiance_img_,
            .src_stage = rr::rhi::PipelineStage::TopOfPipe,
            .src_access = rr::rhi::AccessFlags::None,
            .dst_stage = rr::rhi::PipelineStage::ComputeShader,
            .dst_access = rr::rhi::AccessFlags::ShaderWrite,
            .old_layout = rr::rhi::ImageLayout::Undefined,
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

    recorder.bind_compute_pipeline(pipeline_);

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

    recorder.push_constants(&pc, sizeof(pc));

    // Dispatch one thread per pixel, 8x8 tiles
    uint32_t gx = (extent_.width  + 7) / 8;
    uint32_t gy = (extent_.height + 7) / 8;
    recorder.dispatch(gx, gy, 1);
}

} // namespace rr::passes::pathtracer
