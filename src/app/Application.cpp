#include "app/Application.h"

#include "core/Log.h"
#include "passes/accumulate/AccumulatePass.h"
#include "passes/denoise/AtrousPass.h"
#include "passes/gbuffer/GBufferPass.h"
#include "passes/imgui/ImGuiPass.h"
#include "passes/pathtracer_reference/PathTracerPass.h"
#include "passes/restir_di/ReSTIRDIPass.h"
#include "passes/restir_gi/ReSTIRGIPass.h"
#include "passes/tonemap/TonemapPass.h"
#include "scene/GltfLoader.h"

// stb_image_write: implementation is in GltfLoader.cpp; only include header here.
#include <stb_image_write.h>

#include <algorithm>  // std::clamp
#include <cctype>     // std::tolower
#include <cmath>      // std::pow
#include <ctime>      // std::time
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rr::app
{
Application::Application() = default;

Application::~Application()
{
    shutdown();
}

void Application::run()
{
    initialize();
    last_frame_time_ = std::chrono::steady_clock::now();
    main_loop();
}

void Application::initialize()
{
    initialize_window();
    initialize_vulkan();
    initialize_renderer();
}

void Application::shutdown()
{
    if (device_.device() != nullptr)
    {
        device_.wait_idle();
    }

    if (gbuffer_pass_)    { gbuffer_pass_->shutdown(device_);    gbuffer_pass_    = nullptr; }
    if (pathtracer_pass_) { pathtracer_pass_->shutdown(device_); pathtracer_pass_ = nullptr; }
    if (accumulate_pass_) { accumulate_pass_->shutdown(device_); accumulate_pass_ = nullptr; }
    if (tonemap_pass_)    { tonemap_pass_->shutdown(device_);    tonemap_pass_    = nullptr; }
    if (restir_di_pass_)  { restir_di_pass_->shutdown(device_);  restir_di_pass_  = nullptr; }
    if (restir_gi_pass_)  { restir_gi_pass_->shutdown(device_);  restir_gi_pass_  = nullptr; }
    if (atrous_pass_)     { atrous_pass_->shutdown(device_);     atrous_pass_     = nullptr; }

    hot_reload_.shutdown();

    mse_staging_pt_.destroy(device_);
    mse_staging_ri_.destroy(device_);

    if (imgui_pass_ != nullptr)
    {
        imgui_pass_->shutdown();
        imgui_pass_ = nullptr;
    }

    scene_.destroy(device_);

    command_buffer_.shutdown();
    frame_.shutdown();
    swapchain_.shutdown();

    bindless_registry_.shutdown(device_);

    surface_.shutdown();

    device_.shutdown();

    if (window_ != nullptr)
    {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }

    if (glfw_initialized_)
    {
        glfwTerminate();
        glfw_initialized_ = false;
    }
}

void Application::main_loop()
{
    while (!glfwWindowShouldClose(window_))
    {
        glfwPollEvents();

        const auto now      = std::chrono::steady_clock::now();
        delta_time_seconds_ = std::chrono::duration<float>(now - last_frame_time_).count();
        last_frame_time_    = now;

        render_frame();
    }
    device_.wait_idle();
}

void Application::initialize_window()
{
    if (glfwInit() == GLFW_FALSE)
    {
        throw std::runtime_error("Failed to initialize GLFW.");
    }
    glfw_initialized_ = true;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window_ = glfwCreateWindow(width_, height_, title_.c_str(), nullptr, nullptr);
    if (window_ == nullptr)
    {
        throw std::runtime_error("Failed to create the application window.");
    }
    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_, &Application::glfw_resize_callback);
    glfwSetMouseButtonCallback(window_, &Application::glfw_mouse_button_callback);
    glfwSetCursorPosCallback(window_, &Application::glfw_cursor_pos_callback);
    glfwSetScrollCallback(window_, &Application::glfw_scroll_callback);
    glfwSetDropCallback(window_, &Application::glfw_drop_callback);

    rr::core::log()->info("Created window {} ({}x{})", title_, width_, height_);
}

void Application::initialize_vulkan()
{
    // Device is brought up in two phases: the surface (needed for queue
    // present-capability checks) can only be created once the instance exists.
    rr::rhi::Device::CreateInfo create_info{};
    create_info.application_name  = title_;
    create_info.presentation      = rr::rhi::PresentationSupport::Window;
    create_info.enable_validation = true;

    device_.create_instance(create_info);
    surface_.initialize(device_, {.window = static_cast<void*>(window_)});
    device_.create_device_with_surface(surface_);

    int fb_w = 0;
    int fb_h = 0;
    glfwGetFramebufferSize(window_, &fb_w, &fb_h);
    swapchain_.initialize(device_, surface_,
                          static_cast<uint32_t>(fb_w),
                          static_cast<uint32_t>(fb_h));
    frame_.initialize(device_);
    command_buffer_.initialize(device_);
}

void Application::initialize_renderer()
{
    bindless_registry_.initialize(device_);
    slang_session_.initialize("assets/shaders/include");

    // Build Cornell Box and upload to GPU.
    // one_time_submit provides a fresh VkCommandBuffer that gets submitted and
    // waited on before returning.
    scene_.build_cornell_box();
    scene_.upload(device_, bindless_registry_,
        [this](std::function<void(rr::rhi::CommandRecorder)> fn)
        {
            one_time_submit(fn);
        });

    camera_.on_resize(static_cast<float>(width_) / static_cast<float>(height_));

    rr::rhi::Extent2D extent{static_cast<uint32_t>(width_), static_cast<uint32_t>(height_)};

    // GBufferPass — rasterize scene into position/normal/material_id targets.
    // Must run before ReSTIRDIPass which reads its outputs.
    gbuffer_pass_ = renderer_.add_pass<rr::passes::gbuffer::GBufferPass>();
    gbuffer_pass_->initialize(device_, slang_session_, bindless_registry_, extent);

    pathtracer_pass_ = renderer_.add_pass<rr::passes::pathtracer::PathTracerPass>();
    pathtracer_pass_->initialize(device_, slang_session_, bindless_registry_, extent);

    accumulate_pass_ = renderer_.add_pass<rr::passes::accumulate::AccumulatePass>();
    accumulate_pass_->initialize(device_, slang_session_, bindless_registry_, extent);
    accumulate_pass_->radiance_storage_idx = pathtracer_pass_->radiance_storage_idx;

    // ReSTIR DI pass — must run before TonemapPass so its output is ready for sampling
    restir_di_pass_ = renderer_.add_pass<rr::passes::restir_di::ReSTIRDIPass>();
    restir_di_pass_->initialize(device_, slang_session_, bindless_registry_, extent);
    restir_di_pass_->set_gbuffer_indices(
        gbuffer_pass_->position_storage_idx, gbuffer_pass_->position_image_handle(),
        gbuffer_pass_->normal_storage_idx,   gbuffer_pass_->normal_image_handle());

    restir_gi_pass_ = renderer_.add_pass<rr::passes::restir_gi::ReSTIRGIPass>();
    restir_gi_pass_->initialize(device_, slang_session_, bindless_registry_, extent);
    restir_gi_pass_->set_inputs(
        gbuffer_pass_->position_storage_idx,
        gbuffer_pass_->normal_storage_idx,
        restir_di_pass_->output_texture_idx,
        restir_di_pass_->output_image_handle());

    atrous_pass_ = renderer_.add_pass<rr::passes::denoise::AtrousPass>();
    atrous_pass_->initialize(device_, slang_session_, bindless_registry_, extent);
    atrous_pass_->set_gbuffer_indices(
        gbuffer_pass_->position_storage_idx,
        gbuffer_pass_->normal_storage_idx);

    tonemap_pass_ = renderer_.add_pass<rr::passes::tonemap::TonemapPass>();
    tonemap_pass_->initialize(device_, slang_session_, bindless_registry_,
                              swapchain_.image_format());
    tonemap_pass_->accumulated_texture_idx = accumulate_pass_->accumulated_texture_idx;
    tonemap_pass_->linear_sampler_idx      = scene_.gpu_handles().linear_sampler_idx;

    imgui_pass_ = renderer_.add_pass<rr::passes::imgui::ImGuiPass>();
    imgui_pass_->initialize(device_, window_, swapchain_);

    // Pre-allocate persistent MSE staging buffers (64×64 crop, RGBA32F).
    {
        const VkDeviceSize mse_buf_size =
            static_cast<VkDeviceSize>(kMseCropSize * kMseCropSize) * 4 * sizeof(float);
        rr::rhi::BufferDesc desc{};
        desc.size         = mse_buf_size;
        desc.usage        = rr::rhi::BufferUsage::TransferDst;
        desc.memory_usage = rr::rhi::MemoryUsage::Auto;
        desc.alloc_flags  = rr::rhi::AllocFlags::HostAccessRandom;
        mse_staging_pt_.create(device_, desc);
        mse_staging_ri_.create(device_, desc);
    }

    // Initialize persistent storage images to VK_IMAGE_LAYOUT_GENERAL before the
    // first frame.  The bindless heap is bound at the start of each frame; the
    // Vulkan validation layer checks that every registered image is in the layout
    // it was registered with.  Without this transition, images that are only
    // transitioned inside their own execute() would still be UNDEFINED when an
    // earlier pass dispatches.
    one_time_submit([this](rr::rhi::CommandRecorder recorder)
    {
        VkCommandBuffer cmd = static_cast<VkCommandBuffer>(recorder.handle());
        // accumulated_image (AccumulatePass): registered as GENERAL storage image.
        {
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask        = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            b.srcAccessMask       = 0;
            b.dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.dstAccessMask       = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
            b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image               = rr::rhi::from_handle<VkImage>(accumulate_pass_->accumulated_image_handle());
            b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
            VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers    = &b;
            vkCmdPipelineBarrier2(cmd, &dep);
        }
        // GBufferPass storage images: transition UNDEFINED → GENERAL so the
        // bindless heap sees the correct layout before the first frame executes.
        if (gbuffer_pass_)
            gbuffer_pass_->pre_transition_to_general(rr::rhi::CommandRecorder{static_cast<void*>(cmd)});
        // ReSTIR DI pass: all owned storage images must be GENERAL before first frame.
        if (restir_di_pass_)
            restir_di_pass_->pre_transition_to_general(rr::rhi::CommandRecorder{static_cast<void*>(cmd)});
        if (restir_gi_pass_)
            restir_gi_pass_->pre_transition_to_general(rr::rhi::CommandRecorder{static_cast<void*>(cmd)});
        if (atrous_pass_)
            atrous_pass_->pre_transition_to_general(rr::rhi::CommandRecorder{static_cast<void*>(cmd)});
    });

    // Hot reload — watch shaders directory
    hot_reload_.initialize("assets/shaders");
    hot_reload_.register_shader("assets/shaders/passes/pathtracer_reference/pathtracer.slang",
        [this]() { return pathtracer_pass_->reload_shader(slang_session_); });
    hot_reload_.register_shader("assets/shaders/passes/accumulate/accumulate.slang",
        [this]() { return accumulate_pass_->reload_shader(slang_session_); });
    hot_reload_.register_shader("assets/shaders/passes/tonemap/tonemap.slang",
        [this]() { return tonemap_pass_->reload_shader(slang_session_); });
    hot_reload_.register_shader("assets/shaders/passes/restir_di/restir_di.slang",
        [this]() { return restir_di_pass_->reload_shader(slang_session_); });
    hot_reload_.register_shader("assets/shaders/passes/restir_gi/restir_gi.slang",
        [this]() { return restir_gi_pass_->reload_shader(slang_session_); });
    hot_reload_.register_shader("assets/shaders/passes/denoise/atrous.slang",
        [this]() { return atrous_pass_->reload_shader(slang_session_); });
}

void Application::render_frame()
{
    int fb_w = 0;
    int fb_h = 0;
    glfwGetFramebufferSize(window_, &fb_w, &fb_h);
    if (fb_w == 0 || fb_h == 0)
    {
        // Window minimised: skip the frame entirely.
        return;
    }

    if (framebuffer_resized_)
    {
        recreate_swapchain();
        framebuffer_resized_ = false;
    }

    const uint32_t frame_index = frame_.current();
    frame_.wait_for_in_flight(frame_index);

    uint32_t image_index = 0;
    const VkDevice vk_device = rr::rhi::from_opaque_handle<VkDevice>(device_.device());
    const VkQueue graphics_queue = rr::rhi::from_opaque_handle<VkQueue>(device_.graphics_queue());
    const VkSwapchainKHR swapchain_handle = rr::rhi::from_opaque_handle<VkSwapchainKHR>(swapchain_.handle());
    const VkSemaphore image_available = rr::rhi::from_opaque_handle<VkSemaphore>(frame_.image_available_semaphore(frame_index));
    VkResult acquire = vkAcquireNextImageKHR(
        vk_device, swapchain_handle, UINT64_MAX,
        image_available, VK_NULL_HANDLE, &image_index);
    if (acquire == VK_ERROR_OUT_OF_DATE_KHR)
    {
        recreate_swapchain();
        return;
    }
    if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR)
    {
        throw std::runtime_error("vkAcquireNextImageKHR failed.");
    }

    frame_.reset_in_flight_fence(frame_index);

    // ── ImGui per-frame state machine (before command recording) ──────────
    imgui_pass_->new_frame();
    // Compute correct count and offset for the circular MSE history buffer.
    const uint32_t mse_count  = std::min(mse_history_pos_, kMseHistoryLen);
    const uint32_t mse_offset = (mse_history_pos_ >= kMseHistoryLen)
                                    ? (mse_history_pos_ % kMseHistoryLen)
                                    : 0u;
    editor_ui_.build(renderer_, delta_time_seconds_, accumulated_spp_, save_screenshot_,
                     use_di_, use_gi_, use_denoise_, mse_compare_,
                     gltf_path_input_, load_cornell_request_, load_gltf_request_,
                     current_scene_name_, atrous_pass_,
                     &hot_reload_,
                     mse_history_.data(), mse_count, mse_offset,
                     mse_latest_, &mse_auto_update_);

    if (!(use_di_ || use_gi_))
        use_denoise_ = false;

    // ── Scene reload requests ─────────────────────────────────────────────
    if (load_cornell_request_)
    {
        load_cornell_request_ = false;
        reload_scene_cornell();
    }
    else if (load_gltf_request_)
    {
        load_gltf_request_ = false;
        reload_scene_gltf(gltf_path_input_);
    }

    // ── Display mode switch detection ─────────────────────────────────────
    if (use_di_ != prev_use_di_ || use_gi_ != prev_use_gi_ || use_denoise_ != prev_use_denoise_)
    {
        accumulated_spp_ = 0;
        camera_moved_    = true;
        if (restir_di_pass_) restir_di_pass_->reset_history();
        if (restir_gi_pass_) restir_gi_pass_->reset_history();
        prev_use_di_      = use_di_;
        prev_use_gi_      = use_gi_;
        prev_use_denoise_ = use_denoise_;
    }
    // ── Pass enable/disable based on display mode ─────────────────────────
    const bool use_realtime = use_di_ || use_gi_;
    const bool use_denoise = use_denoise_ && use_realtime;
    if (restir_gi_pass_)
        restir_gi_pass_->include_direct_lighting = use_di_;
    if (!mse_compare_)
    {
        if (accumulate_pass_)  accumulate_pass_->set_enabled(!use_realtime);
        if (pathtracer_pass_)  pathtracer_pass_->set_enabled(!use_realtime);
        if (restir_di_pass_)   restir_di_pass_->set_enabled(use_di_);
        if (restir_gi_pass_)   restir_gi_pass_->set_enabled(use_gi_);
        if (atrous_pass_)      atrous_pass_->set_enabled(use_denoise);
    }
    else
    {
        if (accumulate_pass_)  accumulate_pass_->set_enabled(true);
        if (pathtracer_pass_)  pathtracer_pass_->set_enabled(true);
        if (restir_di_pass_)   restir_di_pass_->set_enabled(use_di_);
        if (restir_gi_pass_)   restir_gi_pass_->set_enabled(use_gi_);
        if (atrous_pass_)      atrous_pass_->set_enabled(use_denoise);
    }

    // ── Hot reload ────────────────────────────────────────────────────────
    if (hot_reload_.pump(device_))
    {
        accumulated_spp_ = 0;
        camera_moved_    = true;
        if (restir_di_pass_) restir_di_pass_->reset_history();
        if (restir_gi_pass_) restir_gi_pass_->reset_history();
    }

    // ── Camera update ─────────────────────────────────────────────────────
    bool cam_moved = camera_.update(window_, delta_time_seconds_);
    if (cam_moved)
    {
        camera_moved_    = true;
        accumulated_spp_ = 0;
    }
    // Update GPU camera data
    int fb_w2 = 0, fb_h2 = 0;
    glfwGetFramebufferSize(window_, &fb_w2, &fb_h2);
    scene_.update_camera(camera_,
                          static_cast<uint32_t>(fb_w2),
                          static_cast<uint32_t>(fb_h2),
                          frame_index);

    // ── Wire accumulate pass state ────────────────────────────────────────
    if (accumulate_pass_)
    {
        accumulate_pass_->camera_moved    = camera_moved_;
        accumulate_pass_->accumulated_spp = accumulated_spp_;
    }

    // ── Command buffer recording ──────────────────────────────────────────
    rr::rhi::CommandRecorder recorder = command_buffer_.begin_frame(frame_index);
    VkCommandBuffer cmd = static_cast<VkCommandBuffer>(recorder.handle());

    // Ensure descriptor heap writes (done during upload) are visible to GPU
    bindless_registry_.heap_write_barrier(cmd);
    // Bind bindless heaps once per frame
    bindless_registry_.bind_heaps(cmd);

    // Transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL
    const rr::rhi::ImageBarrier swapchain_to_color{
        .image = &swapchain_.image(image_index),
        .src_stage = rr::rhi::PipelineStage::TopOfPipe,
        .src_access = rr::rhi::AccessFlags::None,
        .dst_stage = rr::rhi::PipelineStage::ColorAttachmentOutput,
        .dst_access = rr::rhi::AccessFlags::ColorAttachmentWrite,
        .old_layout = rr::rhi::ImageLayout::Undefined,
        .new_layout = rr::rhi::ImageLayout::ColorAttachment,
        .subresource = {
            .aspect = rr::rhi::ImageAspect::Color,
            .base_mip = 0,
            .mip_count = 1,
            .base_layer = 0,
            .layer_count = 1,
        },
    };
    recorder.pipeline_barrier({&swapchain_to_color, 1});

    // Build the frame context
    rr::render::FrameContext frame_context{};
    frame_context.command_recorder     = recorder;
    frame_context.device               = &device_;
    frame_context.bindless_registry    = &bindless_registry_;
    frame_context.scene                = &scene_;
    frame_context.renderer             = &renderer_;
    frame_context.swapchain_image_view = rr::rhi::to_handle(swapchain_.image(image_index).view());
    frame_context.swapchain_extent     = {swapchain_.extent().width, swapchain_.extent().height};
    frame_context.image_index          = image_index;
    frame_context.frame_index          = frame_index;
    frame_context.delta_time_seconds   = delta_time_seconds_;
    uint32_t realtime_texture_idx = UINT32_MAX;
    rr::rhi::ImageHandle realtime_image = 0;
    if (use_gi_ && restir_gi_pass_)
    {
        realtime_texture_idx = restir_gi_pass_->output_texture_idx;
        realtime_image       = restir_gi_pass_->output_image_handle();
    }
    else if (use_di_ && restir_di_pass_)
    {
        realtime_texture_idx = restir_di_pass_->output_texture_idx;
        realtime_image       = restir_di_pass_->output_image_handle();
    }

    if (atrous_pass_)
        atrous_pass_->set_input(realtime_texture_idx, realtime_image);

    if (use_denoise && atrous_pass_)
    {
        tonemap_pass_->accumulated_texture_idx = atrous_pass_->output_texture_idx();
        frame_context.accumulated_image        = atrous_pass_->output_image_handle();
    }
    else if (realtime_image != 0)
    {
        tonemap_pass_->accumulated_texture_idx = realtime_texture_idx;
        frame_context.accumulated_image        = realtime_image;
    }
    else if (accumulate_pass_)
        frame_context.accumulated_image = accumulate_pass_->accumulated_image_handle();

    renderer_.render(frame_context);

    // Reset camera_moved flag after executing accumulate pass this frame
    camera_moved_ = false;

    // Sync SPP back from accumulate pass
    if (accumulate_pass_)
        accumulated_spp_ = accumulate_pass_->accumulated_spp;

    // Reset tonemap source back to accumulate pass for next frame
    if (tonemap_pass_ && accumulate_pass_)
        tonemap_pass_->accumulated_texture_idx = accumulate_pass_->accumulated_texture_idx;

    // Periodic MSE computation
    if (mse_auto_update_ && (use_di_ || use_gi_) && ++mse_frame_counter_ % kMseInterval == 0)
        compute_mse();

    // Transition the swapchain image to PRESENT_SRC_KHR for presentation.
    const rr::rhi::ImageBarrier swapchain_to_present{
        .image = &swapchain_.image(image_index),
        .src_stage = rr::rhi::PipelineStage::ColorAttachmentOutput,
        .src_access = rr::rhi::AccessFlags::ColorAttachmentWrite,
        .dst_stage = rr::rhi::PipelineStage::None,
        .dst_access = rr::rhi::AccessFlags::None,
        .old_layout = rr::rhi::ImageLayout::ColorAttachment,
        .new_layout = rr::rhi::ImageLayout::Present,
        .subresource = {
            .aspect = rr::rhi::ImageAspect::Color,
            .base_mip = 0,
            .mip_count = 1,
            .base_layer = 0,
            .layer_count = 1,
        },
    };
    recorder.pipeline_barrier({&swapchain_to_present, 1});

    command_buffer_.end_frame(recorder);

    // ── Queue submit ──────────────────────────────────────────────────────
    const VkSemaphore render_finished = rr::rhi::from_opaque_handle<VkSemaphore>(swapchain_.render_finished(image_index));
    constexpr VkPipelineStageFlags kWaitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit{};
    submit.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount   = 1;
    submit.pWaitSemaphores      = &image_available;
    submit.pWaitDstStageMask    = &kWaitStage;
    submit.commandBufferCount   = 1;
    submit.pCommandBuffers      = &cmd;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores    = &render_finished;
    const VkFence in_flight_fence = rr::rhi::from_opaque_handle<VkFence>(frame_.in_flight_fence(frame_index));
    if (vkQueueSubmit(graphics_queue, 1, &submit, in_flight_fence) != VK_SUCCESS)
    {
        throw std::runtime_error("vkQueueSubmit failed.");
    }

    // ── Presentation ──────────────────────────────────────────────────────
    VkSwapchainKHR swap = swapchain_handle;
    VkPresentInfoKHR present{};
    present.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores    = &render_finished;
    present.swapchainCount     = 1;
    present.pSwapchains        = &swap;
    present.pImageIndices      = &image_index;

    const VkResult present_result = vkQueuePresentKHR(graphics_queue, &present);
    if (present_result == VK_ERROR_OUT_OF_DATE_KHR ||
        present_result == VK_SUBOPTIMAL_KHR         ||
        framebuffer_resized_)
    {
        framebuffer_resized_ = false;
        recreate_swapchain();
    }
    else if (present_result != VK_SUCCESS)
    {
        throw std::runtime_error("vkQueuePresentKHR failed.");
    }

    // Screenshot capture: triggered by UI button or auto at 4096 spp
    if (save_screenshot_)
    {
        save_screenshot_ = false;
        capture_screenshot();
    }

    (void)frame_.advance();
}

void Application::recreate_swapchain()
{
    int fb_w = 0;
    int fb_h = 0;
    glfwGetFramebufferSize(window_, &fb_w, &fb_h);
    while (fb_w == 0 || fb_h == 0)
    {
        glfwWaitEvents();
        glfwGetFramebufferSize(window_, &fb_w, &fb_h);
    }
    device_.wait_idle();
    const uint32_t previous_image_count = swapchain_.image_count();
    swapchain_.recreate(static_cast<uint32_t>(fb_w), static_cast<uint32_t>(fb_h));
    const bool swapchain_image_count_changed = swapchain_.image_count() != previous_image_count;
    width_  = fb_w;
    height_ = fb_h;
    camera_.on_resize(static_cast<float>(fb_w) / static_cast<float>(fb_h));

    // Resized render targets invalidate progressive accumulation history.
    camera_moved_   = true;
    accumulated_spp_ = 0;

    const rr::rhi::Extent2D swapchain_extent = swapchain_.extent();
    renderer_.on_resize({swapchain_extent.width, swapchain_extent.height});

    // Pass resize recreates images and re-registers them in the bindless heap.
    // Refresh all cached downstream indices and handles before the next frame.
    if (gbuffer_pass_ && restir_di_pass_)
    {
        restir_di_pass_->set_gbuffer_indices(
            gbuffer_pass_->position_storage_idx,
            gbuffer_pass_->position_image_handle(),
            gbuffer_pass_->normal_storage_idx,
            gbuffer_pass_->normal_image_handle());
    }
    if (pathtracer_pass_ && accumulate_pass_)
    {
        accumulate_pass_->radiance_storage_idx = pathtracer_pass_->radiance_storage_idx;
    }
    if (gbuffer_pass_ && atrous_pass_)
    {
        atrous_pass_->set_gbuffer_indices(
            gbuffer_pass_->position_storage_idx,
            gbuffer_pass_->normal_storage_idx);
    }
    if (gbuffer_pass_ && restir_gi_pass_ && restir_di_pass_)
    {
        restir_gi_pass_->set_inputs(
            gbuffer_pass_->position_storage_idx,
            gbuffer_pass_->normal_storage_idx,
            restir_di_pass_->output_texture_idx,
            restir_di_pass_->output_image_handle());
    }
    if (tonemap_pass_ && accumulate_pass_)
    {
        tonemap_pass_->accumulated_texture_idx = accumulate_pass_->accumulated_texture_idx;
        tonemap_pass_->linear_sampler_idx      = scene_.gpu_handles().linear_sampler_idx;
    }

    if (imgui_pass_ != nullptr && swapchain_image_count_changed)
    {
        imgui_pass_->shutdown();
        imgui_pass_->initialize(device_, window_, swapchain_);
    }

    // Recreated persistent images are back in UNDEFINED, but the bindless heap is
    // bound before pass execution each frame. Transition them up front again.
    one_time_submit([this](rr::rhi::CommandRecorder recorder)
    {
        VkCommandBuffer cmd = static_cast<VkCommandBuffer>(recorder.handle());

        if (accumulate_pass_)
        {
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask        = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            b.srcAccessMask       = 0;
            b.dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.dstAccessMask       = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
            b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image               = rr::rhi::from_handle<VkImage>(accumulate_pass_->accumulated_image_handle());
            b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
            VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers    = &b;
            vkCmdPipelineBarrier2(cmd, &dep);
        }

        if (gbuffer_pass_)
            gbuffer_pass_->pre_transition_to_general(rr::rhi::CommandRecorder{static_cast<void*>(cmd)});
        if (restir_di_pass_)
            restir_di_pass_->pre_transition_to_general(rr::rhi::CommandRecorder{static_cast<void*>(cmd)});
        if (restir_gi_pass_)
            restir_gi_pass_->pre_transition_to_general(rr::rhi::CommandRecorder{static_cast<void*>(cmd)});
        if (atrous_pass_)
            atrous_pass_->pre_transition_to_general(rr::rhi::CommandRecorder{static_cast<void*>(cmd)});
    });
}

void Application::glfw_resize_callback(GLFWwindow* window, int /*width*/, int /*height*/)
{
    auto* self = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (self != nullptr) self->framebuffer_resized_ = true;
}

void Application::glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int /*mods*/)
{
    auto* self = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (self != nullptr) self->camera_.on_mouse_button(button, action, 0);
}

void Application::glfw_cursor_pos_callback(GLFWwindow* window, double x, double y)
{
    auto* self = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (self != nullptr) self->camera_.on_mouse_move(static_cast<float>(x),
                                                       static_cast<float>(y));
}

void Application::glfw_scroll_callback(GLFWwindow* /*window*/, double /*xoff*/, double /*yoff*/)
{
}

void Application::glfw_drop_callback(GLFWwindow* window, int count, const char** paths)
{
    auto* self = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (self == nullptr || count <= 0 || paths == nullptr) return;

    // Accept the first dropped file that looks like a glTF/glb.
    for (int i = 0; i < count; ++i)
    {
        if (paths[i] == nullptr) continue;
        std::string p(paths[i]);
        const auto ext_pos = p.rfind('.');
        if (ext_pos == std::string::npos) continue;
        std::string ext = p.substr(ext_pos);
        // lowercase comparison
        for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (ext == ".gltf" || ext == ".glb")
        {
            self->gltf_path_input_   = p;
            self->load_gltf_request_ = true;
            break;
        }
    }
}

void Application::reload_scene_cornell()
{
    device_.wait_idle();
    scene_.destroy(device_);
    scene_.clear_cpu_data();
    scene_.build_cornell_box();
    scene_.upload(device_, bindless_registry_,
        [this](std::function<void(rr::rhi::CommandRecorder)> fn)
        {
            one_time_submit(fn);
        });
    camera_.on_resize(static_cast<float>(width_) / static_cast<float>(height_));
    camera_ = rr::scene::Camera{};
    camera_.on_resize(static_cast<float>(width_) / static_cast<float>(height_));
    // Update passes that cache scene handles
    if (tonemap_pass_) tonemap_pass_->linear_sampler_idx = scene_.gpu_handles().linear_sampler_idx;
    accumulated_spp_      = 0;
    camera_moved_         = true;
    current_scene_name_   = "Cornell Box";
    if (restir_di_pass_) restir_di_pass_->reset_history();
    if (restir_gi_pass_) restir_gi_pass_->reset_history();
    rr::core::log()->info("[Scene] Reloaded Cornell Box");
}

void Application::reload_scene_gltf(const std::string& path)
{
    if (path.empty()) return;
    device_.wait_idle();
    scene_.destroy(device_);
    scene_.clear_cpu_data();
    if (!rr::scene::GltfLoader::load(path, scene_))
    {
        // Load failed: fall back to Cornell Box
        rr::core::log()->warn("[Scene] glTF load failed for '{}', falling back to Cornell Box", path);
        scene_.build_cornell_box();
        current_scene_name_ = "Cornell Box";
    }
    else
    {
        current_scene_name_ = std::filesystem::path(path).filename().string();
    }
    scene_.upload(device_, bindless_registry_,
        [this](std::function<void(rr::rhi::CommandRecorder)> fn)
        {
            one_time_submit(fn);
        });
    camera_ = rr::scene::Camera{};
    camera_.on_resize(static_cast<float>(width_) / static_cast<float>(height_));
    // Update passes that cache scene handles
    if (tonemap_pass_) tonemap_pass_->linear_sampler_idx = scene_.gpu_handles().linear_sampler_idx;
    accumulated_spp_  = 0;
    camera_moved_     = true;
    if (restir_di_pass_) restir_di_pass_->reset_history();
    if (restir_gi_pass_) restir_gi_pass_->reset_history();
    rr::core::log()->info("[Scene] Loaded '{}'", current_scene_name_);
}

void Application::one_time_submit(std::function<void(rr::rhi::CommandRecorder)> fn)
{
    const VkDevice vk_device = rr::rhi::from_opaque_handle<VkDevice>(device_.device());
    const VkQueue graphics_queue = rr::rhi::from_opaque_handle<VkQueue>(device_.graphics_queue());

    // Allocate a temporary command buffer from the pool
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool        = rr::rhi::from_opaque_handle<VkCommandPool>(command_buffer_.pool());
    alloc_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer tmp_cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(vk_device, &alloc_info, &tmp_cmd) != VK_SUCCESS)
        throw std::runtime_error("one_time_submit: vkAllocateCommandBuffers failed.");

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(tmp_cmd, &begin_info);

    fn(rr::rhi::CommandRecorder{static_cast<void*>(tmp_cmd)});

    vkEndCommandBuffer(tmp_cmd);

    VkSubmitInfo submit_info{};
    submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers    = &tmp_cmd;
    vkQueueSubmit(graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphics_queue);

    const VkCommandPool one_time_pool = rr::rhi::from_opaque_handle<VkCommandPool>(command_buffer_.pool());
    vkFreeCommandBuffers(vk_device, one_time_pool, 1, &tmp_cmd);
}

void Application::capture_screenshot()
{
    if (!accumulate_pass_) return;

    // Wait for all GPU work to complete before reading back.
    device_.wait_idle();

    const rr::rhi::Extent2D ext = swapchain_.extent();
    const uint32_t   pixel_count = ext.width * ext.height;
    const VkDeviceSize buf_size  = static_cast<VkDeviceSize>(pixel_count) * 4 * sizeof(float);

    // Create a host-readable staging buffer (GPU writes → CPU reads).
    // VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT = 0x00000800
    rr::rhi::Buffer staging;
    rr::rhi::BufferDesc bd{};
    bd.size         = buf_size;
    bd.usage        = rr::rhi::BufferUsage::TransferDst;
    bd.memory_usage = rr::rhi::MemoryUsage::Auto;
    bd.alloc_flags  = rr::rhi::AllocFlags::HostAccessRandom;
    staging.create(device_, bd);

    VkImage src = rr::rhi::from_handle<VkImage>(accumulate_pass_->accumulated_image_handle());

    // GPU commands: transition accumulated image to TRANSFER_SRC, copy, transition back.
    one_time_submit([&](rr::rhi::CommandRecorder recorder)
    {
        VkCommandBuffer cmd = static_cast<VkCommandBuffer>(recorder.handle());
        // GENERAL → TRANSFER_SRC_OPTIMAL
        VkImageMemoryBarrier2 b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask        = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        b.srcAccessMask       = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT;
        b.dstStageMask        = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        b.dstAccessMask       = VK_ACCESS_2_TRANSFER_READ_BIT;
        b.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = src;
        b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{};
        dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &b;
        vkCmdPipelineBarrier2(cmd, &dep);

        // Copy image → buffer (RGBA32F tightly packed)
        VkBufferImageCopy region{};
        region.bufferOffset      = 0;
        region.bufferRowLength   = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset       = {0, 0, 0};
        region.imageExtent       = {ext.width, ext.height, 1};
        vkCmdCopyImageToBuffer(cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               rr::rhi::from_handle<VkBuffer>(staging.handle()), 1, &region);

        // TRANSFER_SRC_OPTIMAL → GENERAL (restore for next frame)
        b.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        b.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        b.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        b.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
        vkCmdPipelineBarrier2(cmd, &dep);
    });

    // Map buffer and convert RGBA32F → RGBA8 with ACES tone mapping + sRGB gamma.
    const float* hdr = static_cast<const float*>(staging.map(device_));

    const float exposure = tonemap_pass_ ? tonemap_pass_->exposure : 1.0f;

    std::vector<uint8_t> ldr(static_cast<size_t>(pixel_count) * 4);
    for (uint32_t i = 0; i < pixel_count; ++i)
    {
        for (int c = 0; c < 3; ++c)
        {
            float v = hdr[i * 4 + c] * exposure;
            // ACES filmic approx (Krzysztof Narkowicz)
            v = (v * (2.51f * v + 0.03f)) / (v * (2.43f * v + 0.59f) + 0.14f);
            v = std::clamp(v, 0.0f, 1.0f);
            // sRGB gamma
            v = std::pow(v, 1.0f / 2.2f);
            ldr[i * 4 + c] = static_cast<uint8_t>(v * 255.0f + 0.5f);
        }
        ldr[i * 4 + 3] = 255u;
    }
    staging.unmap(device_);
    staging.destroy(device_);

    const std::time_t ts   = std::time(nullptr);
    const std::string path = "screenshot_" + std::to_string(accumulated_spp_) + "spp_" + std::to_string(ts) + ".png";
    stbi_write_png(path.c_str(),
                   static_cast<int>(ext.width),
                   static_cast<int>(ext.height),
                   4, ldr.data(),
                   static_cast<int>(ext.width) * 4);
    rr::core::log()->info("[Screenshot] Saved '{}' ({}x{}, {} spp)",
                          path, ext.width, ext.height, accumulated_spp_);
}

void Application::compute_mse()
{
    if (!accumulate_pass_) return;

    rr::rhi::ImageHandle img_ri = 0;
    if (use_denoise_ && atrous_pass_ && (use_di_ || use_gi_))
        img_ri = atrous_pass_->output_image_handle();
    else if (use_gi_ && restir_gi_pass_)
        img_ri = restir_gi_pass_->output_image_handle();
    else if (use_di_ && restir_di_pass_)
        img_ri = restir_di_pass_->output_image_handle();
    if (img_ri == 0) return;

    const rr::rhi::Extent2D ext = swapchain_.extent();
    const uint32_t   crop = kMseCropSize;
    if (ext.width < crop || ext.height < crop) return;

    const int32_t    off_x       = static_cast<int32_t>((ext.width  - crop) / 2);
    const int32_t    off_y       = static_cast<int32_t>((ext.height - crop) / 2);
    const uint32_t   pixel_count = crop * crop;

    // Use persistent staging buffers (allocated in initialize_renderer).
    // No alloc/free per call — avoids the per-frame GPU stall.
    const VkImage img_pt = rr::rhi::from_handle<VkImage>(accumulate_pass_->accumulated_image_handle());
    const VkImage img_ri_vk = rr::rhi::from_handle<VkImage>(img_ri);

    one_time_submit([&](rr::rhi::CommandRecorder recorder)
    {
        VkCommandBuffer cmd = static_cast<VkCommandBuffer>(recorder.handle());
        // Helper lambda — captures nothing from outer compute_mse; all by param.
        // Barrier: GENERAL -> TRANSFER_SRC
        auto barrier_to_src = [](VkCommandBuffer c, VkImage img)
        {
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask       = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.srcAccessMask      = VK_ACCESS_2_SHADER_WRITE_BIT;
            b.dstStageMask       = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            b.dstAccessMask      = VK_ACCESS_2_TRANSFER_READ_BIT;
            b.oldLayout          = VK_IMAGE_LAYOUT_GENERAL;
            b.newLayout          = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            b.image              = img;
            b.subresourceRange   = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkDependencyInfo dep = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers    = &b;
            vkCmdPipelineBarrier2(c, &dep);
        };
        auto barrier_to_general = [](VkCommandBuffer c, VkImage img)
        {
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask       = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            b.srcAccessMask      = VK_ACCESS_2_TRANSFER_READ_BIT;
            b.dstStageMask       = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.dstAccessMask      = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
            b.oldLayout          = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            b.newLayout          = VK_IMAGE_LAYOUT_GENERAL;
            b.image              = img;
            b.subresourceRange   = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkDependencyInfo dep = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers    = &b;
            vkCmdPipelineBarrier2(c, &dep);
        };

        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset      = {off_x, off_y, 0};
        region.imageExtent      = {crop, crop, 1};

        barrier_to_src(cmd, img_pt);
        vkCmdCopyImageToBuffer(cmd, img_pt, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               rr::rhi::from_handle<VkBuffer>(mse_staging_pt_.handle()), 1, &region);
        barrier_to_general(cmd, img_pt);

        barrier_to_src(cmd, img_ri_vk);
        vkCmdCopyImageToBuffer(cmd, img_ri_vk, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               rr::rhi::from_handle<VkBuffer>(mse_staging_ri_.handle()), 1, &region);
        barrier_to_general(cmd, img_ri_vk);
    });

    const float* pt = static_cast<const float*>(mse_staging_pt_.map(device_));
    const float* ri = static_cast<const float*>(mse_staging_ri_.map(device_));

    double mse_sum = 0.0;
    for (uint32_t i = 0; i < pixel_count; ++i)
    {
        for (int c = 0; c < 3; ++c)
        {
            const double d = static_cast<double>(pt[i * 4 + c]) - static_cast<double>(ri[i * 4 + c]);
            mse_sum += d * d;
        }
    }
    mse_sum /= static_cast<double>(pixel_count) * 3.0;

    mse_staging_pt_.unmap(device_);
    mse_staging_ri_.unmap(device_);

    mse_latest_ = static_cast<float>(mse_sum);
    mse_history_[mse_history_pos_ % kMseHistoryLen] = mse_latest_;
    ++mse_history_pos_;
}

} // namespace rr::app
