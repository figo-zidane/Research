#include "app/Application.h"

#include "core/Log.h"
#include "passes/accumulate/AccumulatePass.h"
#include "passes/gbuffer/GBufferPass.h"
#include "passes/imgui/ImGuiPass.h"
#include "passes/pathtracer_reference/PathTracerPass.h"
#include "passes/tonemap/TonemapPass.h"

#include <stdexcept>
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
    if (device_.device() != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(device_.device());
    }

    if (gbuffer_pass_)    { gbuffer_pass_->shutdown(device_);    gbuffer_pass_    = nullptr; }
    if (pathtracer_pass_) { pathtracer_pass_->shutdown(device_); pathtracer_pass_ = nullptr; }
    if (accumulate_pass_) { accumulate_pass_->shutdown(device_); accumulate_pass_ = nullptr; }
    if (tonemap_pass_)    { tonemap_pass_->shutdown(device_);    tonemap_pass_    = nullptr; }

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

    if (surface_ != VK_NULL_HANDLE && device_.instance() != VK_NULL_HANDLE)
    {
        vkDestroySurfaceKHR(device_.instance(), surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
    }

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
    vkDeviceWaitIdle(device_.device());
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

    rr::core::log()->info("Created window {} ({}x{})", title_, width_, height_);
}

void Application::initialize_vulkan()
{
    uint32_t extension_count = 0;
    const char** extension_names = glfwGetRequiredInstanceExtensions(&extension_count);
    if (extension_names == nullptr || extension_count == 0)
    {
        throw std::runtime_error("GLFW did not provide Vulkan instance extensions.");
    }
    std::vector<const char*> required_extensions(extension_names, extension_names + extension_count);

    // Device is brought up in two phases: the surface (needed for queue
    // present-capability checks) can only be created once the instance exists.
    rr::rhi::Device::CreateInfo create_info{};
    create_info.application_name             = title_;
    create_info.required_instance_extensions = std::move(required_extensions);
    create_info.enable_validation            = true;

    device_.create_instance(create_info);
    if (glfwCreateWindowSurface(device_.instance(), window_, nullptr, &surface_) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vulkan window surface.");
    }
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
        [this](std::function<void(VkCommandBuffer)> fn) { one_time_submit(fn); });

    camera_.on_resize(static_cast<float>(width_) / static_cast<float>(height_));

    VkExtent2D extent{static_cast<uint32_t>(width_), static_cast<uint32_t>(height_)};

    pathtracer_pass_ = renderer_.add_pass<rr::passes::pathtracer::PathTracerPass>();
    pathtracer_pass_->initialize(device_, slang_session_, bindless_registry_, extent);

    accumulate_pass_ = renderer_.add_pass<rr::passes::accumulate::AccumulatePass>();
    accumulate_pass_->initialize(device_, slang_session_, bindless_registry_, extent);
    accumulate_pass_->radiance_storage_idx = pathtracer_pass_->radiance_storage_idx;

    tonemap_pass_ = renderer_.add_pass<rr::passes::tonemap::TonemapPass>();
    tonemap_pass_->initialize(device_, slang_session_, bindless_registry_, swapchain_.image_format());
    tonemap_pass_->accumulated_texture_idx = accumulate_pass_->accumulated_texture_idx;
    tonemap_pass_->linear_sampler_idx      = scene_.gpu_handles().linear_sampler_idx;

    imgui_pass_ = renderer_.add_pass<rr::passes::imgui::ImGuiPass>();
    imgui_pass_->initialize(device_, window_, swapchain_);
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
    const auto&    sync        = frame_.sync(frame_index);

    vkWaitForFences(device_.device(), 1, &sync.in_flight, VK_TRUE, UINT64_MAX);

    uint32_t image_index = 0;
    VkResult acquire = vkAcquireNextImageKHR(
        device_.device(), swapchain_.handle(), UINT64_MAX,
        sync.image_available, VK_NULL_HANDLE, &image_index);
    if (acquire == VK_ERROR_OUT_OF_DATE_KHR)
    {
        recreate_swapchain();
        return;
    }
    if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR)
    {
        throw std::runtime_error("vkAcquireNextImageKHR failed.");
    }

    vkResetFences(device_.device(), 1, &sync.in_flight);

    // ── ImGui per-frame state machine (before command recording) ──────────
    imgui_pass_->new_frame();
    editor_ui_.build(renderer_, delta_time_seconds_);

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
    VkCommandBuffer cmd = command_buffer_.begin_frame(frame_index);

    // Ensure descriptor heap writes (done during upload) are visible to GPU
    bindless_registry_.heap_write_barrier(cmd);
    // Bind bindless heaps once per frame
    bindless_registry_.bind_heaps(cmd);

    // Transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL
    rr::rhi::CommandBuffer::image_barrier(
        cmd, swapchain_.image(image_index),
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // Build the frame context
    rr::render::FrameContext frame_context{};
    frame_context.command_buffer       = cmd;
    frame_context.device               = &device_;
    frame_context.bindless_registry    = &bindless_registry_;
    frame_context.scene                = &scene_;
    frame_context.renderer             = &renderer_;
    frame_context.swapchain_image_view = swapchain_.image_view(image_index);
    frame_context.swapchain_extent     = swapchain_.extent();
    frame_context.image_index          = image_index;
    frame_context.frame_index          = frame_index;
    frame_context.delta_time_seconds   = delta_time_seconds_;
    if (accumulate_pass_)
        frame_context.accumulated_image = accumulate_pass_->accumulated_image_handle();

    renderer_.render(frame_context);

    // Reset camera_moved flag after executing accumulate pass this frame
    camera_moved_ = false;

    // Transition the swapchain image to PRESENT_SRC_KHR for presentation.
    rr::rhi::CommandBuffer::image_barrier(
        cmd, swapchain_.image(image_index),
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    command_buffer_.end_frame(cmd);

    // ── Queue submit ──────────────────────────────────────────────────────
    const VkSemaphore render_finished = swapchain_.render_finished(image_index);
    constexpr VkPipelineStageFlags kWaitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit{};
    submit.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount   = 1;
    submit.pWaitSemaphores      = &sync.image_available;
    submit.pWaitDstStageMask    = &kWaitStage;
    submit.commandBufferCount   = 1;
    submit.pCommandBuffers      = &cmd;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores    = &render_finished;
    if (vkQueueSubmit(device_.graphics_queue(), 1, &submit, sync.in_flight) != VK_SUCCESS)
    {
        throw std::runtime_error("vkQueueSubmit failed.");
    }

    // ── Presentation ──────────────────────────────────────────────────────
    VkSwapchainKHR swap = swapchain_.handle();
    VkPresentInfoKHR present{};
    present.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores    = &render_finished;
    present.swapchainCount     = 1;
    present.pSwapchains        = &swap;
    present.pImageIndices      = &image_index;

    const VkResult present_result = vkQueuePresentKHR(device_.graphics_queue(), &present);
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
    vkDeviceWaitIdle(device_.device());
    swapchain_.recreate(static_cast<uint32_t>(fb_w), static_cast<uint32_t>(fb_h));
    renderer_.on_resize(swapchain_.extent());
    if (imgui_pass_ != nullptr)
    {
        imgui_pass_->set_min_image_count(swapchain_.image_count());
    }
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

void Application::glfw_scroll_callback(GLFWwindow* window, double /*xoff*/, double yoff)
{
    auto* self = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (self != nullptr) self->camera_.on_scroll(0.0, static_cast<double>(yoff));
}

void Application::one_time_submit(std::function<void(VkCommandBuffer)> fn)
{
    // Allocate a temporary command buffer from the pool
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool        = command_buffer_.pool();
    alloc_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer tmp_cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device_.device(), &alloc_info, &tmp_cmd) != VK_SUCCESS)
        throw std::runtime_error("one_time_submit: vkAllocateCommandBuffers failed.");

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(tmp_cmd, &begin_info);

    fn(tmp_cmd);

    vkEndCommandBuffer(tmp_cmd);

    VkSubmitInfo submit_info{};
    submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers    = &tmp_cmd;
    vkQueueSubmit(device_.graphics_queue(), 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(device_.graphics_queue());

    vkFreeCommandBuffers(device_.device(), command_buffer_.pool(), 1, &tmp_cmd);
}

} // namespace rr::app
