#include "app/Application.h"

#include "core/Log.h"

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
    main_loop();
}

void Application::initialize()
{
    initialize_window();
    initialize_vulkan();
    scene_.set_name("Bootstrap Scene");
}

void Application::shutdown()
{
    if (device_.device() != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(device_.device());
    }

    command_buffer_.shutdown();
    frame_.shutdown();
    swapchain_.shutdown();

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

    // Device is brought up in two phases because the surface (needed for queue
    // present-capability checks) can only be created once the instance exists.
    rr::rhi::Device::CreateInfo create_info{};
    create_info.application_name = title_;
    create_info.required_instance_extensions = std::move(required_extensions);
    create_info.enable_validation = true;

    device_.create_instance(create_info);
    if (glfwCreateWindowSurface(device_.instance(), window_, nullptr, &surface_) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vulkan window surface.");
    }
    device_.create_device_with_surface(surface_);

    int fb_w = 0;
    int fb_h = 0;
    glfwGetFramebufferSize(window_, &fb_w, &fb_h);
    swapchain_.initialize(device_, surface_, static_cast<uint32_t>(fb_w), static_cast<uint32_t>(fb_h));
    frame_.initialize(device_);
    command_buffer_.initialize(device_);
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
    const auto& sync = frame_.sync(frame_index);

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

    VkCommandBuffer cmd = command_buffer_.begin_frame(frame_index);

    rr::rhi::CommandBuffer::image_barrier(
        cmd, swapchain_.image(image_index),
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    VkRenderingAttachmentInfo color_attachment{};
    color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color_attachment.imageView = swapchain_.image_view(image_index);
    color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.clearValue.color = {{0.05f, 0.10f, 0.15f, 1.0f}};

    VkRenderingInfo rendering{};
    rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering.renderArea.extent = swapchain_.extent();
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &color_attachment;

    vkCmdBeginRendering(cmd, &rendering);

    rr::render::FrameContext frame_context{};
    frame_context.command_buffer = cmd;
    frame_context.device = &device_;
    frame_context.scene = &scene_;
    frame_context.renderer = &renderer_;
    frame_context.frame_index = frame_index;
    renderer_.render(frame_context);

    vkCmdEndRendering(cmd);

    rr::rhi::CommandBuffer::image_barrier(
        cmd, swapchain_.image(image_index),
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    command_buffer_.end_frame(cmd);

    const VkSemaphore render_finished = swapchain_.render_finished(image_index);
    constexpr VkPipelineStageFlags kWaitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &sync.image_available;
    submit.pWaitDstStageMask = &kWaitStage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &render_finished;
    if (vkQueueSubmit(device_.graphics_queue(), 1, &submit, sync.in_flight) != VK_SUCCESS)
    {
        throw std::runtime_error("vkQueueSubmit failed.");
    }

    VkSwapchainKHR swap = swapchain_.handle();
    VkPresentInfoKHR present{};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &render_finished;
    present.swapchainCount = 1;
    present.pSwapchains = &swap;
    present.pImageIndices = &image_index;

    const VkResult present_result = vkQueuePresentKHR(device_.graphics_queue(), &present);
    if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR || framebuffer_resized_)
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
}

void Application::glfw_resize_callback(GLFWwindow* window, int /*width*/, int /*height*/)
{
    auto* self = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (self != nullptr)
    {
        self->framebuffer_resized_ = true;
    }
}
}
