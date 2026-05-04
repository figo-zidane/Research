#include "rhi/Swapchain.h"

#include "core/Log.h"
#include "rhi/Device.h"
#include "rhi/Surface.h"
#include "rhi/Types.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace rr::rhi
{
namespace
{
VkSurfaceFormatKHR pick_surface_format(VkPhysicalDevice device, VkSurfaceKHR surface)
{
    uint32_t count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &count, formats.data());

    for (const auto& f : formats)
    {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            return f;
        }
    }
    return formats.front();
}

VkPresentModeKHR pick_present_mode(VkPhysicalDevice device, VkSurfaceKHR surface)
{
    uint32_t count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &count, nullptr);
    std::vector<VkPresentModeKHR> modes(count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &count, modes.data());
    for (auto m : modes)
    {
        if (m == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            return m;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D pick_extent(const VkSurfaceCapabilitiesKHR& caps, uint32_t width, uint32_t height)
{
    if (caps.currentExtent.width != UINT32_MAX)
    {
        return caps.currentExtent;
    }
    VkExtent2D extent{width, height};
    extent.width  = std::clamp(extent.width,  caps.minImageExtent.width,  caps.maxImageExtent.width);
    extent.height = std::clamp(extent.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    return extent;
}
}

Swapchain::~Swapchain()
{
    shutdown();
}

void Swapchain::initialize(Device& device, VkSurfaceKHR surface, uint32_t width, uint32_t height)
{
    device_ = &device;
    surface_ = surface;
    surface_format_ = pick_surface_format(device.physical_device(), surface);
    present_mode_ = pick_present_mode(device.physical_device(), surface);
    create_swapchain(width, height);
}

void Swapchain::initialize(Device& device, const Surface& surface, uint32_t width, uint32_t height)
{
    if (!surface.is_valid())
    {
        throw std::runtime_error("Swapchain::initialize requires an initialized Surface.");
    }
    if (surface.instance_ != to_handle(device.instance()))
    {
        throw std::runtime_error("Swapchain::initialize requires a Surface created from this Device instance.");
    }

    initialize(device, from_handle<VkSurfaceKHR>(surface.surface_), width, height);
}

void Swapchain::recreate(uint32_t width, uint32_t height)
{
    if (device_ == nullptr)
    {
        throw std::runtime_error("Swapchain::recreate called before initialize.");
    }
    destroy_swapchain();
    create_swapchain(width, height);
}

void Swapchain::create_swapchain(uint32_t width, uint32_t height)
{
    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device_->physical_device(), surface_, &caps);
    extent_ = pick_extent(caps, width, height);

    uint32_t desired = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && desired > caps.maxImageCount)
    {
        desired = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    info.surface = surface_;
    info.minImageCount = desired;
    info.imageFormat = surface_format_.format;
    info.imageColorSpace = surface_format_.colorSpace;
    info.imageExtent = extent_;
    info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.preTransform = caps.currentTransform;
    info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    info.presentMode = present_mode_;
    info.clipped = VK_TRUE;
    info.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(device_->device(), &info, nullptr, &swapchain_) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create VkSwapchainKHR.");
    }

    uint32_t image_count = 0;
    vkGetSwapchainImagesKHR(device_->device(), swapchain_, &image_count, nullptr);
    images_.resize(image_count);
    vkGetSwapchainImagesKHR(device_->device(), swapchain_, &image_count, images_.data());

    image_views_.resize(image_count);
    render_finished_.resize(image_count);
    VkSemaphoreCreateInfo sem_info{};
    sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (uint32_t i = 0; i < image_count; ++i)
    {
        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = images_[i];
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = surface_format_.format;
        view_info.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                 VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
        view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(device_->device(), &view_info, nullptr, &image_views_[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create swapchain image view.");
        }
        if (vkCreateSemaphore(device_->device(), &sem_info, nullptr, &render_finished_[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create per-image render-finished semaphore.");
        }
    }

    rr::core::log()->info(
        "Created swapchain {}x{}, {} images, format {} present mode {}",
        extent_.width, extent_.height, image_count,
        static_cast<int>(surface_format_.format),
        static_cast<int>(present_mode_));
}

void Swapchain::destroy_swapchain()
{
    if (device_ == nullptr || device_->device() == VK_NULL_HANDLE)
    {
        return;
    }
    for (VkImageView view : image_views_)
    {
        if (view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device_->device(), view, nullptr);
        }
    }
    image_views_.clear();
    for (VkSemaphore sem : render_finished_)
    {
        if (sem != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(device_->device(), sem, nullptr);
        }
    }
    render_finished_.clear();
    images_.clear();

    if (swapchain_ != VK_NULL_HANDLE)
    {
        vkDestroySwapchainKHR(device_->device(), swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

void Swapchain::shutdown()
{
    destroy_swapchain();
    device_ = nullptr;
    surface_ = VK_NULL_HANDLE;
    extent_ = {};
}
}
