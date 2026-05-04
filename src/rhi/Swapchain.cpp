#include "rhi/Swapchain.h"

#include "core/Log.h"
#include "rhi/Device.h"
#include "rhi/Image.h"
#include "rhi/internal/VulkanAccess.h"
#include "rhi/Surface.h"
#include "rhi/Types.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace rr::rhi
{
namespace
{
[[nodiscard]] VkSurfaceKHR as_vk_surface(SurfaceHandle handle)
{
    return from_opaque_handle<VkSurfaceKHR>(handle);
}

[[nodiscard]] VkSwapchainKHR as_vk_swapchain(SwapchainHandle handle)
{
    return from_opaque_handle<VkSwapchainKHR>(handle);
}

[[nodiscard]] VkSemaphore as_vk_semaphore(SemaphoreHandle handle)
{
    return from_opaque_handle<VkSemaphore>(handle);
}

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

uint32_t Swapchain::image_count() const noexcept
{
    return static_cast<uint32_t>(images_.size());
}

const Image& Swapchain::image(uint32_t index) const
{
    return images_[index];
}

SemaphoreHandle Swapchain::render_finished(uint32_t image_index) const
{
    return render_finished_[image_index];
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

    device_ = &device;
    surface_ = surface.surface_;

    const VkSurfaceKHR vk_surface = as_vk_surface(surface_);
    const VkSurfaceFormatKHR surface_format = pick_surface_format(vulkan::get_physical_device(device), vk_surface);
    image_format_ = static_cast<Format>(surface_format.format);
    color_space_ = static_cast<uint32_t>(surface_format.colorSpace);
    present_mode_ = static_cast<uint32_t>(pick_present_mode(vulkan::get_physical_device(device), vk_surface));
    create_swapchain(width, height);
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
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vulkan::get_physical_device(*device_), as_vk_surface(surface_), &caps);
    const VkExtent2D vk_extent = pick_extent(caps, width, height);
    extent_ = {vk_extent.width, vk_extent.height};

    uint32_t desired = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && desired > caps.maxImageCount)
    {
        desired = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    info.surface = as_vk_surface(surface_);
    info.minImageCount = desired;
    info.imageFormat = static_cast<VkFormat>(image_format_);
    info.imageColorSpace = static_cast<VkColorSpaceKHR>(color_space_);
    info.imageExtent = vk_extent;
    info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.preTransform = caps.currentTransform;
    info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    info.presentMode = static_cast<VkPresentModeKHR>(present_mode_);
    info.clipped = VK_TRUE;
    info.oldSwapchain = VK_NULL_HANDLE;

    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    if (vkCreateSwapchainKHR(vulkan::get_device(*device_), &info, nullptr, &swapchain) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create VkSwapchainKHR.");
    }
    swapchain_ = to_opaque_handle<SwapchainHandle>(swapchain);

    uint32_t image_count = 0;
    vkGetSwapchainImagesKHR(vulkan::get_device(*device_), swapchain, &image_count, nullptr);
    std::vector<VkImage> vk_images(image_count);
    vkGetSwapchainImagesKHR(vulkan::get_device(*device_), swapchain, &image_count, vk_images.data());

    images_.clear();
    images_.resize(image_count);
    render_finished_.resize(image_count);
    VkSemaphoreCreateInfo sem_info{};
    sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (uint32_t i = 0; i < image_count; ++i)
    {
        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = vk_images[i];
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = static_cast<VkFormat>(image_format_);
        view_info.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                 VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
        view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageView image_view = VK_NULL_HANDLE;
        if (vkCreateImageView(vulkan::get_device(*device_), &view_info, nullptr, &image_view) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create swapchain image view.");
        }
        images_[i].attach_external(
            to_handle(vk_images[i]),
            to_handle(image_view),
            image_format_,
            extent_);

        VkSemaphore render_finished = VK_NULL_HANDLE;
        if (vkCreateSemaphore(vulkan::get_device(*device_), &sem_info, nullptr, &render_finished) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create per-image render-finished semaphore.");
        }
        render_finished_[i] = to_opaque_handle<SemaphoreHandle>(render_finished);
    }

    rr::core::log()->info(
        "Created swapchain {}x{}, {} images, format {} present mode {}",
        extent_.width, extent_.height, image_count,
        static_cast<int>(image_format_),
        static_cast<int>(present_mode_));
}

void Swapchain::destroy_swapchain()
{
    if (device_ == nullptr || device_->device() == nullptr)
    {
        return;
    }
    for (auto& image : images_)
    {
        image.destroy(*device_);
    }
    images_.clear();
    for (SemaphoreHandle sem : render_finished_)
    {
        if (sem != nullptr)
        {
            vkDestroySemaphore(vulkan::get_device(*device_), as_vk_semaphore(sem), nullptr);
        }
    }
    render_finished_.clear();

    if (swapchain_ != nullptr)
    {
        vkDestroySwapchainKHR(vulkan::get_device(*device_), as_vk_swapchain(swapchain_), nullptr);
        swapchain_ = nullptr;
    }
}

void Swapchain::shutdown()
{
    destroy_swapchain();
    device_ = nullptr;
    surface_ = nullptr;
    swapchain_ = nullptr;
    image_format_ = Format::Undefined;
    color_space_ = 0;
    present_mode_ = 0;
    extent_ = {};
}
}
