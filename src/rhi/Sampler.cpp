#include "rhi/Sampler.h"

#include "rhi/Device.h"
#include "rhi/internal/VulkanAccess.h"
#include "rhi/internal/VulkanTypeCasts.h"

#include <stdexcept>
#include <utility>

namespace rr::rhi
{

Sampler::Sampler(Sampler&& other) noexcept : sampler_(other.sampler_)
{
    other.sampler_ = 0;
}

Sampler& Sampler::operator=(Sampler&& other) noexcept
{
    if (this != &other)
    {
        sampler_       = other.sampler_;
        other.sampler_ = 0;
    }
    return *this;
}

void Sampler::create(Device& device, const SamplerDesc& desc)
{
    if (sampler_ != 0)
    {
        throw std::runtime_error("Sampler::create called on an already-created sampler.");
    }
    const VkSamplerCreateInfo info = to_vk_sampler_create_info(desc);
    VkSampler raw_sampler = VK_NULL_HANDLE;
    if (vkCreateSampler(vulkan::get_device(device), &info, nullptr, &raw_sampler) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateSampler failed.");
    }
    sampler_ = to_handle(raw_sampler);
}

void Sampler::destroy(Device& device)
{
    if (sampler_ == 0)
    {
        return;
    }
    vkDestroySampler(vulkan::get_device(device), from_handle<VkSampler>(sampler_), nullptr);
    sampler_ = 0;
}

SamplerDesc Sampler::linear_repeat() noexcept
{
    return SamplerDesc{};
}

SamplerDesc Sampler::nearest_clamp() noexcept
{
    SamplerDesc info{};
    info.mag_filter     = SamplerFilter::Nearest;
    info.min_filter     = SamplerFilter::Nearest;
    info.mipmap_mode    = SamplerMipmapMode::Nearest;
    info.address_mode_u = SamplerAddressMode::ClampToEdge;
    info.address_mode_v = SamplerAddressMode::ClampToEdge;
    info.address_mode_w = SamplerAddressMode::ClampToEdge;
    return info;
}

} // namespace rr::rhi
