#include "rhi/Sampler.h"

#include "rhi/Device.h"
#include "rhi/VulkanTypeCasts.h"

#include <stdexcept>
#include <utility>

namespace rr::rhi
{

Sampler::Sampler(Sampler&& other) noexcept : sampler_(other.sampler_)
{
    other.sampler_ = VK_NULL_HANDLE;
}

Sampler& Sampler::operator=(Sampler&& other) noexcept
{
    if (this != &other)
    {
        sampler_       = other.sampler_;
        other.sampler_ = VK_NULL_HANDLE;
    }
    return *this;
}

void Sampler::create(Device& device, const SamplerDesc& desc)
{
    if (sampler_ != VK_NULL_HANDLE)
    {
        throw std::runtime_error("Sampler::create called on an already-created sampler.");
    }
    const VkSamplerCreateInfo info = to_vk_sampler_create_info(desc);
    if (vkCreateSampler(device.device(), &info, nullptr, &sampler_) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateSampler failed.");
    }
}

void Sampler::destroy(Device& device)
{
    if (sampler_ == VK_NULL_HANDLE)
    {
        return;
    }
    vkDestroySampler(device.device(), sampler_, nullptr);
    sampler_ = VK_NULL_HANDLE;
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
