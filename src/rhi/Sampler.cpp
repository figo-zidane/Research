#include "rhi/Sampler.h"

#include "rhi/Device.h"

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

void Sampler::create(Device& device, const VkSamplerCreateInfo& info)
{
    if (sampler_ != VK_NULL_HANDLE)
    {
        throw std::runtime_error("Sampler::create called on an already-created sampler.");
    }
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

VkSamplerCreateInfo Sampler::linear_repeat() noexcept
{
    VkSamplerCreateInfo info{};
    info.sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.magFilter               = VK_FILTER_LINEAR;
    info.minFilter               = VK_FILTER_LINEAR;
    info.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    info.addressModeU            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.addressModeV            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.addressModeW            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.mipLodBias              = 0.0f;
    info.anisotropyEnable        = VK_FALSE;
    info.compareEnable           = VK_FALSE;
    info.minLod                  = 0.0f;
    info.maxLod                  = VK_LOD_CLAMP_NONE;
    info.borderColor             = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    info.unnormalizedCoordinates = VK_FALSE;
    return info;
}

VkSamplerCreateInfo Sampler::nearest_clamp() noexcept
{
    VkSamplerCreateInfo info{};
    info.sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.magFilter               = VK_FILTER_NEAREST;
    info.minFilter               = VK_FILTER_NEAREST;
    info.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    info.addressModeU            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.addressModeV            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.addressModeW            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.mipLodBias              = 0.0f;
    info.anisotropyEnable        = VK_FALSE;
    info.compareEnable           = VK_FALSE;
    info.minLod                  = 0.0f;
    info.maxLod                  = VK_LOD_CLAMP_NONE;
    info.borderColor             = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    info.unnormalizedCoordinates = VK_FALSE;
    return info;
}

} // namespace rr::rhi
