#pragma once

#include <volk.h>

namespace rr::rhi
{
class Device;

// Thin RAII wrapper around VkSampler.
class Sampler
{
public:
    Sampler() = default;
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;
    Sampler(Sampler&& other) noexcept;
    Sampler& operator=(Sampler&& other) noexcept;
    ~Sampler() = default; // call destroy() explicitly

    void create(Device& device, const VkSamplerCreateInfo& info);
    void destroy(Device& device);

    [[nodiscard]] VkSampler handle()   const noexcept { return sampler_; }
    [[nodiscard]] bool      is_valid() const noexcept { return sampler_ != VK_NULL_HANDLE; }

    // Convenience factory: linear repeat, no anisotropy.
    [[nodiscard]] static VkSamplerCreateInfo linear_repeat() noexcept;
    // Convenience factory: nearest clamp-to-edge (for render targets).
    [[nodiscard]] static VkSamplerCreateInfo nearest_clamp() noexcept;

private:
    VkSampler sampler_ = VK_NULL_HANDLE;
};

} // namespace rr::rhi
