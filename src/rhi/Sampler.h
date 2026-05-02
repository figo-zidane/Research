#pragma once

#include "rhi/SamplerDesc.h"

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

    void create(Device& device, const SamplerDesc& desc);
    void destroy(Device& device);

    [[nodiscard]] SamplerHandle handle() const noexcept { return to_handle(sampler_); }
    [[nodiscard]] bool          is_valid() const noexcept { return sampler_ != VK_NULL_HANDLE; }

    // Convenience factory: linear repeat, no anisotropy.
    [[nodiscard]] static SamplerDesc linear_repeat() noexcept;
    // Convenience factory: nearest clamp-to-edge (for render targets).
    [[nodiscard]] static SamplerDesc nearest_clamp() noexcept;

private:
    VkSampler sampler_ = VK_NULL_HANDLE;
};

} // namespace rr::rhi
