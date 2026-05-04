#pragma once

#include "rhi/SamplerDesc.h"

namespace rr::rhi
{
class Device;

// Thin RAII wrapper around a sampler object.
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

    [[nodiscard]] SamplerHandle handle() const noexcept { return sampler_; }
    [[nodiscard]] bool          is_valid() const noexcept { return sampler_ != 0; }

    // Convenience factory: linear repeat, no anisotropy.
    [[nodiscard]] static SamplerDesc linear_repeat() noexcept;
    // Convenience factory: nearest clamp-to-edge (for render targets).
    [[nodiscard]] static SamplerDesc nearest_clamp() noexcept;

private:
    SamplerHandle sampler_ = 0;
};

} // namespace rr::rhi
