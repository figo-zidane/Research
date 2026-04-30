#pragma once

#include <cstdint>
#include <string>
#include <vector>

// slang.h is pulled only in the .cpp to keep compile times fast.

namespace rr::shader
{

// Entry-point type classification, matching Slang's stage enum but with only
// the values we actively use in this renderer.
enum class ShaderStage
{
    Unknown      = 0,
    Vertex,
    Fragment,
    Compute,
    RayGeneration,
    ClosestHit,
    Miss,
    AnyHit,
    Intersection,
};

struct EntryPointInfo
{
    std::string name;
    ShaderStage stage = ShaderStage::Unknown;
    uint32_t    index = 0; // index in the ProgramLayout
};

// ShaderReflection provides lightweight access to the parts of a Slang
// ProgramLayout that the pipeline creation code cares about:
//
//   - push-constant block size (bytes)
//   - list of entry-point names and their shader stages
//
// The ProgramLayout pointer is non-owning; it must remain valid for the
// lifetime of this object.
class ShaderReflection
{
public:
    ShaderReflection() = default;

    // Build reflection data from the layout produced by ShaderModule::compile().
    // layout must be a slang::ProgramLayout* (passed as void* to avoid header
    // pollution from slang.h).
    explicit ShaderReflection(void* layout);

    [[nodiscard]] uint32_t                         push_constant_size() const noexcept { return push_constant_size_; }
    [[nodiscard]] const std::vector<EntryPointInfo>& entry_points()      const noexcept { return entry_points_; }
    [[nodiscard]] bool                             is_valid()            const noexcept { return layout_ != nullptr; }

private:
    void*                      layout_            = nullptr;
    uint32_t                   push_constant_size_ = 0;
    std::vector<EntryPointInfo> entry_points_;
};

} // namespace rr::shader
