#include "shader/ShaderReflection.h"

#include "core/Log.h"

#include <slang.h>

namespace rr::shader
{

namespace
{
ShaderStage slang_stage_to_shader_stage(SlangStage stage)
{
    switch (stage)
    {
    case SLANG_STAGE_VERTEX:               return ShaderStage::Vertex;
    case SLANG_STAGE_FRAGMENT:             return ShaderStage::Fragment;
    case SLANG_STAGE_COMPUTE:              return ShaderStage::Compute;
    case SLANG_STAGE_RAY_GENERATION:       return ShaderStage::RayGeneration;
    case SLANG_STAGE_CLOSEST_HIT:          return ShaderStage::ClosestHit;
    case SLANG_STAGE_MISS:                 return ShaderStage::Miss;
    case SLANG_STAGE_ANY_HIT:              return ShaderStage::AnyHit;
    case SLANG_STAGE_INTERSECTION:         return ShaderStage::Intersection;
    default:                               return ShaderStage::Unknown;
    }
}
} // namespace

ShaderReflection::ShaderReflection(void* layout_ptr)
    : layout_(layout_ptr)
{
    if (!layout_ptr)
    {
        return;
    }
    auto* layout = static_cast<slang::ProgramLayout*>(layout_ptr);

    // ── Entry points ─────────────────────────────────────────────────────
    const SlangInt ep_count = layout->getEntryPointCount();
    entry_points_.reserve(static_cast<size_t>(ep_count));
    for (SlangInt i = 0; i < ep_count; ++i)
    {
        slang::EntryPointReflection* ep = layout->getEntryPointByIndex(i);
        if (!ep) { continue; }

        EntryPointInfo info{};
        info.index  = static_cast<uint32_t>(i);
        info.name   = ep->getName() ? ep->getName() : "";
        info.stage  = slang_stage_to_shader_stage(ep->getStage());
        entry_points_.push_back(std::move(info));
    }

    // ── Push-constant size ────────────────────────────────────────────────
    // Walk the global-scope parameter blocks looking for a push-constant range.
    // Slang represents the push-constant block as a parameter block whose
    // VarLayout has a ParameterBlockTypeLayout whose element type is a struct.
    // A simpler way: iterate the global parameter layout's fields and sum up the
    // size of any field tagged as a push-constant.
    //
    // For our renderer we use a single struct as push constants per pass,
    // declared in the shader as:
    //   [[vk::push_constant]] struct PushConstants { ... } g_push;
    //
    // Slang surfaces this as a uniform parameter of kind PUSH_CONSTANT.
    const SlangInt param_count = layout->getParameterCount();
    for (SlangInt i = 0; i < param_count; ++i)
    {
        slang::VariableLayoutReflection* var = layout->getParameterByIndex(i);
        if (!var) { continue; }

        slang::TypeLayoutReflection* type_layout = var->getTypeLayout();
        if (!type_layout) { continue; }

        // Check binding category: if the variable occupies push-constant space,
        // its type layout will report VK_PUSH_CONSTANT as the category.
        const slang::BindingType binding = type_layout->getBindingRangeType(0);
        if (binding == slang::BindingType::PushConstant)
        {
            push_constant_size_ = static_cast<uint32_t>(type_layout->getSize());
            break;
        }

        // Alternative: check by parameter category
        if (var->getCategory() == slang::ParameterCategory::PushConstantBuffer)
        {
            push_constant_size_ = static_cast<uint32_t>(type_layout->getSize());
            break;
        }
    }

    rr::core::log()->debug(
        "ShaderReflection: {} entry points, push_constant_size={}",
        entry_points_.size(), push_constant_size_);
}

} // namespace rr::shader
