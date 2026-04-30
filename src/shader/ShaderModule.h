#pragma once

#include "shader/ShaderReflection.h"
#include "shader/SlangSession.h"

#include <filesystem>
#include <string>
#include <vector>

namespace rr::shader
{

// ShaderModule wraps a compiled Slang module + linked program.
// Usage:
//   ShaderModule mod;
//   mod.compile(session, "assets/shaders/passes/tonemap/tonemap.slang",
//               {"cs_main"});
//   auto spv = mod.spv_code();          // raw SPIR-V bytes
//   auto layout = mod.program_layout(); // for reflection
class ShaderModule
{
public:
    // Entry-point descriptor supplied by the caller.
    struct EntryPointDesc
    {
        std::string  name;          // e.g. "cs_main", "vs_main", "raygen_main"
        ShaderStage  stage = ShaderStage::Unknown;
    };

    ShaderModule() = default;
    ShaderModule(const ShaderModule&) = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;
    ~ShaderModule();

    // Compile the shader.  Throws on error; on warning the diagnostic is
    // logged at WARN level and compilation proceeds.
    void compile(SlangSession&                         session,
                 const std::filesystem::path&          source_path,
                 const std::vector<EntryPointDesc>&    entry_points);

    // Release Slang objects and free the SPIR-V blob.
    void reset();

    // Raw SPIR-V bytes for the linked program.
    [[nodiscard]] const std::vector<uint8_t>& spv_code()      const noexcept { return spv_code_; }
    // Returns a slang::ProgramLayout* cast to void* to avoid header pollution.
    [[nodiscard]] void*                        program_layout() const noexcept { return layout_; }
    [[nodiscard]] size_t                       entry_point_count() const noexcept { return entry_points_.size(); }
    [[nodiscard]] bool                         is_valid()       const noexcept { return !spv_code_.empty(); }

private:
    void* module_  = nullptr; // slang::IModule*
    void* program_ = nullptr; // slang::IComponentType*
    void* layout_  = nullptr; // slang::ProgramLayout*

    std::vector<EntryPointDesc> entry_points_;
    std::vector<uint8_t>        spv_code_;
};

} // namespace rr::shader
