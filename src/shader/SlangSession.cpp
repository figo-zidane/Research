#include "shader/SlangSession.h"

#include "core/Log.h"

#include <slang.h>
#include <slang-com-ptr.h>

#include <stdexcept>
#include <string>

namespace rr::shader
{

SlangSession::~SlangSession()
{
    shutdown();
}

void SlangSession::initialize(const std::filesystem::path&    shader_include_dir,
                               const std::vector<std::string>& extra_defines)
{
    if (global_session_)
    {
        throw std::runtime_error("SlangSession::initialize called twice.");
    }

    // ── 1. Create global session ──────────────────────────────────────────
    {
        slang::IGlobalSession* gs = nullptr;
        if (SLANG_FAILED(slang::createGlobalSession(&gs)))
        {
            throw std::runtime_error("slang::createGlobalSession failed.");
        }
        global_session_ = gs;
    }

    // ── 2. Configure the per-target ISession ─────────────────────────────
    slang::TargetDesc target_desc{};
    target_desc.format  = SLANG_SPIRV;           // output: SPIR-V
    target_desc.profile = global_session_->findProfile("spirv_1_5");
    // Row-major matrices match GLSL/Vulkan convention.
    target_desc.floatingPointMode = SLANG_FLOATING_POINT_MODE_DEFAULT;
    target_desc.forceGLSLScalarBufferLayout = true;

    // RayQuery (inline ray tracing) needs the spvRayQueryKHR capability, which
    // the bare spirv_1_5 profile does not include.  Declare it explicitly via a
    // compiler option so Slang does not emit warning 41012 ("automatically
    // updated to include these capabilities") for every ray-query entry point.
    slang::CompilerOptionEntry capability_option{};
    const SlangCapabilityID ray_query_cap =
        global_session_->findCapability("spvRayQueryKHR");
    if (ray_query_cap != SLANG_CAPABILITY_UNKNOWN)
    {
        capability_option.name = slang::CompilerOptionName::Capability;
        capability_option.value.kind = slang::CompilerOptionValueKind::Int;
        capability_option.value.intValue0 = static_cast<int32_t>(ray_query_cap);
        target_desc.compilerOptionEntries   = &capability_option;
        target_desc.compilerOptionEntryCount = 1;
    }

    // Preprocessor defines.
    std::vector<slang::PreprocessorMacroDesc> macros;
    macros.push_back({"VK_BINDLESS", "1"});
    for (const auto& def : extra_defines)
    {
        const auto eq = def.find('=');
        if (eq == std::string::npos)
        {
            macros.push_back({def.c_str(), "1"});
        }
        else
        {
            // NOTE: strings are captured by pointer; we keep the source strings alive
            // below via macro_strings.
            macros.push_back({def.c_str(), def.c_str() + eq + 1});
        }
    }
    // Adjust pointers now that the vector is stable.
    std::vector<std::string> macro_strings;
    macro_strings.reserve(extra_defines.size());
    for (size_t i = 1; i < macros.size(); ++i) // 0 is VK_BINDLESS which is a literal
    {
        const std::string& raw = extra_defines[i - 1];
        const auto eq = raw.find('=');
        if (eq != std::string::npos)
        {
            macro_strings.push_back(raw.substr(0, eq));
            macro_strings.push_back(raw.substr(eq + 1));
            macros[i].name  = macro_strings[macro_strings.size() - 2].c_str();
            macros[i].value = macro_strings[macro_strings.size() - 1].c_str();
        }
        else
        {
            macro_strings.push_back(raw);
            macros[i].name  = macro_strings.back().c_str();
            macros[i].value = "1";
        }
    }

    // Include paths.
    const std::string include_path = shader_include_dir.string();
    const char* search_paths[] = {include_path.c_str()};

    slang::SessionDesc session_desc{};
    session_desc.targets             = &target_desc;
    session_desc.targetCount         = 1;
    session_desc.preprocessorMacros = macros.data();
    session_desc.preprocessorMacroCount = static_cast<SlangInt>(macros.size());
    session_desc.searchPaths         = search_paths;
    session_desc.searchPathCount     = 1;
    // Enable matrix layout compatible with HLSL/GLSL row-major convention.
    session_desc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_ROW_MAJOR;

    slang::ISession* sess = nullptr;
    if (SLANG_FAILED(global_session_->createSession(session_desc, &sess)))
    {
        throw std::runtime_error("slang IGlobalSession::createSession failed.");
    }
    session_ = sess;

    rr::core::log()->info("SlangSession: initialized (SPIR-V 1.5 target, include={})",
                          include_path);
}

void SlangSession::shutdown()
{
    if (session_)
    {
        session_->release();
        session_ = nullptr;
    }
    if (global_session_)
    {
        global_session_->release();
        global_session_ = nullptr;
    }
}

} // namespace rr::shader
