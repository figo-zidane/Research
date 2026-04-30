#pragma once

#include <filesystem>
#include <string>
#include <vector>

// Slang headers — only pulled into cpp files that need the full API.
// Here we use forward declarations and an opaque handle to keep compile times
// manageable in headers that only use SlangSession by reference.
namespace slang
{
struct IGlobalSession;
struct ISession;
} // namespace slang

namespace rr::shader
{

// SlangSession holds one IGlobalSession (process-wide) and one ISession
// configured to target SPIR-V.  All ShaderModule instances share the same
// SlangSession; creating a SlangSession is expensive (loads the Slang DLL's
// core module), so it should be constructed once at startup.
class SlangSession
{
public:
    SlangSession() = default;
    SlangSession(const SlangSession&) = delete;
    SlangSession& operator=(const SlangSession&) = delete;
    ~SlangSession();

    // Initialize: pass extra shader include directories beyond
    // assets/shaders/include, which is always added automatically.
    // @param shader_include_dir  main include directory (assets/shaders/include)
    // @param extra_defines       optional list of "MACRO=VALUE" define strings
    void initialize(const std::filesystem::path& shader_include_dir,
                    const std::vector<std::string>& extra_defines = {});

    void shutdown();

    [[nodiscard]] slang::IGlobalSession* global_session() const noexcept { return global_session_; }
    [[nodiscard]] slang::ISession*       session()        const noexcept { return session_; }
    [[nodiscard]] bool                   is_valid()       const noexcept { return session_ != nullptr; }

private:
    slang::IGlobalSession* global_session_ = nullptr;
    slang::ISession*       session_        = nullptr;
};

} // namespace rr::shader
