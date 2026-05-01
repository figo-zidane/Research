#include "shader/ShaderModule.h"

#include "core/Log.h"

#include <slang.h>
#include <slang-com-ptr.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace rr::shader
{

namespace
{
// Log Slang diagnostic output (if any) and return true if it indicates an error.
bool handle_diagnostics(ISlangBlob* diag_blob, const char* context)
{
    if (!diag_blob)
    {
        return false;
    }
    const char* msg = static_cast<const char*>(diag_blob->getBufferPointer());
    if (!msg || msg[0] == '\0')
    {
        return false;
    }
    // Slang errors contain "error:" in the message; warnings do not.
    const std::string_view sv(msg);
    if (sv.find("error:") != std::string_view::npos ||
        sv.find("Error:") != std::string_view::npos)
    {
        rr::core::log()->error("[slang] {} diagnostics:\n{}", context, msg);
        return true; // is error
    }
    rr::core::log()->warn("[slang] {} diagnostics:\n{}", context, msg);
    return false;
}
} // namespace

ShaderModule::~ShaderModule()
{
    reset();
}

void ShaderModule::compile(SlangSession&                      session,
                            const std::filesystem::path&       source_path,
                            const std::vector<EntryPointDesc>& entry_points)
{
    reset();

    if (!session.is_valid())
    {
        throw std::runtime_error("ShaderModule::compile: SlangSession is not initialized.");
    }

    slang::ISession* sess = session.session();

    // ── 1. Load the module from file ─────────────────────────────────────
    Slang::ComPtr<ISlangBlob> diag;
    const std::string path_str = source_path.string();

    slang::IModule* slang_module = nullptr;
    slang_module = sess->loadModule(path_str.c_str(), diag.writeRef());
    if (handle_diagnostics(diag.get(), path_str.c_str()) || !slang_module)
    {
        throw std::runtime_error("Slang: failed to load module: " + path_str);
    }
    module_ = slang_module;

    // ── 2. Collect entry points ──────────────────────────────────────────
    std::vector<slang::IComponentType*> components;
    components.push_back(slang_module);

    std::vector<Slang::ComPtr<slang::IEntryPoint>> ep_ptrs;
    ep_ptrs.reserve(entry_points.size());

    for (const auto& ep : entry_points)
    {
        diag.setNull();
        slang::IEntryPoint* entry = nullptr;
        if (SLANG_FAILED(slang_module->findEntryPointByName(ep.name.c_str(), &entry)) || !entry)
        {
            throw std::runtime_error("Slang: entry point '" + ep.name +
                                     "' not found in " + path_str);
        }
        ep_ptrs.emplace_back(entry);
        components.push_back(entry);
    }

    // ── 3. Link the program ──────────────────────────────────────────────
    diag.setNull();
    slang::IComponentType* linked_program = nullptr;
    if (SLANG_FAILED(sess->createCompositeComponentType(
            components.data(),
            static_cast<SlangInt>(components.size()),
            &linked_program,
            diag.writeRef())) ||
        handle_diagnostics(diag.get(), "link") || !linked_program)
    {
        throw std::runtime_error("Slang: program link failed for " + path_str);
    }
    program_ = linked_program;

    // ── 4. Obtain reflection layout ──────────────────────────────────────
    layout_ = linked_program->getLayout();
    if (!layout_)
    {
        throw std::runtime_error("Slang: getLayout() returned null for " + path_str);
    }

    // ── 5. Emit SPIR-V per entry point ──────────────────────────────────
    spv_codes_.resize(entry_points.size());
    for (size_t i = 0; i < entry_points.size(); ++i)
    {
        diag.setNull();
        Slang::ComPtr<ISlangBlob> spv_blob;
        if (SLANG_FAILED(linked_program->getEntryPointCode(
                static_cast<SlangInt>(i),
                0, // targetIndex
                spv_blob.writeRef(),
                diag.writeRef())))
        {
            handle_diagnostics(diag.get(), "emit SPIR-V");
            throw std::runtime_error("Slang: SPIR-V emission failed for entry point '" +
                                     entry_points[i].name + "' in " + path_str);
        }
        handle_diagnostics(diag.get(), "emit SPIR-V");

        const uint8_t* data = static_cast<const uint8_t*>(spv_blob->getBufferPointer());
        const size_t   size = spv_blob->getBufferSize();
        spv_codes_[i].assign(data, data + size);
    }

    entry_points_ = entry_points;
    rr::core::log()->info("ShaderModule: compiled '{}' ({} entry points, {} SPIR-V bytes)",
                          path_str,
                          entry_points.size(),
                          spv_codes_.empty() ? 0 : spv_codes_[0].size());
}

void ShaderModule::reset()
{
    spv_codes_.clear();
    layout_ = nullptr;
    if (program_)
    {
        static_cast<slang::IComponentType*>(program_)->release();
        program_ = nullptr;
    }
    if (module_)
    {
        static_cast<slang::IModule*>(module_)->release();
        module_ = nullptr;
    }
    entry_points_.clear();
    spv_codes_.clear();
}

const std::vector<uint8_t>& ShaderModule::spv_code(uint32_t entry_index) const
{
    if (entry_index >= spv_codes_.size())
    {
        throw std::out_of_range("ShaderModule::spv_code: entry_index out of range");
    }
    return spv_codes_[entry_index];
}

} // namespace rr::shader
