#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace rr::rhi { class Device; }
namespace rr::core { class FileWatcher; }

namespace rr::shader
{

// HotReload watches a shader directory for .slang file changes and triggers
// per-pass recompilation callbacks.
//
// Lifecycle:
//   1. initialize(shader_dir) — start the file watcher
//   2. register_shader(path, cb) — associate a source file with a recompile fn
//   3. pump(device) per frame — process pending file events; calls
//      vkDeviceWaitIdle before any recompile so old pipelines are idle
//   4. shutdown() — stop the watcher
//
// When ANY .slang file in the watched directory changes, ALL registered
// callbacks are invoked.  This is conservative but avoids include-tracking.
class HotReload
{
public:
    // Returns true on success; on false the old pipeline is kept.
    using RecompileCallback = std::function<bool()>;

    HotReload();
    ~HotReload();
    HotReload(const HotReload&) = delete;
    HotReload& operator=(const HotReload&) = delete;

    void initialize(const std::filesystem::path& shader_dir);
    void shutdown();

    // Register a shader source file and the callback to invoke on change.
    void register_shader(const std::filesystem::path& path, RecompileCallback cb);

    // Per-frame: poll for file events, run recompiles.
    // Returns true if at least one reload was attempted.
    bool pump(rr::rhi::Device& device);

    // ImGui panel: shows last reload result and any error message.
    void render_ui() const;

    [[nodiscard]] bool              has_error()          const noexcept { return !last_error_.empty(); }
    [[nodiscard]] const std::string& last_error()         const noexcept { return last_error_; }
    [[nodiscard]] const std::string& last_reloaded_file() const noexcept { return last_reloaded_file_; }

private:
    struct Entry
    {
        std::filesystem::path path;
        RecompileCallback     callback;
    };

    std::unique_ptr<rr::core::FileWatcher> watcher_;
    std::vector<Entry>                     entries_;
    std::string                            last_error_;
    std::string                            last_reloaded_file_;
    bool                                   initialized_ = false;
};

} // namespace rr::shader
