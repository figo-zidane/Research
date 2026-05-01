#pragma once

#include <filesystem>
#include <memory>
#include <vector>

namespace rr::core
{

// Thread-safe shader file watcher backed by efsw.
// Recursively watches a directory for .slang file modifications/creations.
// The efsw callback fires on a background thread; poll() safely transfers
// the pending list to the caller's thread.
class FileWatcher
{
public:
    FileWatcher();
    ~FileWatcher();
    FileWatcher(const FileWatcher&) = delete;
    FileWatcher& operator=(const FileWatcher&) = delete;

    // Start watching dir recursively. No-op if already running.
    void start(const std::filesystem::path& dir);

    // Stop watching.
    void stop();

    // Returns all changed .slang paths since the last call and clears the list.
    // Thread-safe; safe to call from any thread.
    [[nodiscard]] std::vector<std::filesystem::path> poll();

    [[nodiscard]] bool is_running() const noexcept { return running_; }

private:
    // PIMPL hides the efsw header from consumers.
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool running_ = false;
};

} // namespace rr::core
