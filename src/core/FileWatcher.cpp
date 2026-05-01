#include "core/FileWatcher.h"
#include "core/Log.h"

#include <efsw/efsw.hpp>
#include <mutex>

namespace rr::core
{

// Internal PIMPL implementation that also serves as the efsw listener.
struct FileWatcher::Impl : public efsw::FileWatchListener
{
    efsw::FileWatcher watcher;
    efsw::WatchID     watch_id = 0;

    std::mutex                         mutex;
    std::vector<std::filesystem::path> pending;

    void handleFileAction(efsw::WatchID          /*id*/,
                          const std::string&     dir,
                          const std::string&     filename,
                          efsw::Action           action,
                          std::string            /*old_filename*/) override
    {
        // We only care about file modifications and additions (saves / renames).
        if (action != efsw::Actions::Modified && action != efsw::Actions::Add)
            return;

        auto path = std::filesystem::path(dir) / filename;
        if (path.extension() != ".slang")
            return;

        std::lock_guard<std::mutex> lock(mutex);
        // Deduplicate: don't add the same path twice before poll() clears the list.
        for (const auto& p : pending)
            if (p == path) return;

        pending.push_back(std::move(path));
    }
};

// ── Public interface ──────────────────────────────────────────────────────────

FileWatcher::FileWatcher()  : impl_(std::make_unique<Impl>()) {}
FileWatcher::~FileWatcher() { stop(); }

void FileWatcher::start(const std::filesystem::path& dir)
{
    if (running_) return;

    if (!std::filesystem::exists(dir))
    {
        core::log()->warn("[FileWatcher] directory not found: {}", dir.string());
        return;
    }

    impl_->watch_id = impl_->watcher.addWatch(dir.string(), impl_.get(), /*recursive=*/true);
    impl_->watcher.watch();
    running_ = true;
    core::log()->info("[FileWatcher] watching '{}'", dir.string());
}

void FileWatcher::stop()
{
    if (!running_) return;
    impl_->watcher.removeWatch(impl_->watch_id);
    running_ = false;
}

std::vector<std::filesystem::path> FileWatcher::poll()
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::vector<std::filesystem::path> out;
    out.swap(impl_->pending);
    return out;
}

} // namespace rr::core
