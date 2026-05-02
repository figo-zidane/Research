#pragma once

namespace rr::rhi
{
class CommandRecorder
{
public:
    CommandRecorder() = default;
    explicit CommandRecorder(void* cmd) : cmd_(cmd) {}

    [[nodiscard]] void* handle() const noexcept { return cmd_; }
    [[nodiscard]] bool is_valid() const noexcept { return cmd_ != nullptr; }

private:
    void* cmd_ = nullptr;
};
} // namespace rr::rhi