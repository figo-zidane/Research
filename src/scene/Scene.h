#pragma once

#include <string>

namespace rr::scene
{
class Scene
{
public:
    void set_name(std::string name);
    [[nodiscard]] const std::string& name() const noexcept;

private:
    std::string name_ = "Untitled";
};
}
