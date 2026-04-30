#include "scene/Scene.h"

#include <utility>

namespace rr::scene
{
void Scene::set_name(std::string name)
{
    name_ = std::move(name);
}

const std::string& Scene::name() const noexcept
{
    return name_;
}
}
