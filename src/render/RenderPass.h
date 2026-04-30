#pragma once

#include "render/FrameContext.h"

#include <string>
#include <vector>

namespace rr::render
{
class RenderPass
{
public:
    struct ResourceDesc
    {
        enum class Kind
        {
            Texture,
            Buffer
        };

        std::string name;
        Kind kind = Kind::Texture;
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkExtent2D extent{};
        bool persistent = false;
    };

    struct Reflection
    {
        std::vector<ResourceDesc> inputs;
        std::vector<ResourceDesc> outputs;
    };

    virtual ~RenderPass() = default;

    [[nodiscard]] virtual const char* name() const = 0;
    [[nodiscard]] virtual Reflection reflect() const = 0;
    virtual void render_ui() {}
    virtual void execute(FrameContext& frame_context) = 0;
};
}
