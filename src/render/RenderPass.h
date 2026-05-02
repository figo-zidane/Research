#pragma once

#include "render/FrameContext.h"
#include "rhi/Types.h"

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
        rr::rhi::Format format = rr::rhi::Format::Undefined;
        rr::rhi::Extent2D extent{};
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
    virtual void on_resize(rr::rhi::Extent2D /*new_extent*/) {}
    virtual void execute(FrameContext& frame_context) = 0;

    void set_enabled(bool v)          { enabled_ = v; }
    [[nodiscard]] bool is_enabled() const { return enabled_; }

protected:
    bool enabled_ = true;
};
}
