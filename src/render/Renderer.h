#pragma once

#include "render/FrameContext.h"
#include "render/RenderPass.h"

#include <memory>
#include <vector>

namespace rr::render
{
class Renderer
{
public:
    void add_pass(std::unique_ptr<RenderPass> pass);
    void render(FrameContext& frame_context);

private:
    std::vector<std::unique_ptr<RenderPass>> passes_;
};
}
