#include "render/Renderer.h"

#include "render/RenderPass.h"

#include <utility>

namespace rr::render
{
void Renderer::add_pass(std::unique_ptr<RenderPass> pass)
{
    passes_.push_back(std::move(pass));
}

void Renderer::render(FrameContext& frame_context)
{
    for (const auto& pass : passes_)
    {
        pass->execute(frame_context);
    }
}
}
