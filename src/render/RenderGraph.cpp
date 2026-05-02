#include "render/RenderGraph.h"

#include "render/FrameContext.h"

#include <utility>

namespace rr::render
{
void RenderGraph::add_pass(std::unique_ptr<RenderPass> pass)
{
    passes_.push_back(std::move(pass));
}

void RenderGraph::execute(FrameContext& frame_context)
{
    for (const auto& pass : passes_)
    {
        if (pass->is_enabled())
            pass->execute(frame_context);
    }
}

void RenderGraph::on_resize(rr::rhi::Extent2D extent)
{
    for (const auto& pass : passes_)
    {
        pass->on_resize(extent);
    }
}
}
