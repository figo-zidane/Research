#pragma once

#include "render/RenderPass.h"

#include <memory>
#include <vector>

namespace rr::render
{
// RenderGraph holds the ordered list of RenderPasses and drives their
// execution.
// NOTE: Transient off-screen resource allocation will be wired in
// future phase; currently only the swapchain image is used (injected via
// FrameContext) so the graph itself needs no resource management yet.
class RenderGraph
{
public:
    // Append a pass to the end of the linear pass list.
    void add_pass(std::unique_ptr<RenderPass> pass);

    // Execute all passes in insertion order.
    void execute(FrameContext& frame_context);

    // Propagate a framebuffer resize to all passes.
    void on_resize(rr::rhi::Extent2D extent);

    // Read-only access to the pass list (e.g. for EditorUI to iterate renderUI).
    [[nodiscard]] const std::vector<std::unique_ptr<RenderPass>>& passes() const noexcept
    {
        return passes_;
    }

private:
    std::vector<std::unique_ptr<RenderPass>> passes_;
};
}
