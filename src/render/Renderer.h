#pragma once

#include "render/FrameContext.h"
#include "render/RenderGraph.h"
#include "render/RenderPass.h"

#include <memory>
#include <utility>

namespace rr::render
{
class Renderer
{
public:
    // Construct a pass in-place at the back of the linear pass list.
    // Returns a raw (non-owning) pointer for callers that need to call
    // pass-specific methods later (e.g. Application → ImGuiPass::new_frame).
    template<typename T, typename... Args>
    T* add_pass(Args&&... args)
    {
        auto pass = std::make_unique<T>(std::forward<Args>(args)...);
        T* ptr = pass.get();
        graph_.add_pass(std::move(pass));
        return ptr;
    }

    // Notify all passes of a framebuffer resize.
    void on_resize(rr::rhi::Extent2D extent) { graph_.on_resize(extent); }

    // Execute all passes in order.
    void render(FrameContext& frame_context) { graph_.execute(frame_context); }

    [[nodiscard]] RenderGraph&       graph() noexcept       { return graph_; }
    [[nodiscard]] const RenderGraph& graph() const noexcept { return graph_; }

private:
    RenderGraph graph_;
};
}
