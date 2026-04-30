// imgui_impl_vulkan.h needs to see IMGUI_IMPL_VULKAN_USE_VOLK (set as a
// compile definition on the imgui_pass CMake target) before the Vulkan
// include so it pulls in volk.h instead of vulkan/vulkan.h.  This avoids
// conflicts with the project-wide VK_NO_PROTOTYPES flag.
#include "passes/imgui/imgui_impl_glfw.h"
#include "passes/imgui/imgui_impl_vulkan.h"

#include "passes/imgui/ImGuiPass.h"

#include "render/FrameContext.h"
#include "rhi/Device.h"
#include "rhi/Swapchain.h"
#include "core/Log.h"

#include <imgui.h>

#include <stdexcept>

namespace rr::passes::imgui
{
// ─────────────────────────────────────────────────────────────────────────────
// Destructor
// ─────────────────────────────────────────────────────────────────────────────
ImGuiPass::~ImGuiPass()
{
    // shutdown() must be called explicitly before the device is destroyed.
    // If it hasn't been, we can't do much safely here.
}

// ─────────────────────────────────────────────────────────────────────────────
// initialize
// ─────────────────────────────────────────────────────────────────────────────
void ImGuiPass::initialize(rr::rhi::Device&          device,
                            GLFWwindow*               window,
                            const rr::rhi::Swapchain& swapchain)
{
    color_format_ = swapchain.image_format();
    extent_       = swapchain.extent();

    // ── ImGui context ────────────────────────────────────────────────────
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::StyleColorsDark();

    // ── GLFW backend ─────────────────────────────────────────────────────
    ImGui_ImplGlfw_InitForVulkan(window, /*install_callbacks=*/true);

    // ── Vulkan backend ────────────────────────────────────────────────────
    // We use IMGUI_IMPL_VULKAN_USE_VOLK (compile definition on the target)
    // so the backend includes volk.h and resolves function pointers through
    // volk's global dispatch table — no separate LoadFunctions() call needed.

    // PipelineRenderingCreateInfo for dynamic rendering (Vulkan 1.3 core).
    // pColorAttachmentFormats must remain valid for the lifetime of the backend;
    // we store color_format_ as a member for this purpose.
    VkPipelineRenderingCreateInfoKHR pipeline_rendering_ci{};
    pipeline_rendering_ci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
    pipeline_rendering_ci.colorAttachmentCount    = 1;
    pipeline_rendering_ci.pColorAttachmentFormats = &color_format_;

    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.ApiVersion      = VK_API_VERSION_1_4;
    init_info.Instance        = device.instance();
    init_info.PhysicalDevice  = device.physical_device();
    init_info.Device          = device.device();
    init_info.QueueFamily     = device.graphics_queue_family();
    init_info.Queue           = device.graphics_queue();
    // Let the backend create its own descriptor pool (not the bindless one).
    init_info.DescriptorPoolSize = 32;
    init_info.MinImageCount   = 2;
    init_info.ImageCount      = swapchain.image_count();
    init_info.UseDynamicRendering = true;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo = pipeline_rendering_ci;
    // Silence the "allocation < 1 MiB" best-practice warning from the
    // validation layer.
    init_info.MinAllocationSize = 1024u * 1024u;

    if (!ImGui_ImplVulkan_Init(&init_info))
    {
        throw std::runtime_error("ImGui_ImplVulkan_Init failed.");
    }

    initialized_ = true;
    rr::core::log()->info("[ImGuiPass] Initialized (dynamic rendering, format={})", static_cast<int>(color_format_));
}

// ─────────────────────────────────────────────────────────────────────────────
// shutdown
// ─────────────────────────────────────────────────────────────────────────────
void ImGuiPass::shutdown()
{
    if (!initialized_)
        return;

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    initialized_ = false;
}

// ─────────────────────────────────────────────────────────────────────────────
// new_frame
// ─────────────────────────────────────────────────────────────────────────────
void ImGuiPass::new_frame()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

// ─────────────────────────────────────────────────────────────────────────────
// RenderPass interface
// ─────────────────────────────────────────────────────────────────────────────
rr::render::RenderPass::Reflection ImGuiPass::reflect() const
{
    Reflection r;
    r.outputs.push_back({"swapchain", ResourceDesc::Kind::Texture,
                         color_format_, extent_, /*persistent=*/false});
    return r;
}

void ImGuiPass::on_resize(VkExtent2D new_extent)
{
    extent_ = new_extent;
    ImGui_ImplVulkan_SetMinImageCount(2);
}

void ImGuiPass::execute(rr::render::FrameContext& ctx)
{
    // Finalise the ImGui draw list built during this frame.
    ImGui::Render();

    ImDrawData* draw_data = ImGui::GetDrawData();
    if (draw_data == nullptr || draw_data->TotalVtxCount == 0)
        return;

    // Open a dynamic rendering scope on the current swapchain image.
    // In Phase 3, ImGuiPass is the only render pass, so we CLEAR the image to
    // establish a dark background.  Later (Phase 4+), when previous passes have
    // already written to the swapchain, change loadOp to LOAD.
    VkRenderingAttachmentInfo color_attachment{};
    color_attachment.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color_attachment.imageView   = ctx.swapchain_image_view;
    color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attachment.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.clearValue.color = {{0.05f, 0.10f, 0.15f, 1.0f}};

    VkRenderingInfo rendering{};
    rendering.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering.renderArea.extent    = ctx.swapchain_extent;
    rendering.layerCount           = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments    = &color_attachment;

    vkCmdBeginRendering(ctx.command_buffer, &rendering);
    ImGui_ImplVulkan_RenderDrawData(draw_data, ctx.command_buffer);
    vkCmdEndRendering(ctx.command_buffer);
}
}
