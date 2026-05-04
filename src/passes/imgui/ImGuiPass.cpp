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
static_assert(sizeof(rr::rhi::Format) == sizeof(VkFormat));

namespace
{
void initialize_vulkan_backend(rr::rhi::Device& device,
                               const rr::rhi::Swapchain& swapchain,
                               const rr::rhi::Format& color_format)
{
    VkPipelineRenderingCreateInfoKHR pipeline_rendering_ci{};
    pipeline_rendering_ci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
    pipeline_rendering_ci.colorAttachmentCount    = 1;
    pipeline_rendering_ci.pColorAttachmentFormats = reinterpret_cast<const VkFormat*>(&color_format);

    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.ApiVersion      = VK_API_VERSION_1_4;
    init_info.Instance        = device.instance();
    init_info.PhysicalDevice  = device.physical_device();
    init_info.Device          = device.device();
    init_info.QueueFamily     = device.graphics_queue_family();
    init_info.Queue           = device.graphics_queue();
    init_info.DescriptorPoolSize = 32;
    init_info.MinImageCount   = 2;
    init_info.ImageCount      = swapchain.image_count();
    init_info.UseDynamicRendering = true;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo = pipeline_rendering_ci;
    init_info.MinAllocationSize = 1024u * 1024u;

    if (!ImGui_ImplVulkan_Init(&init_info))
        throw std::runtime_error("ImGui_ImplVulkan_Init failed.");
}
} // namespace

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
    extent_       = {swapchain.extent().width, swapchain.extent().height};

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
    initialize_vulkan_backend(device, swapchain, color_format_);

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

void ImGuiPass::on_resize(rr::rhi::Extent2D new_extent)
{
    extent_ = new_extent;
}

void ImGuiPass::execute(rr::render::FrameContext& ctx)
{
    // Finalise the ImGui draw list built during this frame.
    ImGui::Render();

    ImDrawData* draw_data = ImGui::GetDrawData();
    if (draw_data == nullptr || draw_data->TotalVtxCount == 0)
        return;

    // Open a dynamic rendering scope on the current swapchain image.
    // TonemapPass has already written the rendered scene to the swapchain;
    // use LOAD so ImGui is composited on top of the existing contents.
    const rr::rhi::ColorAttachment color_attachment{
        .image = nullptr,
        .image_view = ctx.swapchain_image_view,
        .layout = rr::rhi::ImageLayout::ColorAttachment,
        .load_op = rr::rhi::LoadOp::Load,
        .store_op = rr::rhi::StoreOp::Store,
    };
    const rr::rhi::RenderingInfo rendering{
        .area = ctx.swapchain_extent,
        .layer_count = 1,
        .color_attachments = {&color_attachment, 1},
        .depth_attachment = nullptr,
    };

    const rr::rhi::CommandRecorder recorder = ctx.command_recorder;
    VkCommandBuffer cmd = static_cast<VkCommandBuffer>(recorder.handle());
    recorder.begin_rendering(rendering);
    ImGui_ImplVulkan_RenderDrawData(draw_data, cmd);
    recorder.end_rendering();
}
}
