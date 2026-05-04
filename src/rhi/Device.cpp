#include "rhi/Device.h"

#include "core/Log.h"
#include "rhi/PlatformInternal.h"
#include "rhi/Surface.h"
#include "rhi/Types.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <vector>

// VMA declarations (implementation lives in vma_impl.cpp)
#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <volk.h>
#include <vma/vk_mem_alloc.h>

namespace rr::rhi
{
namespace
{
constexpr std::array<const char*, 1> kValidationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

constexpr std::array<const char*, 6> kRequiredDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_RAY_QUERY_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    "VK_EXT_descriptor_heap"
};

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void* /*user_data*/)
{
    auto& logger = *rr::core::log();
    const char* msg = callback_data && callback_data->pMessage ? callback_data->pMessage : "(no message)";
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    {
        logger.error("[vk] {}", msg);
    }
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    {
        logger.warn("[vk] {}", msg);
    }
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
    {
        logger.info("[vk] {}", msg);
    }
    else
    {
        logger.debug("[vk] {}", msg);
    }
    return VK_FALSE;
}

std::vector<VkExtensionProperties> enumerate_device_extensions(VkPhysicalDevice device)
{
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> available(count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, available.data());
    return available;
}

bool extension_present(const std::vector<VkExtensionProperties>& available, const char* name)
{
    return std::any_of(available.begin(), available.end(), [name](const VkExtensionProperties& p) {
        return std::string_view(p.extensionName) == name;
    });
}

bool device_supports_extensions(const std::vector<VkExtensionProperties>& available,
                                const std::vector<const char*>& required)
{
    for (const char* name : required)
    {
        if (!extension_present(available, name))
        {
            return false;
        }
    }
    return true;
}

void append_unique(std::vector<const char*>& extensions, const char* name)
{
    const bool exists = std::find_if(extensions.begin(), extensions.end(), [name](const char* candidate) {
        return std::strcmp(candidate, name) == 0;
    }) != extensions.end();
    if (!exists)
    {
        extensions.push_back(name);
    }
}

uint32_t find_graphics_present_queue(VkPhysicalDevice device, VkSurfaceKHR surface)
{
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

    constexpr VkQueueFlags kRequired = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
    for (uint32_t i = 0; i < count; ++i)
    {
        if ((families[i].queueFlags & kRequired) != kRequired)
        {
            continue;
        }
        VkBool32 present_supported = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present_supported);
        if (present_supported == VK_TRUE)
        {
            return i;
        }
    }
    return UINT32_MAX;
}
}

Device::~Device()
{
    shutdown();
}

void Device::create_instance(const CreateInfo& create_info)
{
    if (instance_ != VK_NULL_HANDLE)
    {
        throw std::runtime_error("Vulkan instance is already initialized.");
    }

    if (volkInitialize() != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to initialize volk (Vulkan loader not found).");
    }

    uint32_t loader_version = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion(&loader_version) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to query Vulkan instance version.");
    }
    if (loader_version < VK_API_VERSION_1_4)
    {
        throw std::runtime_error("Vulkan 1.4 loader is required.");
    }

    enabled_instance_extensions_ = platform::required_instance_extensions(create_info.presentation);
    validation_enabled_ = create_info.enable_validation && validation_layers_available();

    if (validation_enabled_)
    {
        append_unique(enabled_instance_extensions_, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkApplicationInfo application_info{};
    application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application_info.pApplicationName = create_info.application_name.c_str();
    application_info.applicationVersion = VK_MAKE_API_VERSION(0, 0, 0, 1);
    application_info.pEngineName = "ResearchRenderer";
    application_info.engineVersion = VK_MAKE_API_VERSION(0, 0, 0, 1);
    application_info.apiVersion = VK_API_VERSION_1_4;

    VkInstanceCreateInfo instance_create_info{};
    instance_create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_create_info.pApplicationInfo = &application_info;
    instance_create_info.enabledExtensionCount = static_cast<uint32_t>(enabled_instance_extensions_.size());
    instance_create_info.ppEnabledExtensionNames = enabled_instance_extensions_.data();
    if (validation_enabled_)
    {
        instance_create_info.enabledLayerCount = static_cast<uint32_t>(kValidationLayers.size());
        instance_create_info.ppEnabledLayerNames = kValidationLayers.data();
    }

    if (vkCreateInstance(&instance_create_info, nullptr, &instance_) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vulkan instance.");
    }

    rr::core::log()->info(
        "Created Vulkan instance. Loader API {}.{}.{}",
        VK_API_VERSION_MAJOR(loader_version),
        VK_API_VERSION_MINOR(loader_version),
        VK_API_VERSION_PATCH(loader_version));
    for (const char* name : enabled_instance_extensions_)
    {
        rr::core::log()->info("Enabled instance extension: {}", name);
    }
    if (validation_enabled_)
    {
        rr::core::log()->info("Validation layer enabled: {}", kValidationLayers[0]);
    }

    volkLoadInstance(instance_);

    if (validation_enabled_)
    {
        create_debug_messenger();
    }
}

void Device::create_device_with_surface(const Surface& surface)
{
    if (!surface.is_valid())
    {
        throw std::runtime_error("create_device_with_surface requires an initialized Surface.");
    }
    if (surface.instance_ != to_handle(instance_))
    {
        throw std::runtime_error("create_device_with_surface requires a Surface created from this Device instance.");
    }

    create_device_with_surface(from_opaque_handle<VkSurfaceKHR>(surface.surface_));
}

void Device::create_device_with_surface(VkSurfaceKHR surface)
{
    if (instance_ == VK_NULL_HANDLE)
    {
        throw std::runtime_error("create_device_with_surface called before create_instance.");
    }
    if (surface == VK_NULL_HANDLE)
    {
        throw std::runtime_error("create_device_with_surface requires a non-null VkSurfaceKHR.");
    }

    pick_physical_device(surface);
    create_logical_device();
    volkLoadDevice(device_);

    vkGetDeviceQueue(device_, queue_families_.graphics_compute, 0, &graphics_queue_);

    // Create the VMA allocator now that we have the device and volk function
    // pointers loaded.  We hand volk's proc-addr functions to VMA so it can
    // resolve everything dynamically.
    VmaVulkanFunctions vma_funcs{};
    vma_funcs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vma_funcs.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo vma_info{};
    vma_info.vulkanApiVersion = VK_API_VERSION_1_4;
    vma_info.physicalDevice   = physical_device_;
    vma_info.device           = device_;
    vma_info.instance         = instance_;
    vma_info.pVulkanFunctions = &vma_funcs;
    // BUFFER_DEVICE_ADDRESS_BIT is required so VMA can obtain device addresses
    // for buffers allocated with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT.
    vma_info.flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    if (vmaCreateAllocator(&vma_info, &allocator_) != VK_SUCCESS)
    {
        throw std::runtime_error("vmaCreateAllocator failed.");
    }
    rr::core::log()->info("VMA allocator created.");

    log_enabled_features();
}

void Device::create_debug_messenger()
{
    VkDebugUtilsMessengerCreateInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = &debug_callback;

    if (vkCreateDebugUtilsMessengerEXT(instance_, &info, nullptr, &debug_messenger_) != VK_SUCCESS)
    {
        rr::core::log()->warn("Failed to create VkDebugUtilsMessengerEXT (validation messages will be silent).");
        debug_messenger_ = VK_NULL_HANDLE;
    }
}

void Device::pick_physical_device(VkSurfaceKHR surface)
{
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (count == 0)
    {
        throw std::runtime_error("No Vulkan-capable physical device found.");
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());

    const std::vector<const char*> required(kRequiredDeviceExtensions.begin(), kRequiredDeviceExtensions.end());

    VkPhysicalDevice fallback = VK_NULL_HANDLE;
    uint32_t fallback_queue = UINT32_MAX;
    VkPhysicalDeviceProperties fallback_props{};

    for (VkPhysicalDevice device : devices)
    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(device, &props);
        const auto extensions = enumerate_device_extensions(device);

        rr::core::log()->info(
            "Inspecting physical device: {} (apiVersion {}.{}.{}, type {})",
            props.deviceName,
            VK_API_VERSION_MAJOR(props.apiVersion),
            VK_API_VERSION_MINOR(props.apiVersion),
            VK_API_VERSION_PATCH(props.apiVersion),
            static_cast<int>(props.deviceType));
        for (const char* name : required)
        {
            rr::core::log()->info("  ext {} : {}", name,
                                  extension_present(extensions, name) ? "present" : "MISSING");
        }

        if (props.apiVersion < VK_API_VERSION_1_4)
        {
            continue;
        }
        if (!device_supports_extensions(extensions, required))
        {
            continue;
        }
        const uint32_t queue = find_graphics_present_queue(device, surface);
        if (queue == UINT32_MAX)
        {
            continue;
        }

        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            physical_device_ = device;
            physical_device_properties_ = props;
            queue_families_.graphics_compute = queue;
            break;
        }
        if (fallback == VK_NULL_HANDLE)
        {
            fallback = device;
            fallback_queue = queue;
            fallback_props = props;
        }
    }

    if (physical_device_ == VK_NULL_HANDLE && fallback != VK_NULL_HANDLE)
    {
        physical_device_ = fallback;
        physical_device_properties_ = fallback_props;
        queue_families_.graphics_compute = fallback_queue;
    }

    if (physical_device_ == VK_NULL_HANDLE)
    {
        throw std::runtime_error(
            "No Vulkan 1.4 physical device with required RT and descriptor-heap extensions was found.");
    }

    rr::core::log()->info(
        "Selected physical device: {} (driver API {}.{}.{})",
        physical_device_properties_.deviceName,
        VK_API_VERSION_MAJOR(physical_device_properties_.apiVersion),
        VK_API_VERSION_MINOR(physical_device_properties_.apiVersion),
        VK_API_VERSION_PATCH(physical_device_properties_.apiVersion));
    rr::core::log()->info(
        "Graphics+Compute+Present queue family: {}", queue_families_.graphics_compute);
}

void Device::create_logical_device()
{
    enabled_device_extensions_.assign(kRequiredDeviceExtensions.begin(), kRequiredDeviceExtensions.end());

    // Feature chain. We declare every struct, query support, then enable the bits we need.
    VkPhysicalDeviceDescriptorHeapFeaturesEXT heap_features{};
    heap_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT;

    VkPhysicalDeviceRayQueryFeaturesKHR rq_features{};
    rq_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rq_features.pNext = &heap_features;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_features{};
    rt_pipeline_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rt_pipeline_features.pNext = &rq_features;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR as_features{};
    as_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    as_features.pNext = &rt_pipeline_features;

    VkPhysicalDeviceVulkan14Features v14{};
    v14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
    v14.pNext = &as_features;

    VkPhysicalDeviceVulkan13Features v13{};
    v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    v13.pNext = &v14;

    VkPhysicalDeviceVulkan12Features v12{};
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.pNext = &v13;

    VkPhysicalDeviceVulkan11Features v11{};
    v11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    v11.pNext = &v12;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &v11;

    vkGetPhysicalDeviceFeatures2(physical_device_, &features2);

    auto require = [](VkBool32 supported, const char* what) {
        if (!supported)
        {
            throw std::runtime_error(std::string("Required Vulkan feature is not supported: ") + what);
        }
    };

    require(v12.bufferDeviceAddress, "Vulkan12.bufferDeviceAddress");
    require(v12.descriptorIndexing, "Vulkan12.descriptorIndexing");
    require(v12.runtimeDescriptorArray, "Vulkan12.runtimeDescriptorArray");
    require(v12.descriptorBindingPartiallyBound, "Vulkan12.descriptorBindingPartiallyBound");
    require(v12.descriptorBindingVariableDescriptorCount, "Vulkan12.descriptorBindingVariableDescriptorCount");
    require(v12.timelineSemaphore, "Vulkan12.timelineSemaphore");
    require(v12.scalarBlockLayout, "Vulkan12.scalarBlockLayout");
    require(v13.synchronization2, "Vulkan13.synchronization2");
    require(v13.dynamicRendering, "Vulkan13.dynamicRendering");
    require(v14.pushDescriptor, "Vulkan14.pushDescriptor");
    require(v14.dynamicRenderingLocalRead, "Vulkan14.dynamicRenderingLocalRead");
    require(v14.hostImageCopy, "Vulkan14.hostImageCopy");
    require(v14.maintenance5, "Vulkan14.maintenance5");
    require(v14.maintenance6, "Vulkan14.maintenance6");
    require(as_features.accelerationStructure, "accelerationStructure");
    require(rt_pipeline_features.rayTracingPipeline, "rayTracingPipeline");
    require(rq_features.rayQuery, "rayQuery");
    require(v11.shaderDrawParameters, "Vulkan11.shaderDrawParameters");
    require(heap_features.descriptorHeap, "descriptorHeap");

    constexpr float kQueuePriority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = queue_families_.graphics_compute;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &kQueuePriority;

    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = &features2;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = static_cast<uint32_t>(enabled_device_extensions_.size());
    device_info.ppEnabledExtensionNames = enabled_device_extensions_.data();
    // Device-level layer enablement is deprecated; instance layers cover validation.

    if (vkCreateDevice(physical_device_, &device_info, nullptr, &device_) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vulkan logical device.");
    }
}

void Device::log_enabled_features() const
{
    auto& logger = *rr::core::log();
    logger.info("Enabled device extensions:");
    for (const char* name : enabled_device_extensions_)
    {
        logger.info("  - {}", name);
    }
    logger.info("Enabled features:");
    logger.info("  Vulkan 1.2: bufferDeviceAddress, descriptorIndexing, timelineSemaphore, scalarBlockLayout");
    logger.info("  Vulkan 1.3: synchronization2, dynamicRendering");
    logger.info("  Vulkan 1.4: pushDescriptor, dynamicRenderingLocalRead, hostImageCopy, maintenance5, maintenance6");
    logger.info("  Ray tracing: accelerationStructure, rayTracingPipeline, rayQuery");
    logger.info("  Bindless:    descriptorHeap (VK_EXT_descriptor_heap)");
}

void Device::shutdown()
{
    if (allocator_ != nullptr)
    {
        vmaDestroyAllocator(allocator_);
        allocator_ = nullptr;
    }

    if (device_ != VK_NULL_HANDLE)
    {
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    graphics_queue_ = VK_NULL_HANDLE;
    physical_device_ = VK_NULL_HANDLE;

    if (debug_messenger_ != VK_NULL_HANDLE && instance_ != VK_NULL_HANDLE)
    {
        vkDestroyDebugUtilsMessengerEXT(instance_, debug_messenger_, nullptr);
        debug_messenger_ = VK_NULL_HANDLE;
    }

    if (instance_ != VK_NULL_HANDLE)
    {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }

    enabled_instance_extensions_.clear();
    enabled_device_extensions_.clear();
    validation_enabled_ = false;
    queue_families_ = {};
}

bool Device::validation_layers_available() const
{
    uint32_t property_count = 0;
    if (vkEnumerateInstanceLayerProperties(&property_count, nullptr) != VK_SUCCESS)
    {
        return false;
    }
    std::vector<VkLayerProperties> properties(property_count);
    if (vkEnumerateInstanceLayerProperties(&property_count, properties.data()) != VK_SUCCESS)
    {
        return false;
    }
    for (const char* required_name : kValidationLayers)
    {
        const bool found = std::any_of(properties.begin(), properties.end(), [required_name](const VkLayerProperties& p) {
            return std::string_view(p.layerName) == required_name;
        });
        if (!found)
        {
            rr::core::log()->warn("Validation layer '{}' is not available.", required_name);
            return false;
        }
    }
    return true;
}
}
