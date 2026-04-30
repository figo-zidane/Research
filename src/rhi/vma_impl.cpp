// VMA single-implementation translation unit.
// This is the only file that defines VMA_IMPLEMENTATION; all other TUs that
// need VMA types/functions should #include <vma/vk_mem_alloc.h> WITHOUT the
// macro so they only get declarations.

#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS  0  // don't link against static loader
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1  // function pointers via vkGetDeviceProcAddr
#include <volk.h>
#include <vma/vk_mem_alloc.h>
