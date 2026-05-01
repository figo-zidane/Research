#pragma once

#include <cstdint>
#include <glm/glm.hpp>

// GPU-side scene data structures shared between C++ and Slang shaders.
// Slang counterparts are in assets/shaders/include/GpuScene.slang.
//
// All structs use scalarBlockLayout-compatible alignment (tightly packed)
// thanks to VK_KHR_buffer_device_address + scalarBlockLayout being enabled.
// Verify that sizeof(GpuXxx) matches the Slang side if you change fields.

namespace rr::scene
{

// One vertex in the global vertex buffer. 48 bytes.
struct GpuVertex
{
    float pos[3];      // world-space (pre-transform) or object-space
    float normal[3];
    float uv[2];
    float tangent[4];  // .w = bitangent sign
};
static_assert(sizeof(GpuVertex) == 48);

// One mesh (geometry primitive group).  Index/vertex data lives in the
// global scene buffers; offsets are in bytes or vertex units. 32 bytes.
struct GpuMesh
{
    uint32_t vertex_base;         // first vertex in global vertex buffer
    uint32_t vertex_count;
    uint32_t index_byte_offset;   // byte offset into global index buffer
    uint32_t index_count;
    uint32_t material_index;
    uint32_t _pad[3];
};
static_assert(sizeof(GpuMesh) == 32);

// One PBR material. 80 bytes.
struct GpuMaterial
{
    float    base_color[4];               // linear RGBA
    float    metallic;
    float    roughness;
    float    emissive[3];
    float    ior;                         // index of refraction (default 1.5)
    uint32_t albedo_tex_idx;              // gTextures[] index, 0xFFFFFFFF = none
    uint32_t normal_tex_idx;
    uint32_t metallic_roughness_tex_idx;
    uint32_t emissive_tex_idx;
    uint32_t sampler_idx;                 // gSamplers[] index
    uint32_t _pad[3];
};
static_assert(sizeof(GpuMaterial) == 72);

// One instance (mesh + world transform). 80 bytes.
struct GpuInstance
{
    float    transform[16];   // row-major float4x4
    uint32_t mesh_index;
    uint32_t _pad[3];
};
static_assert(sizeof(GpuInstance) == 80);

// One light. 48 bytes.
// type: 0=point, 1=directional, 2=area_quad (not used in Phase 4).
struct GpuLight
{
    float    position[3];
    uint32_t type;
    float    direction[3];
    float    radius;
    float    emission[3];    // W/m^2 (sr^-1 for direction, total for area)
    float    intensity;
};
static_assert(sizeof(GpuLight) == 48);

// Per-frame camera + scene metadata buffer stored at gScene[kCameraSlot].
// Passes read this to get camera matrices and resource indices. 304 bytes.
struct GpuCameraData
{
    float    view[16];         // 64
    float    inv_view[16];     // 64
    float    inv_proj[16];     // 64
    float    prev_view_proj[16]; // 64 – for motion vectors

    float    position[3];
    float    near_plane;

    float    direction[3];
    float    far_plane;

    uint32_t frame_index;
    uint32_t screen_width;
    uint32_t screen_height;
    uint32_t _pad;
};
static_assert(sizeof(GpuCameraData) == 304);

// Well-known indices into gScene[] allocated at scene-init time.
// The Scene class populates these during build() / upload().
struct SceneGpuHandles
{
    uint32_t camera_buf_idx       = UINT32_MAX; // gScene[x] = GpuCameraData
    uint32_t vertex_buf_idx       = UINT32_MAX; // gScene[x] = GpuVertex[]
    uint32_t index_buf_idx        = UINT32_MAX; // gScene[x] = uint32_t[]
    uint32_t mesh_buf_idx         = UINT32_MAX; // gScene[x] = GpuMesh[]
    uint32_t material_buf_idx     = UINT32_MAX; // gScene[x] = GpuMaterial[]
    uint32_t instance_buf_idx     = UINT32_MAX; // gScene[x] = GpuInstance[]
    uint32_t light_buf_idx        = UINT32_MAX; // gScene[x] = GpuLight[]
    uint32_t tlas_idx             = UINT32_MAX; // gTLAS[x]
    uint32_t linear_sampler_idx   = UINT32_MAX; // gSamplers[x]
    uint32_t nearest_sampler_idx  = UINT32_MAX; // gSamplers[x]
};

} // namespace rr::scene
