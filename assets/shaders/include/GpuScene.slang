// GpuScene.slang
// Slang mirror of src/scene/GpuScene.h.
// All passes #include this to access scene data structures.
//
// scalarBlockLayout is enabled globally, so float3/uint3 are tightly packed.

struct GpuVertex
{
    float3 position;   // 12
    float3 normal;     // 12
    float2 uv;         //  8
    float4 tangent;    // 16
    // total 48 bytes
};

struct GpuMesh
{
    uint vertex_base;
    uint vertex_count;
    uint index_byte_offset;
    uint index_count;
    uint material_index;
    uint3 _pad;
    // total 32 bytes
};

struct GpuMaterial
{
    float4 base_color;
    float  metallic;
    float  roughness;
    float3 emissive;
    float  ior;
    uint   albedo_tex_idx;
    uint   normal_tex_idx;
    uint   metallic_roughness_tex_idx;
    uint   emissive_tex_idx;
    uint   sampler_idx;
    uint3  _pad;
    // total 72 bytes
};

struct GpuInstance
{
    float4x4 transform;   // 64
    uint     mesh_index;  //  4
    uint3    _pad;        // 12
    // total 80 bytes
};

struct GpuLight
{
    float3 position;
    uint   type;       // 0=point, 1=directional
    float3 direction;
    float  radius;
    float3 emission;
    float  intensity;
    // total 48 bytes
};

struct GpuCameraData
{
    float4x4 view;
    float4x4 inv_view;
    float4x4 inv_proj;
    float4x4 prev_view_proj;

    float3   position;
    float    near_plane;

    float3   direction;
    float    far_plane;

    uint     frame_index;
    uint     screen_width;
    uint     screen_height;
    uint     _pad;
    // total 304 bytes
};

// ── Helper: load a GpuVertex from the global vertex buffer ─────────────────
GpuVertex scene_load_vertex(ByteAddressBuffer buf, uint global_vertex_id)
{
    uint base = global_vertex_id * 48u;
    GpuVertex v;
    v.position  = buf.Load<float3>(base);
    v.normal    = buf.Load<float3>(base + 12u);
    v.uv        = buf.Load<float2>(base + 24u);
    v.tangent   = buf.Load<float4>(base + 32u);
    return v;
}

// ── Helper: load triangle indices from the global index buffer ─────────────
uint3 scene_load_triangle(ByteAddressBuffer buf, uint mesh_index_byte_offset, uint prim_idx)
{
    uint base = mesh_index_byte_offset + prim_idx * 12u;
    return buf.Load<uint3>(base);
}
