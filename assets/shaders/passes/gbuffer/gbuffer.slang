// GBuffer Pass shaders
// Vertex shader: fetch vertex from global vertex buffer via SV_VertexID
// Fragment shader: write position, normal, material_id to GBuffer targets

#include "BindlessHeap.slang"
#include "Common.slang"
#include "GpuScene.slang"

struct PushConstants
{
    uint camera_buf_idx;
    uint vertex_buf_idx;
    uint index_buf_idx;
    uint mesh_buf_idx;
    uint material_buf_idx;
    uint instance_buf_idx;
    uint draw_instance_idx;
    uint linear_sampler_idx;
};

[[vk::push_constant]] PushConstants pc;

struct VSOutput
{
    float4 sv_pos   : SV_Position;
    float3 world_pos : TEXCOORD0;
    float3 world_normal : TEXCOORD1;
    float2 uv       : TEXCOORD2;
    uint   material_id : TEXCOORD3;
};

[shader("vertex")]
VSOutput vs_main(uint vertex_id : SV_VertexID)
{
    GpuInstance inst = gScene[pc.instance_buf_idx].Load<GpuInstance>(
        pc.draw_instance_idx * int(sizeof(GpuInstance)));
    GpuMesh mesh = gScene[pc.mesh_buf_idx].Load<GpuMesh>(
        inst.mesh_index * int(sizeof(GpuMesh)));

    // Manual index lookup: read from index buffer using mesh.index_byte_offset
    uint index_byte = mesh.index_byte_offset + vertex_id * 4u;
    uint vi = gScene[pc.index_buf_idx].Load<uint>(int(index_byte));
    GpuVertex v = gScene[pc.vertex_buf_idx].Load<GpuVertex>(
        int((mesh.vertex_base + vi) * sizeof(GpuVertex)));

    float4x4 model = inst.transform;

    GpuCameraData cam = gScene[pc.camera_buf_idx].Load<GpuCameraData>(0);
    float4x4 view = cam.view;

    // Rebuild projection.
    // inv_proj[1][1] = 1/(proj[1][1]).  proj[1][1] = -f after Vulkan Y-flip,
    // so inv_proj[1][1] = -1/f (negative).  Take abs to recover f.
    float tan_half_fov = abs(cam.inv_proj[1][1]);  // = 1/f = tan(fov/2)
    float aspect = float(cam.screen_width) / float(cam.screen_height);
    float near_z = cam.near_plane;
    float far_z  = cam.far_plane;
    float f = 1.0f / tan_half_fov;
    // Row-major proj matrix for Vulkan (Y-flip applied via -f on row 1)
    float4x4 proj = float4x4(
        float4(f / aspect, 0,   0,                              0),
        float4(0,         -f,   0,                              0),
        float4(0,          0,   far_z / (far_z - near_z),      -(far_z * near_z) / (far_z - near_z)),
        float4(0,          0,   1,                              0)
    );

    float4 world = mul(model, float4(v.position, 1.0));
    float4 view_pos = mul(view, world);
    float4 clip_pos = mul(proj, view_pos);

    // Normal transform (ignore scale for now — Cornell box has uniform scale)
    float3 world_normal = normalize(mul((float3x3)model, v.normal));

    VSOutput o;
    o.sv_pos      = clip_pos;
    o.world_pos   = world.xyz;
    o.world_normal = world_normal;
    o.uv          = float2(v.uv[0], v.uv[1]);
    o.material_id = mesh.material_index;
    return o;
}

struct FSOutput
{
    float4 position    : SV_Target0;
    float4 normal      : SV_Target1;
    uint   material_id : SV_Target2;
};

[shader("fragment")]
FSOutput fs_main(VSOutput inp)
{
    FSOutput o;
    o.position    = float4(inp.world_pos, 1.0);
    // normal.w stores material_index as float bits (read back with asuint in shaders)
    o.normal      = float4(normalize(inp.world_normal), asfloat(inp.material_id));
    o.material_id = inp.material_id;
    return o;
}
