// tonemap.slang — Fullscreen triangle + ACES filmic tonemapping

#include "BindlessHeap.slang"
#include "Common.slang"

struct PushConstants
{
    uint  accumulated_idx;
    uint  linear_sampler_idx;
    float exposure;
    uint  _pad;
};

[[vk::push_constant]] PushConstants pc;

// \u2500 Vertex shader: generate fullscreen triangle \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

struct VSOutput
{
    float4 sv_pos : SV_Position;
    float2 uv     : TEXCOORD0;
};

[shader("vertex")]
VSOutput vs_main(uint vid : SV_VertexID)
{
    // 3 vertices that cover the entire screen
    float2 pos = float2(
        (vid == 1) ? 3.0 : -1.0,
        (vid == 2) ? 3.0 : -1.0
    );
    VSOutput o;
    o.sv_pos = float4(pos, 0.0, 1.0);
    o.uv     = pos * 0.5 + 0.5;
    return o;
}

// \u2500 ACES approximate tonemapping (Krzysztof Narkowicz) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

float3 aces_approx(float3 x)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Linear to sRGB gamma
float3 linear_to_srgb(float3 x)
{
    return pow(max(x, 0.0), 1.0 / 2.2);
}

// \u2500 Fragment shader \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

[shader("fragment")]
float4 fs_main(VSOutput inp) : SV_Target
{
    // The accumulated image may be in GENERAL layout and we registered it
    // as both a storage image and a texture. Use gTextures for sampling.
    float4 hdr  = gTextures[pc.accumulated_idx].Sample(
        gSamplers[pc.linear_sampler_idx], inp.uv);

    float3 col  = hdr.rgb * pc.exposure;
    col = aces_approx(col);
    col = linear_to_srgb(col);
    return float4(col, 1.0);
}
