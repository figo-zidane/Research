// accumulate.slang — Temporal accumulation pass
// If camera_moved, reset accumulated = radiance, spp = 1.
// Otherwise blend: acc = acc * (spp/(spp+1)) + radiance * (1/(spp+1))

#include "BindlessHeap.slang"

struct PushConstants
{
    uint radiance_idx;    // gTextures index (sampled)
    uint accumulated_idx; // gStorageImages index (read+write)
    uint screen_width;
    uint screen_height;
    uint camera_moved;    // 1 = reset
    uint accumulated_spp; // spp BEFORE this frame
    uint _pad0;
    uint _pad1;
};

[[vk::push_constant]] PushConstants pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dispatch_id : SV_DispatchThreadID)
{
    uint2 pixel = dispatch_id.xy;
    if (pixel.x >= pc.screen_width || pixel.y >= pc.screen_height) return;

    // Load current frame radiance (from pathtracer storage image, treated as texture here)
    // We registered it as gStorageImages for pathtracer write and gTextures for our read.
    float4 radiance = gStorageImages[pc.radiance_idx][pixel];

    float4 result;
    if (pc.camera_moved != 0 || pc.accumulated_spp == 0)
    {
        result = radiance;
    }
    else
    {
        float4 prev    = gStorageImages[pc.accumulated_idx][pixel];
        float  spp     = float(pc.accumulated_spp);
        float  weight  = 1.0 / (spp + 1.0);
        result = prev * (spp / (spp + 1.0)) + radiance * weight;
    }

    gStorageImages[pc.accumulated_idx][pixel] = result;
}
