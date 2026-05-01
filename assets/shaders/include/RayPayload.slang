// RayPayload.slang
// Ray payload structure shared between path tracer passes.
// Used with VK_KHR_ray_tracing_pipeline for future phases.
// For Phase 4, the RayQuery path tracer processes everything inline.

struct RayPayload
{
    float3 throughput;   // accumulated path throughput
    float3 radiance;     // accumulated radiance along the path
    float3 origin;       // next ray origin
    float3 direction;    // next ray direction
    uint   seed;         // RNG state
    bool   terminated;   // set to true when path ends
    uint   bounce;       // current bounce depth
    float  t_hit;        // hit distance (set by closest-hit / inline query)
};

// Compact miss payload — used for shadow / visibility rays.
struct ShadowPayload
{
    bool occluded;
};
