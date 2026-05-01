// Sampling.slang
// Random number generation (PCG) and direction-sampling utilities.
//
// All random functions take a mutable uint seed and update it in place.

import "Common";  // pcg_hash, uint_to_float01

// ── PCG-based RNG ─────────────────────────────────────────────────────────

// Advance the RNG and return a uniform float in [0, 1).
float rand_float(inout uint seed)
{
    seed = pcg_hash(seed);
    return uint_to_float01(seed);
}

// Return a 2-D uniform sample from the RNG.
float2 rand_float2(inout uint seed)
{
    float a = rand_float(seed);
    float b = rand_float(seed);
    return float2(a, b);
}

// Seed from pixel + frame.  Produces per-pixel-per-frame independent streams.
uint make_seed(uint2 pixel, uint frame_index)
{
    uint h = pixel.x ^ (pixel.y * 1000003u) ^ (frame_index * 2654435761u);
    return pcg_hash(h);
}

// ── Direction sampling ─────────────────────────────────────────────────────

// Sample a direction on a cosine-weighted hemisphere.
// Returns the sampled direction in the tangent frame (z = up).
// PDF = cos(theta) / pi.
float3 sample_cosine_hemisphere(float u0, float u1, out float pdf)
{
    float r   = sqrt(u0);
    float phi = 6.28318530717958647692f * u1;
    float x   = r * cos(phi);
    float y   = r * sin(phi);
    float z   = sqrt(max(0.0f, 1.0f - u0));
    pdf = z * 0.31830988618379067154f;  // z / pi
    return float3(x, y, z);
}

// Sample a direction uniformly on the sphere.
// PDF = 1 / (4 pi).
float3 sample_uniform_sphere(float u0, float u1)
{
    float cos_theta = 2.0f * u0 - 1.0f;
    float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta * cos_theta));
    float phi       = 6.28318530717958647692f * u1;
    return float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

// Build an ONB (orthonormal basis) from a normal vector.
// Returns tangent T and bitangent B such that {T, B, N} form a right-handed frame.
void build_onb(float3 n, out float3 T, out float3 B)
{
    float3 up = (abs(n.x) < 0.9f) ? float3(1.0f, 0.0f, 0.0f)
                                   : float3(0.0f, 1.0f, 0.0f);
    T = normalize(cross(up, n));
    B = cross(n, T);
}

// Transform a direction from the local tangent frame to world space.
float3 tangent_to_world(float3 v_local, float3 T, float3 B, float3 N)
{
    return v_local.x * T + v_local.y * B + v_local.z * N;
}

// Transform a direction from world space to the local tangent frame.
float3 world_to_tangent(float3 v_world, float3 T, float3 B, float3 N)
{
    return float3(dot(v_world, T), dot(v_world, B), dot(v_world, N));
}

// Sample a point uniformly on the surface of a quad light.
// v0 = corner, e0 / e1 = two edge vectors.
float3 sample_quad_light(float3 v0, float3 e0, float3 e1, float u0, float u1)
{
    return v0 + u0 * e0 + u1 * e1;
}
