// restir_di.slang — ReSTIR DI using RayQuery inline ray tracing.
//
// Three compute entry points, all sharing the same PushConstants:
//   cs_initial       — primary ray + initial reservoir (M-sample RIS)
//   cs_temporal      — merge current reservoir with previous frame
//   cs_spatial_shade — spatial reuse from neighbours + shadow test + shading

#include "BindlessHeap.slang"
#include "Common.slang"
#include "GpuScene.slang"
#include "BRDF.slang"
#include "Sampling.slang"

// ── Push constants ────────────────────────────────────────────────────────────

struct PushConstants
{
    uint camera_buf_idx;
    uint tlas_idx;
    uint vertex_buf_idx;
    uint index_buf_idx;
    uint mesh_buf_idx;
    uint material_buf_idx;
    uint instance_buf_idx;
    uint light_buf_idx;
    uint num_instances;
    uint num_lights;
    uint linear_sampler_idx;    // unused in this shader, reserved for future
    uint frame_seed;
    // G-buffer storage image indices
    uint gbuf_pos_idx;          // float4(world_pos.xyz, ray_t); ray_t < 0 → miss
    uint gbuf_norm_idx;         // float4(normal.xyz, asfloat(material_index))
    // Reservoir ping-pong storage image indices
    uint reservoir_curr_idx;    // write target this frame / read next frame
    uint reservoir_prev_idx;    // previous frame reservoir (read-only)
    // Output radiance
    uint output_idx;
    // Per-frame settings
    uint num_initial_candidates;
    uint temporal_enable;       // 0 or 1
    uint spatial_enable;        // 0 or 1
    uint spatial_num_neighbors;
    uint spatial_radius;        // pixels
    uint _pad0;
    uint _pad1;
};

[[vk::push_constant]] PushConstants pc;

// ── Reservoir packing ─────────────────────────────────────────────────────────
// Packed as float4:
//   .x = asfloat(selected_light_index)  — UINT_MAX means empty
//   .y = w_sum  (accumulated importance weight * N_lights)
//   .z = W      (final normalized weight = w_sum / (M * p_hat(y)))
//   .w = asfloat(M)  (sample count)

float4 pack_reservoir(uint selected_light, float w_sum, float W, uint M)
{
    return float4(asfloat(selected_light), w_sum, W, asfloat(M));
}

void unpack_reservoir(float4 v,
                      out uint  selected_light,
                      out float w_sum,
                      out float W,
                      out uint  M)
{
    selected_light = asuint(v.x);
    w_sum          = v.y;
    W              = v.z;
    M              = asuint(v.w);
}

// ── Geometry helpers (shared with pathtracer.slang logic) ─────────────────────

struct HitInfo
{
    float3 pos;
    float3 normal;
    float2 uv;
    uint   material_index;
    float  t;
    bool   hit;
};

HitInfo trace_ray(float3 origin, float3 dir, float t_min, float t_max)
{
    HitInfo info;
    info.hit = false;
    info.t   = t_max;

    RayQuery<RAY_FLAG_CULL_BACK_FACING_TRIANGLES> rq;
    RayDesc ray;
    ray.Origin    = origin;
    ray.TMin      = t_min;
    ray.Direction = dir;
    ray.TMax      = t_max;
    rq.TraceRayInline(gTLAS[pc.tlas_idx],
                      RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                      0xFF, ray);

    while (rq.Proceed())
    {
        if (rq.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            rq.CommitNonOpaqueTriangleHit();
    }

    if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        info.hit      = true;
        info.t        = rq.CommittedRayT();
        uint inst_idx = rq.CommittedInstanceID();
        uint prim_idx = rq.CommittedPrimitiveIndex();
        float2 bary   = rq.CommittedTriangleBarycentrics();

        GpuInstance inst = gScene[pc.instance_buf_idx].Load<GpuInstance>(
            inst_idx * int(sizeof(GpuInstance)));
        GpuMesh mesh = gScene[pc.mesh_buf_idx].Load<GpuMesh>(
            inst.mesh_index * int(sizeof(GpuMesh)));

        uint base = mesh.index_byte_offset + prim_idx * 3 * 4;
        uint i0   = gScene[pc.index_buf_idx].Load<uint>(int(base));
        uint i1   = gScene[pc.index_buf_idx].Load<uint>(int(base + 4));
        uint i2   = gScene[pc.index_buf_idx].Load<uint>(int(base + 8));
        GpuVertex v0 = gScene[pc.vertex_buf_idx].Load<GpuVertex>(
            int((mesh.vertex_base + i0) * sizeof(GpuVertex)));
        GpuVertex v1 = gScene[pc.vertex_buf_idx].Load<GpuVertex>(
            int((mesh.vertex_base + i1) * sizeof(GpuVertex)));
        GpuVertex v2 = gScene[pc.vertex_buf_idx].Load<GpuVertex>(
            int((mesh.vertex_base + i2) * sizeof(GpuVertex)));

        float3 b3 = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
        float3 lp = b3.x * v0.position + b3.y * v1.position + b3.z * v2.position;
        float3 ln = normalize(b3.x * v0.normal + b3.y * v1.normal + b3.z * v2.normal);
        float2 luv = b3.x * v0.uv + b3.y * v1.uv + b3.z * v2.uv;

        float4 wpos = mul(inst.transform, float4(lp, 1.0));
        float3 wnrm = normalize(mul((float3x3)inst.transform, ln));

        info.pos            = wpos.xyz;
        info.normal         = wnrm;
        info.uv             = luv;
        info.material_index = mesh.material_index;
    }
    return info;
}

// ── Target function p_hat (unnormalized importance) ───────────────────────────
// p_hat(light, surface) = max(0, n·l) * luminance(irradiance)
// Used to weight samples in WRS.  Does NOT trace shadow rays.

float compute_p_hat(uint light_idx, float3 hit_pos, float3 normal)
{
    if (light_idx >= pc.num_lights) return 0.0;

    GpuLight light = gScene[pc.light_buf_idx].Load<GpuLight>(
        light_idx * int(sizeof(GpuLight)));

    float3 l;
    float3 irradiance;

    if (light.type == 0) // point light
    {
        float3 to = light.position - hit_pos;
        float  d2 = max(dot(to, to), 1e-6f);
        l          = normalize(to);
        irradiance = light.emission * light.intensity / d2;
    }
    else // directional
    {
        l          = normalize(-light.direction);
        irradiance = light.emission * light.intensity;
    }

    float n_dot_l = max(dot(normal, l), 0.0f);
    float lum     = 0.2126f * irradiance.r + 0.7152f * irradiance.g + 0.0722f * irradiance.b;
    return n_dot_l * lum;
}

// ── cs_initial — primary ray + initial reservoir ──────────────────────────────

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_initial(uint3 id : SV_DispatchThreadID)
{
    GpuCameraData cam = gScene[pc.camera_buf_idx].Load<GpuCameraData>(0);
    uint W = cam.screen_width;
    uint H = cam.screen_height;
    if (id.x >= W || id.y >= H) return;

    uint2 pixel = id.xy;
    uint  seed  = make_seed(pixel, pc.frame_seed);

    // Primary ray (same as pathtracer: jittered NDC → world direction)
    float2 ndc      = (float2(pixel) + float2(rand_float(seed), rand_float(seed)))
                      / float2(W, H) * 2.0 - 1.0;
    float4 ray_clip = float4(ndc.x, ndc.y, 1.0, 1.0);
    float4 ray_view = mul(cam.inv_proj, ray_clip);
    ray_view.z = -1.0; ray_view.w = 0.0;
    float4 ray_world = mul(cam.inv_view, ray_view);
    float3 ray_dir   = normalize(ray_world.xyz);
    float3 ray_orig  = cam.position;

    HitInfo hit = trace_ray(ray_orig, ray_dir, 0.001, 1e9);

    if (!hit.hit)
    {
        gStorageImages[pc.gbuf_pos_idx][pixel]   = float4(0, 0, 0, -1.0); // ray_t < 0 → miss
        gStorageImages[pc.gbuf_norm_idx][pixel]  = float4(0, 0, 0, 0);
        gStorageImages[pc.reservoir_curr_idx][pixel] = pack_reservoir(~0u, 0.0, 0.0, 0);
        return;
    }

    // Ensure normal faces the camera
    float3 normal = hit.normal;
    if (dot(normal, -ray_dir) < 0.0) normal = -normal;

    gStorageImages[pc.gbuf_pos_idx][pixel]  = float4(hit.pos, hit.t);
    gStorageImages[pc.gbuf_norm_idx][pixel] = float4(normal, asfloat(hit.material_index));

    if (pc.num_lights == 0)
    {
        gStorageImages[pc.reservoir_curr_idx][pixel] = pack_reservoir(~0u, 0.0, 0.0, 0);
        return;
    }

    // Weighted Reservoir Sampling over M candidate lights
    // Source distribution: uniform over all lights, p(x) = 1 / num_lights
    // Weight:              w_i = p_hat(x_i) * num_lights
    uint  sel_light = ~0u;
    float w_sum     = 0.0;
    uint  M_count   = 0;

    uint M = pc.num_initial_candidates;
    for (uint i = 0; i < M; ++i)
    {
        uint  cand   = uint(rand_float(seed) * float(pc.num_lights)) % pc.num_lights;
        float ph     = compute_p_hat(cand, hit.pos, normal);
        float w_i    = ph * float(pc.num_lights);  // importance weight

        w_sum  += w_i;
        M_count++;

        // WRS: accept new candidate proportional to its weight
        if (w_sum > 0.0 && rand_float(seed) < w_i / w_sum)
            sel_light = cand;
    }

    // Compute final normalized weight W = w_sum / (M * p_hat(y))
    float W_final = 0.0;
    if (sel_light != ~0u)
    {
        float ph_y = compute_p_hat(sel_light, hit.pos, normal);
        W_final = (ph_y > 1e-10) ? (w_sum / (float(M_count) * ph_y)) : 0.0;
    }

    gStorageImages[pc.reservoir_curr_idx][pixel] = pack_reservoir(sel_light, w_sum, W_final, M_count);
}

// ── cs_temporal — merge with previous frame reservoir ─────────────────────────

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_temporal(uint3 id : SV_DispatchThreadID)
{
    if (!bool(pc.temporal_enable)) return;

    GpuCameraData cam = gScene[pc.camera_buf_idx].Load<GpuCameraData>(0);
    if (id.x >= cam.screen_width || id.y >= cam.screen_height) return;

    uint2 pixel = id.xy;

    float4 gbp = gStorageImages[pc.gbuf_pos_idx][pixel];
    if (gbp.w < 0.0) return;  // primary ray miss — nothing to accumulate

    float3 hit_pos = gbp.xyz;
    float3 normal  = gStorageImages[pc.gbuf_norm_idx][pixel].xyz;

    // Load current frame's reservoir (from cs_initial)
    uint  curr_sel; float curr_wsum; float curr_W; uint curr_M;
    unpack_reservoir(gStorageImages[pc.reservoir_curr_idx][pixel],
                     curr_sel, curr_wsum, curr_W, curr_M);

    // Load previous frame reservoir at the same pixel (static scene assumption)
    uint  prev_sel; float prev_wsum; float prev_W; uint prev_M;
    unpack_reservoir(gStorageImages[pc.reservoir_prev_idx][pixel],
                     prev_sel, prev_wsum, prev_W, prev_M);

    if (prev_sel == ~0u || prev_M == 0)
    {
        // No history: keep current as-is
        return;
    }

    // M-cap to prevent unbounded accumulation
    uint m_cap  = clamp(20u * pc.num_initial_candidates, 1u, 8192u);
    prev_M      = min(prev_M, m_cap);

    // Contribution of previous sample at current pixel's geometry
    float ph_prev  = compute_p_hat(prev_sel, hit_pos, normal);
    float contrib  = ph_prev * prev_W * float(prev_M);

    curr_wsum += contrib;
    curr_M    += prev_M;
    curr_M     = min(curr_M, m_cap);

    uint seed = make_seed(pixel, pc.frame_seed + 1u);
    if (contrib > 0.0 && rand_float(seed) < contrib / curr_wsum)
        curr_sel = prev_sel;

    // Recompute W
    float W_new = 0.0;
    if (curr_sel != ~0u)
    {
        float ph_y = compute_p_hat(curr_sel, hit_pos, normal);
        W_new = (ph_y > 1e-10) ? (curr_wsum / (float(curr_M) * ph_y)) : 0.0;
    }

    gStorageImages[pc.reservoir_curr_idx][pixel] =
        pack_reservoir(curr_sel, curr_wsum, W_new, curr_M);
}

// ── cs_spatial_shade — spatial reuse + final shading ─────────────────────────

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_spatial_shade(uint3 id : SV_DispatchThreadID)
{
    GpuCameraData cam = gScene[pc.camera_buf_idx].Load<GpuCameraData>(0);
    uint W = cam.screen_width;
    uint H = cam.screen_height;
    if (id.x >= W || id.y >= H) return;

    uint2 pixel = id.xy;

    // --- Miss: sky colour ---------------------------------------------------
    float4 gbp = gStorageImages[pc.gbuf_pos_idx][pixel];
    if (gbp.w < 0.0)
    {
        // Reconstruct sky direction from camera matrices
        float2 ndc      = (float2(pixel) + 0.5) / float2(W, H) * 2.0 - 1.0;
        float4 rv       = mul(cam.inv_proj, float4(ndc.x, ndc.y, 1.0, 1.0));
        rv.z = -1.0; rv.w = 0.0;
        float3 dir = normalize(mul(cam.inv_view, rv).xyz);
        float  t   = 0.5 * (dir.y + 1.0);
        gStorageImages[pc.output_idx][pixel] =
            float4(lerp(float3(0.1, 0.1, 0.1), float3(0.2, 0.3, 0.5), t), 1.0);
        return;
    }

    float3 hit_pos = gbp.xyz;
    float4 gbn     = gStorageImages[pc.gbuf_norm_idx][pixel];
    float3 normal  = gbn.xyz;
    uint   mat_idx = asuint(gbn.w);

    GpuMaterial mat = gScene[pc.material_buf_idx].Load<GpuMaterial>(
        mat_idx * int(sizeof(GpuMaterial)));

    // Emissive surfaces: output directly
    if (any(mat.emissive > 0.0))
    {
        gStorageImages[pc.output_idx][pixel] = float4(mat.emissive, 1.0);
        return;
    }

    // Load reservoir from temporal pass
    uint  r_sel; float r_wsum; float r_W; uint r_M;
    unpack_reservoir(gStorageImages[pc.reservoir_curr_idx][pixel],
                     r_sel, r_wsum, r_W, r_M);

    // --- Spatial reuse (read-only: does NOT write back to reservoir_curr) ---
    if (bool(pc.spatial_enable) && pc.spatial_num_neighbors > 0 && pc.num_lights > 0)
    {
        uint seed = make_seed(pixel, pc.frame_seed + 2u);
        for (uint k = 0; k < pc.spatial_num_neighbors; ++k)
        {
            float angle  = rand_float(seed) * 6.28318530717958f;
            float radius = rand_float(seed) * float(pc.spatial_radius);
            int2  offs   = int2(int(radius * cos(angle)), int(radius * sin(angle)));
            uint2 nb     = uint2(clamp(int2(pixel) + offs, int2(0, 0),
                                       int2(int(W) - 1, int(H) - 1)));

            float4 nb_gbp = gStorageImages[pc.gbuf_pos_idx][nb];
            if (nb_gbp.w < 0.0) continue;  // neighbour missed

            uint nb_sel; float nb_wsum; float nb_W; uint nb_M;
            unpack_reservoir(gStorageImages[pc.reservoir_curr_idx][nb],
                             nb_sel, nb_wsum, nb_W, nb_M);

            if (nb_sel == ~0u || nb_M == 0) continue;

            // Re-evaluate importance at current pixel's surface
            float ph_nb    = compute_p_hat(nb_sel, hit_pos, normal);
            float contrib  = ph_nb * nb_W * float(nb_M);

            r_wsum += contrib;
            r_M    += nb_M;

            if (contrib > 0.0 && rand_float(seed) < contrib / r_wsum)
                r_sel = nb_sel;
        }

        // Recompute W after spatial merge
        if (r_sel != ~0u)
        {
            float ph_y = compute_p_hat(r_sel, hit_pos, normal);
            r_W = (ph_y > 1e-10) ? (r_wsum / (float(r_M) * ph_y)) : 0.0;
        }
    }

    // --- Final shading -------------------------------------------------------
    float3 output = float3(0.0);

    if (r_sel != ~0u && r_W > 0.0 && pc.num_lights > 0)
    {
        GpuLight light = gScene[pc.light_buf_idx].Load<GpuLight>(
            r_sel * int(sizeof(GpuLight)));

        float3 l;
        float3 irradiance;
        float  dist;

        if (light.type == 0) // point
        {
            float3 to = light.position - hit_pos;
            dist      = length(to);
            l         = to / max(dist, 1e-6);
            irradiance = light.emission * light.intensity / max(dist * dist, 1e-6);
        }
        else // directional
        {
            l          = normalize(-light.direction);
            dist       = 1e9;
            irradiance = light.emission * light.intensity;
        }

        float n_dot_l = max(dot(normal, l), 0.0);
        if (n_dot_l > 0.0)
        {
            // Shadow test
            HitInfo shadow = trace_ray(hit_pos + normal * 1e-4, l, 0.001, dist - 0.01);
            if (!shadow.hit)
            {
                // View direction (from hit surface toward camera)
                float3 wo = normalize(cam.position - hit_pos);
                float3 brdf_val = brdf_eval(normal, wo, l,
                                             mat.base_color.rgb,
                                             mat.metallic, mat.roughness);
                // Estimator: f(y) * r_W
                // r_W = r_wsum / (r_M * p_hat(y)), and r_wsum already contains
                // the N_lights compensation from the initial sampling weights.
                output = brdf_val * irradiance * n_dot_l * r_W;
            }
        }
    }

    gStorageImages[pc.output_idx][pixel] = float4(output, 1.0);
}
