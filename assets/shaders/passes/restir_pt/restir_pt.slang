// restir_pt.slang — Multi-bounce ReSTIR PT (Path Tracing) with reconnection
// shift mapping.
//
// This is the multi-bounce generalization of restir_gi.slang. Instead of a
// single primary→secondary bounce, cs_initial constructs a full path of up to
// `max_bounces` segments via BRDF sampling + NEE. The first sufficiently diffuse
// secondary vertex is recorded as the "reconnection vertex"; temporal/spatial
// reuse re-attach the cached suffix radiance to the current primary surface
// through that vertex (reconnection shift), with a geometric Jacobian.
//
// Reservoir payload is split across four RGBA32F images:
//   rc_pos = float4(reconnect_pos.xyz,    w_sum)
//   rc_nrm = float4(reconnect_normal.xyz, asfloat(M))
//   rc_rad = float4(suffix_radiance.rgb,  W)
//   rc_pre = float4(prefix_throughput.rgb, asfloat(valid_flag))
//
// where:
//   reconnect_pos/normal : world-space position/normal of the reconnection vertex
//   suffix_radiance      : radiance arriving at the reconnection vertex from the
//                          rest of the path (everything after the reconnection)
//   prefix_throughput    : BRDF*cos/pdf product accumulated from the primary
//                          surface up to (but excluding) the reconnection vertex,
//                          divided by the geometry already folded into the path.
//                          For a 1-segment prefix this is just 1 (direct
//                          reconnection at the first bounce).
//   valid_flag           : 1.0 if the reservoir holds a usable sample.

#include "BindlessHeap.slang"
#include "Common.slang"
#include "GpuScene.slang"
#include "BRDF.slang"
#include "Sampling.slang"
#include "GuidingGrid.slang"

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
    uint frame_seed;
    uint gbuf_pos_idx;
    uint gbuf_norm_idx;
    uint direct_input_idx;
    uint reservoir_curr_pos_idx;
    uint reservoir_curr_nrm_idx;
    uint reservoir_curr_rad_idx;
    uint reservoir_curr_pre_idx;
    uint reservoir_prev_pos_idx;
    uint reservoir_prev_nrm_idx;
    uint reservoir_prev_rad_idx;
    uint reservoir_prev_pre_idx;
    uint history_curr_pos_idx;
    uint history_curr_norm_idx;
    uint history_prev_pos_idx;
    uint history_prev_norm_idx;
    uint output_idx;
    uint num_initial_candidates;
    uint max_bounces;
    uint temporal_enable;
    uint spatial_enable;
    uint spatial_num_neighbors;
    uint spatial_radius;
    uint include_direct_lighting;
    float indirect_strength;
    float ray_bias;
    // ── Path guiding ───────────────────────────────────────────
    uint  guiding_accum_idx;
    uint  guiding_fitted_idx;
    uint  guiding_enable;
    float guiding_alpha;
    float grid_base_cell_size;
    float grid_distance_slope;
    uint  grid_cell_count;
    uint  grid_min_samples;
    float grid_kappa_max;
    uint  _pad0;
};

[[vk::push_constant]] PushConstants pc;

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

float luminance(float3 c)
{
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

float3 sky_radiance(float3 dir)
{
    float t = 0.5f * (dir.y + 1.0f);
    return lerp(float3(0.1f, 0.1f, 0.1f), float3(0.2f, 0.3f, 0.5f), t);
}

float3 sample_lights(float3 hit_pos, float3 normal, float3 wo,
                     GpuMaterial mat, inout uint seed)
{
    if (pc.num_lights == 0) return 0.0;

    uint light_idx = uint(rand_float(seed) * float(pc.num_lights)) % pc.num_lights;
    GpuLight light = gScene[pc.light_buf_idx].Load<GpuLight>(
        light_idx * int(sizeof(GpuLight)));

    float3 light_dir;
    float  dist;
    float  pdf_light;

    if (light.type == 0)
    {
        float3 to_light = light.position - hit_pos;
        dist       = length(to_light);
        light_dir  = to_light / max(dist, 1e-6f);
        pdf_light  = dist * dist;
    }
    else
    {
        light_dir = normalize(-light.direction);
        dist      = 1e9f;
        pdf_light = 1.0f;
    }

    if (dot(light_dir, normal) <= 0.0f) return 0.0f;

    HitInfo shadow = trace_ray(hit_pos + normal * pc.ray_bias,
                               light_dir,
                               pc.ray_bias,
                               dist - 0.01f);
    if (shadow.hit) return 0.0f;

    float3 brdf_val = brdf_eval(normal, wo, light_dir,
                                mat.base_color.rgb,
                                mat.metallic,
                                mat.roughness);
    float n_dot_l = max(dot(normal, light_dir), 0.0f);
    float3 irradiance = light.emission * light.intensity /
                        (light.type == 0 ? (pdf_light + 1e-6f) : 1.0f);
    float pdf_bsdf   = brdf_pdf(wo, light_dir, normal, mat.roughness);
    float mis_weight = (light.type == 0) ? 1.0f :
        (pdf_light * pdf_light) / (pdf_light * pdf_light + pdf_bsdf * pdf_bsdf + 1e-10f);

    return brdf_val * irradiance * n_dot_l * mis_weight * float(pc.num_lights);
}

// ── Reservoir packing ──────────────────────────────────────────────────────

void write_empty_reservoir(uint2 pixel)
{
    gStorageImages[pc.reservoir_curr_pos_idx][pixel] = 0.0;
    gStorageImages[pc.reservoir_curr_nrm_idx][pixel] = float4(0.0, 0.0, 0.0, asfloat(0u));
    gStorageImages[pc.reservoir_curr_rad_idx][pixel] = 0.0;
    gStorageImages[pc.reservoir_curr_pre_idx][pixel] = float4(0.0, 0.0, 0.0, 0.0);
}

struct Reservoir
{
    float3 rc_pos;       // reconnection vertex position
    float3 rc_normal;    // reconnection vertex normal
    float3 suffix_rad;   // radiance arriving at rc vertex from path suffix
    float3 prefix_thp;   // throughput from primary to rc vertex (excl. geometry)
    float  w_sum;
    float  W;
    uint   M;
    float  valid;
};

Reservoir load_reservoir(uint2 pixel,
                         uint pos_idx, uint nrm_idx, uint rad_idx, uint pre_idx)
{
    float4 p = gStorageImages[pos_idx][pixel];
    float4 n = gStorageImages[nrm_idx][pixel];
    float4 r = gStorageImages[rad_idx][pixel];
    float4 e = gStorageImages[pre_idx][pixel];

    Reservoir res;
    res.rc_pos     = p.xyz;
    res.w_sum      = p.w;
    res.rc_normal  = n.xyz;
    res.M          = asuint(n.w);
    res.suffix_rad = r.xyz;
    res.W          = r.w;
    res.prefix_thp = e.xyz;
    res.valid      = e.w;
    return res;
}

void store_reservoir(uint2 pixel,
                     uint pos_idx, uint nrm_idx, uint rad_idx, uint pre_idx,
                     Reservoir res)
{
    gStorageImages[pos_idx][pixel] = float4(res.rc_pos, res.w_sum);
    gStorageImages[nrm_idx][pixel] = float4(res.rc_normal, asfloat(res.M));
    gStorageImages[rad_idx][pixel] = float4(res.suffix_rad, res.W);
    gStorageImages[pre_idx][pixel] = float4(res.prefix_thp, res.valid);
}

// ── Primary surface / history helpers (mirrors restir_gi.slang) ────────────

void write_primary_history(uint2 pixel, uint pos_idx, uint norm_idx,
                           float4 pos_value, float4 norm_value)
{
    gStorageImages[pos_idx][pixel]  = pos_value;
    gStorageImages[norm_idx][pixel] = norm_value;
}

void write_empty_primary_history(uint2 pixel, uint pos_idx, uint norm_idx)
{
    write_primary_history(pixel, pos_idx, norm_idx, float4(0.0f, 0.0f, 0.0f, -1.0f), 0.0f);
}

bool load_surface_from_images(uint2 pixel, uint pos_idx, uint norm_idx,
                              out float4 pos_value, out float4 norm_value,
                              out float3 primary_pos, out float3 primary_n,
                              out uint material_index)
{
    pos_value  = gStorageImages[pos_idx][pixel];
    norm_value = gStorageImages[norm_idx][pixel];
    if (pos_value.w < 0.0f)
    {
        primary_pos     = 0.0f;
        primary_n       = 0.0f;
        material_index  = 0u;
        return false;
    }
    primary_pos    = pos_value.xyz;
    primary_n      = normalize(norm_value.xyz);
    material_index = asuint(norm_value.w);
    return true;
}

bool load_primary_surface(uint2 pixel,
                          out float4 gbuf_pos, out float4 gbuf_norm,
                          out float3 primary_pos, out float3 primary_n,
                          out uint primary_mat_idx, out GpuMaterial primary_mat)
{
    if (!load_surface_from_images(pixel, pc.gbuf_pos_idx, pc.gbuf_norm_idx,
                                  gbuf_pos, gbuf_norm,
                                  primary_pos, primary_n, primary_mat_idx))
    {
        gbuf_pos    = 0.0;
        gbuf_norm   = 0.0;
        primary_mat = (GpuMaterial)0;
        return false;
    }
    primary_mat = gScene[pc.material_buf_idx].Load<GpuMaterial>(
        primary_mat_idx * int(sizeof(GpuMaterial)));
    return true;
}

bool reproject_to_prev_pixel(float4x4 prev_view_proj, float3 world_pos,
                             uint screen_w, uint screen_h, out uint2 prev_pixel)
{
    float4 clip = mul(prev_view_proj, float4(world_pos, 1.0f));
    if (clip.w <= 1.0e-6f) { prev_pixel = 0; return false; }
    float2 ndc = clip.xy / clip.w;
    if (any(abs(ndc) > 1.0f)) { prev_pixel = 0; return false; }
    float2 pixel_f = (ndc * 0.5f + 0.5f) * float2(screen_w, screen_h) - 0.5f;
    int2 pixel_i = int2(floor(pixel_f + 0.5f));
    if (pixel_i.x < 0 || pixel_i.y < 0
     || pixel_i.x >= int(screen_w) || pixel_i.y >= int(screen_h))
    { prev_pixel = 0; return false; }
    prev_pixel = uint2(pixel_i);
    return true;
}

bool is_reprojected_history_compatible(float3 current_pos, float3 current_n,
                                       uint current_material,
                                       float3 history_pos, float3 history_n,
                                       uint history_material, float3 camera_pos)
{
    if (current_material != history_material) return false;
    if (dot(current_n, history_n) < 0.9f)      return false;
    float view_dist   = max(length(current_pos - camera_pos), 1.0f);
    float plane_delta = abs(dot(history_pos - current_pos, current_n));
    float pos_delta   = length(history_pos - current_pos);
    return plane_delta <= 0.01f * view_dist && pos_delta <= 0.05f * view_dist;
}

bool find_reprojected_history_surface(float4x4 prev_view_proj, float3 world_pos,
                                      float3 current_n, uint current_material,
                                      float3 camera_pos, uint screen_w, uint screen_h,
                                      out uint2 history_pixel,
                                      out float3 history_pos, out float3 history_n)
{
    uint2 center_pixel;
    if (!reproject_to_prev_pixel(prev_view_proj, world_pos, screen_w, screen_h, center_pixel))
    {
        history_pixel = 0; history_pos = 0.0f; history_n = 0.0f;
        return false;
    }

    bool  found      = false;
    float best_error = 1e30f;
    history_pixel    = center_pixel;
    history_pos      = 0.0f;
    history_n        = 0.0f;

    for (int oy = -1; oy <= 1; ++oy)
    for (int ox = -1; ox <= 1; ++ox)
    {
        int2 candidate_i = int2(center_pixel) + int2(ox, oy);
        if (candidate_i.x < 0 || candidate_i.y < 0
         || candidate_i.x >= int(screen_w) || candidate_i.y >= int(screen_h))
            continue;

        uint2 candidate = uint2(candidate_i);
        float4 hp4, hn4;
        float3 cpos, cn;
        uint   cmat;
        if (!load_surface_from_images(candidate,
                                      pc.history_prev_pos_idx,
                                      pc.history_prev_norm_idx,
                                      hp4, hn4, cpos, cn, cmat))
            continue;
        if (!is_reprojected_history_compatible(world_pos, current_n,
                                               current_material,
                                               cpos, cn, cmat, camera_pos))
            continue;

        float plane_delta = abs(dot(cpos - world_pos, current_n));
        float pos_delta   = length(cpos - world_pos);
        float error       = plane_delta * 4.0f + pos_delta;
        if (error < best_error)
        {
            best_error    = error;
            history_pixel = candidate;
            history_pos   = cpos;
            history_n     = cn;
            found         = true;
        }
    }
    return found;
}

bool is_spatial_neighbor_compatible(float3 primary_pos, float3 primary_n,
                                    float3 camera_pos, uint2 neighbor_pixel,
                                    out float3 neighbor_primary_pos,
                                    out float3 neighbor_primary_n)
{
    float4 nb_pos4  = gStorageImages[pc.gbuf_pos_idx][neighbor_pixel];
    float4 nb_norm4 = gStorageImages[pc.gbuf_norm_idx][neighbor_pixel];
    if (nb_pos4.w < 0.0f)
    {
        neighbor_primary_pos = 0.0f;
        neighbor_primary_n   = 0.0f;
        return false;
    }
    neighbor_primary_pos = nb_pos4.xyz;
    neighbor_primary_n   = normalize(nb_norm4.xyz);
    if (dot(neighbor_primary_n, primary_n) < 0.85f)
        return false;
    float plane_delta = abs(dot(neighbor_primary_pos - primary_pos, primary_n));
    float dist_scale  = max(length(primary_pos - camera_pos), 1.0f);
    return plane_delta <= 0.02f * dist_scale;
}

// Geometric Jacobian of reconnecting the cached suffix (anchored at the
// reconnection vertex) from a source primary surface to a target primary
// surface. Same form as restir_gi.slang but used for the prefix's last
// segment (primary → reconnection vertex).
float reconnection_jacobian(float3 source_primary_pos,
                            float3 target_primary_pos,
                            float3 rc_pos, float3 rc_normal)
{
    float3 source_delta = rc_pos - source_primary_pos;
    float3 target_delta = rc_pos - target_primary_pos;
    float  source_dist2 = dot(source_delta, source_delta);
    float  target_dist2 = dot(target_delta, target_delta);
    if (source_dist2 <= 1e-8f || target_dist2 <= 1e-8f)
        return 0.0f;

    float3 source_dir = source_delta * rsqrt(source_dist2);
    float3 target_dir = target_delta * rsqrt(target_dist2);
    float  source_cos_y = abs(dot(rc_normal, -source_dir));
    float  target_cos_y = abs(dot(rc_normal, -target_dir));
    if (source_cos_y <= 0.05f || target_cos_y <= 0.05f)
        return 0.0f;

    float jacobian = (target_cos_y * source_dist2) / (source_cos_y * target_dist2);
    return clamp(jacobian, 0.0f, 3.0f);
}

bool trace_visibility(float3 origin, float3 normal, float3 target_pos)
{
    float3 delta = target_pos - origin;
    float  dist  = length(delta);
    if (dist <= pc.ray_bias) return false;
    float3 dir   = delta / dist;
    HitInfo hit = trace_ray(origin + normal * pc.ray_bias,
                            dir, pc.ray_bias, dist - pc.ray_bias);
    return !hit.hit;
}

// Evaluate the unshadowed contribution of a reservoir sample reconnected to
// the given primary surface: f = BRDF(primary) * cos * prefix * suffix.
// (geometry term between primary and rc vertex is folded via the Jacobian
//  applied by the caller during resampling.)
float3 evaluate_contribution(float3 primary_pos, float3 primary_n,
                             GpuMaterial primary_mat, float3 camera_pos,
                             Reservoir res)
{
    float3 to_rc = res.rc_pos - primary_pos;
    float  dist2 = dot(to_rc, to_rc);
    if (dist2 <= 1e-8f) return 0.0f;

    float3 wi    = to_rc * rsqrt(dist2);
    float  cos_x = max(dot(primary_n, wi), 0.0f);
    if (cos_x <= 0.0f) return 0.0f;

    float3 wo = normalize(camera_pos - primary_pos);
    float3 brdf_val = brdf_eval(primary_n, wo, wi,
                                primary_mat.base_color.rgb,
                                primary_mat.metallic,
                                primary_mat.roughness);
    return brdf_val * cos_x * res.prefix_thp * res.suffix_rad;
}

float target_function(float3 primary_pos, float3 primary_n,
                      GpuMaterial primary_mat, float3 camera_pos,
                      Reservoir res)
{
    return luminance(evaluate_contribution(primary_pos, primary_n,
                                           primary_mat, camera_pos, res));
}

// ── Path guiding helpers ─────────────────────────────────────────

GuidingGridParams make_grid_params()
{
    GuidingGridParams gp;
    gp.base_cell_size = pc.grid_base_cell_size;
    gp.distance_slope = pc.grid_distance_slope;
    gp.cell_count     = pc.grid_cell_count;
    gp.enabled        = pc.guiding_enable;
    return gp;
}

// Sample the primary bounce direction using one-sample MIS between a cosine
// hemisphere and the cell's fitted vMF lobe (paper §4.5). Returns the world
// direction `wi` and the *combined* source pdf used by RIS:
//   p(wi) = alpha * p_vmf(wi) + (1-alpha) * p_cos(wi)
// When guiding is disabled or the cell has no lobe, this degrades to a pure
// cosine sample, keeping the estimator unbiased regardless of guiding quality.
float3 sample_primary_dir(float3 primary_pos, float3 primary_n,
                          uint bsdf_id, float3 camera_pos,
                          inout uint seed, out float src_pdf)
{
    float  alpha = 0.0f;
    GuidingLobe lobe;
    lobe.valid = false;

    bool guiding_on = (pc.guiding_enable != 0u) && (pc.guiding_alpha > 0.0f);
    if (guiding_on)
    {
        GuidingGridParams gp = make_grid_params();
        uint cell = guiding_cell_index(primary_pos, bsdf_id, camera_pos, gp);
        lobe = guiding_load_lobe(pc.guiding_fitted_idx, cell);
        if (lobe.valid)
            alpha = pc.guiding_alpha;
    }

    float3 T, B;
    build_onb(primary_n, T, B);

    float3 wi;
    if (rand_float(seed) < alpha)
    {
        // Guided sample from the fitted vMF lobe.
        float vpdf;
        wi = vmf_sample(lobe.mu, lobe.kappa, rand_float2(seed), vpdf);
    }
    else
    {
        // Cosine-hemisphere fallback.
        float cpdf;
        float3 local = sample_cosine_hemisphere(rand_float(seed),
                                                rand_float(seed), cpdf);
        wi = tangent_to_world(local, T, B, primary_n);
    }

    // Combined MIS source pdf evaluated at the chosen direction.
    float cos_i = dot(primary_n, wi);
    float p_cos = (cos_i > 0.0f) ? (cos_i * kInvPi) : 0.0f;
    float p_vmf = lobe.valid ? vmf_pdf(lobe.mu, lobe.kappa, wi) : 0.0f;
    src_pdf = alpha * p_vmf + (1.0f - alpha) * p_cos;
    return wi;
}

// ── cs_initial : multi-bounce path tracing + RIS over candidates ───────────

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

    float4 primary_pos4, primary_norm4;
    float3 primary_pos, primary_n;
    uint primary_mat_idx;
    GpuMaterial primary_mat;
    if (!load_primary_surface(pixel, primary_pos4, primary_norm4,
                              primary_pos, primary_n,
                              primary_mat_idx, primary_mat))
    {
        write_empty_primary_history(pixel, pc.history_curr_pos_idx, pc.history_curr_norm_idx);
        write_empty_reservoir(pixel);
        return;
    }

    write_primary_history(pixel, pc.history_curr_pos_idx, pc.history_curr_norm_idx,
                          primary_pos4, primary_norm4);

    float3 cam_pos = cam.position;

    Reservoir selected;
    selected.rc_pos     = 0.0;
    selected.rc_normal  = 0.0;
    selected.suffix_rad = 0.0;
    selected.prefix_thp = 0.0;
    selected.valid      = 0.0;
    float w_sum  = 0.0;
    uint  M_count = 0;

    uint M = pc.num_initial_candidates;
    uint max_b = max(pc.max_bounces, 1u);

    for (uint c = 0; c < M; ++c)
    {
        // ── Trace one multi-bounce path from the primary surface ──────────
        //
        // The estimator mirrors restir_gi.slang exactly, generalized to
        // multiple bounces:
        //   - The reconnection vertex is the first secondary hit (x2).
        //   - suffix_rad is the *outgoing radiance leaving the rc vertex
        //     toward the primary*: emissive(rc) + NEE(rc) + reflected
        //     indirect from the continued path. (Same role as GI's
        //     evaluate_sample_radiance result, just with extra bounces.)
        //   - prefix_thp stays 1.0; the primary BRDF*cos and the inverse
        //     source pdf are applied at resolve time (W) and via the cosine
        //     source pdf in RIS — NOT folded here. Folding 1/pdf_brdf here
        //     while also dividing by the cosine source pdf double-counted the
        //     primary bounce and is what made the image far too bright.

        // First bounce: one-sample MIS between a cosine hemisphere and the
        // fitted vMF guiding lobe. src_pdf is the combined source pdf consumed by RIS below.
        // With guiding disabled / no lobe this is exactly the cosine sampler.
        float  src_pdf;
        float3 wi0 = sample_primary_dir(primary_pos, primary_n,
                                        primary_mat_idx, cam_pos,
                                        seed, src_pdf);
        if (src_pdf <= 1e-6f || dot(primary_n, wi0) <= 0.0f)
            continue;

        HitInfo rc_hit = trace_ray(primary_pos + primary_n * pc.ray_bias,
                                   wi0, pc.ray_bias, 1e9f);
        if (!rc_hit.hit)
            continue;

        float3 rc_pos    = rc_hit.pos;
        float3 rc_normal = rc_hit.normal;
        if (dot(rc_normal, wi0) > 0.0f) rc_normal = -rc_normal;
        GpuMaterial rc_mat = gScene[pc.material_buf_idx].Load<GpuMaterial>(
            rc_hit.material_index * int(sizeof(GpuMaterial)));

        // Outgoing radiance leaving the rc vertex toward the primary.
        float3 suffix_rad = rc_mat.emissive;
        float3 rc_wo      = -wi0;   // direction rc → primary
        suffix_rad += sample_lights(rc_pos, rc_normal, rc_wo, rc_mat, seed);

        // Continue bouncing from the rc vertex to gather reflected indirect.
        float3 cur_pos = rc_pos;
        float3 cur_n   = rc_normal;
        GpuMaterial cur_mat = rc_mat;
        float3 cur_wo  = rc_wo;
        float3 path_thp = 1.0;   // throughput rc → current vertex (excl. emit)

        for (uint b = 1; b < max_b; ++b)
        {
            float3 wi;
            float  pdf;
            float3 f = brdf_sample(cur_wo, cur_n,
                                   cur_mat.base_color.rgb,
                                   cur_mat.metallic,
                                   cur_mat.roughness,
                                   rand_float2(seed), wi, pdf);
            if (pdf <= 1e-6f || dot(cur_n, wi) <= 0.0f)
                break;

            float cos_at_cur = max(dot(cur_n, wi), 0.0f);
            path_thp *= f * cos_at_cur / pdf;
            if (max(path_thp.x, max(path_thp.y, path_thp.z)) <= 0.0f)
                break;

            HitInfo hit = trace_ray(cur_pos + cur_n * pc.ray_bias,
                                    wi, pc.ray_bias, 1e9f);
            if (!hit.hit)
            {
                suffix_rad += path_thp * sky_radiance(wi);
                break;
            }

            float3 hit_n = hit.normal;
            if (dot(hit_n, wi) > 0.0f) hit_n = -hit_n;
            GpuMaterial hit_mat = gScene[pc.material_buf_idx].Load<GpuMaterial>(
                hit.material_index * int(sizeof(GpuMaterial)));
            float3 hit_wo = -wi;

            // Reflected indirect at the next vertex: NEE only. Emissive of
            // hit surfaces is intentionally not added on continued bounces to
            // avoid double counting with NEE (no MIS between them here),
            // matching the single-bounce behavior of restir_gi.slang.
            suffix_rad += path_thp * sample_lights(hit.pos, hit_n, hit_wo,
                                                   hit_mat, seed);

            cur_pos = hit.pos;
            cur_n   = hit_n;
            cur_mat = hit_mat;
            cur_wo  = hit_wo;
        }

        // prefix_thp == 1.0 (see note above).
        float3 prefix_thp = 1.0;

        Reservoir cand;
        cand.rc_pos     = rc_pos;
        cand.rc_normal  = rc_normal;
        cand.suffix_rad = suffix_rad;
        cand.prefix_thp = prefix_thp;
        cand.M          = 1;
        cand.W          = 0.0;
        cand.w_sum      = 0.0;
        cand.valid      = 1.0;

        float p_hat = target_function(primary_pos, primary_n, primary_mat,
                                      cam_pos, cand);
        if (p_hat <= 0.0f)
            continue;

        // RIS source pdf is the combined MIS pdf of the direction we actually
        // sampled (returned by sample_primary_dir). This is the solid-angle
        // pdf of wi0; it correctly accounts for the guided component so the
        // estimator stays unbiased.
        float3 delta = rc_pos - primary_pos;
        float  d2    = dot(delta, delta);
        if (d2 <= 1e-8f) continue;

        float weight = p_hat / max(src_pdf, 1e-6f);
        w_sum += weight;
        M_count++;

        if (w_sum > 0.0f && rand_float(seed) < weight / w_sum)
            selected = cand;
    }

    if (M_count > 0 && selected.valid > 0.5f)
    {
        float p_hat = target_function(primary_pos, primary_n, primary_mat,
                                      cam_pos, selected);
        if (p_hat > 1e-10f)
        {
            selected.M     = M_count;
            selected.w_sum = w_sum;
            selected.W     = w_sum / (float(M_count) * p_hat);
            store_reservoir(pixel,
                            pc.reservoir_curr_pos_idx,
                            pc.reservoir_curr_nrm_idx,
                            pc.reservoir_curr_rad_idx,
                            pc.reservoir_curr_pre_idx,
                            selected);
            return;
        }
    }
    write_empty_reservoir(pixel);
}

// ── cs_temporal : reconnection-shift reuse of previous frame's reservoir ───

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_temporal(uint3 id : SV_DispatchThreadID)
{
    if (!bool(pc.temporal_enable)) return;

    GpuCameraData cam = gScene[pc.camera_buf_idx].Load<GpuCameraData>(0);
    if (id.x >= cam.screen_width || id.y >= cam.screen_height) return;
    uint2 pixel = id.xy;

    float4 pp4, pn4;
    float3 primary_pos, primary_n;
    uint primary_mat_idx;
    GpuMaterial primary_mat;
    if (!load_primary_surface(pixel, pp4, pn4, primary_pos, primary_n,
                              primary_mat_idx, primary_mat))
        return;

    uint2 history_pixel;
    float3 prev_primary_pos, prev_primary_n;
    if (!find_reprojected_history_surface(cam.prev_view_proj, primary_pos,
                                          primary_n, primary_mat_idx,
                                          cam.position,
                                          cam.screen_width, cam.screen_height,
                                          history_pixel,
                                          prev_primary_pos, prev_primary_n))
        return;

    Reservoir cur = load_reservoir(pixel,
                                   pc.reservoir_curr_pos_idx,
                                   pc.reservoir_curr_nrm_idx,
                                   pc.reservoir_curr_rad_idx,
                                   pc.reservoir_curr_pre_idx);
    Reservoir prv = load_reservoir(history_pixel,
                                   pc.reservoir_prev_pos_idx,
                                   pc.reservoir_prev_nrm_idx,
                                   pc.reservoir_prev_rad_idx,
                                   pc.reservoir_prev_pre_idx);

    if (prv.M == 0 || prv.valid < 0.5f)
        return;

    uint m_cap = clamp(20u * pc.num_initial_candidates, 1u, 8192u);
    prv.M = min(prv.M, m_cap);

    float jacobian = reconnection_jacobian(prev_primary_pos, primary_pos,
                                           prv.rc_pos, prv.rc_normal);
    if (jacobian <= 0.0f)
        return;

    if (!trace_visibility(primary_pos, primary_n, prv.rc_pos))
        return;

    float ph_prev = target_function(primary_pos, primary_n, primary_mat,
                                    cam.position, prv);
    float prev_W_capped = min(prv.W, 50.0f);
    float contrib = ph_prev * prev_W_capped * float(prv.M) * jacobian;
    if (contrib <= 0.0f)
        return;

    cur.w_sum += contrib;
    uint seed = make_seed(pixel, pc.frame_seed + 1u);
    if (cur.w_sum > 0.0f && rand_float(seed) < contrib / cur.w_sum)
    {
        cur.rc_pos     = prv.rc_pos;
        cur.rc_normal  = prv.rc_normal;
        cur.suffix_rad = prv.suffix_rad;
        cur.prefix_thp = prv.prefix_thp;
        cur.valid      = 1.0;
    }

    uint new_M = cur.M + prv.M;
    if (new_M > m_cap)
    {
        cur.w_sum *= float(m_cap) / float(new_M);
        cur.M = m_cap;
    }
    else
    {
        cur.M = new_M;
    }

    if (cur.M > 0 && cur.valid > 0.5f)
    {
        float ph_y = target_function(primary_pos, primary_n, primary_mat,
                                     cam.position, cur);
        cur.W = (ph_y > 1e-10f) ? (cur.w_sum / (float(cur.M) * ph_y)) : 0.0f;
        cur.W = min(cur.W, 50.0f);
    }

    store_reservoir(pixel,
                    pc.reservoir_curr_pos_idx,
                    pc.reservoir_curr_nrm_idx,
                    pc.reservoir_curr_rad_idx,
                    pc.reservoir_curr_pre_idx,
                    cur);
}

// ── cs_spatial_shade : spatial reuse + final shading ───────────────────────

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_spatial_shade(uint3 id : SV_DispatchThreadID)
{
    GpuCameraData cam = gScene[pc.camera_buf_idx].Load<GpuCameraData>(0);
    uint W = cam.screen_width;
    uint H = cam.screen_height;
    if (id.x >= W || id.y >= H) return;
    uint2 pixel = id.xy;

    float3 radiance = 0.0;
    if (pc.include_direct_lighting != 0 && pc.direct_input_idx != 0xffffffffu)
        radiance += gTextures[pc.direct_input_idx].Load(int3(pixel, 0)).rgb;

    float4 pp4, pn4;
    float3 primary_pos, primary_n;
    uint primary_mat_idx;
    GpuMaterial primary_mat;
    if (!load_primary_surface(pixel, pp4, pn4, primary_pos, primary_n,
                              primary_mat_idx, primary_mat))
    {
        if (pc.include_direct_lighting == 0)
        {
            float2 ndc = (float2(pixel) + 0.5f) / float2(W, H) * 2.0f - 1.0f;
            float4 rv  = mul(cam.inv_proj, float4(ndc.x, ndc.y, 1.0f, 1.0f));
            rv.z = -1.0f; rv.w = 0.0f;
            radiance += sky_radiance(normalize(mul(cam.inv_view, rv).xyz));
        }
        gStorageImages[pc.output_idx][pixel] = float4(radiance, 1.0f);
        return;
    }

    if (any(primary_mat.emissive > 0.0f))
    {
        radiance += primary_mat.emissive;
        gStorageImages[pc.output_idx][pixel] = float4(radiance, 1.0f);
        return;
    }

    Reservoir res = load_reservoir(pixel,
                                   pc.reservoir_curr_pos_idx,
                                   pc.reservoir_curr_nrm_idx,
                                   pc.reservoir_curr_rad_idx,
                                   pc.reservoir_curr_pre_idx);

    if (bool(pc.spatial_enable) && pc.spatial_num_neighbors > 0 && res.valid > 0.5f)
    {
        const float base_phase = frac(float(pcg_hash(pixel.x ^ (pixel.y * 65537u)))
                                      * 2.3283064365386963e-10f);
        for (uint k = 0; k < pc.spatial_num_neighbors; ++k)
        {
            float u      = (float(k) + 0.5f) / float(pc.spatial_num_neighbors);
            float angle  = frac(base_phase + 0.61803398875f * float(k)) * kTwoPi;
            float radius = sqrt(u) * float(pc.spatial_radius);
            int2 offs = int2(int(radius * cos(angle)), int(radius * sin(angle)));
            uint2 nb = uint2(clamp(int2(pixel) + offs,
                                   int2(0, 0),
                                   int2(int(W) - 1, int(H) - 1)));
            if (all(nb == pixel))
                continue;

            float3 nb_primary_pos, nb_primary_n;
            if (!is_spatial_neighbor_compatible(primary_pos, primary_n,
                                                cam.position, nb,
                                                nb_primary_pos, nb_primary_n))
                continue;

            Reservoir nbr = load_reservoir(nb,
                                           pc.reservoir_curr_pos_idx,
                                           pc.reservoir_curr_nrm_idx,
                                           pc.reservoir_curr_rad_idx,
                                           pc.reservoir_curr_pre_idx);
            if (nbr.M == 0 || nbr.valid < 0.5f)
                continue;

            float jacobian = reconnection_jacobian(nb_primary_pos, primary_pos,
                                                   nbr.rc_pos, nbr.rc_normal);
            if (jacobian <= 0.0f)
                continue;

            float contrib = target_function(primary_pos, primary_n, primary_mat,
                                            cam.position, nbr)
                          * nbr.W * float(nbr.M) * jacobian;
            if (contrib <= 0.0f)
                continue;

            res.w_sum += contrib;
            res.M     += nbr.M;

            float accept = frac(base_phase + 0.754877666f * float(k + 1u));
            if (res.w_sum > 0.0f && accept < contrib / res.w_sum)
            {
                res.rc_pos     = nbr.rc_pos;
                res.rc_normal  = nbr.rc_normal;
                res.suffix_rad = nbr.suffix_rad;
                res.prefix_thp = nbr.prefix_thp;
            }
        }

        if (res.M > 0)
        {
            float ph_y = target_function(primary_pos, primary_n, primary_mat,
                                         cam.position, res);
            res.W = (ph_y > 1e-10f) ? (res.w_sum / (float(res.M) * ph_y)) : 0.0f;
        }
    }

    if (res.M > 0 && res.W > 0.0f && res.valid > 0.5f)
    {
        float3 indirect = evaluate_contribution(primary_pos, primary_n,
                                                primary_mat, cam.position, res)
                        * res.W;
        if (luminance(indirect) > 0.0f
         && trace_visibility(primary_pos, primary_n, res.rc_pos))
            radiance += indirect * pc.indirect_strength;

        // ── Splat (paper §4.3) ───────────────────────────────────────────
        // The selected reservoir is (approximately) distributed by the path
        // contribution, so its bounce direction off the primary toward the
        // reconnection vertex is a sample of the ideal local guiding
        // distribution. Accumulate it into the primary cell for next-frame
        // guiding.
        if (pc.guiding_enable != 0u)
        {
            float3 to_rc = res.rc_pos - primary_pos;
            float  d2    = dot(to_rc, to_rc);
            if (d2 > 1e-8f)
            {
                float3 wi = to_rc * rsqrt(d2);
                GuidingGridParams gp = make_grid_params();
                uint cell = guiding_cell_index(primary_pos, primary_mat_idx,
                                               cam.position, gp);
                guiding_splat(pc.guiding_accum_idx, cell, wi);
            }
        }
    }

    gStorageImages[pc.output_idx][pixel] = float4(radiance, 1.0f);
}

// ── cs_fit_guiding : per-cell vMF fit (paper §4.4) ─────────────────────────
// One thread per cell. Reads the accumulator, fits a single vMF lobe via the
// Banerjee closed form, writes it to the fitted buffer (read by next frame's
// cs_initial), and resets the accumulator.

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_fit_guiding(uint3 id : SV_DispatchThreadID)
{
    uint cell = id.x;
    if (cell >= pc.grid_cell_count) return;
    if (pc.guiding_enable == 0u) return;

    guiding_fit_cell(pc.guiding_accum_idx, pc.guiding_fitted_idx, cell,
                     pc.grid_min_samples, pc.grid_kappa_max);
}
