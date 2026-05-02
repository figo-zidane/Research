// restir_gi.slang — ReSTIR GI with initial / temporal / spatial passes.
//
// Reservoir payload is split across three RGBA32F images:
//   pos  = float4(sample_pos.xyz, w_sum)
//   rad  = float4(sample_radiance.rgb, W)
//   meta = float4(sample_normal.xyz, asfloat(M))
//
// Temporal reuse validates reprojected primary surfaces against explicit
// history buffers, and temporal/spatial reuse apply a Jacobian correction when
// reconnecting a reused secondary sample to a new primary surface.

#include "BindlessHeap.slang"
#include "Common.slang"
#include "GpuScene.slang"
#include "BRDF.slang"
#include "Sampling.slang"

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
    uint reservoir_curr_rad_idx;
    uint reservoir_curr_meta_idx;
    uint reservoir_prev_pos_idx;
    uint reservoir_prev_rad_idx;
    uint reservoir_prev_meta_idx;
    uint history_curr_pos_idx;
    uint history_curr_norm_idx;
    uint history_prev_pos_idx;
    uint history_prev_norm_idx;
    uint output_idx;
    uint num_initial_candidates;
    uint temporal_enable;
    uint spatial_enable;
    uint spatial_num_neighbors;
    uint spatial_radius;
    uint include_direct_lighting;
    float indirect_strength;
    float ray_bias;
    uint _pad0;
    uint _pad1;
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

void write_empty_reservoir(uint2 pixel, uint pos_idx, uint rad_idx, uint meta_idx)
{
    gStorageImages[pos_idx][pixel]  = 0.0;
    gStorageImages[rad_idx][pixel]  = 0.0;
    gStorageImages[meta_idx][pixel] = float4(0.0, 0.0, 0.0, asfloat(0u));
}

void unpack_reservoir(uint2 pixel,
                      uint pos_idx,
                      uint rad_idx,
                      uint meta_idx,
                      out float3 sample_pos,
                      out float3 sample_radiance,
                      out float3 sample_normal,
                      out float  w_sum,
                      out float  W,
                      out uint   M)
{
    float4 pos_v  = gStorageImages[pos_idx][pixel];
    float4 rad_v  = gStorageImages[rad_idx][pixel];
    float4 meta_v = gStorageImages[meta_idx][pixel];

    sample_pos      = pos_v.xyz;
    sample_radiance = rad_v.xyz;
    sample_normal   = meta_v.xyz;
    w_sum           = pos_v.w;
    W               = rad_v.w;
    M               = asuint(meta_v.w);
}

void pack_reservoir(uint2 pixel,
                    uint pos_idx,
                    uint rad_idx,
                    uint meta_idx,
                    float3 sample_pos,
                    float3 sample_radiance,
                    float3 sample_normal,
                    float  w_sum,
                    float  W,
                    uint   M)
{
    gStorageImages[pos_idx][pixel]  = float4(sample_pos, w_sum);
    gStorageImages[rad_idx][pixel]  = float4(sample_radiance, W);
    gStorageImages[meta_idx][pixel] = float4(sample_normal, asfloat(M));
}

void write_primary_history(uint2 pixel,
                           uint pos_idx,
                           uint norm_idx,
                           float4 pos_value,
                           float4 norm_value)
{
    gStorageImages[pos_idx][pixel]  = pos_value;
    gStorageImages[norm_idx][pixel] = norm_value;
}

void write_empty_primary_history(uint2 pixel, uint pos_idx, uint norm_idx)
{
    write_primary_history(pixel, pos_idx, norm_idx, float4(0.0f, 0.0f, 0.0f, -1.0f), 0.0f);
}

bool load_surface_from_images(uint2 pixel,
                              uint pos_idx,
                              uint norm_idx,
                              out float4 pos_value,
                              out float4 norm_value,
                              out float3 primary_pos,
                              out float3 primary_n,
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
                          out float4 gbuf_pos,
                          out float4 gbuf_norm,
                          out float3 primary_pos,
                          out float3 primary_n,
                          out uint primary_mat_idx,
                          out GpuMaterial primary_mat)
{
    if (!load_surface_from_images(pixel,
                                  pc.gbuf_pos_idx,
                                  pc.gbuf_norm_idx,
                                  gbuf_pos,
                                  gbuf_norm,
                                  primary_pos,
                                  primary_n,
                                  primary_mat_idx))
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

bool reproject_to_prev_pixel(float4x4 prev_view_proj,
                             float3 world_pos,
                             uint   screen_w,
                             uint   screen_h,
                             out uint2 prev_pixel)
{
    float4 clip = mul(prev_view_proj, float4(world_pos, 1.0f));
    if (clip.w <= 1.0e-6f)
    {
        prev_pixel = 0;
        return false;
    }

    float2 ndc = clip.xy / clip.w;
    if (any(abs(ndc) > 1.0f))
    {
        prev_pixel = 0;
        return false;
    }

    float2 pixel_f = (ndc * 0.5f + 0.5f) * float2(screen_w, screen_h) - 0.5f;
    int2 pixel_i = int2(floor(pixel_f + 0.5f));
    if (pixel_i.x < 0 || pixel_i.y < 0 || pixel_i.x >= int(screen_w) || pixel_i.y >= int(screen_h))
    {
        prev_pixel = 0;
        return false;
    }

    prev_pixel = uint2(pixel_i);
    return true;
}

bool is_reprojected_history_compatible(float3 current_pos,
                                       float3 current_n,
                                       uint   current_material,
                                       float3 history_pos,
                                       float3 history_n,
                                       uint   history_material,
                                       float3 camera_pos)
{
    if (current_material != history_material)
        return false;

    if (dot(current_n, history_n) < 0.9f)
        return false;

    float view_dist   = max(length(current_pos - camera_pos), 1.0f);
    float plane_delta = abs(dot(history_pos - current_pos, current_n));
    float pos_delta   = length(history_pos - current_pos);
    return plane_delta <= 0.01f * view_dist && pos_delta <= 0.05f * view_dist;
}

bool find_reprojected_history_surface(float4x4 prev_view_proj,
                                      float3 world_pos,
                                      float3 current_n,
                                      uint   current_material,
                                      float3 camera_pos,
                                      uint   screen_w,
                                      uint   screen_h,
                                      out uint2 history_pixel,
                                      out float3 history_pos,
                                      out float3 history_n)
{
    uint2 center_pixel;
    if (!reproject_to_prev_pixel(prev_view_proj, world_pos, screen_w, screen_h, center_pixel))
    {
        history_pixel = 0;
        history_pos   = 0.0f;
        history_n     = 0.0f;
        return false;
    }

    bool  found      = false;
    float best_error = 1e30f;
    history_pixel    = center_pixel;
    history_pos      = 0.0f;
    history_n        = 0.0f;

    for (int oy = -1; oy <= 1; ++oy)
    {
        for (int ox = -1; ox <= 1; ++ox)
        {
            int2 candidate_i = int2(center_pixel) + int2(ox, oy);
            if (candidate_i.x < 0 || candidate_i.y < 0
             || candidate_i.x >= int(screen_w) || candidate_i.y >= int(screen_h))
                continue;

            uint2 candidate = uint2(candidate_i);
            float4 history_pos4;
            float4 history_norm4;
            float3 candidate_pos;
            float3 candidate_n;
            uint   candidate_material;
            if (!load_surface_from_images(candidate,
                                          pc.history_prev_pos_idx,
                                          pc.history_prev_norm_idx,
                                          history_pos4,
                                          history_norm4,
                                          candidate_pos,
                                          candidate_n,
                                          candidate_material))
                continue;

            if (!is_reprojected_history_compatible(world_pos,
                                                   current_n,
                                                   current_material,
                                                   candidate_pos,
                                                   candidate_n,
                                                   candidate_material,
                                                   camera_pos))
                continue;

            float plane_delta = abs(dot(candidate_pos - world_pos, current_n));
            float pos_delta   = length(candidate_pos - world_pos);
            float error       = plane_delta * 4.0f + pos_delta;
            if (error < best_error)
            {
                best_error   = error;
                history_pixel = candidate;
                history_pos   = candidate_pos;
                history_n     = candidate_n;
                found         = true;
            }
        }
    }

    return found;
}

bool is_spatial_neighbor_compatible(float3 primary_pos,
                                    float3 primary_n,
                                    float3 camera_pos,
                                    uint2  neighbor_pixel,
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

float sample_reconnection_jacobian(float3 source_primary_pos,
                                   float3 target_primary_pos,
                                   float3 sample_pos,
                                   float3 sample_normal)
{
    float3 source_delta = sample_pos - source_primary_pos;
    float3 target_delta = sample_pos - target_primary_pos;
    float  source_dist2 = dot(source_delta, source_delta);
    float  target_dist2 = dot(target_delta, target_delta);
    if (source_dist2 <= 1e-8f || target_dist2 <= 1e-8f)
        return 0.0f;

    float3 source_dir = source_delta * rsqrt(source_dist2);
    float3 target_dir = target_delta * rsqrt(target_dist2);
    float  source_cos_y = abs(dot(sample_normal, -source_dir));
    float  target_cos_y = abs(dot(sample_normal, -target_dir));
    // Reject grazing reconnections: when source_cos_y is tiny, the jacobian explodes
    // and even after clamping it amplifies the prior contribution, causing single-pixel
    // spikes that persist via temporal reuse.
    if (source_cos_y <= 0.05f || target_cos_y <= 0.05f)
        return 0.0f;

    float jacobian = (target_cos_y * source_dist2) / (source_cos_y * target_dist2);
    return clamp(jacobian, 0.0f, 3.0f);
}

float3 evaluate_indirect_radiance(float3 primary_pos,
                                  float3 primary_n,
                                  GpuMaterial primary_mat,
                                  float3 sample_pos,
                                  float3 sample_radiance,
                                  float3 camera_pos)
{
    float3 to_sample = sample_pos - primary_pos;
    float  dist2     = dot(to_sample, to_sample);
    if (dist2 <= 1e-8f) return 0.0f;

    float3 wi     = to_sample * rsqrt(dist2);
    float  cos_x  = max(dot(primary_n, wi), 0.0f);
    if (cos_x <= 0.0f) return 0.0f;

    float3 wo = normalize(camera_pos - primary_pos);
    float3 brdf_val = brdf_eval(primary_n, wo, wi,
                                primary_mat.base_color.rgb,
                                primary_mat.metallic,
                                primary_mat.roughness);
    return brdf_val * sample_radiance * cos_x;
}

float sample_primary_pdf(float3 primary_pos,
                         float3 primary_n,
                         float3 sample_pos)
{
    float3 delta = sample_pos - primary_pos;
    float  dist2 = dot(delta, delta);
    if (dist2 <= 1e-8f)
        return 0.0f;

    float3 wi    = delta * rsqrt(dist2);
    float  cos_x = max(dot(primary_n, wi), 0.0f);
    return max(cos_x * kInvPi, 0.0f);
}

float compute_target_function(float3 primary_pos,
                              float3 primary_n,
                              GpuMaterial primary_mat,
                              float3 sample_pos,
                              float3 sample_radiance,
                              float3 camera_pos)
{
    return luminance(evaluate_indirect_radiance(primary_pos,
                                                primary_n,
                                                primary_mat,
                                                sample_pos,
                                                sample_radiance,
                                                camera_pos));
}

float3 evaluate_sample_radiance(HitInfo secondary, float3 wo, inout uint seed)
{
    float3 secondary_n = secondary.normal;
    if (dot(secondary_n, wo) < 0.0f)
        secondary_n = -secondary_n;

    GpuMaterial secondary_mat = gScene[pc.material_buf_idx].Load<GpuMaterial>(
        secondary.material_index * int(sizeof(GpuMaterial)));

    float3 incoming = secondary_mat.emissive;
    incoming += sample_lights(secondary.pos, secondary_n, wo, secondary_mat, seed);
    return incoming;
}

bool trace_visibility(float3 origin, float3 normal, float3 target_pos)
{
    float3 delta = target_pos - origin;
    float  dist  = length(delta);
    if (dist <= pc.ray_bias) return false;
    float3 dir   = delta / dist;
    HitInfo hit = trace_ray(origin + normal * pc.ray_bias,
                            dir,
                            pc.ray_bias,
                            dist - pc.ray_bias);
    return !hit.hit;
}

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

    float4 primary_pos4;
    float4 primary_norm4;
    float3 primary_pos;
    float3 primary_n;
    uint primary_mat_idx;
    GpuMaterial primary_mat;
    if (!load_primary_surface(pixel,
                              primary_pos4,
                              primary_norm4,
                              primary_pos,
                              primary_n,
                              primary_mat_idx,
                              primary_mat))
    {
        write_empty_primary_history(pixel, pc.history_curr_pos_idx, pc.history_curr_norm_idx);
        write_empty_reservoir(pixel,
                              pc.reservoir_curr_pos_idx,
                              pc.reservoir_curr_rad_idx,
                              pc.reservoir_curr_meta_idx);
        return;
    }

    write_primary_history(pixel,
                          pc.history_curr_pos_idx,
                          pc.history_curr_norm_idx,
                          primary_pos4,
                          primary_norm4);

    float3 selected_pos      = 0.0;
    float3 selected_radiance = 0.0;
    float3 selected_normal   = 0.0;
    float  w_sum             = 0.0;
    uint   M_count           = 0;

    uint M = pc.num_initial_candidates;
    for (uint i = 0; i < M; ++i)
    {
        float pdf_dir;
        float3 local_dir = sample_cosine_hemisphere(rand_float(seed), rand_float(seed), pdf_dir);
        float3 T, B;
        build_onb(primary_n, T, B);
        float3 bounce_dir = tangent_to_world(local_dir, T, B, primary_n);
        if (dot(primary_n, bounce_dir) <= 0.0f)
            continue;

        HitInfo secondary = trace_ray(primary_pos + primary_n * pc.ray_bias,
                                      bounce_dir,
                                      pc.ray_bias,
                                      1e9f);
        if (!secondary.hit)
            continue;

        float3 candidate_radiance = evaluate_sample_radiance(secondary, -bounce_dir, seed);
        float candidate_target = compute_target_function(primary_pos,
                                                         primary_n,
                                                         primary_mat,
                                                         secondary.pos,
                                                         candidate_radiance,
                                                         cam.position);
        if (candidate_target <= 0.0f)
            continue;

        float pdf = sample_primary_pdf(primary_pos, primary_n, secondary.pos);
        if (pdf <= 1e-6f)
            continue;

        float weight = candidate_target / pdf;
        w_sum += weight;
        M_count++;

        if (w_sum > 0.0f && rand_float(seed) < weight / w_sum)
        {
            selected_pos      = secondary.pos;
            selected_radiance = candidate_radiance;
            selected_normal   = secondary.normal;
        }
    }

    if (M_count > 0)
    {
        float ph_y = compute_target_function(primary_pos,
                                             primary_n,
                                             primary_mat,
                                             selected_pos,
                                             selected_radiance,
                                             cam.position);
        if (ph_y > 1e-10f)
        {
            float W_final = w_sum / (float(M_count) * ph_y);
            pack_reservoir(pixel,
                           pc.reservoir_curr_pos_idx,
                           pc.reservoir_curr_rad_idx,
                           pc.reservoir_curr_meta_idx,
                           selected_pos,
                           selected_radiance,
                           selected_normal,
                           w_sum,
                           W_final,
                           M_count);
            return;
        }
    }
    write_empty_reservoir(pixel,
                          pc.reservoir_curr_pos_idx,
                          pc.reservoir_curr_rad_idx,
                          pc.reservoir_curr_meta_idx);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_temporal(uint3 id : SV_DispatchThreadID)
{
    if (!bool(pc.temporal_enable)) return;

    GpuCameraData cam = gScene[pc.camera_buf_idx].Load<GpuCameraData>(0);
    if (id.x >= cam.screen_width || id.y >= cam.screen_height) return;

    uint2 pixel = id.xy;

    float4 primary_pos4;
    float4 primary_norm4;
    float3 primary_pos;
    float3 primary_n;
    uint primary_mat_idx;
    GpuMaterial primary_mat;
    if (!load_primary_surface(pixel,
                              primary_pos4,
                              primary_norm4,
                              primary_pos,
                              primary_n,
                              primary_mat_idx,
                              primary_mat))
        return;

    uint2 history_pixel;
    float3 prev_primary_pos;
    float3 prev_primary_n;
    if (!find_reprojected_history_surface(cam.prev_view_proj,
                                          primary_pos,
                                          primary_n,
                                          primary_mat_idx,
                                          cam.position,
                                          cam.screen_width,
                                          cam.screen_height,
                                          history_pixel,
                                          prev_primary_pos,
                                          prev_primary_n))
        return;

    float3 curr_pos;
    float3 curr_rad;
    float3 curr_normal;
    float  curr_wsum;
    float  curr_W;
    uint   curr_M;
    unpack_reservoir(pixel,
                     pc.reservoir_curr_pos_idx,
                     pc.reservoir_curr_rad_idx,
                     pc.reservoir_curr_meta_idx,
                     curr_pos,
                     curr_rad,
                     curr_normal,
                     curr_wsum,
                     curr_W,
                     curr_M);

    float3 prev_pos;
    float3 prev_rad;
    float3 prev_normal;
    float  prev_wsum;
    float  prev_W;
    uint   prev_M;
    unpack_reservoir(history_pixel,
                     pc.reservoir_prev_pos_idx,
                     pc.reservoir_prev_rad_idx,
                     pc.reservoir_prev_meta_idx,
                     prev_pos,
                     prev_rad,
                     prev_normal,
                     prev_wsum,
                     prev_W,
                     prev_M);

    if (prev_M == 0)
        return;

    uint m_cap = clamp(20u * pc.num_initial_candidates, 1u, 8192u);
    prev_M = min(prev_M, m_cap);

    float jacobian = sample_reconnection_jacobian(prev_primary_pos,
                                                  primary_pos,
                                                  prev_pos,
                                                  prev_normal);
    if (jacobian <= 0.0f)
        return;

    // Visibility re-check: a sample that was visible from prev_primary_pos may be
    // occluded from the current primary_pos. Without this test, stale occluded
    // samples persist in the reservoir for many frames (M-cap = 160), producing
    // dark blocks where the spatial-shade visibility test then rejects them and
    // bright blocks where they happen to pass while still being numerically wrong.
    if (!trace_visibility(primary_pos, primary_n, prev_pos))
        return;

    float ph_prev = compute_target_function(primary_pos,
                                            primary_n,
                                            primary_mat,
                                            prev_pos,
                                            prev_rad,
                                            cam.position);
    // Defensive cap mirroring W_new clamp below: a stale reservoir with an inflated
    // W must not pour a giant contrib into curr_wsum.
    float prev_W_capped = min(prev_W, 50.0f);
    float contrib = ph_prev * prev_W_capped * float(prev_M) * jacobian;
    if (contrib <= 0.0f)
        return;

    // WRS sample selection MUST use the unscaled wsum, otherwise the accept ratio
    // becomes inconsistent with the candidate weight and the prev sample is over-/
    // under-accepted, producing persistent patches.
    curr_wsum += contrib;
    uint seed = make_seed(pixel, pc.frame_seed + 1u);
    if (curr_wsum > 0.0f && rand_float(seed) < contrib / curr_wsum)
    {
        curr_pos    = prev_pos;
        curr_rad    = prev_rad;
        curr_normal = prev_normal;
    }

    // Apply M-cap *after* WRS selection so that the chosen sample's selection
    // probability is unaffected by the scaling.
    uint new_M = curr_M + prev_M;
    if (new_M > m_cap)
    {
        float scale = float(m_cap) / float(new_M);
        curr_wsum *= scale;
        curr_M = m_cap;
    }
    else
    {
        curr_M = new_M;
    }

    float W_new = 0.0f;
    if (curr_M > 0)
    {
        float ph_y = compute_target_function(primary_pos,
                                             primary_n,
                                             primary_mat,
                                             curr_pos,
                                             curr_rad,
                                             cam.position);
        W_new = (ph_y > 1e-10f) ? (curr_wsum / (float(curr_M) * ph_y)) : 0.0f;
        // Tight cap on UCW: a well-behaved reservoir has W ≈ 1/p_hat. Allowing W to
        // grow into the thousands lets a single bad sample dominate temporal reuse
        // for many subsequent frames, producing persistent spike noise.
        W_new = min(W_new, 50.0f);
    }

    pack_reservoir(pixel,
                   pc.reservoir_curr_pos_idx,
                   pc.reservoir_curr_rad_idx,
                   pc.reservoir_curr_meta_idx,
                   curr_pos,
                   curr_rad,
                   curr_normal,
                   curr_wsum,
                   W_new,
                   curr_M);
}

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

    float4 primary_pos4;
    float4 primary_norm4;
    float3 primary_pos;
    float3 primary_n;
    uint primary_mat_idx;
    GpuMaterial primary_mat;
    if (!load_primary_surface(pixel,
                              primary_pos4,
                              primary_norm4,
                              primary_pos,
                              primary_n,
                              primary_mat_idx,
                              primary_mat))
    {
        if (pc.include_direct_lighting == 0)
        {
            float2 ndc = (float2(pixel) + 0.5f) / float2(W, H) * 2.0f - 1.0f;
            float4 rv  = mul(cam.inv_proj, float4(ndc.x, ndc.y, 1.0f, 1.0f));
            rv.z = -1.0f;
            rv.w = 0.0f;
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

    float3 sample_pos;
    float3 sample_rad;
    float3 sample_normal;
    float  r_wsum;
    float  r_W;
    uint   r_M;
    unpack_reservoir(pixel,
                     pc.reservoir_curr_pos_idx,
                     pc.reservoir_curr_rad_idx,
                     pc.reservoir_curr_meta_idx,
                     sample_pos,
                     sample_rad,
                     sample_normal,
                     r_wsum,
                     r_W,
                     r_M);

    if (bool(pc.spatial_enable) && pc.spatial_num_neighbors > 0)
    {
        const float base_phase = frac(float(pcg_hash(pixel.x ^ (pixel.y * 65537u))) * 2.3283064365386963e-10f);
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

            float3 neighbor_primary_pos;
            float3 neighbor_primary_n;
            if (!is_spatial_neighbor_compatible(primary_pos,
                                                primary_n,
                                                cam.position,
                                                nb,
                                                neighbor_primary_pos,
                                                neighbor_primary_n))
                continue;

            float3 nb_pos;
            float3 nb_rad;
            float3 nb_normal;
            float  nb_wsum;
            float  nb_W;
            uint   nb_M;
            unpack_reservoir(nb,
                             pc.reservoir_curr_pos_idx,
                             pc.reservoir_curr_rad_idx,
                             pc.reservoir_curr_meta_idx,
                             nb_pos,
                             nb_rad,
                             nb_normal,
                             nb_wsum,
                             nb_W,
                             nb_M);

            if (nb_M == 0)
                continue;

            float jacobian = sample_reconnection_jacobian(neighbor_primary_pos,
                                                          primary_pos,
                                                          nb_pos,
                                                          nb_normal);
            if (jacobian <= 0.0f)
                continue;

            float contrib = compute_target_function(primary_pos,
                                                    primary_n,
                                                    primary_mat,
                                                    nb_pos,
                                                    nb_rad,
                                                    cam.position)
                          * nb_W * float(nb_M) * jacobian;
            if (contrib <= 0.0f)
                continue;

            r_wsum += contrib;
            r_M    += nb_M;

            float accept = frac(base_phase + 0.754877666f * float(k + 1u));
            if (r_wsum > 0.0f && accept < contrib / r_wsum)
            {
                sample_pos    = nb_pos;
                sample_rad    = nb_rad;
                sample_normal = nb_normal;
            }
        }

        if (r_M > 0)
        {
            float ph_y = compute_target_function(primary_pos,
                                                 primary_n,
                                                 primary_mat,
                                                 sample_pos,
                                                 sample_rad,
                                                 cam.position);
            r_W = (ph_y > 1e-10f) ? (r_wsum / (float(r_M) * ph_y)) : 0.0f;
        }
    }

    if (r_M > 0 && r_W > 0.0f)
    {
        float3 indirect = evaluate_indirect_radiance(primary_pos,
                                                     primary_n,
                                                     primary_mat,
                                                     sample_pos,
                                                     sample_rad,
                                                     cam.position) * r_W;
        if (luminance(indirect) > 0.0f && trace_visibility(primary_pos, primary_n, sample_pos))
            radiance += indirect * pc.indirect_strength;
    }

    gStorageImages[pc.output_idx][pixel] = float4(radiance, 1.0f);
}