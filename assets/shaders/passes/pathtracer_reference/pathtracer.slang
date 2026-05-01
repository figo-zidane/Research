// pathtracer.slang — Reference path tracer using RayQuery (inline ray tracing)
// Compute shader: one thread per pixel, 8x8 tiles.

#include "BindlessHeap.slang"
#include "Common.slang"
#include "GpuScene.slang"
#include "BRDF.slang"
#include "Sampling.slang"

struct PushConstants
{
    uint camera_buf_idx;
    uint tlas_idx;
    uint radiance_img_idx;
    uint vertex_buf_idx;
    uint index_buf_idx;
    uint mesh_buf_idx;
    uint material_buf_idx;
    uint instance_buf_idx;
    uint light_buf_idx;
    uint num_instances;
    uint num_lights;
    uint linear_sampler_idx;
    uint frame_seed;
    uint max_bounces;
    uint _pad1;
    uint _pad2;
};

[[vk::push_constant]] PushConstants pc;

// \u2500 Intersection helper \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

struct HitInfo
{
    float3 pos;
    float3 normal;
    float3 geom_normal;
    float2 uv;
    uint   material_index;
    bool   hit;
    float  t;
};

HitInfo trace(float3 origin, float3 dir, float t_min, float t_max)
{
    HitInfo info;
    info.hit = false;

    RayQuery<RAY_FLAG_CULL_BACK_FACING_TRIANGLES> rq;
    RayDesc ray;
    ray.Origin    = origin;
    ray.TMin      = t_min;
    ray.Direction = dir;
    ray.TMax      = t_max;
    rq.TraceRayInline(
        gTLAS[pc.tlas_idx],
        RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
        0xFF,
        ray
    );

    while (rq.Proceed())
    {
        if (rq.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            rq.CommitNonOpaqueTriangleHit();
    }

    if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        info.hit       = true;
        info.t         = rq.CommittedRayT();
        uint inst_idx  = rq.CommittedInstanceID();
        uint prim_idx  = rq.CommittedPrimitiveIndex();
        float2 bary    = rq.CommittedTriangleBarycentrics();

        // Fetch mesh info
        GpuInstance inst = gScene[pc.instance_buf_idx].Load<GpuInstance>(
            inst_idx * int(sizeof(GpuInstance)));
        GpuMesh mesh = gScene[pc.mesh_buf_idx].Load<GpuMesh>(
            inst.mesh_index * int(sizeof(GpuMesh)));

        // Fetch triangle vertices
        uint base_byte = mesh.index_byte_offset + prim_idx * 3 * 4;
        uint i0 = gScene[pc.index_buf_idx].Load<uint>(int(base_byte));
        uint i1 = gScene[pc.index_buf_idx].Load<uint>(int(base_byte + 4));
        uint i2 = gScene[pc.index_buf_idx].Load<uint>(int(base_byte + 8));
        GpuVertex v0 = gScene[pc.vertex_buf_idx].Load<GpuVertex>(
            int((mesh.vertex_base + i0) * sizeof(GpuVertex)));
        GpuVertex v1 = gScene[pc.vertex_buf_idx].Load<GpuVertex>(
            int((mesh.vertex_base + i1) * sizeof(GpuVertex)));
        GpuVertex v2 = gScene[pc.vertex_buf_idx].Load<GpuVertex>(
            int((mesh.vertex_base + i2) * sizeof(GpuVertex)));

        float3 b3 = float3(1.0 - bary.x - bary.y, bary.x, bary.y);

        // Interpolate
        float3 lp = b3.x * v0.position + b3.y * v1.position + b3.z * v2.position;
        float3 ln = normalize(b3.x * v0.normal   + b3.y * v1.normal   + b3.z * v2.normal);
        float2 luv = b3.x * v0.uv      + b3.y * v1.uv      + b3.z * v2.uv;

        // Transform to world space using instance transform
        float4 world_pos  = mul(inst.transform, float4(lp, 1.0));
        // For normals use inverse-transpose (skip for uniform scale Cornell box)
        float3 world_norm = normalize(mul((float3x3)inst.transform, ln));

        // Geometric normal from cross product
        float3 e0 = v1.position - v0.position;
        float3 e1 = v2.position - v0.position;
        float3 geom_n = normalize(mul((float3x3)inst.transform, cross(e0, e1)));
        if (dot(geom_n, -dir) < 0.0) geom_n = -geom_n;

        info.pos           = world_pos.xyz;
        info.normal        = world_norm;
        info.geom_normal   = geom_n;
        info.uv            = luv;
        info.material_index = mesh.material_index;
    }
    return info;
}

// \u2500 Direct light sampling (NEE) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

float3 sample_lights(float3 hit_pos, float3 normal, float3 wo,
                      GpuMaterial mat, inout uint seed)
{
    if (pc.num_lights == 0) return 0.0;

    // Pick a random light
    uint light_idx = uint(rand_float(seed) * float(pc.num_lights)) % pc.num_lights;
    GpuLight light = gScene[pc.light_buf_idx].Load<GpuLight>(
        light_idx * int(sizeof(GpuLight)));

    float3 light_dir;
    float  dist;
    float  pdf_light;

    if (light.type == 0) // point
    {
        float3 to_light = light.position - hit_pos;
        dist       = length(to_light);
        light_dir  = to_light / dist;
        float r2   = dist * dist;
        pdf_light  = r2; // proportional for MIS (simplified: no area measure here)
    }
    else // directional
    {
        light_dir = normalize(-light.direction);
        dist      = 1e9;
        pdf_light = 1.0;
    }

    if (dot(light_dir, normal) <= 0.0) return 0.0;

    // Shadow ray
    HitInfo shadow = trace(hit_pos + normal * 1e-4, light_dir, 0.001, dist - 0.01);
    if (shadow.hit) return 0.0;

    // Evaluate BRDF
    float3 wi_world = light_dir;
    float3 brdf_val = brdf_eval(normal, wo, wi_world, mat.base_color.rgb,
                                  mat.metallic, mat.roughness);

    float n_dot_l = max(dot(normal, wi_world), 0.0);

    // Point light irradiance: emission / r^2
    float3 irradiance = light.emission * light.intensity /
                        (light.type == 0 ? (pdf_light + 1e-6) : 1.0);

    // MIS weight (simplified: only NEE for point lights)
    float pdf_bsdf   = brdf_pdf(wo, wi_world, normal, mat.roughness);
    float mis_weight = (light.type == 0) ? 1.0 :
        (pdf_light * pdf_light) / (pdf_light * pdf_light + pdf_bsdf * pdf_bsdf + 1e-10);

    return brdf_val * irradiance * n_dot_l * mis_weight
           * float(pc.num_lights); // compensate for random light pick
}

// \u2500 Main compute shader \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dispatch_id : SV_DispatchThreadID)
{
    GpuCameraData cam = gScene[pc.camera_buf_idx].Load<GpuCameraData>(0);
    uint W = cam.screen_width;
    uint H = cam.screen_height;

    if (dispatch_id.x >= W || dispatch_id.y >= H) return;

    uint2 pixel = dispatch_id.xy;
    uint  seed  = make_seed(pixel, pc.frame_seed);

    // Generate primary ray using inv_proj + inv_view
    float2 ndc = (float2(pixel) + float2(rand_float(seed), rand_float(seed)))
                 / float2(W, H) * 2.0 - 1.0;
    // Vulkan NDC: y increases downward in clip space. Our inv_proj already handles flip.
    float4 ray_clip  = float4(ndc.x, ndc.y, 1.0, 1.0);
    float4 ray_view  = mul(cam.inv_proj, ray_clip);
    ray_view.z = 1.0; ray_view.w = 0.0;
    float4 ray_world = mul(cam.inv_view, ray_view);
    float3 ray_dir   = normalize(ray_world.xyz);
    float3 ray_orig  = cam.position;

    float3 radiance   = 0.0;
    float3 throughput = 1.0;

    for (uint bounce = 0; bounce < pc.max_bounces; ++bounce)
    {
        HitInfo hit = trace(ray_orig, ray_dir, 0.001, 1e9);
        if (!hit.hit)
        {
            // Sky / environment — simple gradient
            float t = 0.5 * (ray_dir.y + 1.0);
            radiance += throughput * lerp(float3(0.1, 0.1, 0.1), float3(0.2, 0.3, 0.5), t);
            break;
        }

        GpuMaterial mat = gScene[pc.material_buf_idx].Load<GpuMaterial>(
            hit.material_index * int(sizeof(GpuMaterial)));

        // Emissive contribution
        if (any(mat.emissive > 0.0))
        {
            radiance += throughput * mat.emissive;
            break; // emissive surfaces don't scatter
        }

        float3 wo = -ray_dir;
        float3 n  = hit.normal;
        // Ensure normal faces the ray origin
        if (dot(n, wo) < 0.0) n = -n;

        // NEE direct lighting
        radiance += throughput * sample_lights(hit.pos, n, wo, mat, seed);

        // Sample BRDF for indirect lighting
        float3 wi;
        float  pdf;
        float3 f = brdf_sample(wo, n, mat.base_color.rgb,
                                mat.metallic, mat.roughness,
                                rand_float2(seed), wi, pdf);

        if (pdf < 1e-6) break;

        float n_dot_wi = max(dot(n, wi), 0.0);
        throughput *= f * n_dot_wi / pdf;

        // Russian roulette after bounce 3
        if (bounce >= 3)
        {
            float survive = min(max(throughput.r, max(throughput.g, throughput.b)), 0.95);
            if (rand_float(seed) > survive) break;
            throughput /= survive;
        }

        ray_orig = hit.pos + n * 1e-4;
        ray_dir  = wi;
    }

    // Write radiance
    gStorageImages[pc.radiance_img_idx][pixel] = float4(radiance, 1.0);
}
