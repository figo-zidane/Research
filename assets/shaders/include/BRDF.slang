// BRDF.slang
// GGX / Lambertian BRDF utilities used by path tracer and ReSTIR passes.
//
// Conventions:
//   - All vectors assumed unit-length unless noted otherwise.
//   - wo = outgoing direction (toward camera), wi = incoming (toward light).
//   - Normal n points outward from the surface.

static const float kPi      = 3.14159265358979323846f;
static const float kTwoPi   = 6.28318530717958647692f;
static const float kInvPi   = 0.31830988618379067154f;
static const float kEpsilon = 1e-4f;

// ── Fresnel ────────────────────────────────────────────────────────────────

float3 fresnel_schlick(float cos_theta, float3 F0)
{
    float x = clamp(1.0f - cos_theta, 0.0f, 1.0f);
    float x2 = x * x;
    return F0 + (1.0f - F0) * (x2 * x2 * x);
}

// ── GGX Distribution ───────────────────────────────────────────────────────

float distribution_ggx(float NdotH, float alpha)
{
    float a2 = alpha * alpha;
    float d  = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    return a2 / max(kPi * d * d, kEpsilon);
}

// ── Smith G (separable) ────────────────────────────────────────────────────

float geometry_smith_ggx(float NdotV, float NdotL, float alpha)
{
    float k  = alpha * 0.5f;
    float gv = NdotV / max(NdotV * (1.0f - k) + k, kEpsilon);
    float gl = NdotL / max(NdotL * (1.0f - k) + k, kEpsilon);
    return gv * gl;
}

// ── Combined Cook-Torrance specular ────────────────────────────────────────

float3 brdf_specular_ggx(float3 n, float3 wo, float3 wi,
                          float3 F0, float roughness)
{
    float alpha  = roughness * roughness;
    float3 h     = normalize(wo + wi);
    float NdotH  = max(dot(n, h),  0.0f);
    float NdotV  = max(dot(n, wo), 0.0f);
    float NdotL  = max(dot(n, wi), 0.0f);
    float VdotH  = max(dot(wo, h), 0.0f);

    float3 F = fresnel_schlick(VdotH, F0);
    float  D = distribution_ggx(NdotH, alpha);
    float  G = geometry_smith_ggx(NdotV, NdotL, alpha);

    return (D * G * F) / max(4.0f * NdotV * NdotL, kEpsilon);
}

// ── Lambertian diffuse ─────────────────────────────────────────────────────

float3 brdf_diffuse_lambert(float3 albedo)
{
    return albedo * kInvPi;
}

// ── Full PBR BRDF (metallic-roughness model) ───────────────────────────────
// Returns BRDF value (not multiplied by NdotL).

float3 brdf_eval(float3 n, float3 wo, float3 wi,
                 float3 base_color, float metallic, float roughness)
{
    // Clamp roughness to avoid degenerate GGX behaviour.
    float r   = clamp(roughness, 0.04f, 1.0f);
    float3 F0 = lerp(float3(0.04f, 0.04f, 0.04f), base_color, metallic);
    float3 kS = fresnel_schlick(max(dot(n, wo), 0.0f), F0);
    float3 kD = (1.0f - kS) * (1.0f - metallic);

    float3 diffuse  = kD * brdf_diffuse_lambert(base_color);
    float3 specular = brdf_specular_ggx(n, wo, wi, F0, r);
    return diffuse + specular;
}

// ── GGX importance-sampled half-vector ────────────────────────────────────

float3 sample_ggx_vndf(float3 wo_local, float roughness, float u0, float u1)
{
    float alpha = roughness * roughness;
    // Stretch viewing direction
    float3 Vh = normalize(float3(alpha * wo_local.x, alpha * wo_local.y, wo_local.z));
    // Orthonormal basis
    float lensq = dot(Vh.xy, Vh.xy);
    float3 T1 = (lensq > 0.0f) ? float3(-Vh.y, Vh.x, 0.0f) / sqrt(lensq)
                                : float3(1.0f, 0.0f, 0.0f);
    float3 T2 = cross(Vh, T1);
    // Disk sample
    float r2  = sqrt(u0);
    float phi = kTwoPi * u1;
    float t1  = r2 * cos(phi);
    float t2  = r2 * sin(phi);
    float s   = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrt(1.0f - t1 * t1) + s * t2;
    // Hemisphere point
    float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;
    // Unstretch
    return normalize(float3(alpha * Nh.x, alpha * Nh.y, max(0.0f, Nh.z)));
}

// ── BRDF sampling (metallic-roughness) ────────────────────────────────────
// Samples a wi direction given wo and surface parameters.
// Returns the BRDF value f(wo, wi) and outputs wi, pdf.

import "Sampling";  // build_onb, tangent_to_world, world_to_tangent

float3 brdf_sample(float3 wo, float3 n,
                    float3 base_color, float metallic, float roughness,
                    float2 u, out float3 wi, out float pdf)
{
    float r   = clamp(roughness, 0.04f, 1.0f);
    float3 F0 = lerp(float3(0.04f, 0.04f, 0.04f), base_color, metallic);

    // Fresnel at normal incidence decides specular vs diffuse probability
    float3 kS_approx = fresnel_schlick(max(dot(n, wo), 0.0f), F0);
    float  p_spec    = (kS_approx.r + kS_approx.g + kS_approx.b) / 3.0f;
    p_spec = clamp(p_spec, 0.1f, 0.9f);

    float3 T, B;
    build_onb(n, T, B);
    float3 wo_local = world_to_tangent(wo, T, B, n);

    float3 wi_local;
    float  pdf_local;

    if (u.x < p_spec)
    {
        // Specular GGX VNDF sample
        float2 u2 = float2(u.x / p_spec, u.y);
        float3 h_local = sample_ggx_vndf(wo_local, r, u2.x, u2.y);
        wi_local = reflect(-wo_local, h_local);
        if (wi_local.z <= 0.0f) { wi = float3(0,1,0); pdf = 1.0f; return 0.0f; }

        float NdotH = max(h_local.z, 1e-6f);
        float VdotH = max(dot(wo_local, h_local), 1e-6f);
        float D     = distribution_ggx(NdotH, r * r);
        pdf_local   = D * NdotH / max(4.0f * VdotH, 1e-6f);
    }
    else
    {
        // Diffuse cosine hemisphere
        float2 u2 = float2((u.x - p_spec) / (1.0f - p_spec), u.y);
        wi_local  = sample_cosine_hemisphere(u2.x, u2.y, pdf_local);
    }

    wi  = tangent_to_world(wi_local, T, B, n);
    pdf = lerp(pdf_local, pdf_local, 1.0f); // placeholder — below combines
    // Combine: pdf = p_spec * pdf_spec + (1-p_spec) * pdf_diff
    // We compute pdf properly below after sampling.
    float NdotWI = max(wi_local.z, 0.0f);

    // Compute pdf for both lobes at wi
    float3 h = normalize(wo_local + wi_local);
    float NdotH = max(h.z, 1e-6f);
    float VdotH = max(dot(wo_local, h), 1e-6f);
    float D     = distribution_ggx(NdotH, r * r);
    float pdf_spec_lobe = D * NdotH / max(4.0f * VdotH, 1e-6f);
    float pdf_diff_lobe = NdotWI * kInvPi;
    pdf = p_spec * pdf_spec_lobe + (1.0f - p_spec) * pdf_diff_lobe;
    pdf = max(pdf, 1e-6f);

    return brdf_eval(n, wo, wi, base_color, metallic, roughness);
}

// ── BRDF PDF evaluation ────────────────────────────────────────────────────

float brdf_pdf(float3 wo, float3 wi, float3 n, float roughness)
{
    float r  = clamp(roughness, 0.04f, 1.0f);
    float3 h = normalize(wo + wi);
    float NdotH = max(dot(n, h), 1e-6f);
    float VdotH = max(dot(wo, h), 1e-6f);
    float NdotL = max(dot(n, wi), 0.0f);
    float D     = distribution_ggx(NdotH, r * r);
    float pdf_spec = D * NdotH / max(4.0f * VdotH, 1e-6f);
    float pdf_diff = NdotL * kInvPi;
    return 0.5f * pdf_spec + 0.5f * pdf_diff;
}
