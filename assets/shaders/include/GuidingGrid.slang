// GuidingGrid.slang — ReSTIR PG world-space guiding grid.
//
// A non-uniform world-space hash grid storing one von Mises-Fisher (vMF) lobe
// per cell. Cells are addressed by quantizing the world position with a
// resolution that adapts to viewing distance (à la [Pantaleoni 2020]) and by
// mixing a BSDF id into the hash so cells with different materials do not
// share a directional distribution (paper §4.2).
//
// We keep TWO buffers (ping-pong handled on the C++ side):
//
//   accum buffer  — integer fixed-point accumulators written via atomics
//                   during the splat stage, read by the fit stage. Layout
//                   per cell (kAccumStride bytes):
//                     int  acc_dir_x   (Σ ω_i.x in fixed point)
//                     int  acc_dir_y
//                     int  acc_dir_z
//                     uint acc_count   (number of splatted directions)
//
//   fitted buffer — float vMF parameters written by the fit stage, read by
//                   the guide stage on the *next* frame. Layout per cell
//                   (kFittedStride bytes):
//                     float mu_x
//                     float mu_y
//                     float mu_z
//                     float kappa     (<= 0  → cell has no usable lobe)
//
// Float atomics are not available on RWByteAddressBuffer, so directions are
// accumulated as scaled integers (kFixedScale) and converted back in the fit
// stage. Each component of a unit direction is in [-1,1]; with a per-cell
// budget of a few thousand splats kFixedScale = 2^20 keeps well clear of
// 32-bit overflow.

import "Common";    // pcg_hash
import "Sampling";  // build_onb, tangent_to_world, world_to_tangent

static const uint  kAccumStride   = 16u;   // 3*int + 1*uint
static const uint  kFittedStride  = 16u;   // 4*float
static const float kFixedScale    = 1048576.0f;          // 2^20
static const float kInvFixedScale = 1.0f / 1048576.0f;

// Grid sizing. The cell edge grows linearly with the distance from the
// camera so that distant regions use coarser cells (cheaper, still adequate
// because they cover fewer pixels). base_cell_size and the slope are tuned on
// the C++ side and passed via push constants.
struct GuidingGridParams
{
    float base_cell_size;   // world-space edge at the camera
    float distance_slope;   // additional edge per unit camera distance
    uint  cell_count;       // hash table size (power of two recommended)
    uint  enabled;          // 0 = guiding disabled (pure BSDF sampling)
};

// Quantize a world position + bsdf id into a cell index in [0, cell_count).
uint guiding_cell_index(float3 world_pos, uint bsdf_id,
                        float3 camera_pos, GuidingGridParams gp)
{
    float view_dist = length(world_pos - camera_pos);
    float cell      = max(gp.base_cell_size + gp.distance_slope * view_dist,
                          1.0e-4f);
    int3  q = int3(floor(world_pos / cell));

    uint h = pcg_hash(uint(q.x) * 73856093u
                    ^ uint(q.y) * 19349663u
                    ^ uint(q.z) * 83492791u
                    ^ bsdf_id   * 2654435761u);
    return h % max(gp.cell_count, 1u);
}

// ── Splat ──────────────────────────────────────────────────────────────────
// Atomically accumulate one incident direction into the cell's accumulator.
// `dir` is the world-space incident direction at the vertex (toward the next
// path vertex / incoming radiance direction), expected unit length.
void guiding_splat(uint accum_buf_idx, uint cell, float3 dir)
{
    uint base = cell * kAccumStride;
    // Two's-complement integer add is bit-identical for signed/unsigned, so
    // accumulate the (possibly negative) fixed-point components as uint and
    // reinterpret with asint() during the fit. This keeps every atomic the
    // same type (Slang's RWByteAddressBuffer.InterlockedAdd requires the
    // value and out-original to match).
    uint dx = asuint(int(round(dir.x * kFixedScale)));
    uint dy = asuint(int(round(dir.y * kFixedScale)));
    uint dz = asuint(int(round(dir.z * kFixedScale)));

    uint old;
    gSceneRW[accum_buf_idx].InterlockedAdd(base + 0u,  dx, old);
    gSceneRW[accum_buf_idx].InterlockedAdd(base + 4u,  dy, old);
    gSceneRW[accum_buf_idx].InterlockedAdd(base + 8u,  dz, old);
    gSceneRW[accum_buf_idx].InterlockedAdd(base + 12u, 1u, old);
}

// ── Fit ────────────────────────────────────────────────────────────────────
// Banerjee et al. maximum-likelihood estimate of a single vMF lobe from the
// mean resultant length r = |Σω| / N:
//   mu    = normalize(Σω)
//   kappa = r (3 - r^2) / (1 - r^2)
// with clamping/regularization for robustness (paper §4.4 refers to the
// supplemental; this is the standard closed form). Writes the fitted lobe to
// the fitted buffer and resets the accumulator for the next frame.
void guiding_fit_cell(uint accum_buf_idx, uint fitted_buf_idx, uint cell,
                      uint min_samples, float kappa_max)
{
    uint abase = cell * kAccumStride;
    uint fbase = cell * kFittedStride;

    int  ix = asint(gSceneRW[accum_buf_idx].Load<uint>(abase + 0u));
    int  iy = asint(gSceneRW[accum_buf_idx].Load<uint>(abase + 4u));
    int  iz = asint(gSceneRW[accum_buf_idx].Load<uint>(abase + 8u));
    uint n  = gSceneRW[accum_buf_idx].Load<uint>(abase + 12u);

    // Reset accumulator for the next frame regardless of outcome.
    gSceneRW[accum_buf_idx].Store<uint>(abase + 0u,  0u);
    gSceneRW[accum_buf_idx].Store<uint>(abase + 4u,  0u);
    gSceneRW[accum_buf_idx].Store<uint>(abase + 8u,  0u);
    gSceneRW[accum_buf_idx].Store<uint>(abase + 12u, 0u);

    if (n < max(min_samples, 1u))
    {
        // Not enough evidence — mark cell unusable.
        gSceneRW[fitted_buf_idx].Store<float>(fbase + 12u, -1.0f);
        return;
    }

    float3 sum = float3(float(ix), float(iy), float(iz)) * kInvFixedScale;
    float  len = length(sum);
    float  r   = len / float(n);
    r = clamp(r, 1.0e-4f, 0.9999f);

    float3 mu    = (len > 1.0e-6f) ? (sum / len) : float3(0.0f, 0.0f, 1.0f);
    float  kappa = r * (3.0f - r * r) / (1.0f - r * r);
    kappa = clamp(kappa, 0.5f, kappa_max);

    gSceneRW[fitted_buf_idx].Store<float>(fbase + 0u,  mu.x);
    gSceneRW[fitted_buf_idx].Store<float>(fbase + 4u,  mu.y);
    gSceneRW[fitted_buf_idx].Store<float>(fbase + 8u,  mu.z);
    gSceneRW[fitted_buf_idx].Store<float>(fbase + 12u, kappa);
}

// ── vMF sample / pdf (read fitted lobe) ───────────────────────────────────
struct GuidingLobe
{
    float3 mu;
    float  kappa;
    bool   valid;
};

GuidingLobe guiding_load_lobe(uint fitted_buf_idx, uint cell)
{
    uint fbase = cell * kFittedStride;
    GuidingLobe lobe;
    lobe.mu    = float3(gSceneRW[fitted_buf_idx].Load<float>(fbase + 0u),
                        gSceneRW[fitted_buf_idx].Load<float>(fbase + 4u),
                        gSceneRW[fitted_buf_idx].Load<float>(fbase + 8u));
    lobe.kappa = gSceneRW[fitted_buf_idx].Load<float>(fbase + 12u);
    lobe.valid = (lobe.kappa > 0.0f) && (dot(lobe.mu, lobe.mu) > 0.25f);
    return lobe;
}

// Sample a direction from a vMF(mu, kappa) lobe. u in [0,1)^2.
float3 vmf_sample(float3 mu, float kappa, float2 u, out float pdf)
{
    // Numerically stable inverse CDF for cos(theta) around +z, then rotate.
    float k = max(kappa, 1.0e-3f);
    float w = 1.0f + (1.0f / k) * log(u.x + (1.0f - u.x) * exp(-2.0f * k));
    w = clamp(w, -1.0f, 1.0f);
    float s   = sqrt(max(0.0f, 1.0f - w * w));
    float phi = 6.28318530717958647692f * u.y;

    float3 local = float3(s * cos(phi), s * sin(phi), w);

    float3 T, B;
    build_onb(mu, T, B);
    float3 dir = tangent_to_world(local, T, B, mu);

    // pdf(w) = k / (4π sinh k) e^{k w};  (with normalization for kappa>0)
    // Use the stable form e^{k(w-1)} * k / (2π (1 - e^{-2k})).
    float denom = 6.28318530717958647692f * (1.0f - exp(-2.0f * k));
    pdf = k * exp(k * (w - 1.0f)) / max(denom, 1.0e-8f);
    return dir;
}

// pdf of direction `dir` under vMF(mu, kappa).
float vmf_pdf(float3 mu, float kappa, float3 dir)
{
    float k = max(kappa, 1.0e-3f);
    float w = clamp(dot(mu, dir), -1.0f, 1.0f);
    float denom = 6.28318530717958647692f * (1.0f - exp(-2.0f * k));
    return k * exp(k * (w - 1.0f)) / max(denom, 1.0e-8f);
}
