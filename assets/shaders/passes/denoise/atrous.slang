#include "BindlessHeap.slang"

struct PushConstants
{
    uint  input_idx;
    uint  output_idx;
    uint  gbuf_pos_idx;
    uint  gbuf_norm_idx;
    uint  screen_width;
    uint  screen_height;
    uint  iter;
    uint  stride;
    float sigma_position;
    float sigma_normal;
    float sigma_luminance;
    uint  _pad;
};

[[vk::push_constant]] PushConstants pc;

float luminance(float3 c)
{
    return dot(c, float3(0.299f, 0.587f, 0.114f));
}

float kernel_weight(int offset)
{
    switch (offset)
    {
    case -2:
    case 2:
        return 1.0f;
    case -1:
    case 1:
        return 4.0f;
    default:
        return 6.0f;
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= pc.screen_width || id.y >= pc.screen_height) return;

    const uint2 pixel = id.xy;
    const float4 center_pos = gStorageImages[pc.gbuf_pos_idx][pixel];
    const float4 center_norm = gStorageImages[pc.gbuf_norm_idx][pixel];
    const float4 center_radiance = gTextures[pc.input_idx].Load(int3(pixel, 0));

    if (center_pos.w < 0.0f)
    {
        gStorageImages[pc.output_idx][pixel] = center_radiance;
        return;
    }

    const float3 center_normal = normalize(center_norm.xyz);
    const float  center_luma = luminance(center_radiance.rgb);
    const float  sigma_pos_sq = max(pc.sigma_position * pc.sigma_position, 1.0e-6f);
    const float  sigma_norm_sq = max(pc.sigma_normal * pc.sigma_normal, 1.0e-6f);
    const float  sigma_luma_sq = max(pc.sigma_luminance * pc.sigma_luminance, 1.0e-6f);

    float3 filtered = 0.0f;
    float  weight_sum = 0.0f;
    const int2 max_pixel = int2(int(pc.screen_width) - 1, int(pc.screen_height) - 1);
    const int stride = int(pc.stride);

    for (int tap_y = -2; tap_y <= 2; ++tap_y)
    {
        for (int tap_x = -2; tap_x <= 2; ++tap_x)
        {
            int2 sample_pixel = int2(pixel) + int2(tap_x * stride, tap_y * stride);
            sample_pixel = clamp(sample_pixel, int2(0, 0), max_pixel);

            const float4 sample_pos = gStorageImages[pc.gbuf_pos_idx][sample_pixel];
            if (sample_pos.w < 0.0f)
                continue;

            const float4 sample_norm = gStorageImages[pc.gbuf_norm_idx][sample_pixel];
            const float4 sample_radiance = gTextures[pc.input_idx].Load(int3(sample_pixel, 0));

            const float base_w = (kernel_weight(tap_x) * kernel_weight(tap_y)) / 256.0f;

            const float3 pos_delta = sample_pos.xyz - center_pos.xyz;
            const float pos_w = exp(-dot(pos_delta, pos_delta) / sigma_pos_sq);

            const float3 sample_normal = normalize(sample_norm.xyz);
            const float normal_delta = max(0.0f, 1.0f - dot(center_normal, sample_normal));
            const float normal_w = exp(-normal_delta / sigma_norm_sq);

            const float sample_luma = luminance(sample_radiance.rgb);
            const float luma_delta = sample_luma - center_luma;
            const float luma_w = exp(-(luma_delta * luma_delta) / sigma_luma_sq);

            const float weight = base_w * pos_w * normal_w * luma_w;
            filtered += sample_radiance.rgb * weight;
            weight_sum += weight;
        }
    }

    const float3 output_rgb = (weight_sum > 0.0f)
        ? (filtered / weight_sum)
        : center_radiance.rgb;
    gStorageImages[pc.output_idx][pixel] = float4(output_rgb, center_radiance.a);
}