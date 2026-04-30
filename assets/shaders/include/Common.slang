// Common.slang
// 全パス共通の定数バッファ型と汎用ユーティリティ。

// フレームごとに更新される定数 (push constant で渡すか UBO で渡す)。
struct FrameConstants
{
    uint  frameIndex;
    float deltaTimeSeconds;
    uint  screenWidth;
    uint  screenHeight;
    float cameraNear;
    float cameraFar;
    uint  _pad0;
    uint  _pad1;
};

// 軽量な乱数生成 (PCG ベース)。
uint pcg_hash(uint seed)
{
    uint state = seed * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float uint_to_float01(uint v)
{
    return float(v >> 8) * (1.0 / float(1u << 24));
}

