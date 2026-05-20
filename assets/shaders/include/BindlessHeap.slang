// BindlessHeap.slang
// VK_EXT_descriptor_heap の untyped モードで全リソースを宣言するヘッダ。
// 全パスが #include "BindlessHeap.slang" の 1 行でリソース配列にアクセスできる。
//
// ヒープ構成 (C++ 側 BindlessRegistry と対応):
//   resource heap (set 0):
//     binding 0 → gTextures[]       (SAMPLED_IMAGE)
//     binding 1 → gStorageImages[]  (STORAGE_IMAGE)
//     binding 2 → gScene[]          (STORAGE_BUFFER, SceneData など)
//     binding 3 → gTLAS[]           (ACCELERATION_STRUCTURE)
//   sampler heap (set 1):
//     binding 0 → gSamplers[]       (SAMPLER)
//
// C++ 側 BindlessRegistry がヒープ内のオフセットと stride を管理し、
// Pipeline 生成時に VkDescriptorMappingSourceConstantOffsetEXT でこのバインディングと
// ヒープを紐付ける。
//
// SceneData の定義は GpuScene.slang で行い、そちらを先に include すること。

[[vk::binding(0, 0)]] Texture2D                       gTextures[];
[[vk::binding(1, 0)]] RWTexture2D<float4>              gStorageImages[];
[[vk::binding(2, 0)]] ByteAddressBuffer                gScene[];
// Writable alias of the same SSBO heap binding (set 0, binding 2). Used by
// passes that need atomic read-modify-write on a registered buffer (e.g. the
// ReSTIR PG world-space guiding grid). The buffer must be created with
// Storage usage. Index is the same gScene[] index from register_buffer().
[[vk::binding(2, 0)]] RWByteAddressBuffer              gSceneRW[];
[[vk::binding(3, 0)]] RaytracingAccelerationStructure  gTLAS[];
[[vk::binding(0, 1)]] SamplerState                     gSamplers[];
