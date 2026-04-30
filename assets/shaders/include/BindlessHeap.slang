// BindlessHeap.slang
// VK_EXT_descriptor_heap の untyped モードで全リソースを宣言するヘッダ。
// 全パスが #include "BindlessHeap.slang" の 1 行でリソース配列にアクセスできる。
//
// ヒープ構成 (C++ 側 BindlessRegistry と対応):
//   resource heap: gTextures / gStorageImages / gScene / gTLAS
//   sampler heap : gSamplers

layout(descriptor_heap) Texture2D gTextures[];
layout(descriptor_heap) RWTexture2D<float4> gStorageImages[];
layout(descriptor_heap) SamplerState gSamplers[];
// SceneData の定義は GpuScene.slang で行い、そちらを先に include すること。
// layout(descriptor_heap) StructuredBuffer<SceneData> gScene[];
layout(descriptor_heap) RaytracingAccelerationStructure gTLAS[];
