#pragma once

#include "scene/Scene.h"

#include <filesystem>

namespace rr::rhi
{
class Device;
class BindlessRegistry;
} // namespace rr::rhi

namespace rr::scene
{

// GltfLoader loads a glTF 2.0 file and populates a Scene with the
// parsed geometry, materials, and instances.
//
// Notes:
//  - Only the first scene in the glTF is loaded.
//  - Textures are uploaded using Image::upload_host (hostImageCopy path).
//  - Animation / skinning / morph targets are not supported.
//  - The caller must subsequently call scene.upload() to push GPU data.
class GltfLoader
{
public:
    // Load a glTF file (binary .glb or JSON .gltf) into a Scene.
    // Returns true on success, false and logs an error on failure.
    static bool load(const std::filesystem::path& path, Scene& scene);
};

} // namespace rr::scene
